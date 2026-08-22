import argparse
import json
import logging
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from monai.data.utils import list_data_collate

from .utils import (
    AverageMeter,
    ProgressMeter,
    configure_logger,
    exp_lr_scheduler_with_warmup,
    filter_validation_results,
    get_optimizer,
    load_fold_filenames,
    log_evaluation_result,
    resume_load_model_checkpoint,
    resume_load_optimizer_checkpoint,
    save_configure,
    update_ema_variables,
)
from .losses import DiceLoss
from .validation import validation

from ..model.utils import get_model
from ..dataset.pants_dataset import PanTSDataset

warnings.filterwarnings("ignore", category=UserWarning)

REPO_SRC = Path(__file__).resolve().parent.parent


def train_net(net, args, ema_net=None, fold_idx=0):

    ################################################################################
    # Dataset Creation
    data_dir = REPO_SRC / "data"
    train_filenames, test_filenames = load_fold_filenames(data_dir, args.k_fold, fold_idx)
    fingerprint = json.loads((data_dir / "fingerprint.json").read_text())

    trainset = PanTSDataset(train_filenames, Path(args.cache_dir).expanduser(), fingerprint, train=True)

    trainLoader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(args.aug_device != 'gpu'),
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        collate_fn=list_data_collate,  # RandCropByPosNegLabeld(num_samples>1) returns a list of crops per index
    )

    testset = PanTSDataset(test_filenames, Path(args.cache_dir).expanduser(), fingerprint, train=False)
    testLoader = DataLoader(testset, batch_size=1, pin_memory=True, shuffle=False, num_workers=2)

    logging.info("Created Dataset and DataLoader")

    ################################################################################
    # Initialize tensorboard, optimizer and etc
    writer = SummaryWriter(f"{args.log_path}{args.unique_name}/fold_{fold_idx}")

    optimizer = get_optimizer(args, net)

    if args.resume:
        resume_load_optimizer_checkpoint(optimizer, args)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight)).cuda()
    criterion_dl = DiceLoss()

    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    ################################################################################
    # Start training
    best_Dice = np.zeros(args.classes)
    best_HD = np.ones(args.classes) * 1000
    best_ASD = np.ones(args.classes) * 1000

    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
        exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=args.base_lr, epoch=epoch, warmup_epoch=args.warmup_epoch, max_epoch=args.epochs)
        logging.info(f"Current lr: {exp_scheduler:.4e}")

        train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, criterion, criterion_dl, scaler, args)

        ########################################################################################
        # Evaluation, save checkpoint and log training info
        net_for_eval = ema_net if args.ema else net

        # save the latest checkpoint, including net, ema_net, and optimizer
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict() if not args.torch_compile else net._orig_mod.state_dict(),
            'ema_model_state_dict': ema_net.state_dict() if args.ema else None,
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{args.cp_dir}/fold_{fold_idx}_latest.pth")

        if (epoch + 1) % args.val_freq == 0:

            dice_list_test, ASD_list_test, HD_list_test = validation(net_for_eval, testLoader, args)
            dice_list_test, ASD_list_test, HD_list_test = filter_validation_results(dice_list_test, ASD_list_test, HD_list_test, args)
            log_evaluation_result(writer, dice_list_test, ASD_list_test, HD_list_test, 'test', epoch, args)

            if dice_list_test.mean() >= best_Dice.mean():
                best_Dice = dice_list_test
                best_HD = HD_list_test
                best_ASD = ASD_list_test

                # Save the checkpoint with best performance
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': net.state_dict() if not args.torch_compile else net._orig_mod.state_dict(),
                    'ema_model_state_dict': ema_net.state_dict() if args.ema else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{args.cp_dir}/fold_{fold_idx}_best.pth")

            logging.info("Evaluation Done")
            logging.info(f"Dice: {dice_list_test.mean():.4f}/Best Dice: {best_Dice.mean():.4f}")

        writer.add_scalar('LR', exp_scheduler, epoch + 1)

    return best_Dice, best_HD, best_ASD


def train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, criterion, criterion_dl, scaler, args):
    batch_time = AverageMeter("Time", ":6.2f")
    epoch_loss = AverageMeter("Loss", ":.2f")
    progress = ProgressMeter(
        len(trainLoader) if args.dimension == '2d' else args.iter_per_epoch,
        [batch_time, epoch_loss],
        prefix="Epoch: [{}]".format(epoch + 1),
    )

    net.train()

    tic = time.time()
    iter_num_per_epoch = 0
    for i, inputs in enumerate(trainLoader):
        img, label = inputs["image"], inputs["label"].float()
        if args.aug_device != 'gpu':
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

        step = i + epoch * len(trainLoader)  # global steps

        optimizer.zero_grad()

        if args.amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result = net(img)

                loss = 0

                if isinstance(result, tuple) or isinstance(result, list):
                    # if use deep supervision, add all loss together
                    for j in range(len(result)):
                        loss += args.aux_weight[j] * (criterion(result[j], label) + criterion_dl(result[j], label))
                else:
                    loss = criterion(result, label) + criterion_dl(result, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            result = net(img)
            loss = 0
            if isinstance(result, tuple) or isinstance(result, list):
                # If use deep supervision, add all loss together
                for j in range(len(result)):
                    loss += args.aux_weight[j] * (criterion(result[j], label) + criterion_dl(result[j], label))
            else:
                loss = criterion(result, label) + criterion_dl(result, label)

            loss.backward()
            optimizer.step()

        if args.ema:
            update_ema_variables(net, ema_net, args.ema_alpha, step)

        epoch_loss.update(loss.item(), img.shape[0])
        batch_time.update(time.time() - tic)
        tic = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        if args.dimension == '3d':
            iter_num_per_epoch += 1
            if iter_num_per_epoch > args.iter_per_epoch:
                break

    writer.add_scalar('Train/Loss', epoch_loss.avg, epoch + 1)


def get_parser():
    parser = argparse.ArgumentParser(description='PanTS Pancreatic Tumor Segmentation')
    parser.add_argument('--dataset', type=str, default='pants', help='dataset name')
    parser.add_argument('--model', type=str, default='medformer', help='model name')
    parser.add_argument('--dimension', type=str, default='3d', help='2d model or 3d model')
    parser.add_argument('--pretrain', action='store_true', help='if use pretrained weight for init')
    parser.add_argument('--amp', action='store_true', help='if use automatic mixed precision for faster training')
    parser.add_argument('--torch_compile', action='store_true', help='use torch.compile, only supported by pytorch2.0')

    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--resume', action='store_true', help='if resume training from checkpoint')
    parser.add_argument('--load', type=str, default=False, help='checkpoint path, used with --resume or --pretrain')
    parser.add_argument('--cp_path', type=str, default='./exp/', help='checkpoint path')
    parser.add_argument('--log_path', type=str, default='./log/', help='log path')
    parser.add_argument('--unique_name', type=str, default='test', help='unique experiment name')

    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    config_path = REPO_SRC / "config" / args.dataset / f"{args.model}_{args.dimension}.yaml"
    if not config_path.exists():
        raise ValueError(f"The specified configuration doesn't exist: {config_path}")

    print(f"Loading configurations from {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)

    args.start_epoch = 0

    return args


def init_network(args):
    net = get_model(args, pretrain=args.pretrain)

    if args.ema:
        ema_net = get_model(args, pretrain=args.pretrain)
        for p in ema_net.parameters():
            p.requires_grad_(False)
        logging.info("Use EMA model for evaluation")
    else:
        ema_net = None

    if args.resume:
        resume_load_model_checkpoint(net, ema_net, args)

    if args.torch_compile:
        net = torch.compile(net)
    return net, ema_net


if __name__ == '__main__':

    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')

    args.log_path = args.log_path + '%s/' % args.dataset

    if args.reproduce_seed is not None:
        random.seed(args.reproduce_seed)
        np.random.seed(args.reproduce_seed)
        torch.manual_seed(args.reproduce_seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    Dice_list, HD_list, ASD_list = [], [], []

    for fold_idx in range(args.k_fold):

        args.cp_dir = f"{args.cp_path}/{args.dataset}/{args.unique_name}"
        os.makedirs(args.cp_dir, exist_ok=True)
        configure_logger(0, args.cp_dir + f"/fold_{fold_idx}.txt")
        save_configure(args)
        logging.info(
            f"\nDataset: {args.dataset},\n"
            + f"Model: {args.model},\n"
            + f"Dimension: {args.dimension}"
        )

        net, ema_net = init_network(args)

        net.cuda()
        if args.ema:
            ema_net.cuda()
        logging.info("Created Model")
        best_Dice, best_HD, best_ASD = train_net(net, args, ema_net, fold_idx=fold_idx)

        logging.info(f"Training and evaluation on Fold {fold_idx} is done")

        Dice_list.append(best_Dice)
        HD_list.append(best_HD)
        ASD_list.append(best_ASD)

    ############################################################################################
    # Save the cross validation results
    total_Dice = np.vstack(Dice_list)
    total_HD = np.vstack(HD_list)
    total_ASD = np.vstack(ASD_list)

    with open(f"{args.cp_path}/{args.dataset}/{args.unique_name}/cross_validation.txt", 'w') as f:
        np.set_printoptions(precision=4, suppress=True)
        f.write('Dice\n')
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {Dice_list[i]}\n")
        f.write(f"Each Class Dice Avg: {np.mean(total_Dice, axis=0)}\n")
        f.write(f"Each Class Dice Std: {np.std(total_Dice, axis=0)}\n")
        f.write(f"All classes Dice Avg: {total_Dice.mean()}\n")
        f.write(f"All classes Dice Std: {np.mean(total_Dice, axis=1).std()}\n")

        f.write("\n")

        f.write("HD\n")
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {HD_list[i]}\n")
        f.write(f"Each Class HD Avg: {np.mean(total_HD, axis=0)}\n")
        f.write(f"Each Class HD Std: {np.std(total_HD, axis=0)}\n")
        f.write(f"All classes HD Avg: {total_HD.mean()}\n")
        f.write(f"All classes HD Std: {np.mean(total_HD, axis=1).std()}\n")

        f.write("\n")

        f.write("ASD\n")
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {ASD_list[i]}\n")
        f.write(f"Each Class ASD Avg: {np.mean(total_ASD, axis=0)}\n")
        f.write(f"Each Class ASD Std: {np.std(total_ASD, axis=0)}\n")
        f.write(f"All classes ASD Avg: {total_ASD.mean()}\n")
        f.write(f"All classes ASD Std: {np.mean(total_ASD, axis=1).std()}\n")

    print(f'All {args.k_fold} folds done.')

    sys.exit(0)
