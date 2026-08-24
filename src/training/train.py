import argparse
import json
import logging
import os
import random
import sys
import time
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
    unwrap_model,
    update_ema_variables,
    write_cross_validation_results,
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

    trainset = PanTSDataset(
        train_filenames, Path(args.cache_dir).expanduser(), fingerprint, train=True,
        spatial_size=args.spatial_size, pos=args.pos, neg=args.neg, num_samples=args.num_samples,
    )

    # each rank's DataLoader only sees its own shard of the dataset under DDP; batch_size
    # below is therefore per-GPU -- the effective global batch is args.batch_size * world_size
    train_sampler = DistributedSampler(trainset, shuffle=True) if args.distributed else None
    trainLoader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=(args.aug_device != 'gpu'),
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        collate_fn=list_data_collate,  # RandCropByPosNegLabeld(num_samples>1) returns a list of crops per index
    )

    val_filenames = test_filenames
    if getattr(args, 'val_subset_size', None):
        # fixed seed so every validation pass in this run samples the same subset --
        # otherwise "Dice improved" comparisons across epochs would be apples-to-oranges
        val_filenames = random.Random(42).sample(test_filenames, min(args.val_subset_size, len(test_filenames)))
        logging.info(f"Validating on a fixed {len(val_filenames)}-case subset of the {len(test_filenames)}-case test fold")

    testset = PanTSDataset(val_filenames, Path(args.cache_dir).expanduser(), fingerprint, train=False)
    testLoader = DataLoader(testset, batch_size=1, pin_memory=True, shuffle=False, num_workers=2)

    logging.info("Created Dataset and DataLoader")

    ################################################################################
    # Initialize tensorboard, optimizer and etc
    # validation/checkpointing/tensorboard only happen on rank 0 -- every rank has an
    # identical copy of the full testset, so re-running eval on every rank would be
    # redundant work (and N processes writing the same files would race)
    writer = SummaryWriter(f"{args.log_path}{args.unique_name}/fold_{fold_idx}") if args.rank == 0 else None

    optimizer = get_optimizer(args, net)

    if args.resume:
        resume_load_optimizer_checkpoint(optimizer, args)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight)).cuda()
    criterion_dl = DiceLoss()

    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    ################################################################################
    # Start training
    best_results = {
        'dice': np.zeros(args.classes),
        'nsd': np.zeros(args.classes),
        'hd95': np.ones(args.classes) * 1000,
        'asd': np.ones(args.classes) * 1000,
        'sensitivity': 0.0,
        'specificity': 0.0,
        'f1': 0.0,
        'auc': 0.5,  # random-chance baseline
    }

    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # reshuffles differently per epoch across ranks

        logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
        exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=args.base_lr, epoch=epoch, warmup_epoch=args.warmup_epoch, max_epoch=args.epochs)
        logging.info(f"Current lr: {exp_scheduler:.4e}")

        train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, criterion, criterion_dl, scaler, args)

        ########################################################################################
        # Evaluation, save checkpoint and log training info
        net_for_eval = ema_net if args.ema else net

        if args.rank == 0:
            # save the latest checkpoint, including net, ema_net, and optimizer
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': unwrap_model(net).state_dict(),
                'ema_model_state_dict': unwrap_model(ema_net).state_dict() if args.ema else None,
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{args.cp_dir}/fold_{fold_idx}_latest.pth")

        if (epoch + 1) % args.val_freq == 0:

            if args.rank == 0:
                results = validation(unwrap_model(net_for_eval), testLoader, args)
                results = filter_validation_results(results, args)
                log_evaluation_result(writer, results, 'test', epoch, args)

                if results['dice'].mean() >= best_results['dice'].mean():
                    best_results = results

                    # Save the checkpoint with best performance
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': unwrap_model(net).state_dict(),
                        'ema_model_state_dict': unwrap_model(ema_net).state_dict() if args.ema else None,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f"{args.cp_dir}/fold_{fold_idx}_best.pth")

                logging.info("Evaluation Done")
                logging.info(f"Dice: {results['dice'].mean():.4f}/Best Dice: {best_results['dice'].mean():.4f}")

            if args.distributed:
                # rank 0 is off doing validation + checkpointing (no DDP collective calls
                # happen during that); without this, other ranks would race ahead into next
                # epoch's training before rank 0 rejoins
                dist.barrier()

        if args.rank == 0:
            writer.add_scalar('LR', exp_scheduler, epoch + 1)

    if writer is not None:
        writer.close()

    return best_results


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

    if writer is not None:
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
    parser.add_argument('--spatial_size', type=int, nargs=3, default=None, help='override yaml spatial_size, e.g. --spatial_size 96 96 96')
    parser.add_argument('--num_samples', type=int, default=None, help='override yaml num_samples (crops per case)')
    parser.add_argument('--sw_batch_size', type=int, default=None, help='override yaml sw_batch_size (sliding-window inference batch, validation only)')
    parser.add_argument('--epochs', type=int, default=None, help='override yaml epochs')
    parser.add_argument('--val_freq', type=int, default=None, help='override yaml val_freq (epochs between validation passes)')
    parser.add_argument('--fold_idx', type=int, default=None, help='run only this one fold (0-indexed) instead of looping over all k_fold folds')
    parser.add_argument('--start_fold', type=int, default=None, help='resume the fold loop from this index (0-indexed) through k_fold-1, instead of starting at 0. Ignored if --fold_idx is set.')
    parser.add_argument('--pos_weight', type=float, default=None, help='override yaml pos_weight (BCE positive-class weight)')
    parser.add_argument('--val_subset_size', type=int, default=None, help='validate on a random subset of this many test cases instead of the full fold (fixed seed, same subset each val pass this run)')
    parser.add_argument('--resume', action='store_true', help='if resume training from checkpoint')
    parser.add_argument('--load', type=str, default=False, help='checkpoint path, used with --resume or --pretrain')
    parser.add_argument('--cp_path', type=str, default='./exp/', help='checkpoint path')
    parser.add_argument('--log_path', type=str, default='./log/', help='log path')
    parser.add_argument('--unique_name', type=str, default='test', help='unique experiment name')

    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()
    # snapshot before the yaml loop below unconditionally overwrites same-named attributes
    cli_spatial_size = args.spatial_size
    cli_num_samples = args.num_samples
    cli_sw_batch_size = args.sw_batch_size
    cli_epochs = args.epochs
    cli_val_freq = args.val_freq
    cli_pos_weight = args.pos_weight
    # val_subset_size has no yaml key to clobber it, no snapshot needed

    config_path = REPO_SRC / "config" / args.dataset / f"{args.model}_{args.dimension}.yaml"
    if not config_path.exists():
        raise ValueError(f"The specified configuration doesn't exist: {config_path}")

    print(f"Loading configurations from {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)

    # CLI overrides win over the yaml
    if cli_spatial_size is not None:
        args.spatial_size = list(cli_spatial_size)
    if cli_num_samples is not None:
        args.num_samples = cli_num_samples
    if cli_sw_batch_size is not None:
        args.sw_batch_size = cli_sw_batch_size
    if cli_epochs is not None:
        args.epochs = cli_epochs
    if cli_val_freq is not None:
        args.val_freq = cli_val_freq
    if cli_pos_weight is not None:
        args.pos_weight = cli_pos_weight

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
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')

    # torchrun sets WORLD_SIZE/RANK/LOCAL_RANK; a plain `python -m src.training.train` run
    # leaves them unset, so this auto-detects single-GPU vs DDP with no extra flag needed
    args.distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if args.distributed:
        # default NCCL collective timeout is 10 minutes -- too short for the dist.barrier()
        # after full-fold validation (rank 0 alone runs sliding-window inference over the
        # whole test fold while other ranks wait there), which can easily take longer
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=3))
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        args.local_rank = 0
        args.rank = 0
        args.world_size = 1
        device = torch.device('cuda')

    args.log_path = args.log_path + '%s/' % args.dataset

    if args.reproduce_seed is not None:
        seed = args.reproduce_seed + args.rank  # avoid identical augmentation streams across ranks
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    fold_results = []
    if args.fold_idx is not None:
        fold_indices = [args.fold_idx]
    elif args.start_fold is not None:
        fold_indices = range(args.start_fold, args.k_fold)
    else:
        fold_indices = range(args.k_fold)

    for fold_idx in fold_indices:

        args.cp_dir = f"{args.cp_path}/{args.dataset}/{args.unique_name}"
        if args.rank == 0:
            os.makedirs(args.cp_dir, exist_ok=True)
        configure_logger(args.rank, args.cp_dir + f"/fold_{fold_idx}.txt")
        if args.rank == 0:
            save_configure(args)
        logging.info(
            f"\nDataset: {args.dataset},\n"
            + f"Model: {args.model},\n"
            + f"Dimension: {args.dimension},\n"
            + f"Distributed: {args.distributed} (world_size={args.world_size})"
        )

        net, ema_net = init_network(args)

        net.to(device)
        if args.ema:
            ema_net.to(device)

        if args.distributed:
            # no-op for InstanceNorm3d (this config's default) -- only matters if norm: bn
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = DistributedDataParallel(net, device_ids=[args.local_rank])
            # ema_net is intentionally left unwrapped: it's updated by directly averaging
            # .data in-place (update_ema_variables), never through backward(), so it needs
            # no gradient sync. DDP already broadcasts net's initial weights from rank 0;
            # only rank 0's ema_net is ever read (validation/checkpointing are rank-0-only),
            # so the other ranks' independently-initialized ema_net copies are harmless.

        logging.info("Created Model")
        best_results = train_net(net, args, ema_net, fold_idx=fold_idx)

        logging.info(f"Training and evaluation on Fold {fold_idx} is done")

        fold_results.append(best_results)

    ############################################################################################
    # Save the cross validation results
    # only rank 0 ever runs validation (see train_net), so only its fold_results are real
    if args.rank == 0:
        write_cross_validation_results(fold_results, args.cp_path, args.dataset, args.unique_name, len(fold_results))
        print(f'All {len(fold_results)} fold(s) done.')

    if args.distributed:
        dist.destroy_process_group()

    sys.exit(0)
