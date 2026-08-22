import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import optim


def load_fold_filenames(data_dir, k_fold, test_fold_idx):
    folds = [
        (Path(data_dir) / f"fold_{i}_ids.txt").read_text().split()
        for i in range(1, k_fold + 1)
    ]
    test_filenames = folds[test_fold_idx]
    train_filenames = [fid for i, fold in enumerate(folds) if i != test_fold_idx for fid in fold]
    return train_filenames, test_filenames


def get_optimizer(args, net):
    return optim.AdamW(
        net.parameters(),
        lr=args.base_lr,
        betas=getattr(args, 'betas', (0.9, 0.999)),
        weight_decay=getattr(args, 'weight_decay', 0.05),
        eps=getattr(args, 'eps', 1e-5),  # larger eps has better stability during AMP training
    )


def exp_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, max_epoch, power=0.9):
    if epoch < warmup_epoch:
        lr = init_lr * (epoch + 1) / warmup_epoch
    else:
        lr = init_lr * (1 - (epoch - warmup_epoch) / (max_epoch - warmup_epoch)) ** power

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def update_ema_variables(net, ema_net, alpha, global_step):
    alpha = min(alpha, 1 - 1 / (global_step + 1))
    for ema_param, param in zip(ema_net.parameters(), net.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    for ema_buffer, buffer in zip(ema_net.buffers(), net.buffers()):
        ema_buffer.data.copy_(buffer.data)


class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def configure_logger(rank, log_file=None):
    if rank != 0:
        return

    handlers = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='a'))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers,
        force=True,
    )


def save_configure(args):
    cp_dir = Path(args.cp_dir)
    cp_dir.mkdir(parents=True, exist_ok=True)

    serializable = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, list, tuple, type(None)))}
    with open(cp_dir / 'config.yaml', 'w') as f:
        yaml.dump(serializable, f, default_flow_style=False)


def unwrap_model(net):
    # peel off DistributedDataParallel's `.module` and/or torch.compile's `._orig_mod`
    # wrapper so state_dict() keys and eval-time calls always hit the raw nn.Module
    if net is None:
        return None
    if hasattr(net, 'module'):
        net = net.module
    if hasattr(net, '_orig_mod'):
        net = net._orig_mod
    return net


def resume_load_model_checkpoint(net, ema_net, args):
    checkpoint = torch.load(args.load, map_location='cpu', weights_only=True)
    net.load_state_dict(checkpoint['model_state_dict'])

    if ema_net is not None and checkpoint.get('ema_model_state_dict') is not None:
        ema_net.load_state_dict(checkpoint['ema_model_state_dict'])

    args.start_epoch = checkpoint.get('epoch', 0)
    logging.info(f"Resumed model from {args.load}, starting at epoch {args.start_epoch}")


def resume_load_optimizer_checkpoint(optimizer, args):
    checkpoint = torch.load(args.load, map_location='cpu', weights_only=True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logging.info(f"Resumed optimizer from {args.load}")


# per-class segmentation-quality metrics (voxel/surface overlap); 'sensitivity' etc. below
# are case-level detection scalars, not per-class, and logged separately
ARRAY_METRICS = ('dice', 'nsd', 'hd95', 'asd')
SCALAR_METRICS = ('sensitivity', 'specificity', 'f1', 'auc')


def log_evaluation_result(writer, results, tag, epoch, args):
    for name in ARRAY_METRICS:
        values = results[name]
        for i in range(args.classes):
            writer.add_scalar(f'{tag}/{name}_class_{i}', values[i], epoch + 1)
        writer.add_scalar(f'{tag}/{name}_mean', values.mean(), epoch + 1)

    for name in SCALAR_METRICS:
        writer.add_scalar(f'{tag}/{name}', results[name], epoch + 1)

    array_str = ", ".join(f"{name.upper()}: {results[name]}" for name in ARRAY_METRICS)
    scalar_str = ", ".join(f"{name}: {results[name]:.4f}" for name in SCALAR_METRICS)
    logging.info(f"{tag} - {array_str}, {scalar_str}")


def filter_validation_results(results, args):
    # single-class binary task: nothing to filter out, kept only for interface parity
    return results


def write_cross_validation_results(fold_results, cp_path, dataset, unique_name, k_fold):
    # fold_results: list of the per-fold best-checkpoint result dicts returned by train_net()
    np.set_printoptions(precision=4, suppress=True)
    out_path = Path(cp_path) / dataset / unique_name / "cross_validation.txt"

    with open(out_path, 'w') as f:
        for name in ARRAY_METRICS:
            stacked = np.vstack([r[name] for r in fold_results])
            f.write(f"{name.upper()}\n")
            for i in range(k_fold):
                f.write(f"Fold {i}: {fold_results[i][name]}\n")
            f.write(f"Each Class Avg: {np.mean(stacked, axis=0)}\n")
            f.write(f"Each Class Std: {np.std(stacked, axis=0)}\n")
            f.write(f"All classes Avg: {stacked.mean()}\n")
            f.write(f"All classes Std: {np.mean(stacked, axis=1).std()}\n\n")

        f.write("Detection metrics (case-level tumor presence/absence, not segmentation quality --\n")
        f.write("see validation.py docstring)\n")
        for name in SCALAR_METRICS:
            values = np.array([r[name] for r in fold_results], dtype=float)
            f.write(f"{name}: per-fold {values}, avg {np.nanmean(values):.4f}, std {np.nanstd(values):.4f}\n")
