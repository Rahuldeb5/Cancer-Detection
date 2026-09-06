"""
End-to-end smoke test for the PanTS training pipeline against the cached .npy scans.

Phase A (CPU):  dataset + MONAI transforms  -- crop shapes, pos-voxel fraction per crop
Phase B (CPU):  one forward pass through MedFormer at a reduced spatial size -- catches
                shape/wiring bugs cheaply, without waiting on a full 128^3 CPU forward
Phase C (GPU):  forward + backward + optimizer step, timed, with peak VRAM reported.
                Uses the yaml's spatial_size/batch_size/num_samples unless overridden --
                on a 12GB card the full config (128^3, batch_size=2, num_samples=4 ->
                8 patches/step) does not fit; see README notes for a card that does.

Run from the repo root:
    python -m testing.PanTS.smoke_test
    python -m testing.PanTS.smoke_test --n_cases 5 --iters 5 --amp
    python -m testing.PanTS.smoke_test --skip_gpu          # data-pipeline check only
    python -m testing.PanTS.smoke_test --amp --spatial_size 96 96 96 --num_samples 1 --batch_size 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from monai.data.utils import list_data_collate

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.pants_dataset import PanTSDataset
from src.training.losses import DiceLoss
from src.training.utils import load_fold_filenames, get_optimizer
from src.model.utils import get_model

DATA_DIR = REPO_ROOT / "src" / "data"
CONFIG_PATH = REPO_ROOT / "src" / "config" / "pants" / "medformer_3d.yaml"


def load_args(overrides: dict) -> argparse.Namespace:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    # these mirror train.py's argparse defaults -- not in the yaml, but get_model() needs them
    config.setdefault("batch_size", 2)
    config.setdefault("model", "medformer")
    config.setdefault("dataset", "pants")
    config.setdefault("dimension", "3d")
    config.update(overrides)
    args = argparse.Namespace(**config)
    args.cache_dir = Path(args.cache_dir).expanduser()
    return args


def compute_loss(result, label, criterion, criterion_dl, aux_weight):
    if isinstance(result, (tuple, list)):
        loss = 0
        for j in range(len(result)):
            loss = loss + aux_weight[j] * (criterion(result[j], label) + criterion_dl(result[j], label))
        return loss
    return criterion(result, label) + criterion_dl(result, label)


def phase_dataset(args, filenames, fingerprint, n_cases):
    print(f"\n== Phase A: dataset + transforms (CPU), {n_cases} case(s), "
          f"spatial_size={args.spatial_size}, pos={args.pos}, neg={args.neg}, num_samples={args.num_samples} ==")
    subset = filenames[:n_cases]
    ds = PanTSDataset(subset, args.cache_dir, fingerprint, train=True,
                       spatial_size=args.spatial_size, pos=args.pos, neg=args.neg, num_samples=args.num_samples)

    for i, case_id in enumerate(subset):
        t0 = time.time()
        item = ds[i]
        dt = time.time() - t0
        crops = item if isinstance(item, list) else [item]
        pos_frac = [c["label"].float().mean().item() for c in crops]
        print(
            f"  {case_id}: {len(crops)} crop(s) in {dt:.2f}s, "
            f"image {tuple(crops[0]['image'].shape)}, label {tuple(crops[0]['label'].shape)}, "
            f"pos-voxel frac/crop: {[f'{p:.4%}' for p in pos_frac]}"
        )

    val_ds = PanTSDataset(subset, args.cache_dir, fingerprint, train=False)
    t0 = time.time()
    val_item = val_ds[0]
    print(f"  val transform (full volume): image {tuple(val_item['image'].shape)} in {time.time()-t0:.2f}s")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=list_data_collate)
    batch = next(iter(loader))
    print(f"  DataLoader batch: image {tuple(batch['image'].shape)}, label {tuple(batch['label'].shape)}")
    return batch


def phase_cpu_model(args, spatial_size):
    print(f"\n== Phase B: CPU forward pass @ spatial_size={spatial_size} ==")
    args.spatial_size = list(spatial_size)
    net = get_model(args)
    net.eval()

    x = torch.randn(1, 1, *spatial_size)
    t0 = time.time()
    with torch.no_grad():
        out = net(x)
    dt = time.time() - t0

    if isinstance(out, (list, tuple)):
        shapes = [tuple(o.shape) for o in out]
        print(f"  forward ok in {dt:.2f}s, outputs (main + aux): {shapes}")
    else:
        print(f"  forward ok in {dt:.2f}s, output {tuple(out.shape)}")

    n_params = sum(p.numel() for p in net.parameters())
    print(f"  params: {n_params/1e6:.1f}M")


def phase_gpu_train_step(args, real_batch, iters, amp):
    if not torch.cuda.is_available():
        print("\n== Phase C: SKIPPED (no CUDA device visible) ==")
        return

    print(f"\n== Phase C: GPU forward+backward, batch_size={args.batch_size}, "
          f"num_samples={args.num_samples}, spatial_size={args.spatial_size}, amp={amp} ==")

    device = torch.device("cuda")
    net = get_model(args).to(device)
    net.train()

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight)).to(device)
    criterion_dl = DiceLoss()
    optimizer = get_optimizer(args, net)
    scaler = torch.amp.GradScaler("cuda") if amp else None

    img = real_batch["image"].to(device, non_blocking=True)
    label = real_batch["label"].float().to(device, non_blocking=True)
    print(f"  batch on device: image {tuple(img.shape)}, label {tuple(label.shape)}")

    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for it in range(iters):
        t0 = time.time()
        optimizer.zero_grad()

        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                result = net(img)
                loss = compute_loss(result, label, criterion, criterion_dl, args.aux_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            result = net(img)
            loss = compute_loss(result, label, criterion, criterion_dl, args.aux_weight)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        dt = time.time() - t0
        times.append(dt)
        print(f"  iter {it+1}/{iters}: loss={loss.item():.4f}, {dt:.2f}s")

    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    print(f"  peak VRAM: {peak_gb:.2f} GiB / {total_gb:.2f} GiB  |  avg iter (after warmup): "
          f"{np.mean(times[1:]) if len(times) > 1 else times[0]:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cases", type=int, default=3, help="cases to run through Phase A")
    parser.add_argument("--cpu_spatial_size", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--iters", type=int, default=3, help="GPU train-step iterations to time")
    parser.add_argument("--batch_size", type=int, default=None, help="override yaml batch_size")
    parser.add_argument("--num_samples", type=int, default=None, help="override yaml num_samples (crops per case)")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=None, help="override yaml spatial_size (patch size)")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip_cpu_model", action="store_true")
    parser.add_argument("--skip_gpu", action="store_true")
    cli = parser.parse_args()

    overrides = {}
    if cli.batch_size is not None:
        overrides["batch_size"] = cli.batch_size
    if cli.num_samples is not None:
        overrides["num_samples"] = cli.num_samples
    if cli.spatial_size is not None:
        overrides["spatial_size"] = cli.spatial_size
    args = load_args(overrides)

    fingerprint = json.loads((DATA_DIR / "fingerprint.json").read_text())
    train_filenames, _ = load_fold_filenames(DATA_DIR, args.k_fold, test_fold_idx=0)

    real_batch = phase_dataset(args, train_filenames, fingerprint, cli.n_cases)

    if not cli.skip_cpu_model:
        phase_cpu_model(load_args(overrides), cli.cpu_spatial_size)  # separate Namespace: leaves args.spatial_size alone
    else:
        print("\n== Phase B: skipped ==")

    if not cli.skip_gpu:
        phase_gpu_train_step(args, real_batch, cli.iters, cli.amp)
    else:
        print("\n== Phase C: skipped ==")

    print("\nAll requested phases completed.")


if __name__ == "__main__":
    main()
