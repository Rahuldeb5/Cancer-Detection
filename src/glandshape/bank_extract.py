"""S15 Part B step 7 (SERVER, after bank_predict.sh): keep only the pancreas-region probability of each bank case.

Per case: npz['probabilities'] is (C, z, y, x) float32 with channel 0 = pancreas region (502 regions_class_order
[1, 2]), as for the cohort. Channel 0 is transposed to nibabel (x, y, z), cropped to the bbox of p > 0.01 padded by
2 voxels, and saved as float16 with the crop offset and the full shape:
    OUT/{case}.npz : p0 (float16, cropped), offset (3,), shape (3,), n_gt05 (#voxels p0 > 0.5, full precision)
Channel 1 (lesion) is not kept. The big nnU-Net npz files are left in place (delete only after the PC has the
extracted files and Rahul agrees).

Usage (server): ~/nnunet_setup/.venv/bin/python bank_extract.py [--root ~/research/gland_shape_bank]
"""
import argparse
from pathlib import Path

import numpy as np

FLOOR = 0.01
PAD = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.home() / "research/gland_shape_bank")
    a = ap.parse_args()
    out = a.root / "ch0"
    out.mkdir(exist_ok=True)
    done = bad = 0
    for f in sorted((a.root / "pred").glob("PanTS_*.npz")):
        dst = out / f.name
        if dst.exists():
            continue
        try:
            with np.load(f) as d:
                P = d["probabilities"]
        except Exception as e:  # nnU-Net may still be writing this file; a later run picks it up
            print(f"{f.stem}: not readable yet ({type(e).__name__}); skipped")
            continue
        if P.ndim != 4 or P.shape[0] != 2:
            print(f"{f.stem}: unexpected {P.shape}")
            bad += 1
            continue
        p0 = np.ascontiguousarray(P[0].transpose(2, 1, 0))          # (x, y, z)
        del P
        n05 = int((p0 > 0.5).sum())
        idx = np.argwhere(p0 > FLOOR)
        if len(idx):
            lo = np.maximum(idx.min(0) - PAD, 0)
            hi = np.minimum(idx.max(0) + 1 + PAD, p0.shape)
        else:
            lo = hi = np.zeros(3, int)
        crop = p0[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float16)
        tmp = dst.with_name(dst.stem + ".tmp.npz")
        np.savez_compressed(tmp, p0=crop, offset=lo, shape=np.array(p0.shape), n_gt05=np.array(n05))
        tmp.rename(dst)
        done += 1
        print(f"{f.stem}: shape {p0.shape} crop {crop.shape} p>0.5 {n05}", flush=True)
    print(f"extracted {done}, problems {bad}, total in ch0/ {len(list(out.glob('*.npz')))}")


if __name__ == "__main__":
    main()
