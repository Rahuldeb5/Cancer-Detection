"""Verify the region-based .npz export of a Dataset502 fold (run on the server).

For a region-based dataset nnU-Net applies a SIGMOID (not softmax) and builds the
segmentation with
    for i, c in enumerate(regions_class_order): seg[prob[i] > 0.5] = c
(nnunetv2/utilities/label_handling/label_handling.py). With
labels = {background:0, pancreas:[1,2], lesion:2} and regions_class_order [1,2] the
channels therefore are
    channel 0 = P(pancreas region, i.e. gland INCLUDING tumor)
    channel 1 = P(lesion)
and, because the lesion assignment runs last, (prob[1] > 0.5) must equal (seg == 2)
exactly. The channel probabilities do NOT sum to 1. This script proves all of that on
real exported files instead of trusting the reading.

    source ~/research/nnunet_env/env.sh
    python src/nnunet-multiclass/verify_npz_regions.py \
        --pred_dir $nnUNet_results/Dataset502_PanTSPancLesion/nnUNetTrainer_1epoch__nnUNetPlans__3d_fullres/fold_0/validation \
        --n 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()

    npzs = sorted(a.pred_dir.glob("*.npz"))
    if not npzs:
        print(f"no .npz in {a.pred_dir}")
        sys.exit(1)
    print(f"{len(npzs)} npz files in {a.pred_dir}; checking the first {a.n}\n")

    bad = 0
    for p in npzs[: a.n]:
        cid = p.stem
        seg_img = nib.load(a.pred_dir / f"{cid}.nii.gz")
        seg = np.asarray(seg_img.dataobj).astype(np.int16)          # nibabel (x,y,z)
        with np.load(p) as d:
            keys = list(d.files)
            prob = d["probabilities"] if "probabilities" in d else d[keys[0]]

        print(f"--- {cid}")
        print(f"    npz keys {keys}  probabilities {prob.shape} {prob.dtype}")
        print(f"    seg (nibabel order) {seg.shape}  values {sorted(np.unique(seg).tolist())}")
        if prob.ndim != 4:
            print("    FAIL: probabilities is not 4D")
            bad += 1
            continue

        # nnU-Net stores (C, z, y, x); channel c in nibabel order is prob[c].T on the spatial axes
        if prob.shape[1:] == seg.shape[::-1]:
            order, ch = "(C, z, y, x)", lambda c: prob[c].transpose(2, 1, 0)
        elif prob.shape[1:] == seg.shape:
            order, ch = "(C, x, y, z)", lambda c: prob[c]
        else:
            print(f"    FAIL: probabilities spatial shape {prob.shape[1:]} matches neither "
                  f"{seg.shape} nor {seg.shape[::-1]}")
            bad += 1
            continue
        print(f"    spatial order {order}, channels {prob.shape[0]}")

        p_panc, p_les = ch(0), ch(1)
        print(f"    channel 0 max {float(p_panc.max()):.6f}  channel 1 max {float(p_les.max()):.6f}")
        s = (p_panc.astype(np.float64) + p_les.astype(np.float64))
        print(f"    channel sum: min {float(s.min()):.4f} max {float(s.max()):.4f} "
              f"(sigmoid regions -> need NOT be 1)")

        d2 = int(((p_les > 0.5) != (seg == 2)).sum())
        d1 = int((((p_panc > 0.5) | (p_les > 0.5)) != (seg > 0)).sum())
        print(f"    (channel1 > 0.5) vs (seg == 2): {d2} mismatched voxels")
        print(f"    (channel0>0.5 | channel1>0.5) vs (seg > 0): {d1} mismatched voxels")
        n2 = int((seg == 2).sum())
        print(f"    seg==2 voxels {n2}   seg==1 voxels {int((seg == 1).sum())}")
        if d2:
            print("    FAIL: channel 1 is not the lesion channel (or thresholding differs)")
            bad += 1
        if d1:
            print("    WARN: union of the two channels does not reproduce seg > 0")

    summ = a.pred_dir.parent / "validation" / "summary.json"
    summ = a.pred_dir / "summary.json"
    if summ.is_file():
        s = json.loads(summ.read_text())
        print(f"\nsummary.json foreground_mean: {s.get('foreground_mean')}")
        for k, v in (s.get("mean") or {}).items():
            print(f"  region {k}: Dice {v.get('Dice')}  IoU {v.get('IoU')}  n_ref {v.get('n_ref')}")

    print(f"\n{'FAILED' if bad else 'PASS'}: {bad} case(s) with a hard mismatch")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
