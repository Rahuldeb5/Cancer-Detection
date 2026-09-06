"""Ground-truth pancreatic-lesion size distribution for a fold's validation set.

Per connected component: volume, equivalent-sphere diameter, and 3D max
("Feret") diameter = max spacing-weighted distance between component voxels
(via convex hull). Bins the lesions and prints how many fall in each.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

FOLD_DIR = Path("src/data")
LABELS = Path(os.environ["nnUNet_raw"]) / "Dataset501_PanTSTumor" / "labelsTr"
BINS_MM = [0, 5, 10, 20, 40, np.inf]
BIN_LABELS = ["<5", "5-10", "10-20", "20-40", ">=40"]


def val_ids(fold_1indexed: int) -> list[str]:
    return sorted(
        l.strip()
        for l in (FOLD_DIR / f"fold_{fold_1indexed}_ids.txt").read_text().splitlines()
        if l.strip()
    )


def max_feret_mm(coords_vox: np.ndarray, spacing_zyx) -> float:
    pts = coords_vox * np.asarray(spacing_zyx)
    if len(pts) < 2:
        return 0.0
    if len(pts) > 12:
        try:
            pts = pts[ConvexHull(pts).vertices]
        except Exception:  # noqa: BLE001 - degenerate (coplanar) hull
            pass
    return float(pdist(pts).max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1, help="1-indexed fold file to use as val set")
    args = ap.parse_args()

    ids = val_ids(args.fold)
    print(f"fold_{args.fold} val set: {len(ids)} cases\n")

    lesions = []  # (case, vol_mm3, eq_diam, feret)
    n_pos = n_multi = 0
    for cid in ids:
        img = sitk.ReadImage(str(LABELS / f"{cid}.nii.gz"))
        arr = sitk.GetArrayFromImage(img) > 0
        sp_xyz = img.GetSpacing()
        sp_zyx = (sp_xyz[2], sp_xyz[1], sp_xyz[0])
        voxvol = float(np.prod(sp_xyz))
        if not arr.any():
            continue
        n_pos += 1
        lab, n = label(arr)
        if n > 1:
            n_multi += 1
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            vol = len(coords) * voxvol
            eq = (6 * vol / np.pi) ** (1 / 3)
            fer = max_feret_mm(coords, sp_zyx)
            lesions.append((cid, vol, eq, fer))

    feret = np.array([x[3] for x in lesions])
    eq = np.array([x[2] for x in lesions])
    print(f"cases with tumor: {n_pos}/{len(ids)}   multi-focal: {n_multi}   total lesions: {len(lesions)}\n")

    def hist(vals, name):
        idx = np.digitize(vals, BINS_MM[1:-1])
        c = Counter(idx)
        print(f"  by {name}:")
        for i, bl in enumerate(BIN_LABELS):
            v = vals[idx == i]
            extra = f"   median {np.median(v):.1f}mm" if len(v) else ""
            print(f"    {bl:>6} mm : {c.get(i,0):3d} lesions{extra}")
        print()

    hist(feret, "3D max (Feret) diameter")
    hist(eq, "equivalent-sphere diameter")

    # per-scan: largest lesion's Feret diameter
    per_scan = {}
    for cid, _, _, fer in lesions:
        per_scan[cid] = max(per_scan.get(cid, 0), fer)
    ps = np.array(list(per_scan.values()))
    idx = np.digitize(ps, BINS_MM[1:-1])
    print("  per-scan, binned by largest lesion Feret diameter:")
    for i, bl in enumerate(BIN_LABELS):
        print(f"    {bl:>6} mm : {int((idx==i).sum()):3d} scans")


if __name__ == "__main__":
    main()
