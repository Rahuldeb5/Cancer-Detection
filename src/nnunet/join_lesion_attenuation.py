"""Join per-lesion nnU-Net Dice (recomputed straight from validation predictions +
gt_segmentations, all 5 folds) with the per-lesion attenuation labels from
results/attenuation_labels.csv.

Dice is recomputed here (not read from per_lesion_metrics.csv) because that CSV
never stored the connected-component index, so there was nothing to key a join
on. The whole thing -- GT, prediction, connected-component labeling, spacing --
is done with **nibabel only**, in nibabel's native (x, y, z) axis order,
because that is what src/attenuation-labeling/main.py used to assign
lesion_id. `evaluate.py`'s own `load_mask` uses SimpleITK, which returns
arrays transposed to (z, y, x); full 26-connectivity (`generate_binary_structure(3,3)`
== `np.ones((3,3,3))`) makes the connected *components* identical either way,
but `scipy.ndimage.label`'s numbering depends on raster-scan order, which
differs between the two axis orderings -- so component "1" in one script is
not reliably component "1" in the other. Mixing the two produced silently
swapped/permuted (case_id, lesion_id) keys (~21% of lesions) on the first
version of this script; staying entirely in nibabel's convention removes the
mismatch by construction. Verified below by comparing this run's voxel volume
to the attenuation script's own vol_mm3 for the same key.

Usage (on the server, from ~/Cancer-Detection):
    python src/nnunet/join_lesion_attenuation.py --atten_csv ATTEN.csv --out_csv OUT.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as cc_label
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

CC_STRUCT = generate_binary_structure(3, 3)  # full 26-connectivity, axis-order agnostic
SIZE_EDGES_MM = (5.0, 10.0, 20.0, 40.0)
VOL_MISMATCH_TOL_MM3 = 5.0


def bin_labels() -> list[str]:
    e = [0.0, *SIZE_EDGES_MM, np.inf]
    out = []
    for lo, hi in zip(e[:-1], e[1:]):
        out.append(f"<{hi:g}" if lo == 0 else (f">={lo:g}" if np.isinf(hi) else f"{lo:g}-{hi:g}"))
    return out


def bin_index(diam_mm: float) -> int:
    return int(np.digitize([diam_mm], SIZE_EDGES_MM)[0])


def feret_mm(coords_vox: np.ndarray, spacing) -> float:
    pts = coords_vox * np.asarray(spacing)
    if len(pts) < 2:
        return 0.0
    if len(pts) > 200:
        try:
            pts = pts[ConvexHull(pts).vertices]
        except Exception:  # noqa: BLE001 - coplanar / degenerate
            idx = np.random.default_rng(0).choice(len(pts), 200, replace=False)
            pts = pts[idx]
    return float(pdist(pts).max())


def load_bool(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(path)
    raw = np.nan_to_num(np.asanyarray(img.dataobj), nan=0.0)
    return raw > 0.5, img.header.get_zooms()[:3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atten_csv", required=True, type=Path)
    ap.add_argument("--out_csv", required=True, type=Path)
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    results_root = Path(os.environ["nnUNet_results"]) / "Dataset501_PanTSTumor" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    gt_dir = Path(os.environ["nnUNet_preprocessed"]) / "Dataset501_PanTSTumor" / "gt_segmentations"

    atten = pd.read_csv(args.atten_csv)
    atten_idx = {(r.case_id, int(r.lesion_id)): r for r in atten.itertuples()}
    seen_keys: set[tuple[str, int]] = set()

    labels = bin_labels()
    rows: list[dict] = []
    unmatched: list[tuple[str, int]] = []
    vol_mismatch: list[tuple] = []

    for fold in range(args.n_folds):
        val_dir = results_root / f"fold_{fold}" / "validation"
        preds = sorted(val_dir.glob("*.nii.gz"))
        print(f"fold {fold}: {len(preds)} validation cases", flush=True)
        for i, pp in enumerate(preds, 1):
            cid = pp.name[:-7]
            g, zooms = load_bool(gt_dir / pp.name)
            if not g.any():
                continue
            p, _ = load_bool(pp)
            voxvol = float(np.prod(zooms))

            glab, gn = cc_label(g, structure=CC_STRUCT)
            plab, _ = cc_label(p, structure=CC_STRUCT)

            for k in range(1, gn + 1):
                gk = glab == k
                gk_vox = int(gk.sum())
                coords = np.argwhere(gk)
                diam = feret_mm(coords, zooms)
                overlap_labels = np.unique(plab[gk])
                overlap_labels = overlap_labels[overlap_labels > 0]
                if overlap_labels.size:
                    pk = np.isin(plab, overlap_labels)
                    inter = int(np.logical_and(gk, pk).sum())
                    dice = 2 * inter / (gk_vox + int(pk.sum()))
                else:
                    dice = 0.0

                key = (cid, k)
                a = atten_idx.get(key)
                row = {
                    "fold": fold, "case": cid, "lesion_id": k,
                    "gt_vox": gk_vox, "vol_mm3_eval": gk_vox * voxvol,
                    "diam_mm": diam, "bin": labels[bin_index(diam)],
                    "lesion_dice": dice,
                }
                if a is None:
                    unmatched.append(key)
                    row["attenuation"] = None
                else:
                    seen_keys.add(key)
                    if abs(row["vol_mm3_eval"] - a.vol_mm3) > VOL_MISMATCH_TOL_MM3:
                        vol_mismatch.append((cid, k, row["vol_mm3_eval"], a.vol_mm3))
                    row.update({
                        "vol_mm3_atten": a.vol_mm3,
                        "attenuation": a.attenuation,
                        "delta_hu": a.delta_hu,
                        "n_components_atten": a.n_components,
                        "mpd_present": a.mpd_present,
                        "dist_to_mpd_mm": a.dist_to_mpd_mm,
                    })
                rows.append(row)
            if i % 50 == 0 or i == len(preds):
                print(f"  {i}/{len(preds)}", flush=True)

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    missing_from_eval = set(atten_idx) - seen_keys
    print(f"\nlesions written: {len(df)}")
    print(f"attenuation rows total: {len(atten_idx)}")
    print(f"joined (matched): {len(seen_keys)}")
    print(f"eval lesions with no attenuation row: {len(unmatched)}")
    print(f"attenuation rows with no eval lesion: {len(missing_from_eval)}")
    print(f"volume mismatches (>{VOL_MISMATCH_TOL_MM3} mm3, join misaligned): {len(vol_mismatch)}")
    for m in vol_mismatch[:20]:
        print("  MISMATCH", m)
    for u in unmatched[:20]:
        print("  UNMATCHED (eval side)", u)
    for m in list(missing_from_eval)[:20]:
        print("  MISSING (atten side)", m)
    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
