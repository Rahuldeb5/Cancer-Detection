"""Build a validation set with chosen GT lesions excluded, so evaluate.py can be
run on it UNCHANGED (same metrics, directly comparable to the baseline).

Excluded lesions are keyed by (case_id, lesion_id) from
results/lesion_location_results/excluded_lesions.csv (every row is excluded).
lesion_id comes from a
nibabel-order, 26-connectivity labeling of the GT -- the same numbering as
join_lesion_attenuation.py -- so this script labels with nibabel too, then
transposes the resulting mask into SimpleITK's (z,y,x) frame (evaluate.py's frame).
Before anything is written it checks, for every case in the CSV, that the GT has the
component count the CSV records and that each excluded lesion_id has the recorded voxel
count, and aborts on any mismatch, because a silently misaligned key would exclude the
wrong lesion.

Per case with >=1 excluded lesion ("ignore region" semantics):
  - GT: excluded components are set to background.
  - Prediction: a predicted component that overlaps an excluded GT lesion and NO kept
    GT lesion is removed (it is the model finding an ignored lesion, not a false
    positive). If it also overlaps a kept lesion it stays, minus the voxels inside the
    excluded lesion.
  - <case>.npz: replaced by a 1-voxel stand-in whose tumor channel holds the max tumor
    probability outside the ignored region, because evaluate.py only reads
    prob[TUMOR].max() from it (used for the max_prob AUROC).
  - Cases with no lesion left: dropped from the evaluation by default (their remaining
    pancreas-tumor status is unknown), or kept as tumor-negative cases with
    --all_excluded negative.
Unchanged cases are symlinked.

Usage (on the server):
  python prepare_excluded_eval.py --pred_dir .../fold_0/validation --gt_dir .../gt_segmentations \
      --work_dir WORK/fold_0 --excluded_csv results/lesion_location_results/excluded_lesions.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as cc_label

CC_STRUCT = generate_binary_structure(3, 3)
TUMOR = 1


def link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def write_like(arr: np.ndarray, ref_img: sitk.Image, dtype, path: Path) -> None:
    out = sitk.GetImageFromArray(arr.astype(dtype))
    out.CopyInformation(ref_img)
    sitk.WriteImage(out, str(path), useCompression=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, type=Path)
    ap.add_argument("--gt_dir", required=True, type=Path)
    ap.add_argument("--work_dir", required=True, type=Path)
    ap.add_argument("--excluded_csv", required=True, type=Path)
    ap.add_argument("--all_excluded", choices=["drop", "negative"], default="drop")
    args = ap.parse_args()

    inc = pd.read_csv(args.excluded_csv)
    by_case = {c: g.set_index("lesion_id") for c, g in inc.groupby("case_id")}

    out_pred, out_gt = args.work_dir / "pred", args.work_dir / "gt"
    out_pred.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)

    preds = sorted(args.pred_dir.glob("*.nii.gz"))
    assert preds, f"no predictions in {args.pred_dir}"

    key_errors: list[str] = []
    manifest: list[dict] = []
    for pp in preds:
        cid = pp.name[:-7]
        gp = args.gt_dir / pp.name
        npz = args.pred_dir / f"{cid}.npz"

        g_nib = np.nan_to_num(np.asanyarray(nib.load(gp).dataobj), nan=0.0) > 0.5
        glab, gn = cc_label(g_nib, structure=CC_STRUCT)
        rows = by_case.get(cid)

        # --- key-alignment checks (abort before writing if the join is not trustworthy)
        rows = by_case.get(cid)
        excluded: list[int] = []
        if rows is not None:
            if gn != int(rows["n_components_in_case"].iloc[0]):
                key_errors.append(f"{cid}: GT has {gn} components, csv says {int(rows['n_components_in_case'].iloc[0])}")
            for lid, r in rows.iterrows():
                nvox = int((glab == lid).sum()) if lid <= gn else -1
                if nvox != int(r["n_vox"]):
                    key_errors.append(f"{cid} lesion {lid}: GT {nvox} vox vs csv {int(r['n_vox'])}")
            excluded = [int(l) for l in rows.index]

        row = {"case": cid, "n_gt_lesions": gn, "n_excluded": len(excluded),
               "action": "unchanged", "n_pred_comps_removed": 0}

        if not excluded:
            link(pp, out_pred / pp.name)
            link(gp, out_gt / pp.name)
            if npz.is_file():
                link(npz, out_pred / npz.name)
            manifest.append(row)
            continue

        if len(excluded) == gn and args.all_excluded == "drop":
            row["action"] = "dropped"
            manifest.append(row)
            continue

        pred_img, gt_img = sitk.ReadImage(str(pp)), sitk.ReadImage(str(gp))
        pred, gt = sitk.GetArrayFromImage(pred_img), sitk.GetArrayFromImage(gt_img)  # (z,y,x)
        excl = np.isin(glab, excluded).transpose(2, 1, 0)
        if excl.shape != gt.shape or not np.array_equal(gt == TUMOR, g_nib.transpose(2, 1, 0)):
            key_errors.append(f"{cid}: nibabel (x,y,z) GT does not transpose onto SimpleITK (z,y,x) GT")
            continue

        kept = (gt == TUMOR) & ~excl
        plab, pn = cc_label(pred == TUMOR, structure=CC_STRUCT)
        ov_excl = np.bincount(plab[excl], minlength=pn + 1)
        ov_kept = np.bincount(plab[kept], minlength=pn + 1)
        remove = (ov_excl > 0) & (ov_kept == 0)
        remove[0] = False
        removed_mask = np.isin(plab, np.nonzero(remove)[0])

        new_pred = pred.copy()
        new_pred[removed_mask] = 0
        new_pred[excl] = 0
        new_gt = gt.copy()
        new_gt[excl] = 0

        write_like(new_pred, pred_img, pred.dtype, out_pred / pp.name)
        write_like(new_gt, gt_img, gt.dtype, out_gt / pp.name)

        if npz.is_file():
            with np.load(npz) as d:
                prob = d["probabilities"] if "probabilities" in d else d[d.files[0]]
            tumor_p = prob[TUMOR]
            ignore = excl | removed_mask
            if tumor_p.shape != ignore.shape:
                if tumor_p.shape == ignore.shape[::-1]:
                    ignore = ignore.transpose(2, 1, 0)
                else:
                    raise RuntimeError(f"{cid}: prob shape {tumor_p.shape} matches neither zyx nor xyz {ignore.shape}")
            outside = tumor_p[~ignore]
            stand_in = np.zeros((2, 1, 1, 1), dtype=np.float32)
            stand_in[TUMOR, 0, 0, 0] = float(outside.max()) if outside.size else 0.0
            np.savez(out_pred / npz.name, probabilities=stand_in)

        row["action"] = "modified"
        row["n_pred_comps_removed"] = int(remove.sum())
        manifest.append(row)

    if key_errors:
        print(f"\nABORT: {len(key_errors)} key/orientation mismatch(es), nothing trustworthy to evaluate:")
        for e in key_errors[:30]:
            print("  ", e)
        sys.exit(1)

    m = pd.DataFrame(manifest)
    m.to_csv(args.work_dir / "manifest.csv", index=False)
    print(f"{len(m)} cases: {m['action'].value_counts().to_dict()}; "
          f"{int(m['n_excluded'].sum())} lesions excluded; "
          f"{int(m['n_pred_comps_removed'].sum())} predicted components removed")


if __name__ == "__main__":
    main()
