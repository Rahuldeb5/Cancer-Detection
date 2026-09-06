"""Comprehensive evaluation of an nnU-Net fold's validation predictions.

Computes, for the tumor class (label 1):

  overlap    : Dice (DSC), IoU / Jaccard
  boundary   : HD95, ASSD (average symmetric surface distance),
               NSD (normalized surface dice) at 1 / 2 / 3 mm tolerance
  volume     : predicted vs reference tumor volume, relative volume difference

  case-level tumor detection (over ALL validation cases):
    Sensitivity, Specificity, Precision/PPV, NPV, F1, Accuracy, Balanced Acc,
    AUROC + AUPRC using (a) predicted tumor volume and (b) max tumor probability
    (from the --npz softmax, if present). "Detected" is reported at volume
    thresholds 0 / 50 / 100 mm^3.

Segmentation metrics are aggregated over the cases whose reference actually
contains tumor (the only ones where a boundary is defined); a reference-positive
case with an empty prediction contributes Dice/IoU = 0 and NaN boundary metrics
(counted separately as "missed").

Usage:
  python evaluate.py \
      --pred_dir  $nnUNet_results/Dataset501_PanTSTumor/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation \
      --gt_dir    $nnUNet_preprocessed/Dataset501_PanTSTumor/gt_segmentations \
      --out_dir   $nnUNet_results/Dataset501_PanTSTumor/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/evaluation
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import SimpleITK as sitk

try:
    from surface_distance import (
        compute_average_surface_distance,
        compute_robust_hausdorff,
        compute_surface_dice_at_tolerance,
        compute_surface_distances,
    )
    HAVE_SD = True
except Exception:  # noqa: BLE001
    HAVE_SD = False

from scipy.ndimage import label as cc_label
from sklearn.metrics import average_precision_score, roc_auc_score

TUMOR = 1
NSD_TOL_MM = (1.0, 2.0, 3.0)
VOL_THRESH_MM3 = (0.0, 50.0, 100.0)
SIZE_EDGES_MM = (5.0, 10.0, 20.0, 40.0)  # -> bins <5, 5-10, 10-20, 20-40, >=40
CC_STRUCT = np.ones((3, 3, 3), dtype=int)  # 26-connectivity


# --------------------------------------------------------------------------- io
def load_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    spacing_xyz = img.GetSpacing()
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    return arr, spacing_zyx


def max_tumor_prob(npz_path: Path, ref_shape_zyx: tuple[int, ...]) -> float | None:
    if not npz_path.is_file():
        return None
    try:
        with np.load(npz_path) as d:
            key = "probabilities" if "probabilities" in d else d.files[0]
            prob = d[key]
    except Exception:  # noqa: BLE001
        return None
    if prob.ndim != 4 or prob.shape[0] <= TUMOR:
        return None
    return float(prob[TUMOR].max())


# ---------------------------------------------------------------- seg metrics
def overlap_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    g = gt == TUMOR
    p = pred == TUMOR
    inter = np.logical_and(g, p).sum(dtype=np.int64)
    gs, ps = int(g.sum()), int(p.sum())
    union = gs + ps - inter
    dice = (2 * inter / (gs + ps)) if (gs + ps) else np.nan
    iou = (inter / union) if union else np.nan
    return {"dice": dice, "iou": iou, "ref_voxels": gs, "pred_voxels": ps}


def boundary_metrics(gt: np.ndarray, pred: np.ndarray, spacing_zyx) -> dict:
    g = np.ascontiguousarray(gt == TUMOR)
    p = np.ascontiguousarray(pred == TUMOR)
    out = {"hd95": np.nan, "assd": np.nan, **{f"nsd_{t}mm": np.nan for t in NSD_TOL_MM}}
    if not g.any() or not p.any():
        return out
    if HAVE_SD:
        sd = compute_surface_distances(g, p, spacing_mm=spacing_zyx)
        out["hd95"] = float(compute_robust_hausdorff(sd, 95.0))
        d_gt2p, d_p2gt = compute_average_surface_distance(sd)
        out["assd"] = float(np.mean([d_gt2p, d_p2gt]))
        for t in NSD_TOL_MM:
            out[f"nsd_{t}mm"] = float(compute_surface_dice_at_tolerance(sd, t))
    else:
        from scipy.ndimage import binary_erosion, distance_transform_edt

        def surf(m):
            return m ^ binary_erosion(m, iterations=1, border_value=0)

        sg, sp = surf(g), surf(p)
        dt_to_g = distance_transform_edt(~sg, sampling=spacing_zyx)
        dt_to_p = distance_transform_edt(~sp, sampling=spacing_zyx)
        d_p2g = dt_to_g[sp]
        d_g2p = dt_to_p[sg]
        both = np.hstack([d_p2g, d_g2p])
        out["hd95"] = float(np.percentile(both, 95))
        out["assd"] = float(both.mean())
        for t in NSD_TOL_MM:
            num = (d_g2p <= t).sum() + (d_p2g <= t).sum()
            out[f"nsd_{t}mm"] = float(num / (len(d_g2p) + len(d_p2g)))
    return out


# ----------------------------------------------------- per-lesion / size bins
def feret_mm(coords_vox: np.ndarray, spacing_zyx) -> float:
    """3D max caliper (Feret) diameter: max spacing-weighted pairwise distance."""
    from scipy.spatial.distance import pdist

    pts = coords_vox * np.asarray(spacing_zyx)
    if len(pts) < 2:
        return 0.0
    if len(pts) > 200:
        try:
            from scipy.spatial import ConvexHull

            pts = pts[ConvexHull(pts).vertices]
        except Exception:  # noqa: BLE001 - coplanar / degenerate
            idx = np.random.default_rng(0).choice(len(pts), 200, replace=False)
            pts = pts[idx]
    return float(pdist(pts).max())


def bin_index(diam_mm: float) -> int:
    return int(np.digitize([diam_mm], SIZE_EDGES_MM)[0])


def bin_labels() -> list[str]:
    e = [0.0, *SIZE_EDGES_MM, np.inf]
    out = []
    for lo, hi in zip(e[:-1], e[1:]):
        out.append(f"<{hi:g}" if lo == 0 else (f">={lo:g}" if np.isinf(hi) else f"{lo:g}-{hi:g}"))
    return out


def lesion_level(gt: np.ndarray, pred: np.ndarray, spacing_zyx) -> list[dict]:
    """One row per GT lesion: size bin, lesion-wise Dice, detected flags."""
    g = gt == TUMOR
    p = pred == TUMOR
    voxvol = float(np.prod(spacing_zyx))
    glab, gn = cc_label(g, structure=CC_STRUCT)
    plab, _ = cc_label(p, structure=CC_STRUCT)
    rows = []
    for k in range(1, gn + 1):
        gk = glab == k
        gk_vox = int(gk.sum())
        coords = np.argwhere(gk)
        diam = feret_mm(coords, spacing_zyx)
        overlap_labels = np.unique(plab[gk])
        overlap_labels = overlap_labels[overlap_labels > 0]
        if overlap_labels.size:
            pk = np.isin(plab, overlap_labels)
            inter = int(np.logical_and(gk, pk).sum())
            dice = 2 * inter / (gk_vox + int(pk.sum()))
            iou = inter / (gk_vox + int(pk.sum()) - inter)
            covered = inter / gk_vox
        else:
            dice = iou = covered = 0.0
        rows.append({
            "gt_vox": gk_vox, "vol_mm3": gk_vox * voxvol, "diam_mm": diam,
            "bin": bin_index(diam), "lesion_dice": dice, "lesion_iou": iou,
            "detected_any": int(covered > 0), "detected_iou10": int(iou >= 0.1),
            "detected_cov10": int(covered >= 0.1),
        })
    return rows


# ---------------------------------------------------------------- aggregation
def _stats(vals: list[float]) -> dict:
    a = np.array([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
    if a.size == 0:
        return {"mean": None, "std": None, "median": None, "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std()),
            "median": float(np.median(a)), "n": int(a.size)}


def detection_block(gt_pos: np.ndarray, score: np.ndarray, det: np.ndarray) -> dict:
    tp = int(np.sum(det & gt_pos))
    fp = int(np.sum(det & ~gt_pos))
    tn = int(np.sum(~det & ~gt_pos))
    fn = int(np.sum(~det & gt_pos))
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    acc = (tp + tn) / len(gt_pos)
    bacc = np.nanmean([sens, spec])
    d = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "sensitivity": sens,
         "specificity": spec, "precision_ppv": ppv, "npv": npv, "f1": f1,
         "accuracy": acc, "balanced_accuracy": float(bacc)}
    if len(np.unique(gt_pos)) == 2:
        d["auroc"] = float(roc_auc_score(gt_pos, score))
        d["auprc"] = float(average_precision_score(gt_pos, score))
    return d


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, type=Path)
    ap.add_argument("--gt_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not HAVE_SD:
        warnings.warn("surface_distance not installed - using scipy voxel-count NSD "
                      "(less accurate on anisotropic spacing). "
                      "pip install surface-distance", stacklevel=1)

    preds = sorted(p for p in args.pred_dir.glob("*.nii.gz"))
    assert preds, f"no predictions in {args.pred_dir}"
    print(f"{len(preds)} validation cases")

    per_case = []
    lesion_rows = []  # one per GT lesion, across all cases
    for i, pp in enumerate(preds, 1):
        cid = pp.name[:-7]
        gp = args.gt_dir / pp.name
        pred, sp = load_mask(pp)
        gt, spg = load_mask(gp)
        assert pred.shape == gt.shape, f"{cid}: shape {pred.shape} vs {gt.shape}"

        ov = overlap_metrics(gt, pred)
        gt_pos = ov["ref_voxels"] > 0
        bd = boundary_metrics(gt, pred, sp) if gt_pos else {
            "hd95": np.nan, "assd": np.nan, **{f"nsd_{t}mm": np.nan for t in NSD_TOL_MM}}

        voxvol = float(np.prod(sp))
        row = {
            "case": cid,
            "gt_tumor": int(gt_pos),
            "dice": ov["dice"] if gt_pos else np.nan,
            "iou": ov["iou"] if gt_pos else np.nan,
            **bd,
            "pred_vol_mm3": ov["pred_voxels"] * voxvol,
            "ref_vol_mm3": ov["ref_voxels"] * voxvol,
            "max_tumor_prob": max_tumor_prob(args.pred_dir / f"{cid}.npz", gt.shape),
        }
        rv, pv = row["ref_vol_mm3"], row["pred_vol_mm3"]
        row["rel_vol_diff"] = ((pv - rv) / rv) if rv > 0 else np.nan
        row["missed"] = int(gt_pos and ov["pred_voxels"] == 0)

        if gt_pos:
            lr = lesion_level(gt, pred, sp)
            largest = max((x["diam_mm"] for x in lr), default=0.0)
            for x in lr:
                x["case"] = cid
                x["scan_largest_diam_mm"] = largest
                lesion_rows.append(x)
            row["n_gt_lesions"] = len(lr)
            row["largest_lesion_diam_mm"] = largest
        per_case.append(row)
        if i % 25 == 0 or i == len(preds):
            print(f"  {i}/{len(preds)}")

    # -------- per-case csv
    #  tumor-negative rows lack the n_gt_lesions / largest_lesion_diam_mm keys,
    #  so take the union of keys across all rows (order-preserving) and fill blanks
    csv_path = args.out_dir / "per_case_metrics.csv"
    fieldnames = list(dict.fromkeys(k for r in per_case for k in r))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(per_case)

    # -------- aggregate
    pos = [r for r in per_case if r["gt_tumor"]]
    seg = {
        "n_ref_positive": len(pos),
        "n_missed_entirely": sum(r["missed"] for r in pos),
        "dice": _stats([r["dice"] for r in pos]),
        "iou": _stats([r["iou"] for r in pos]),
        "hd95": _stats([r["hd95"] for r in pos]),
        "assd": _stats([r["assd"] for r in pos]),
        **{f"nsd_{t}mm": _stats([r[f"nsd_{t}mm"] for r in pos]) for t in NSD_TOL_MM},
        "rel_vol_diff": _stats([r["rel_vol_diff"] for r in pos]),
    }

    gt_pos = np.array([bool(r["gt_tumor"]) for r in per_case])
    vol = np.array([r["pred_vol_mm3"] for r in per_case])
    probs = [r["max_tumor_prob"] for r in per_case]
    have_prob = all(p is not None for p in probs)

    detection = {"n_cases": len(per_case), "n_positive": int(gt_pos.sum()),
                 "by_volume_threshold": {}}
    for thr in VOL_THRESH_MM3:
        detection["by_volume_threshold"][f">{thr:g}mm3"] = detection_block(
            gt_pos, vol, vol > thr)
    detection["score=pred_volume"] = detection_block(gt_pos, vol, vol > 0)
    if have_prob:
        pr = np.array(probs, dtype=float)
        detection["score=max_prob"] = detection_block(gt_pos, pr, pr > 0.5)

    # -------- size-stratified (per GT lesion, binned by Feret diameter)
    labels = bin_labels()
    by_size = {}
    for bi, bl in enumerate(labels):
        grp = [x for x in lesion_rows if x["bin"] == bi]
        if not grp:
            by_size[bl] = {"n_lesions": 0}
            continue
        d = np.array([x["lesion_dice"] for x in grp])
        by_size[bl] = {
            "n_lesions": len(grp),
            "diam_mm_median": float(np.median([x["diam_mm"] for x in grp])),
            "lesion_dice_mean": float(d.mean()),
            "lesion_dice_median": float(np.median(d)),
            "lesion_dice_mean_detected_only": float(
                np.mean([x["lesion_dice"] for x in grp if x["detected_any"]])
                if any(x["detected_any"] for x in grp) else 0.0),
            "detection_rate_any_overlap": float(np.mean([x["detected_any"] for x in grp])),
            "detection_rate_iou>=0.1": float(np.mean([x["detected_iou10"] for x in grp])),
        }
    # per-scan Dice binned by the scan's largest lesion
    by_size_scan = {}
    for bi, bl in enumerate(labels):
        cids = {x["case"] for x in lesion_rows if bin_index(x["scan_largest_diam_mm"]) == bi}
        dvals = [r["dice"] for r in pos if r["case"] in cids and not np.isnan(r["dice"])]
        by_size_scan[bl] = {"n_scans": len(cids), **_stats(dvals)} if cids else {"n_scans": 0}

    size_strat = {
        "size_metric": "3D max (Feret) diameter of GT lesion connected component",
        "bin_edges_mm": list(SIZE_EDGES_MM),
        "total_gt_lesions": len(lesion_rows),
        "per_lesion_by_size": by_size,
        "per_scan_dice_by_largest_lesion_size": by_size_scan,
    }

    with open(args.out_dir / "per_lesion_metrics.csv", "w", newline="") as f:
        if lesion_rows:
            w = csv.DictWriter(f, fieldnames=list(lesion_rows[0].keys()))
            w.writeheader()
            w.writerows(lesion_rows)

    summary = {"pred_dir": str(args.pred_dir), "surface_distance_pkg": HAVE_SD,
               "segmentation_tumor": seg, "detection_tumor": detection,
               "size_stratified": size_strat}
    (args.out_dir / "evaluation.json").write_text(json.dumps(summary, indent=2))

    # -------- console
    def g(d):  # noqa: E306
        return "n/a" if d["mean"] is None else f"{d['mean']:.4f} +/- {d['std']:.4f}  (med {d['median']:.4f}, n={d['n']})"

    print("\n================  TUMOR SEGMENTATION  (ref-positive cases)  ================")
    print(f"  ref-positive cases : {seg['n_ref_positive']}   missed entirely: {seg['n_missed_entirely']}")
    print(f"  Dice   : {g(seg['dice'])}")
    print(f"  IoU    : {g(seg['iou'])}")
    print(f"  HD95   : {g(seg['hd95'])} mm")
    print(f"  ASSD   : {g(seg['assd'])} mm")
    for t in NSD_TOL_MM:
        print(f"  NSD@{t:g}mm: {g(seg[f'nsd_{t}mm'])}")
    print(f"  rel.vol.diff : {g(seg['rel_vol_diff'])}")

    print("\n================  CASE-LEVEL TUMOR DETECTION  ================")
    for name in ("score=pred_volume", "score=max_prob"):
        if name not in detection:
            continue
        b = detection[name]
        print(f"  [{name}]  Se {b['sensitivity']:.3f}  Sp {b['specificity']:.3f}  "
              f"PPV {b['precision_ppv']:.3f}  F1 {b['f1']:.3f}  "
              f"BalAcc {b['balanced_accuracy']:.3f}"
              + (f"  AUROC {b['auroc']:.3f}  AUPRC {b['auprc']:.3f}" if "auroc" in b else ""))
    print("\n================  DSC BY GT TUMOR DIAMETER (per lesion)  ================")
    print(f"  {'bin (mm)':>9} | {'n':>4} | {'med Ø':>6} | {'les.Dice':>9} | {'Dice|det':>9} | {'det@any':>7} | {'det@IoU.1':>9}")
    for bl in bin_labels():
        b = by_size[bl]
        if not b["n_lesions"]:
            print(f"  {bl:>9} |    0 |")
            continue
        print(f"  {bl:>9} | {b['n_lesions']:>4} | {b['diam_mm_median']:>5.1f} | "
              f"{b['lesion_dice_mean']:>9.3f} | {b['lesion_dice_mean_detected_only']:>9.3f} | "
              f"{b['detection_rate_any_overlap']:>7.2f} | {b['detection_rate_iou>=0.1']:>9.2f}")
    print("\n  per-scan Dice, binned by the scan's largest lesion:")
    for bl in bin_labels():
        b = by_size_scan[bl]
        if not b.get("n_scans"):
            continue
        m = "n/a" if b.get("mean") is None else f"{b['mean']:.3f} (med {b['median']:.3f})"
        print(f"    {bl:>9} mm : {b['n_scans']:>3} scans   Dice {m}")

    print(f"\n  wrote {csv_path}")
    print(f"  wrote {args.out_dir / 'per_lesion_metrics.csv'}")
    print(f"  wrote {args.out_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
