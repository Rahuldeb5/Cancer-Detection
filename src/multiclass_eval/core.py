"""Pure per-case evaluation for S8. Both models go through evaluate_case(); only how the lesion mask and the
lesion probability are read differs, and that is confined to lesion_mask() / lesion_channels().

Lesion probability p:
  501  npz['probabilities'][1]  softmax tumor channel; p0 = channel 0 (= 1 - p up to float error)
  502  npz['probabilities'][1]  sigmoid lesion REGION; channel 0 is the pancreas region (1,2), NOT background,
       so p0 is taken as 1 - p (OneMinus) and only feeds S2's raw-logit columns
Both are (C, z, y, x); channel c in nibabel order is probs[c].transpose(2, 1, 0) (S2 / S7 section 7).

Lesion mask: 501 seg == 1, 502 seg == 2. For 502 (seg == 2) == (p > 0.5) exactly (S7 section 7), so the
ladder's t = 0.5 rung is the saved segmentation for both models.

Reused, not copied: S2's analyze() (search region, lesion peaks, components of p > t with peak/touch hits) and
evaluate.py's overlap_metrics / boundary_metrics / lesion_level (per-lesion Dice against the union of touching
predicted components). All arrays here are nibabel (x, y, z) with (x, y, z) spacing; evaluate.py's functions
are axis-order agnostic given matching spacing, and lesion_level's cc_label numbering on an (x, y, z) array is
tumorlib's lesion_id (same 26-connectivity, raster order is translation invariant so a bbox crop that holds
every GT voxel keeps the numbering).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label as cc_label

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "nnunet-subthreshold"))
sys.path.insert(0, str(REPO / "testing" / "nnunet"))
import case_pass as S2  # noqa: E402  (S2's per-case pass; sets tumorlib env defaults via common)
import common as C  # noqa: E402
import evaluate as EV  # noqa: E402  (testing/nnunet/evaluate.py, imported unchanged)

from tumorlib.lesions import CC_STRUCT, bbox_slices, label_lesions  # noqa: E402

MODELS = ("501", "502")
LESION_LABEL = {"501": 1, "502": 2}
BOUNDARY_PAD = 2  # voxels around the gt|pred bbox before surface distances (crop is exact: no surface on its edge)


# ------------------------------------------------------------------ model-specific reading
def lesion_mask(seg: np.ndarray, model: str) -> np.ndarray:
    """uint8 {0,1} lesion mask from a saved nnU-Net segmentation."""
    return (seg == LESION_LABEL[model]).astype(np.uint8)


class OneMinus:
    """Lazy 1 - p for the 502 sigmoid lesion channel; only indexed at a few points by S2's analyze()."""

    def __init__(self, p):
        self.p = p
        self.shape = p.shape

    def __getitem__(self, idx):
        return np.float32(1.0) - np.asarray(self.p[idx], dtype=np.float32)


def lesion_channels(P: np.ndarray, model: str):
    """(p, p0) in nibabel order for S2's analyze()."""
    p = C.xyz(P, 1)
    return (p, C.xyz(P, 0)) if model == "501" else (p, OneMinus(p))


def scan_502(P: np.ndarray, seg: np.ndarray, z_chunk: int = 32) -> dict:
    """Region-npz consistency (S7 section 7) over the whole volume, chunked over the stored z axis, plus what
    S2's analyze() needs from a volume scan: the bbox of p > min(THRESHOLDS) and the volume peak."""
    t_lo = min(C.THRESHOLDS)
    nz = P.shape[1]
    mis_les = mis_gland = n_one = 0
    pmax = -np.inf
    any_x, any_y, any_z = np.zeros(P.shape[3], bool), np.zeros(P.shape[2], bool), np.zeros(nz, bool)
    for s in range(0, nz, z_chunk):
        b0, b1 = P[0, s:s + z_chunk], P[1, s:s + z_chunk]
        sg = seg[:, :, s:s + z_chunk].transpose(2, 1, 0)
        mis_les += int(((b1 > 0.5) != (sg == 2)).sum())
        mis_gland += int((((b0 > 0.5) | (b1 > 0.5)) != (sg > 0)).sum())
        n_one += int((b1 == 1.0).sum())
        pmax = max(pmax, float(b1.max()))
        m = b1 > t_lo
        if m.any():
            any_z[s:s + z_chunk] |= m.any(axis=(1, 2))
            any_y |= m.any(axis=(0, 2))
            any_x |= m.any(axis=(0, 1))
    bb = None
    if any_x.any():
        rng = [np.nonzero(a)[0] for a in (any_x, any_y, any_z)]
        bb = tuple(slice(int(r[0]), int(r[-1]) + 1) for r in rng)
    return dict(argmax_mismatch=mis_les, gland_mismatch=mis_gland, max_sum_dev=np.nan, n_p1_eq_1=n_one,
                n_p0_eq_0=np.nan, vol_peak_p=pmax, vol_peak_raw_logit=np.nan, bbox_lo=bb)


# ------------------------------------------------------------------ prepare_excluded_eval semantics
def exclusion_variant(gt_labels: np.ndarray, pred: np.ndarray, excluded: list[int]):
    """src/nnunet/prepare_excluded_eval.py's "ignore region" rule, in nibabel order (the rule is per voxel and
    per 26-connected component, so axis order does not matter):
      GT: excluded components -> background.
      pred: a component overlapping an excluded lesion and NO kept lesion is removed; every predicted voxel
            inside an excluded lesion is removed.
    Returns (gt_kept uint8, pred_kept uint8, ignore bool = excluded | removed, n_removed)."""
    excl = np.isin(gt_labels, excluded)
    kept = (gt_labels > 0) & ~excl
    plab, pn = cc_label(pred > 0, structure=CC_STRUCT)
    ov_excl = np.bincount(plab[excl], minlength=pn + 1)
    ov_kept = np.bincount(plab[kept], minlength=pn + 1)
    remove = (ov_excl > 0) & (ov_kept == 0)
    remove[0] = False
    removed = np.isin(plab, np.nonzero(remove)[0])
    new_pred = (pred > 0) & ~removed & ~excl
    return kept.astype(np.uint8), new_pred.astype(np.uint8), excl | removed, int(remove.sum())


# ------------------------------------------------------------------ evaluate.py metrics on one case
def evaluate_metrics(gt: np.ndarray, pred: np.ndarray, sp, max_prob: float) -> tuple[dict, list[dict]]:
    """evaluate.py main()'s per-case row and lesion_level() rows, computed on the gt|pred bbox crop.
    gt/pred uint8 {0,1} (TUMOR = 1). Lesion rows get lesion_id = 1..n in nibabel raster order."""
    sp = tuple(float(s) for s in sp)
    ov = EV.overlap_metrics(gt, pred)
    gt_pos = ov["ref_voxels"] > 0
    nan_bd = {"hd95": np.nan, "assd": np.nan, **{f"nsd_{t}mm": np.nan for t in EV.NSD_TOL_MM}}
    sl = bbox_slices((gt > 0) | (pred > 0), BOUNDARY_PAD)
    lesions = []
    if gt_pos:
        g, p = gt[sl], pred[sl]
        bd = EV.boundary_metrics(g, p, sp)
        lesions = EV.lesion_level(g, p, sp)
        for j, r in enumerate(lesions, 1):
            r["lesion_id"] = j
    else:
        bd = nan_bd
    voxvol = float(np.prod(sp))
    row = {"gt_tumor": int(gt_pos),
           "dice": ov["dice"] if gt_pos else np.nan, "iou": ov["iou"] if gt_pos else np.nan, **bd,
           "pred_vol_mm3": ov["pred_voxels"] * voxvol, "ref_vol_mm3": ov["ref_voxels"] * voxvol,
           "max_tumor_prob": max_prob,
           # nnU-Net summary.json counts and Dice (regression target)
           **{f"nn_{k}": v for k, v in nn_counts(gt > 0, pred > 0).items()}}
    rv, pv = row["ref_vol_mm3"], row["pred_vol_mm3"]
    row["rel_vol_diff"] = ((pv - rv) / rv) if rv > 0 else np.nan
    row["missed"] = int(gt_pos and ov["pred_voxels"] == 0)
    if gt_pos:
        largest = max((x["diam_mm"] for x in lesions), default=0.0)
        row["n_gt_lesions"] = len(lesions)
        row["largest_lesion_diam_mm"] = largest
        for x in lesions:
            x["scan_largest_diam_mm"] = largest
    return row, lesions


def nn_counts(gt_b: np.ndarray, pred_b: np.ndarray) -> dict:
    """nnU-Net's per-case region metric: Dice = 2TP / (2TP + FP + FN), NaN when all three are 0."""
    tp = int(np.logical_and(gt_b, pred_b).sum())
    n_ref, n_pred = int(gt_b.sum()), int(pred_b.sum())
    fp, fn = n_pred - tp, n_ref - tp
    d = 2 * tp + fp + fn
    return dict(tp=tp, fp=fp, fn=fn, n_ref=n_ref, n_pred=n_pred, dice=(2 * tp / d) if d else np.nan)


def localization(seg_any: bool, touches: bool) -> str:
    """Case-level localization-aware outcome for a tumor+ case."""
    return "localized" if touches else ("wrong_place" if seg_any else "nothing")


# ------------------------------------------------------------------ one case, everything
def evaluate_case(model: str, case_id: str, fold: int, seg: np.ndarray, gt: np.ndarray, P: np.ndarray,
                  sr: np.ndarray | None, sp, excluded: list[int], gt_gland: np.ndarray | None = None) -> dict:
    """seg: saved segmentation (x,y,z), labels {0,1} (501) or {0,1,2} (502). gt: uint8 {0,1} lesion GT.
    P: npz probabilities (C, z, y, x). sr: search region or None. excluded: lesion_ids (tumorlib numbering)
    of this case's S1 `separated` lesions. gt_gland: 502 only, uint8 {0,1} (502 GT label > 0)."""
    pred = lesion_mask(seg, model)
    p, p0 = lesion_channels(P, model)
    if not (p.shape == seg.shape == gt.shape):
        raise ValueError(f"{case_id}: shape p {p.shape} seg {seg.shape} gt {gt.shape}")
    vol = S2.volume_scan(P, pred) if model == "501" else scan_502(P, seg)
    res = S2.analyze(case_id, fold, p, p0, pred, gt, sr, sp, vol)

    max_prob = float(vol["vol_peak_p"])
    ev, ev_les = evaluate_metrics(gt, pred, sp, max_prob)
    labels, n = label_lesions(gt)
    s2_vox = [r["gt_vox"] for r in sorted(res["lesions"], key=lambda r: r["lesion_id"])]
    if [r["gt_vox"] for r in ev_les] != s2_vox:  # evaluate.py numbering == tumorlib numbering, checked per case
        raise AssertionError(f"{case_id}: evaluate.lesion_level ids disagree with tumorlib label_lesions")
    out = dict(model=model, case=res["case"], lesions=res["lesions"], comps=res["comps"], hist=res["hist"],
               ev_case=ev, ev_lesions=ev_les)
    out["case"]["seg_labels"] = ";".join(str(int(v)) for v in np.unique(seg))

    # ---- without the S1 `separated` lesions (prepare_excluded_eval semantics)
    excluded = sorted(int(j) for j in excluded)
    if excluded:
        if max(excluded) > n:
            raise ValueError(f"{case_id}: excluded ids {excluded} but only {n} GT lesions")
        if len(excluded) == n:
            out["excl"] = dict(action="dropped")
        else:
            g2, p2, ignore, n_rm = exclusion_variant(labels, pred, excluded)
            outside = np.asarray(p, dtype=np.float32)[~ignore]
            ev2, les2 = evaluate_metrics(g2, p2, sp, float(outside.max()) if outside.size else 0.0)
            out["excl"] = dict(action="modified", n_pred_comps_removed=n_rm, ev_case=ev2, ev_lesions=les2,
                               seg_any=bool(p2.any()), seg_touches_gt=bool((p2.astype(bool) & g2.astype(bool)).any()))
    else:
        out["excl"] = dict(action="unchanged")

    if gt_gland is not None:
        out["gland"] = nn_counts(gt_gland.astype(bool), seg > 0)
    return out
