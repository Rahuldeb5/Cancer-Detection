"""Shared paths, loaders and logit helpers for session S2 (nnU-Net sub-threshold probe).

Runs on the deep-server, where the out-of-fold nnU-Net predictions live. The server has
no CTs, so tumorlib's CT grid is pointed at the nnU-Net gt_segmentations headers (a
symlink farm in work/, one <case>/ct.nii.gz per case). tumorlib only reads the header
from it (shape, affine, spacing); the spacing it yields is checked against the master
table's CT-affine spacing in summarize.py.

Axis order: everything is nibabel (x, y, z). nnU-Net's npz stores probabilities as
(C, z, y, x); channel c in nibabel order is probs[c].transpose(2, 1, 0) (a view).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HOME = Path.home()
REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work" / "nnunet_subthreshold"
PRED_ROOT = HOME / "research/nnunet_env/nnUNet_results/Dataset501_PanTSTumor/nnUNetTrainer__nnUNetPlans__3d_fullres"
GT_DIR = HOME / "research/nnunet_env/nnUNet_preprocessed/Dataset501_PanTSTumor/gt_segmentations"

# tumorlib reads these at import time
os.environ.setdefault("TUMORLIB_MASK_ROOT", str(HOME / "research/datasets/pants/duct_masks"))
os.environ.setdefault("TUMORLIB_CT_ROOT", str(WORK / "ctgrid"))
sys.path.insert(0, str(REPO / "src"))

import nibabel as nib  # noqa: E402
from tumorlib import io as tio  # noqa: E402

TUMOR = 1
EPS = 1e-7  # probability clip for the logit scale: logit range +-16.118
THRESHOLDS = (0.9, 0.7, 0.5, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
DIL_MM = 2.0
EXPORT_MARGIN_MM = 20.0


def fold_of_case() -> dict[str, int]:
    """case -> nnU-Net fold (fold_k_ids.txt <-> fold k-1)."""
    return {c: k - 1 for k in range(1, 6) for c in tio.fold_ids(k)}


def pred_paths(case_id: str, fold: int) -> tuple[Path, Path]:
    v = PRED_ROOT / f"fold_{fold}" / "validation"
    return v / f"{case_id}.nii.gz", v / f"{case_id}.npz"


def load_seg(case_id: str, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Saved nnU-Net segmentation (uint8 {0,1}, nibabel order) and its affine."""
    img = nib.load(pred_paths(case_id, fold)[0])
    return (np.asarray(img.dataobj) > 0).astype(np.uint8), img.affine


def load_gt(case_id: str) -> tuple[np.ndarray, np.ndarray]:
    """nnU-Net gt_segmentations label (uint8 {0,1}) and its affine. Same raw>0.5 rule as tumorlib."""
    img = nib.load(GT_DIR / f"{case_id}.nii.gz")
    raw = img.dataobj.get_unscaled()
    thr = tio._raw_threshold(img.dataobj)
    return (raw > thr).astype(np.uint8), img.affine


def load_probs(case_id: str, fold: int) -> np.ndarray:
    """Full (C, z, y, x) softmax array exactly as saved (dtype untouched)."""
    with np.load(pred_paths(case_id, fold)[1]) as d:
        return d["probabilities"]


def xyz(probs: np.ndarray, c: int) -> np.ndarray:
    """Channel c in nibabel (x, y, z) order, as a view."""
    return probs[c].transpose(2, 1, 0)


def clip_logit(p) -> np.ndarray:
    """logit(clip(p, EPS, 1-EPS)) in float64. Range +-16.118."""
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def raw_logit(p1, p0) -> np.ndarray:
    """log(p1) - log(p0) in float64, no clipping: recovers logits above the float32
    ceiling of p1 (p1 == 1.0 while p0 is still a tiny positive float). +-inf where a channel is 0."""
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(p1, dtype=np.float64)) - np.log(np.asarray(p0, dtype=np.float64))
