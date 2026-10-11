"""Gland QC (v2, Rahul, 2026-10-06; fixed before any profile was scored).

DROP: empty outline; outline volume < 5 or > 250 cm^3; centerline failure = the centerline raises, or its length
      exceeds 1.5 x the outline's bounding-box diagonal (a skeleton path that spirals or doubles back), or the
      profile raises.
FLAG (kept, largest component used): fragmented outline (2nd component > 10% of the 1st). Part C reports targets
      with and without flagged cases.
"""
from __future__ import annotations

import numpy as np

from glandshape import core as K
from glandshape.outline import Outline


def bbox_diag_mm(outline: Outline) -> float:
    idx = np.argwhere(outline.mask)
    return float(np.linalg.norm((idx.max(0) - idx.min(0) + 1) * K.ISO_MM))


def outline_drop(outline: Outline | None) -> str | None:
    if outline is None:
        return "empty"
    v = outline.vol_mm3 / 1000.0
    if v < K.QC_VOL_MIN_CM3:
        return "vol_lt_5cm3"
    if v > K.QC_VOL_MAX_CM3:
        return "vol_gt_250cm3"
    return None


def centerline_drop(outline: Outline, cl_length_mm: float) -> str | None:
    if cl_length_mm > K.QC_CL_DIAG_RATIO * bbox_diag_mm(outline):
        return "centerline_too_long"
    return None


def flags(outline: Outline | None) -> list[str]:
    return ["fragmented"] if outline is not None and outline.fragmented else []
