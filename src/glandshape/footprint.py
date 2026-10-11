"""Part C lesion footprint and chance rate (EVALUATION ONLY; fixed 2026-10-06 in round 3, before any Part C run).

The GT lesion is read only here, never by the profile or the scores.

footprint(stop i, axis j) = True if EITHER half of axis j's diameter at stop i passes through the GT lesion. The two
halves are the in-gland ray segments [O_i, O_i + r(theta_j) d_j] and [O_i, O_i + r(theta_j + 180) d_(j+12)], i.e.
from the ray origin to each half-ray's first exit (the 40 mm cap for a censored half). A segment "passes through"
the lesion if any of its points, sampled every 0.25 mm, falls in a GT lesion voxel (nearest native voxel; the
lesion is never resampled). Only scoreable cells (finite width) can be footprint cells.
A lesion with no footprint cell is NOT EVALUABLE (reported by tier in Part C).
v5: extrapolated tip stops are not scoreable, so they are never footprint cells either.

Chance (S11 lesson: report beside every hit rate) uses the same cell definition:
  chance_fraction = number of footprint cells / number of scoreable cells of the gland's width map.
Footprint stops = stops with at least one footprint cell; a lesion is "fold-heavy" when more than half of its
footprint stops are fold-over stops (Part C reports targets with and without fold-heavy lesions).
Censored footprint = at least one footprint cell whose width is censored (a half-ray reached the 40 mm cap).
"""
from __future__ import annotations

import numpy as np

from glandshape import core as K
from glandshape.centerline import Centerline
from glandshape.outline import Outline
from glandshape.profile import N_AXES, Profile, ray_dirs

SEG_STEP_MM = 0.25
_T = np.arange(0.0, K.RAY_MAX_MM + 1e-9, SEG_STEP_MM)


def footprint(outline: Outline, cl: Centerline, pr: Profile, lesion_xyz: np.ndarray,
              exclude_tip: bool = True) -> np.ndarray:
    """(n, 12) bool footprint of one lesion given its native voxel indices lesion_xyz (N, 3). v5: restricted to
    scoreable cells (finite width, not an extrapolated tip stop); exclude_tip=False gives the pre-v5 footprint."""
    n = len(cl.s)
    fp = np.zeros((n, N_AXES), bool)
    lesion_xyz = np.asarray(lesion_xyz, int)
    vi = np.nonzero(pr.valid)[0]
    if len(lesion_xyz) == 0 or len(vi) == 0:
        return fp
    lo = lesion_xyz.min(0)
    box = np.zeros(tuple(lesion_xyz.max(0) - lo + 1), bool)
    box[tuple((lesion_xyz - lo).T)] = True
    dirs = ray_dirs(cl)[vi]                                              # (nv, 24, 3)
    pts = pr.origin[vi, None, None, :] + _T[None, None, :, None] * dirs[:, :, None, :]
    within = _T[None, None, :] <= np.nan_to_num(pr.r[vi], nan=-1.0)[:, :, None]
    nat = np.rint(outline.iso_to_native(pts.reshape(-1, 3))).astype(int) - lo
    ok = np.all((nat >= 0) & (nat < np.array(box.shape)), axis=1) & within.reshape(-1)
    hit = np.zeros(len(nat), bool)
    hit[ok] = box[tuple(nat[ok].T)]
    hit_dir = hit.reshape(len(vi), K.N_DIRS, len(_T)).any(-1)          # (nv, 24)
    fp[vi] = hit_dir[:, :N_AXES] | hit_dir[:, N_AXES:]
    return fp & (scoreable(pr) if exclude_tip else np.isfinite(pr.w))


def scoreable(pr: Profile) -> np.ndarray:
    ok = np.isfinite(pr.w)
    if getattr(pr, "tip", None) is not None:
        ok &= np.logical_not(pr.tip)[:, None]
    return ok


def chance_fraction(fp: np.ndarray, pr: Profile) -> float:
    n = int(scoreable(pr).sum())
    return float(fp.sum() / n) if n else float("nan")


def footprint_stops(fp: np.ndarray) -> np.ndarray:
    return np.nonzero(fp.any(1))[0]


def fold_heavy(fp: np.ndarray, pr: Profile) -> bool:
    st = footprint_stops(fp)
    return bool(len(st) and pr.fold[st].mean() > 0.5)


def censored_footprint(fp: np.ndarray, pr: Profile) -> bool:
    return bool((fp & pr.w_censored).any())
