"""Steps 3-4: perpendicular cross-sections, fold-over flags, rays -> r(s, theta), WIDTH w(s, axis), A(s), HU rays.

Cross-section at stop i: the plane through P_i perpendicular to T_i, sampled on a 0.5 mm grid (+-40 mm) from the
trilinear-interpolated iso outline (inside = value >= 0.5). Only the 4-connected 2D region containing the ray origin
is kept, so a neighbouring limb of the gland cut by the same plane does not count. A(s) = region samples x 0.25 mm^2.

Ray origin O_i (v3, round 3): the centerline point P_i. If P_i is outside the gland in its own plane, O_i = the
nearest inside sample of that plane (origin_fallback, shift recorded); a plane with no inside sample is invalid.
(v2, superseded, kept only for comparison: the centroid of the kept region; it split a bulge's signal between the
bulge side and the opposite side and left seams where the region changed between stops.)

Fold-over: stop i is flagged when some point of its kept region lies within FOLD_TOL_MM of another stop j's plane
AND inside stop j's kept region, i.e. the two planes intersect inside the gland (expected at tight bends). Fold-over
stops stay in the profile and in the scoring (v3); they are only flagged.

Rays: 24 directions theta_k = k * 15 deg, d = cos(theta) N + sin(theta) B, cast in the section plane from O_i.
r = distance to the FIRST exit (value < 0.5), marched at 0.1 mm and refined by linear interpolation; no exit by
40 mm -> r = 40, censored. The radius map r is a SECONDARY output (which side) and never a score (v3).
WIDTH (v3, PRIMARY measurement): 12 axes j = 0..11 (axis j = directions j and j + 12, i.e. theta and theta + 180):
w[:, j] = r[:, j] + r[:, j + 12]; censored if either half is capped. To first order the width does not depend on
where the origin sits on the axis line (moving the origin along the axis changes nothing; a sideways offset delta
shortens a circular chord by ~delta^2 / R, second order).
HU along each ray at 0, 1, ..., 40 mm (trilinear on the native CT) when a CT sampler is given.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label, map_coordinates

from glandshape import core as K
from glandshape.centerline import Centerline
from glandshape.outline import Outline

GRID = np.arange(-K.RAY_MAX_MM, K.RAY_MAX_MM + 1e-9, K.PLANE_STEP_MM)
C0 = len(GRID) // 2
T_RAY = np.arange(0.0, K.RAY_MAX_MM + 1e-9, K.RAY_STEP_MM)
T_HU = np.arange(0.0, K.RAY_MAX_MM + 1e-9, K.HU_STEP_MM)
N_AXES = K.N_DIRS // 2


@dataclass
class Profile:
    s: np.ndarray            # (n,) mm
    u: np.ndarray            # (n,)
    r: np.ndarray            # (n, 24) mm radius map (secondary), NaN at invalid stops
    censored: np.ndarray     # (n, 24) bool: no exit within 40 mm
    A: np.ndarray            # (n,) mm^2
    valid: np.ndarray        # (n,) bool
    fold: np.ndarray         # (n,) bool fold-over flag
    hu: np.ndarray | None    # (n, 24, 41) float16
    L: float
    origin: np.ndarray = None           # (n, 3) ray origins, iso coords (NaN at invalid stops)
    origin_fallback: np.ndarray = None  # (n,) bool: origin is not the centerline point (see module doc)
    origin_shift: np.ndarray = None     # (n,) mm from the centerline point to the origin
    w: np.ndarray = None                # (n, 12) mm WIDTH (primary)
    w_censored: np.ndarray = None       # (n, 12) bool: either half-ray capped at 40 mm
    tip: np.ndarray = None              # (n,) bool: extrapolated tip stop (v5: excluded from every score)


def _sample(f: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return map_coordinates(f, pts.reshape(-1, 3).T, order=1, mode="constant", cval=0.0).reshape(pts.shape[:-1])


def ray_dirs(cl: Centerline) -> np.ndarray:
    """(n, 24, 3) unit ray directions in iso coords."""
    c, s = np.cos(K.THETAS), np.sin(K.THETAS)
    return c[None, :, None] * cl.N[:, None, :] + s[None, :, None] * cl.B[:, None, :]


def width_from_radius(r: np.ndarray, cens: np.ndarray):
    h = r.shape[1] // 2
    return r[:, :h] + r[:, h:], cens[:, :h] | cens[:, h:]


def compute_profile(outline: Outline, cl: Centerline, ct_sampler=None, do_fold: bool = True,
                    origin_mode: str | None = None) -> Profile:
    mode = origin_mode or K.RAY_ORIGIN
    f = outline.mask.astype(np.float32)
    n = len(cl.s)
    ga, gb = np.meshgrid(GRID, GRID, indexing="ij")
    regions, region_pts = [None] * n, [None] * n
    A = np.full(n, np.nan)
    valid = np.zeros(n, bool)
    O = np.full((n, 3), np.nan)
    fallback = np.zeros(n, bool)
    shift = np.full(n, np.nan)
    for i in range(n):
        pts = cl.P[i] + ga[..., None] * cl.N[i] + gb[..., None] * cl.B[i]
        inside = _sample(f, pts) >= 0.5
        if inside[C0, C0]:
            ia, ib = C0, C0
        elif mode == "centerline" and inside.any():
            cand = np.argwhere(inside)
            d2 = ((cand - C0) ** 2).sum(1)
            ia, ib = (int(v) for v in cand[np.argmin(d2)])
            fallback[i] = True
        else:
            continue
        lab, _ = label(inside)
        reg = lab == lab[ia, ib]
        regions[i], region_pts[i] = reg, pts[reg]
        A[i] = reg.sum() * K.PLANE_STEP_MM ** 2
        valid[i] = True
        O[i] = cl.P[i] + GRID[ia] * cl.N[i] + GRID[ib] * cl.B[i]
        if mode == "centroid":  # v2, superseded (comparison only)
            ac, bc = ga[reg].mean(), gb[reg].mean()
            ja, jb = (int(np.rint((x + K.RAY_MAX_MM) / K.PLANE_STEP_MM)) for x in (ac, bc))
            oc = cl.P[i] + ac * cl.N[i] + bc * cl.B[i]
            if reg[ja, jb] and _sample(f, oc[None])[0] >= 0.5:
                O[i] = oc
            else:
                fallback[i] = True
        shift[i] = float(np.linalg.norm(O[i] - cl.P[i]))

    dirs = ray_dirs(cl)
    r = np.full((n, K.N_DIRS), np.nan)
    cens = np.zeros((n, K.N_DIRS), bool)
    vi = np.nonzero(valid)[0]
    if len(vi):
        pts = O[vi, None, None, :] + T_RAY[None, None, :, None] * dirs[vi, :, None, :]
        v = _sample(f, pts)                                  # (nv, 24, nt)
        outside = v < 0.5
        has = outside.any(-1)
        k = np.argmax(outside, axis=-1)                      # first outside sample (k >= 1 when v[0] >= 0.5)
        k = np.where(has, k, len(T_RAY) - 1)
        vp = np.take_along_axis(v, np.maximum(k - 1, 0)[..., None], -1)[..., 0]
        vk = np.take_along_axis(v, k[..., None], -1)[..., 0]
        frac = np.clip((vp - 0.5) / np.maximum(vp - vk, 1e-9), 0, 1)
        rr = T_RAY[np.maximum(k - 1, 0)] + frac * K.RAY_STEP_MM
        r[vi] = np.where(has, rr, K.RAY_MAX_MM)
        cens[vi] = np.logical_not(has)
    w, wc = width_from_radius(r, cens)

    fold = fold_over(cl, regions, region_pts) if do_fold else np.zeros(n, bool)

    hu = None
    if ct_sampler is not None:
        hu = np.full((n, K.N_DIRS, len(T_HU)), np.nan, np.float16)
        if len(vi):
            pts = O[vi, None, None, :] + T_HU[None, None, :, None] * dirs[vi, :, None, :]
            hu[vi] = ct_sampler(pts.reshape(-1, 3)).reshape(pts.shape[:-1]).astype(np.float16)
    return Profile(s=cl.s, u=cl.u, r=r, censored=cens, A=A, valid=valid, fold=fold, hu=hu, L=cl.L,
                   origin=O, origin_fallback=fallback, origin_shift=shift, w=w, w_censored=wc,
                   tip=cl.tip if cl.tip is not None else np.zeros(n, bool))


def fold_over(cl: Centerline, regions, region_pts) -> np.ndarray:
    n = len(cl.s)
    fold = np.zeros(n, bool)
    vi = np.array([i for i in range(n) if regions[i] is not None], int)
    if len(vi) < 2:
        return fold
    Pv, Tv, Nv, Bv = cl.P[vi], cl.T[vi], cl.N[vi], cl.B[vi]
    off_T = np.einsum("jk,jk->j", Pv, Tv)
    for a, i in enumerate(vi):
        p = region_pts[i]                                     # (m, 3)
        D = p @ Tv.T - off_T[None, :]                         # signed distance to every valid stop's plane
        D[:, a] = np.inf
        mi, mj = np.nonzero(np.abs(D) <= K.FOLD_TOL_MM)
        if not len(mi):
            continue
        rel = p[mi] - Pv[mj]
        ia = np.rint((np.einsum("mk,mk->m", rel, Nv[mj]) + K.RAY_MAX_MM) / K.PLANE_STEP_MM).astype(int)
        ib = np.rint((np.einsum("mk,mk->m", rel, Bv[mj]) + K.RAY_MAX_MM) / K.PLANE_STEP_MM).astype(int)
        ok = (ia >= 0) & (ia < len(GRID)) & (ib >= 0) & (ib < len(GRID))
        for jj, x, y in zip(mj[ok], ia[ok], ib[ok]):
            if regions[vi[jj]][x, y]:
                fold[i] = fold[vi[jj]] = True
                break
    return fold


def make_ct_sampler(outline: Outline, ct_shape, load_ct_block):
    """CT sampler for rays. load_ct_block(slices) -> float32 HU block of the native CT (e.g.
    lambda sl: tumorlib.io.load_ct(case_id, sl)). The block covers the iso outline's native extent + the ray length,
    so the full CT is never held in memory."""
    ext = np.array(outline.mask.shape, float) - 1
    corners = np.array([[x, y, z] for x in (0, ext[0]) for y in (0, ext[1]) for z in (0, ext[2])])
    nat = outline.iso_to_native(corners)
    pad = np.ceil((K.RAY_MAX_MM + 2) / outline.spacing)
    lo = np.maximum(np.floor(nat.min(0) - pad).astype(int), 0)
    hi = np.minimum(np.ceil(nat.max(0) + pad).astype(int) + 1, np.asarray(ct_shape, int))
    block = load_ct_block(tuple(slice(int(a), int(b)) for a, b in zip(lo, hi)))
    if block is None:
        return None

    def sample(iso_pts: np.ndarray) -> np.ndarray:
        q = outline.iso_to_native(iso_pts) - lo
        return map_coordinates(block, q.T, order=1, mode="nearest")
    return sample
