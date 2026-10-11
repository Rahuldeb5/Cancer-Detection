"""Step 5: scores on r(s, theta) and A(s). Fixed before any real gland was scored (core.py).

v5 (round 5, 2026-10-08): PRIMARY = multi-scale bulge_local on RAW width with the extrapolated tip stops set to NaN
(scoring_width); the residual score is SECONDARY (reported only). Use primary_scores(pr, ref).
v4 (superseded): PRIMARY = bulge_residual = the multi-scale rule on the RESIDUAL width
w - median_normal_width(u bin, axis) from the 250 reference bank glands; raw-width bulge_local is SECONDARY.
Part A's known-answer gates still use raw width (they have no bank); the residual version is reported there with
synthetic references.

All bulge scores take a (stops x columns) map. v3 (round 3): the PRIMARY input is the WIDTH map (12 axes,
profile.Profile.w); the 24-direction radius map is a secondary output only and is never scored. Smoothing and the
column wrap work the same for 12 axes (axis 11 neighbours axis 0) as for 24 directions.

bulge_local(s, axis)    PRIMARY (v2 rule, v3 on width, multi-scale). At each half-window h in BULGE_SCALES_MM (15, 30, 45 mm):
                        residual_h = r - median of r over stops within s +- h at the same theta (window truncated
                        at the gland ends; NaN ignored), z_h = residual_h / the gland's own scaled MAD of residual_h.
                        Primary = max over scales of max(z_h, 0) (outward only), then nan-mean smoothed over
                        3 stops x 3 directions (theta wraps). Secondary, report-only: dent side = min over scales of
                        min(z_h, 0), and |bulge| = max over scales of |z_h|, smoothed the same way.
                        v1 (superseded, kept as bulge_local_v1): one +-10 mm window, signed z; it absorbed bulges
                        longer than ~20 mm (Part A).
bulge_global(s, axis)   (w - median_normal(u_bin, axis)) / MAD_normal(u_bin, axis), u bins of 0.05, normal reference
                        of widths (v3). Secondary.
atrophy(s)              log(median A over stops tailward of s / median A over stops headward of s), each side needing
                        >= 5 finite stops; robust z against the normal reference in the same u bin
                        ((x - median) / scaled MAD). CASE LEVEL ONLY (v2): case atrophy = max |z| (signed value kept).
                        Whole-side medians cannot localize a step (a plateau by construction, Part A).
atrophy_win(s)          secondary localizer only: the same ratio over +-20 mm windowed sides, z-scored the same way.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from glandshape import core as K

N_UBINS = int(round(1 / K.U_BIN))


def u_bin(u: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(np.asarray(u) / K.U_BIN).astype(int), 0, N_UBINS - 1)


def _mad(x: np.ndarray, axis=None) -> np.ndarray:
    med = np.nanmedian(x, axis=axis, keepdims=True)
    return K.MAD_SCALE * np.nanmedian(np.abs(x - med), axis=axis)


def local_residual(r: np.ndarray, half_mm: float = K.LOCAL_HALF_MM) -> np.ndarray:
    """r - running median over +- half_mm at the same theta."""
    h = int(round(half_mm / K.STOP_MM))
    n = len(r)
    med = np.full_like(r, np.nan)
    for i in range(n):
        w = r[max(0, i - h):min(n, i + h + 1)]
        with np.errstate(all="ignore"):
            if np.isfinite(w).any():
                med[i] = np.nanmedian(w, axis=0)
    return r - med


def smooth_map(z: np.ndarray) -> np.ndarray:
    """nan-mean over SMOOTH_STOPS x SMOOTH_DIRS (theta wraps, s truncates); NaN where the centre is NaN."""
    hs, hd = K.SMOOTH_STOPS // 2, K.SMOOTH_DIRS // 2
    n, m = z.shape
    acc = np.zeros_like(z)
    cnt = np.zeros_like(z)
    fin = np.isfinite(z)
    zz = np.where(fin, z, 0.0)
    for di in range(-hs, hs + 1):
        lo, hi = max(0, -di), min(n, n - di)
        for dj in range(-hd, hd + 1):
            sh = np.roll(zz, -dj, axis=1)
            fh = np.roll(fin, -dj, axis=1)
            acc[lo:hi] += sh[lo + di:hi + di]
            cnt[lo:hi] += fh[lo + di:hi + di]
    out = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    out[np.logical_not(fin)] = np.nan
    return out


def _scale_z(r: np.ndarray, half_mm: float) -> tuple[np.ndarray, float]:
    res = local_residual(r, half_mm)
    mad = float(_mad(res)) if np.isfinite(res).any() else np.nan
    z = res / max(mad, 1e-6) if np.isfinite(mad) else res * np.nan
    return z, mad


def bulge_local_v1(r: np.ndarray) -> tuple[np.ndarray, float]:
    """v1 (superseded): signed single-scale +-10 mm score; (score map, MAD in mm)."""
    z, mad = _scale_z(r, K.LOCAL_HALF_MM)
    return smooth_map(z), mad


def _nanreduce(fn, stack):
    """fn (np.fmax / np.fmin) over scales ignoring NaN; NaN only where every scale is NaN."""
    out = stack[0]
    for z in stack[1:]:
        out = fn(out, z)
    return out


def bulge_multiscale(r: np.ndarray) -> dict:
    """v2 scores: primary (outward), dent (inward, <= 0), absval, per-scale MADs (mm) and the winning scale index."""
    zs, mads = [], []
    for h in K.BULGE_SCALES_MM:
        z, m = _scale_z(r, h)
        zs.append(z); mads.append(m)
    zs = np.stack(zs)
    pos = np.where(np.isfinite(zs), np.maximum(zs, 0.0), np.nan)
    neg = np.where(np.isfinite(zs), np.minimum(zs, 0.0), np.nan)
    with np.errstate(all="ignore"):
        best = np.where(np.isfinite(pos).any(0), np.nanargmax(np.where(np.isfinite(pos), pos, -np.inf), 0), -1)
    return dict(primary=smooth_map(_nanreduce(np.fmax, pos)), dent=smooth_map(_nanreduce(np.fmin, neg)),
                absval=smooth_map(_nanreduce(np.fmax, np.abs(zs))), mads=mads, best_scale=best)


def bulge_local(r: np.ndarray) -> tuple[np.ndarray, list]:
    """PRIMARY v2 score map (outward, >= 0) and the per-scale MADs (mm)."""
    b = bulge_multiscale(r)
    return b["primary"], b["mads"]


# ---------------------------------------------------------------- normal reference
@dataclass
class Reference:
    w_med: np.ndarray        # (20, 12) normal width median per (u bin, axis)
    w_mad: np.ndarray        # (20, 12)
    atro_med: np.ndarray     # (20,)
    atro_mad: np.ndarray     # (20,)
    n_glands: int
    atw_med: np.ndarray = None   # (20,) windowed (secondary localizer)
    atw_mad: np.ndarray = None


def atrophy_raw(A: np.ndarray) -> np.ndarray:
    n = len(A)
    out = np.full(n, np.nan)
    for i in range(n):
        head, tail = A[:i], A[i + 1:]
        head, tail = head[np.isfinite(head)], tail[np.isfinite(tail)]
        if len(head) >= K.ATROPHY_MIN_SIDE and len(tail) >= K.ATROPHY_MIN_SIDE:
            mh, mt = np.median(head), np.median(tail)
            if mh > 0 and mt > 0:
                out[i] = np.log(mt / mh)
    return out


def atrophy_win_raw(A: np.ndarray, half_mm: float = K.ATROPHY_WIN_HALF_MM) -> np.ndarray:
    """Secondary localizer: log(median A over (s, s + half] / median A over [s - half, s)), >= 5 stops per side."""
    h = int(round(half_mm / K.STOP_MM))
    n = len(A)
    out = np.full(n, np.nan)
    for i in range(n):
        hd, tl = A[max(0, i - h):i], A[i + 1:i + 1 + h]
        hd, tl = hd[np.isfinite(hd)], tl[np.isfinite(tl)]
        if len(hd) >= K.ATROPHY_MIN_SIDE and len(tl) >= K.ATROPHY_MIN_SIDE:
            mh, mt = np.median(hd), np.median(tl)
            if mh > 0 and mt > 0:
                out[i] = np.log(mt / mh)
    return out


def build_reference(profiles) -> Reference:
    """profiles: iterable of objects with .u, .w (width map), .A (normal glands)."""
    profiles = list(profiles)
    ncol = profiles[0].w.shape[1] if profiles else K.N_DIRS // 2
    rs = [[[] for _ in range(ncol)] for _ in range(N_UBINS)]
    at = [[] for _ in range(N_UBINS)]
    aw = [[] for _ in range(N_UBINS)]
    n = 0
    for p in profiles:
        n += 1
        b = u_bin(p.u)
        a = atrophy_raw(p.A)
        awin = atrophy_win_raw(p.A)
        for i, bi in enumerate(b):
            if np.isfinite(awin[i]):
                aw[bi].append(awin[i])
            for k in range(ncol):
                if np.isfinite(p.w[i, k]):
                    rs[bi][k].append(p.w[i, k])
            if np.isfinite(a[i]):
                at[bi].append(a[i])
    width_med = np.array([[np.median(v) if v else np.nan for v in row] for row in rs])
    width_mad = np.array([[_mad(np.asarray(v)) if v else np.nan for v in row] for row in rs])
    a_med = np.array([np.median(v) if v else np.nan for v in at])
    a_mad = np.array([_mad(np.asarray(v)) if v else np.nan for v in at])
    aw_med = np.array([np.median(v) if v else np.nan for v in aw])
    aw_mad = np.array([_mad(np.asarray(v)) if v else np.nan for v in aw])
    return Reference(width_med, width_mad, a_med, a_mad, n, aw_med, aw_mad)


def bulge_global(w: np.ndarray, u: np.ndarray, ref: Reference) -> np.ndarray:
    b = u_bin(u)
    return (w - ref.w_med[b]) / np.maximum(ref.w_mad[b], 1e-6)


def atrophy_z(A: np.ndarray, u: np.ndarray, ref: Reference) -> np.ndarray:
    b = u_bin(u)
    return (atrophy_raw(A) - ref.atro_med[b]) / np.maximum(ref.atro_mad[b], 1e-6)


def scoring_width(w: np.ndarray, tip: np.ndarray | None) -> np.ndarray:
    """v5: the width map used by every score; extrapolated tip stops are NaN (not scored, in no window)."""
    out = np.array(w, float, copy=True)
    if tip is not None:
        out[np.asarray(tip, bool)] = np.nan
    return out


def primary_scores(pr, ref: "Reference | None" = None) -> dict:
    """v5: {'primary': raw-width multi-scale (outward), 'dent', 'residual' (secondary; needs ref), 'w': scored map}."""
    w = scoring_width(pr.w, getattr(pr, "tip", None))
    b = bulge_multiscale(w)
    out = dict(primary=b["primary"], dent=b["dent"], best_scale=b["best_scale"], mads=b["mads"], w=w)
    if ref is not None:
        out["residual"] = bulge_residual(w, pr.u, ref)["primary"]
    return out


def residual_width(w: np.ndarray, u: np.ndarray, ref: Reference) -> np.ndarray:
    """v4: width minus the normal median width at the same (u bin, axis)."""
    return w - ref.w_med[u_bin(u)]


def bulge_residual(w: np.ndarray, u: np.ndarray, ref: Reference) -> dict:
    """v4 PRIMARY (Part B onward): the multi-scale bulge_local rule applied to residual_width. Same keys as
    bulge_multiscale. Raw-width bulge_multiscale(w) is kept as the secondary score."""
    return bulge_multiscale(residual_width(w, u, ref))


def atrophy_win_z(A: np.ndarray, u: np.ndarray, ref: Reference) -> np.ndarray:
    b = u_bin(u)
    return (atrophy_win_raw(A) - ref.atw_med[b]) / np.maximum(ref.atw_mad[b], 1e-6)


def case_atrophy(z: np.ndarray) -> tuple[float, float, int]:
    """(max |z|, signed z at that stop, its stop index); NaN if no finite stop."""
    if not np.isfinite(z).any():
        return np.nan, np.nan, -1
    i = int(np.nanargmax(np.abs(z)))
    return float(abs(z[i])), float(z[i]), i
