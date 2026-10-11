"""Part C metric definitions (fixed 2026-10-06 in round 4, before any Part C run; nothing here reads a GT mask).

All metrics take a gland's PRIMARY score map (v4: bulge_residual(...)["primary"], shape (stops, 12 axes)) and a
lesion footprint (glandshape.footprint, same shape). Axes are lines 15 deg apart; axis distance between j and k is
15 * min(|j - k|, 12 - |j - k|) deg (max 90).

a. Within-scan percentile. Window = the footprint's bounding box: (its stop extent) x (the shortest circular arc of
   axes covering its axes). The window slides over every position in the same gland; axes wrap, stops do not.
   Window score = max primary cell score in the window (NaN ignored; all-NaN windows are skipped). The lesion's
   score = max primary score over its footprint cells. It is compared against every window that shares no cell with
   the footprint; percentile = 100 * (#windows below + 0.5 * #ties) / #windows.
b. Top-3. The 3 highest peaks per gland after greedy non-max suppression: cells are taken in decreasing score; a cell
   is suppressed if it is < 10 mm along the centerline AND < 45 deg in axis from an already kept peak (peaks must be
   >= 10 mm apart OR >= 45 deg apart). Hit = any of the 3 peaks inside the footprint.
c. Chance for top-3, two versions; the gate uses the LARGER:
   - uniform: exact probability that 3 suppressed peaks, placed one after another uniformly over the scoreable cells
     still allowed by the suppression rule, put at least one peak in the footprint (closed form over all first and
     second placements; no sampling);
   - empirical: fraction of the 50 held-out normal glands whose top-3 peaks include one inside the footprint's
     (u range) x (axis arc), mapped by u.
"""
from __future__ import annotations

import numpy as np

N_AX = 12
AX_DEG = 15.0
SEP_MM = 10.0
SEP_DEG = 45.0
TOPK = 3


def axis_dist_deg(j, k):
    d = np.abs(np.asarray(j) - np.asarray(k)) % N_AX
    return AX_DEG * np.minimum(d, N_AX - d)


def axis_arc(axes) -> tuple[int, int]:
    """Shortest circular arc covering the axes: (start axis, length in axes)."""
    occ = np.zeros(N_AX, bool)
    occ[np.asarray(list(axes), int) % N_AX] = True
    if occ.all():
        return 0, N_AX
    idx = np.nonzero(occ)[0]
    # largest circular gap of unoccupied axes; the arc starts right after it
    gaps = [((idx[(k + 1) % len(idx)] - idx[k]) % N_AX or N_AX) - 1 for k in range(len(idx))]
    k = int(np.argmax(gaps))
    start = int(idx[(k + 1) % len(idx)])
    return start, N_AX - gaps[k]


def arc_axes(start: int, length: int) -> np.ndarray:
    return (start + np.arange(length)) % N_AX


def _sliding_any_max(a: np.ndarray, fp: np.ndarray, h: int, wa: int):
    """Window max of a (nan ignored) and window-overlaps-footprint for windows (h stops) x (wa axes, wrapping)."""
    n = a.shape[0]
    m1 = np.full(a.shape, np.nan)
    o1 = np.zeros(a.shape, bool)
    for d in range(wa):
        m1 = np.fmax(m1, np.roll(a, -d, axis=1))
        o1 |= np.roll(fp, -d, axis=1)
    nw = n - h + 1
    m2 = np.full((nw, N_AX), np.nan)
    o2 = np.zeros((nw, N_AX), bool)
    for d in range(h):
        m2 = np.fmax(m2, m1[d:d + nw])
        o2 |= o1[d:d + nw]
    if wa == N_AX:  # every axis offset covers the same cells
        m2, o2 = m2[:, :1], o2[:, :1]
    return m2, o2


def within_scan_percentile(score: np.ndarray, fp: np.ndarray) -> dict:
    out = dict(lesion_score=np.nan, percentile=np.nan, n_windows=0, win_stops=0, win_axes=0)
    cells = fp & np.isfinite(score)
    if not cells.any():
        return out
    st = np.nonzero(fp.any(1))[0]
    h = int(st.max() - st.min() + 1)
    _, wa = axis_arc(np.nonzero(fp.any(0))[0])
    out.update(win_stops=h, win_axes=wa, lesion_score=float(np.nanmax(score[cells])))
    if h > score.shape[0]:
        return out
    m, o = _sliding_any_max(score, fp, h, wa)
    null = m[np.logical_not(o) & np.isfinite(m)]
    if not len(null):
        return out
    L = out["lesion_score"]
    out["n_windows"] = int(len(null))
    out["percentile"] = float(100.0 * (np.sum(null < L) + 0.5 * np.sum(null == L)) / len(null))
    return out


def top_peaks(score: np.ndarray, s_mm: np.ndarray, k: int = TOPK) -> list[tuple[int, int]]:
    """Greedy non-max suppression; returns up to k (stop, axis) cells."""
    fin = np.argwhere(np.isfinite(score))
    if not len(fin):
        return []
    order = np.argsort(-score[tuple(fin.T)], kind="stable")
    peaks = []
    for c in fin[order]:
        i, j = int(c[0]), int(c[1])
        if all(abs(s_mm[i] - s_mm[p]) >= SEP_MM or axis_dist_deg(j, q) >= SEP_DEG for p, q in peaks):
            peaks.append((i, j))
            if len(peaks) == k:
                break
    return peaks


def top3_hit(score, s_mm, fp) -> bool:
    return any(fp[i, j] for i, j in top_peaks(score, s_mm))


def chance_top3_uniform(fp: np.ndarray, scoreable: np.ndarray, s_mm: np.ndarray) -> float:
    """Exact P(at least one of 3 suppressed uniform placements lands in the footprint)."""
    cells = np.argwhere(scoreable)
    n = len(cells)
    if n == 0:
        return np.nan
    si = s_mm[cells[:, 0]]
    ax = cells[:, 1]
    Z = ((np.abs(si[:, None] - si[None, :]) < SEP_MM) & (axis_dist_deg(ax[:, None], ax[None, :]) < SEP_DEG))
    Z = Z.astype(np.float64)                                  # includes the diagonal (a cell suppresses itself)
    nf = np.logical_not(fp[tuple(cells.T)]).astype(np.float64)
    zrow = Z.sum(1)
    zrow_nf = Z @ nf
    a1 = n - zrow                                             # |A1(i)|: cells still allowed after peak i
    a1_nf = nf.sum() - zrow_nf
    ZZ = Z @ Z                                                # |zone(i) & zone(j)|
    ZZnf = Z @ (Z * nf[None, :]).T                            # |zone(i) & zone(j) & not fp|
    a2 = a1[:, None] - (zrow[None, :] - ZZ)                   # |A2(i, j)| for j in A1(i)
    a2_nf = a1_nf[:, None] - (zrow_nf[None, :] - ZZnf)
    with np.errstate(invalid="ignore", divide="ignore"):
        p3 = np.where(a2 > 0, a2_nf / a2, 1.0)                # P(peak 3 misses | i, j), 1 if no third peak
    allowed = (1.0 - Z) * nf[None, :]                         # j allowed after i and j misses
    inner = (allowed * p3).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p2 = np.where(a1 > 0, inner / a1, 1.0)                # P(peaks 2 and 3 miss | i)
    p_miss = float((nf * p2).sum() / n)
    return 1.0 - p_miss


def footprint_window(fp: np.ndarray, u: np.ndarray) -> tuple[float, float, np.ndarray]:
    st = np.nonzero(fp.any(1))[0]
    start, length = axis_arc(np.nonzero(fp.any(0))[0])
    return float(u[st].min()), float(u[st].max()), arc_axes(start, length)


def chance_top3_empirical(fp: np.ndarray, u: np.ndarray, normal_peaks: list) -> float:
    """normal_peaks: per held-out normal gland, a list of (u, axis) of its top-3 peaks."""
    if not fp.any() or not normal_peaks:
        return np.nan
    ulo, uhi, axes = footprint_window(fp, u)
    hits = [any(ulo <= pu <= uhi and int(pa) in set(axes.tolist()) for pu, pa in pk) for pk in normal_peaks]
    return float(np.mean(hits))


def gate_chance(uniform: float, empirical: float) -> float:
    return float(np.nanmax([uniform, empirical]))
