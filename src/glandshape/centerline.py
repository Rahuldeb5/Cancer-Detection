"""Step 2: centerline of the iso outline, stops every 2 mm, parallel-transport frame.

skeleton (skimage 3D) -> 26-neighbour graph, edge weight = step length (mm) -> largest graph component ->
longest geodesic path by double-sweep Dijkstra (exact on a tree; skeleton loops are rare and small) ->
parametric smoothing spline (splprep, s = N * SPLINE_RMS_MM^2) -> each end trimmed back inside the mask if the
smoothing pulled it out, then extended along its end tangent to the first exit from the mask.

Head->tail (S10's PC1 convention, but signed WITHOUT the GT head/tail masks): e1 = PC1 of the outline voxels in
world mm, signed so that e1 points to patient left (-x in RAS+); the curve runs so that (tail end - head end) . e1 > 0.

Frame: rotation-minimizing (double reflection, Wang et al. 2008) along the stops, so the frame does not twist. The one
free constant rotation is chosen so that N is as close as possible to "anterior" over the whole gland
(phi = atan2(sum B.A, sum N.A)). B = T x N in WORLD handedness, so theta has the same sense for every gland
whatever the CT axis order. With T ~ patient left and N ~ anterior, theta = 90 deg ~ inferior, 180 ~ posterior.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.ndimage import map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from skimage.morphology import skeletonize

from glandshape import core as K
from glandshape.outline import Outline


@dataclass
class Centerline:
    s: np.ndarray            # (n,) arc length of each stop (mm), 0 = head end
    u: np.ndarray            # (n,) s / L
    P: np.ndarray            # (n, 3) stop positions, iso index coords (mm)
    T: np.ndarray            # (n, 3) unit tangents (head -> tail), iso coords
    N: np.ndarray            # (n, 3) theta = 0 direction (anterior-aligned), iso coords
    B: np.ndarray            # (n, 3) theta = 90 deg direction
    L: float                 # total length incl. extensions (mm)
    dense: np.ndarray        # (m, 3) dense curve (iso), head -> tail
    info: dict = field(default_factory=dict)
    tip: np.ndarray = None   # (n,) bool: stop lies on the extrapolated extension past the skeleton path's ends (v5)


def _sample(mask_f: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return map_coordinates(mask_f, np.asarray(pts, float).T, order=1, mode="constant", cval=0.0)


def skeleton_path(mask: np.ndarray) -> np.ndarray:
    """Ordered (k, 3) iso voxel coordinates of the longest geodesic path through the 3D skeleton."""
    sk = skeletonize(mask.astype(bool))
    pts = np.argwhere(sk)
    n = len(pts)
    if n < 2:
        return pts.astype(float)
    index = -np.ones(sk.shape, np.int64)
    index[tuple(pts.T)] = np.arange(n)
    offs = [np.array(o) for o in np.ndindex(3, 3, 3)]
    offs = [o - 1 for o in offs]
    offs = [o for o in offs if tuple(o) > (0, 0, 0)]          # 13 forward neighbours
    rows, cols, w = [], [], []
    for o in offs:
        q = pts + o
        ok = np.all((q >= 0) & (q < sk.shape), axis=1)
        j = np.full(n, -1)
        j[ok] = index[tuple(q[ok].T)]
        hit = j >= 0
        rows.append(np.nonzero(hit)[0]); cols.append(j[hit]); w.append(np.full(hit.sum(), np.linalg.norm(o)))
    rows, cols, w = map(np.concatenate, (rows, cols, w))
    G = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    ncc, lab = connected_components(G, directed=False)
    big = np.argmax(np.bincount(lab))
    nodes = np.nonzero(lab == big)[0]
    d0 = dijkstra(G, directed=False, indices=nodes[0])
    a = nodes[np.argmax(np.where(np.isfinite(d0[nodes]), d0[nodes], -1))]
    da, pred = dijkstra(G, directed=False, indices=a, return_predecessors=True)
    b = nodes[np.argmax(np.where(np.isfinite(da[nodes]), da[nodes], -1))]
    path = [b]
    while path[-1] != a:
        path.append(pred[path[-1]])
    return pts[np.array(path[::-1])].astype(float)


def _arclen(c: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]


def _resample_curve(c: np.ndarray, step: float) -> np.ndarray:
    s = _arclen(c)
    if s[-1] <= 0:
        return c[:1]
    q = np.arange(0.0, s[-1] + 1e-9, step)
    if s[-1] - q[-1] > 1e-6:
        q = np.r_[q, s[-1]]
    return np.stack([np.interp(q, s, c[:, k]) for k in range(3)], 1)


def smooth_path(path: np.ndarray) -> np.ndarray:
    """Dense (DENSE_STEP_MM) smoothing-spline curve through an ordered path."""
    # drop duplicate consecutive points (splprep needs distinct parameter values)
    keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9]
    path = path[keep]
    if len(path) < 4:
        return _resample_curve(path, K.DENSE_STEP_MM)
    t = _arclen(path)
    tck, _ = splprep(path.T, u=t / t[-1], k=3, s=len(path) * K.SPLINE_RMS_MM ** 2)
    fine = np.stack(splev(np.linspace(0, 1, max(20 * len(path), 200)), tck), 1)
    return _resample_curve(fine, K.DENSE_STEP_MM)


def _trim_and_extend(dense: np.ndarray, mask_f: np.ndarray) -> tuple[np.ndarray, dict]:
    """Trim each end inward until inside the mask (<= MAX_TRIM_MM), then extend along the end tangent to the
    first exit (<= MAX_EXTEND_MM). Returns the new dense curve and the trim/extension lengths."""
    info = {}
    inside = _sample(mask_f, dense) >= 0.5
    n_trim = int(round(K.MAX_TRIM_MM / K.DENSE_STEP_MM))
    if not inside.any():
        raise ValueError("centerline never inside the mask")
    i0 = int(np.argmax(inside))
    i1 = len(dense) - 1 - int(np.argmax(inside[::-1]))
    if i0 > n_trim or (len(dense) - 1 - i1) > n_trim:
        info["trim_capped"] = True
    i0, i1 = min(i0, n_trim), max(i1, len(dense) - 1 - n_trim)
    i0 = min(i0, i1)
    info["trim_start_mm"] = i0 * K.DENSE_STEP_MM
    info["trim_end_mm"] = (len(dense) - 1 - i1) * K.DENSE_STEP_MM
    dense = dense[i0:i1 + 1]
    k = max(1, int(round(K.END_TANGENT_MM / K.DENSE_STEP_MM)))
    out = [dense]
    for end in ("start", "end"):
        if end == "start":
            p, q = dense[0], dense[min(k, len(dense) - 1)]
        else:
            p, q = dense[-1], dense[max(len(dense) - 1 - k, 0)]
        d = p - q
        nd = np.linalg.norm(d)
        ext = np.zeros((0, 3))
        if nd > 1e-9 and _sample(mask_f, p[None])[0] >= 0.5:
            d /= nd
            steps = np.arange(1, int(K.MAX_EXTEND_MM / K.DENSE_STEP_MM) + 1) * K.DENSE_STEP_MM
            ray = p + steps[:, None] * d
            v = _sample(mask_f, ray)
            out_i = np.nonzero(v < 0.5)[0]
            n_in = out_i[0] if len(out_i) else len(steps)
            if n_in > 0:
                # exact crossing between the last inside step and the first outside one
                v_prev = v[n_in - 1]
                if n_in < len(steps):
                    frac = (v_prev - 0.5) / max(v_prev - v[n_in], 1e-9)
                    tip = p + (steps[n_in - 1] + frac * K.DENSE_STEP_MM) * d
                    ext = np.vstack([ray[:n_in], tip])
                else:
                    ext = ray[:n_in]
        info[f"extend_{end}_mm"] = float(np.linalg.norm(ext[-1] - p)) if len(ext) else 0.0
        out.insert(0 if end == "start" else len(out), ext[::-1] if end == "start" else ext)
    return np.vstack([o for o in out if len(o)]), info


def head_tail_axis(outline: Outline) -> np.ndarray:
    """PC1 of the outline voxels in world mm, signed toward patient left (S10's PC1 axis; sign from the CT
    affine instead of the GT head/tail masks)."""
    w = outline.iso_to_world(np.argwhere(outline.mask))
    w = w - w.mean(0)
    evals, evecs = np.linalg.eigh(w.T @ w)
    e1 = evecs[:, -1]
    return e1 if e1 @ K.WORLD_LEFT >= 0 else -e1


def rmf(P: np.ndarray, T: np.ndarray, n0: np.ndarray) -> np.ndarray:
    """Rotation-minimizing normals along stops P with tangents T (double reflection), starting from n0."""
    N = np.zeros_like(P)
    r = n0 - (n0 @ T[0]) * T[0]
    N[0] = r / np.linalg.norm(r)
    for i in range(len(P) - 1):
        v1 = P[i + 1] - P[i]
        c1 = v1 @ v1
        if c1 < 1e-12:
            rL, tL = N[i], T[i]
        else:
            rL = N[i] - (2 / c1) * (v1 @ N[i]) * v1
            tL = T[i] - (2 / c1) * (v1 @ T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = v2 @ v2
        r = rL if c2 < 1e-12 else rL - (2 / c2) * (v2 @ rL) * v2
        r = r - (r @ T[i + 1]) * T[i + 1]
        N[i + 1] = r / np.linalg.norm(r)
    return N


def build_centerline(outline: Outline, anterior_world: np.ndarray = K.WORLD_ANTERIOR) -> Centerline:
    mask_f = outline.mask.astype(np.float32)
    path = skeleton_path(outline.mask)
    if len(path) < 2:
        raise ValueError("skeleton has < 2 voxels")
    dense = smooth_path(path)
    dense, info = _trim_and_extend(dense, mask_f)
    info["skeleton_path_mm"] = float(_arclen(path)[-1])

    # head -> tail
    e1 = head_tail_axis(outline)
    chord_w = (dense[-1] - dense[0]) @ outline.R.T
    flipped = bool(chord_w @ e1 < 0)
    if flipped:
        dense = dense[::-1]
    # extension lengths were measured before the head->tail flip: swap them when the curve was reversed
    info["ext_head_mm"] = info["extend_end_mm"] if flipped else info["extend_start_mm"]
    info["ext_tail_mm"] = info["extend_start_mm"] if flipped else info["extend_end_mm"]
    info["chord_dot_e1"] = float(abs(chord_w @ e1) / max(np.linalg.norm(chord_w), 1e-9))
    info["e1_world"] = e1

    sd = _arclen(dense)
    L = float(sd[-1])
    s = np.arange(0.0, L + 1e-9, K.STOP_MM)
    P = np.stack([np.interp(s, sd, dense[:, k]) for k in range(3)], 1)
    a = np.stack([np.interp(np.clip(s + K.TANGENT_HALF_MM, 0, L), sd, dense[:, k]) for k in range(3)], 1)
    b = np.stack([np.interp(np.clip(s - K.TANGENT_HALF_MM, 0, L), sd, dense[:, k]) for k in range(3)], 1)
    T = a - b
    T /= np.linalg.norm(T, axis=1, keepdims=True)

    A = outline.dir_world_to_iso(anterior_world)
    A /= np.linalg.norm(A)
    # any start normal, transported, then one constant rotation that best aligns N with anterior
    seed = A if abs(A @ T[0]) < 0.99 else np.array([0.0, 0.0, 1.0])
    N = rmf(P, T, seed)
    h = outline.handedness
    B = h * np.cross(T, N)
    phi = np.arctan2(np.sum(B @ A), np.sum(N @ A))
    N, B = np.cos(phi) * N + np.sin(phi) * B, -np.sin(phi) * N + np.cos(phi) * B
    info["anterior_alignment"] = float(np.mean(N @ A))
    tip = (s < info["ext_head_mm"] - 1e-6) | (s > L - info["ext_tail_mm"] + 1e-6)
    return Centerline(s=s, u=s / L if L > 0 else s, P=P, T=T, N=N, B=B, L=L, dense=dense, info=info, tip=tip)
