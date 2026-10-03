"""PHASE 1 pure array functions: patch grid, rotation-invariant patch features, gland frame (u),
sphere insertion, grid candidates and the same-footprint random-set comparison.

No file I/O and no lesion mask anywhere in the feature path: case_pass.py hands these functions a
1 mm isotropic CT crop and the search region, nothing else. Arrays are nibabel (x, y, z) order.

Features per spherical patch of radius r (fixed a priori, before any cohort lesion was scored):
  hu_mean, hu_std, hu_q05/25/50/75/95  clipped HU [-100, 300] inside the ball
  rad_{a}_{b}       mean HU in 2 mm shells a < d <= b (the centre voxel joins the first shell),
                    minus the patch median -> the radial profile, insensitive to the phase's
                    overall brightness
  ring_contrast     patch mean - mean of the 5 mm ring r < d <= r + 5
  grad_mean/std     Gaussian gradient magnitude (sigma 1 mm) over the ball
  entropy           Shannon entropy (bits) of the ball's HU histogram, 25 HU bins
  log_raw           sigma^2 * LoG at the patch centre, sigma = r / sqrt(3) (the scale whose response
                    peaks for a ball of radius r), NOT divided by S4's per-case MAD
All of these are invariant to rotating the patch about its centre (up to voxelization).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude, gaussian_laplace, maximum_filter
from scipy.spatial import cKDTree
from scipy.special import erfc

from normative import common as C

QUANTILES = (5, 25, 50, 75, 95)
N_ENTROPY_BINS = int(round((C.HU_CLIP[1] - C.HU_CLIP[0]) / C.ENTROPY_BIN_HU))


# ------------------------------------------------------------------ geometry
def ball(radius_mm: float, inner_mm: float = -1.0) -> tuple[np.ndarray, np.ndarray]:
    """(offsets (M, 3) int, distances (M,)) of 1 mm voxels with inner_mm < d <= radius_mm."""
    r = int(np.ceil(radius_mm))
    g = np.stack(np.meshgrid(*[np.arange(-r, r + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
    d = np.sqrt((g ** 2).sum(1))
    keep = (d <= radius_mm + 1e-9) & (d > inner_mm + 1e-9)
    return g[keep], d[keep]


def shell_edges(radius_mm: float, width: float = C.SHELL_MM) -> list[tuple[float, float]]:
    n = int(np.ceil(radius_mm / width - 1e-9))
    return [(k * width, min((k + 1) * width, radius_mm)) for k in range(n)]


def feature_names(radius_mm: float) -> list[str]:
    return (["hu_mean", "hu_std"] + [f"hu_q{q:02d}" for q in QUANTILES]
            + [f"rad_{a:g}_{b:g}" for a, b in shell_edges(radius_mm)]
            + ["ring_contrast", "grad_mean", "grad_std", "entropy", "log_raw"])


def grid_centres(region: np.ndarray, step: int = C.GRID_MM) -> np.ndarray:
    """(N, 3) voxel indices on a `step`-voxel lattice anchored at index 0, inside `region`.
    On the 1 mm grid this is a step-mm lattice in the crop frame."""
    sub = region[::step, ::step, ::step]
    return np.argwhere(sub) * step


def gather(vol: np.ndarray, centres: np.ndarray, offs: np.ndarray) -> np.ndarray:
    """(N, M) values of vol at centres + offsets, indices clamped to the array (mode 'nearest',
    which matches how the CT crop was resampled)."""
    hi = np.asarray(vol.shape) - 1
    q = centres[:, None, :] + offs[None, :, :]
    np.clip(q, 0, hi, out=q)
    return vol[q[..., 0], q[..., 1], q[..., 2]]


# ------------------------------------------------------------------ feature maps + features
def clip_hu(ct: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(ct, dtype=np.float32), *C.HU_CLIP)


def log_raw_map(f: np.ndarray, radius_mm: float) -> np.ndarray:
    """sigma^2 * LoG(f) at sigma = r/sqrt(3) on a 1 mm grid. Positive for a dark blob (S4 sign)."""
    s = radius_mm / np.sqrt(3.0)
    return (s * s) * gaussian_laplace(f, s, mode="nearest", truncate=C.LOG_TRUNCATE)


def case_maps(ct_iso_hu: np.ndarray, radii=C.RADII_MM) -> dict:
    f = clip_hu(ct_iso_hu)
    grad = gaussian_gradient_magnitude(f, C.GRAD_SIGMA_MM, mode="nearest", truncate=C.LOG_TRUNCATE)
    return {"f": f, "grad": grad.astype(np.float32),
            "log": {r: log_raw_map(f, r).astype(np.float32) for r in radii}}


def entropy_rows(v: np.ndarray) -> np.ndarray:
    """Shannon entropy (bits) of each row's histogram over the clipped HU range."""
    idx = np.clip(((v - C.HU_CLIP[0]) / C.ENTROPY_BIN_HU).astype(np.int64), 0, N_ENTROPY_BINS - 1)
    counts = np.zeros((len(v), N_ENTROPY_BINS), np.float64)
    for b in range(N_ENTROPY_BINS):
        counts[:, b] = (idx == b).sum(1)
    p = counts / v.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log2(p), 0.0).sum(1)
    return h


def patch_features(maps: dict, centres: np.ndarray, radius_mm: float, chunk: int = 1024) -> np.ndarray:
    """(N, F) float32 features in feature_names(radius_mm) order."""
    offs, d = ball(radius_mm)
    ring_offs, _ = ball(radius_mm + C.RING_MM, inner_mm=radius_mm)
    shells = [(d > a) | ((a == 0) & (d == 0)) for a, _ in shell_edges(radius_mm)]
    shells = [s & (d <= b + 1e-9) for s, (_, b) in zip(shells, shell_edges(radius_mm))]
    lmap = maps["log"][radius_mm]
    out = np.empty((len(centres), len(feature_names(radius_mm))), np.float32)
    for i in range(0, len(centres), chunk):
        c = centres[i:i + chunk]
        v = gather(maps["f"], c, offs)
        q = np.percentile(v, QUANTILES, axis=1).T
        med = q[:, QUANTILES.index(50)]
        mean = v.mean(1)
        cols = [mean, v.std(1), *q.T]
        cols += [v[:, s].mean(1) - med for s in shells]
        cols.append(mean - gather(maps["f"], c, ring_offs).mean(1))
        g = gather(maps["grad"], c, offs)
        cols += [g.mean(1), g.std(1), entropy_rows(v), lmap[c[:, 0], c[:, 1], c[:, 2]]]
        out[i:i + chunk] = np.stack(cols, 1)
    return out


# ------------------------------------------------------------------ gland frame (u)
def gland_frame(panc_mm: np.ndarray, head_centroid_mm: np.ndarray, tail_centroid_mm: np.ndarray,
                pct=(1.0, 99.0)) -> dict:
    """S1 / lesion-sectioning u frame: PCA of the pancreas point cloud in world mm, PC1 sign-fixed to
    point from the head centroid to the tail centroid, normalized by the 1st-99th percentile of the
    pancreas's own PC1 extent. Same maths as pca_hitrate.build_frame."""
    cen = panc_mm.mean(0)
    w, v = np.linalg.eigh(np.cov((panc_mm - cen).T))
    a1 = v[:, np.argmax(w)].copy()
    if np.dot(a1, tail_centroid_mm - head_centroid_mm) < 0:
        a1 = -a1
    p = (panc_mm - cen) @ a1
    lo, hi = np.percentile(p, pct)
    return {"centroid": cen, "axis": a1, "lo": float(lo), "hi": float(hi)}


def u_of(points_mm: np.ndarray, frame: dict | None) -> np.ndarray:
    if frame is None:
        return np.full(len(points_mm), np.nan)
    return ((points_mm - frame["centroid"]) @ frame["axis"] - frame["lo"]) / (frame["hi"] - frame["lo"])


# ------------------------------------------------------------------ insertion
def sphere_profile(shape, spacing, centre_mm, radius_mm: float, dhu: float,
                   edge_mm: float = C.INSERT_EDGE_MM, sub_mm: float = 0.5) -> tuple[tuple[slice, ...], np.ndarray]:
    """Additive HU offset of a smooth sphere on a NATIVE (anisotropic) grid, partial-volume averaged.

    profile(d) = dHU * 0.5 * erfc((d - R) / (sqrt(2) * edge)): a ball blurred by a Gaussian of
    sigma = edge (the "soft 1 mm edge"); it is dHU at the centre and dHU/2 at d = R. Each native voxel
    gets the mean of the profile over its own extent (sub-sampled every <= sub_mm per axis), so a 5 mm
    slice sees the same partial-volume dilution a real lesion would. Added to the real CT, the native
    noise texture is untouched. Positions are in mm from voxel 0's centre (index * spacing).
    Returns (slices of the affected block, float32 block).
    """
    sp = np.asarray(spacing, float)
    c = np.asarray(centre_mm, float)
    reach = radius_mm + 4.0 * edge_mm
    lo = np.maximum(np.floor((c - reach) / sp).astype(int) - 1, 0)
    hi = np.minimum(np.ceil((c + reach) / sp).astype(int) + 2, np.asarray(shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    nsub = np.maximum(np.ceil(sp / sub_mm).astype(int), 1)
    subs = [((np.arange(n) + 0.5) / n - 0.5) * s for n, s in zip(nsub, sp)]   # within-voxel offsets
    axes = [np.arange(a, b) * s for a, b, s in zip(lo, hi, sp)]
    acc = np.zeros(tuple(hi - lo), np.float64)
    for ox in subs[0]:
        dx2 = (axes[0] + ox - c[0]) ** 2
        for oy in subs[1]:
            dy2 = (axes[1] + oy - c[1]) ** 2
            for oz in subs[2]:
                dz2 = (axes[2] + oz - c[2]) ** 2
                d = np.sqrt(dx2[:, None, None] + dy2[None, :, None] + dz2[None, None, :])
                acc += erfc((d - radius_mm) / (np.sqrt(2.0) * edge_mm))
    acc *= 0.5 * dhu / float(np.prod(nsub))
    return sl, acc.astype(np.float32)


# ------------------------------------------------------------------ candidates on the grid
def grid_local_maxima(centres: np.ndarray, scores: np.ndarray, step: int = C.GRID_MM,
                      sep_mm: float = C.CAND_SEP_MM, top_k: int = C.CAND_TOP_K) -> np.ndarray:
    """Indices into `centres` of the top_k local maxima of the score map on the patch lattice.

    A centre is a local maximum if its score is >= every existing lattice neighbour (26-connected,
    i.e. 4 / 5.7 / 6.9 mm away); then greedy thinning keeps peaks >= sep_mm apart (on a 4 mm lattice
    that removes only tied face neighbours). Returned in descending score order.
    """
    if len(centres) == 0:
        return np.zeros(0, int)
    g = centres // step
    g0 = g.min(0)
    g = g - g0
    vol = np.full(tuple(g.max(0) + 1), -np.inf, np.float64)
    vol[g[:, 0], g[:, 1], g[:, 2]] = scores
    mx = maximum_filter(vol, size=3, mode="constant", cval=-np.inf)
    is_max = vol[g[:, 0], g[:, 1], g[:, 2]] >= mx[g[:, 0], g[:, 1], g[:, 2]]
    cand = np.flatnonzero(is_max & np.isfinite(scores))
    order = cand[np.argsort(-scores[cand], kind="stable")]
    kept: list[int] = []
    pts = centres.astype(float)
    for i in order:
        if kept and np.min(np.linalg.norm(pts[kept] - pts[i], axis=1)) < sep_mm:
            continue
        kept.append(int(i))
        if len(kept) >= top_k:
            break
    return np.asarray(kept, int)


# ------------------------------------------------------------------ same-footprint random sets
def footprint_sets(far_xyz: np.ndarray, n_foot: int, rng, n_sets: int = C.N_RANDOM_SETS,
                   tree: cKDTree | None = None) -> np.ndarray | None:
    """(n_sets, n_foot) indices into far_xyz: each row is the n_foot far-region centres nearest to a
    random far-region anchor -- a compact set with exactly the lesion's number of patch centres.
    Equal COUNT matters because the set score is a max, and a max over more patches is larger."""
    if n_foot < 1 or len(far_xyz) < max(n_foot, 10):
        return None
    tree = tree or cKDTree(far_xyz)
    anchors = rng.integers(0, len(far_xyz), n_sets)
    _, idx = tree.query(far_xyz[anchors], k=n_foot)
    return np.asarray(idx).reshape(n_sets, n_foot)


def set_percentile(lesion_score: float, rand_scores: np.ndarray) -> tuple[float, float]:
    """(percentile, frac_ge). percentile = 100 * fraction of random sets scoring strictly below the
    lesion; frac_ge = fraction scoring >= it (S4's 'beaten by' metric), so percentile = 100(1-frac_ge)."""
    r = np.asarray(rand_scores, float)
    r = r[np.isfinite(r)]
    if r.size == 0 or not np.isfinite(lesion_score):
        return np.nan, np.nan
    fge = float((r >= lesion_score).mean())
    return 100.0 * (1.0 - fge), fge
