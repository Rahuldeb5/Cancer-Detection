"""Constants fixed a priori (2026-10-03, before any real-data measurement) and pure functions.

Nothing here does I/O, so the oracle exercises exactly the code the cohort run uses. Arrays are in
nibabel (x, y, z) order; `spacing` is mm per array axis.

Vessel HU (step 2)
    Each vessel mask is eroded by one voxel (6-connected: every voxel face-adjacent to the outside is
    dropped), which removes the layer that straddles the vessel wall and surrounding fat. Only the
    middle 80% of the slices that contain the (un-eroded) vessel are used: the ends of the aorta mask
    run into the thorax / iliac bifurcation and the ends of the vein mask into intrahepatic branches
    and the SMV, all thin and partial-volume dominated. Statistic = median HU of the eroded voxels on
    those slices. Fewer than MIN_VOX eroded voxels -> flagged (still measured if >0).

Gland HU (descriptive only, never enters the phase rule)
    S3 v3 global pool, same lines as src/attenuation-v3/measure_v3.py and the same frozen constants
    (attn_v3.SHELL_MM, MIN_REF_VOX, robust_stats): hole-filled envelope - (all lesions + 5 mm) - ducts.

Phase rule (step 4)
    3-component full-covariance Gaussian mixture on (aorta HU, portal-vein HU) over all measurable
    cohort cases. Cluster names come from the centroids only: lowest aorta mean -> non-contrast; of
    the other two, the larger (aorta - PV) mean -> arterial; the last -> venous. Max posterior
    < POST_MIN -> "uncertain". No lesion mask, lesion HU, tumor status or metadata phase is an input.

Early arterial (descriptive, defined before looking at the scatter)
    PV enhancement fraction f = (pv - mu_pv[non-contrast]) / (mu_pv[venous] - mu_pv[non-contrast]),
    with the GMM centroids. An estimated-arterial scan with f < EARLY_ART_F (portal vein less than
    half-way to venous enhancement) is called early arterial; f >= EARLY_ART_F is late arterial
    (pancreatic phase).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt, generate_binary_structure

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "attenuation-v3"))
import attn_v3 as av  # noqa: E402  (S3 frozen constants + robust_stats, reused unchanged)

# ------------------------------------------------------------------ fixed a priori
AORTA_MASK = "aorta"
PV_MASK = "veins"          # PanTS "veins" = portal + splenic venous system (checked S12, see README)
IVC_MASK = "postcava"      # QC only (overlap with the two measured vessels)
DUCTS = ("pancreatic_duct", "common_bile_duct")   # same as S3 measure_v3.DUCTS

ERODE_STRUCT = generate_binary_structure(3, 1)    # 6-connected one-voxel erosion
SLICE_KEEP = 0.80          # middle fraction of vessel-containing slices
MIN_VOX = 200              # flag below this many eroded voxels on the kept slices

N_COMPONENTS = 3
GMM_SEED = 0
GMM_N_INIT = 20
POST_MIN = 0.80            # max posterior below this -> "uncertain"
EARLY_ART_F = 0.5
PHASES = ("non-contrast", "arterial", "venous")


# ------------------------------------------------------------------ geometry helpers
def slice_axis(affine: np.ndarray) -> int:
    """Voxel axis whose direction has the largest world-z component (S1 case_pass.spacing_info rule)."""
    import nibabel as nib
    vs = nib.affines.voxel_sizes(affine)
    return int(np.argmax(np.abs(affine[2, :3]) / vs))


def middle_slices(mask: np.ndarray, axis: int, keep: float = SLICE_KEEP) -> np.ndarray:
    """Sorted indices (along `axis`) of the middle `keep` fraction of the slices that contain `mask`.

    Drops round(n * (1 - keep) / 2) slices at each end; with n <= 2 nothing is dropped.
    """
    other = tuple(a for a in range(3) if a != axis)
    idx = np.flatnonzero(mask.any(axis=other))
    cut = int(round(len(idx) * (1.0 - keep) / 2.0))
    return idx[cut:len(idx) - cut] if len(idx) - 2 * cut > 0 else idx


def vessel_stats(ct: np.ndarray, mask: np.ndarray, axis: int, keep: float = SLICE_KEEP) -> dict:
    """Median / IQR HU over the one-voxel-eroded mask, restricted to its middle slices.

    `ct` and `mask` are the same (cropped) grid; a crop should have >= 1 voxel of padding, though
    border voxels are treated as outside anyway (scipy erosion border_value=0).
    """
    mask = mask.astype(bool, copy=False)
    out = {"n_slices": 0, "n_vox_raw": int(mask.sum()), "n_vox": 0,
           "median": np.nan, "iqr": np.nan}
    if not mask.any():
        return out
    keep_idx = middle_slices(mask, axis, keep)
    sel = np.zeros(mask.shape[axis], dtype=bool)
    sel[keep_idx] = True
    shape = [1, 1, 1]
    shape[axis] = -1
    eroded = binary_erosion(mask, structure=ERODE_STRUCT) & sel.reshape(shape)
    vals = ct[eroded]
    out.update(n_slices=int(len(keep_idx)), n_vox=int(vals.size))
    if vals.size:
        q1, q2, q3 = np.percentile(vals, [25, 50, 75])
        out.update(median=float(q2), iqr=float(q3 - q1))
    return out


def gland_pool(env: np.ndarray, les: np.ndarray, ducts: np.ndarray, spacing) -> np.ndarray:
    """S3 v3 global reference pool on a crop (measure_v3.measure_case lines, unchanged logic).

    For a case with no lesion voxels there is no shell to remove (scipy's EDT of an all-ones array
    is not a distance to anything), so the pool is the hole-filled envelope minus ducts.
    """
    env_filled = binary_fill_holes(env.astype(bool))
    pool = env_filled & ~ducts.astype(bool)
    if les.any():
        d_any = distance_transform_edt(~les.astype(bool), sampling=np.asarray(spacing, dtype=np.float64))
        pool &= d_any > av.SHELL_MM
    return pool


def gland_median(ct: np.ndarray, pool: np.ndarray) -> tuple[float, int]:
    """(median HU, n voxels) over the pool; NaN below S3's MIN_REF_VOX."""
    n = int(pool.sum())
    if n < av.MIN_REF_VOX:
        return np.nan, n
    return av.robust_stats(ct[pool])[0], n


# ------------------------------------------------------------------ phase rule
def name_clusters(means: np.ndarray) -> dict[int, str]:
    """GMM component index -> phase name from centroids (columns: aorta, pv)."""
    means = np.asarray(means, dtype=np.float64)
    nc = int(np.argmin(means[:, 0]))
    rest = [i for i in range(len(means)) if i != nc]
    art = max(rest, key=lambda i: means[i, 0] - means[i, 1])
    ven = [i for i in rest if i != art][0]
    return {nc: "non-contrast", art: "arterial", ven: "venous"}


def assign(posteriors: np.ndarray, names: dict[int, str], post_min: float = POST_MIN):
    """(label array, max posterior, argmax phase) per row; label "uncertain" below post_min."""
    k = posteriors.argmax(axis=1)
    pmax = posteriors.max(axis=1)
    best = np.array([names[i] for i in k], dtype=object)
    label = np.where(pmax < post_min, "uncertain", best).astype(object)
    return label, pmax, best


def pv_enhancement_fraction(pv, means: np.ndarray, names: dict[int, str]):
    """(pv - mu_pv[non-contrast]) / (mu_pv[venous] - mu_pv[non-contrast]) from GMM centroids."""
    inv = {v: k for k, v in names.items()}
    lo, hi = means[inv["non-contrast"], 1], means[inv["venous"], 1]
    return (np.asarray(pv, dtype=np.float64) - lo) / (hi - lo)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
