"""v3 attenuation measurements: pure array functions, shared by the phantom and the real run.

Nothing here does I/O, so tests/phantom.py exercises exactly the code the cohort run uses.
Arrays are in nibabel (x, y, z) order; `spacing` is mm per array axis.

Tumor side -- partial-volume-aware core
    A mask voxel's value is the tissue averaged over its own footprint (the voxel box, blurred
    by the scanner PSF). A voxel on the mask's outer layer straddles the true edge, and on a
    5 mm slice that straddle is up to 5 mm deep. So "depth" is measured *after* removing the
    outer one-voxel layer (26-connected erosion): the EDT of that eroded mask, in mm, is the
    clearance between the voxel centre and the nearest boundary voxel. It equals
    (distance to outside) - (voxel size along that direction), i.e. how far the voxel's own
    box sits from the edge. The core is every voxel with depth >= CORE_DEPTH_MM; if fewer than
    CORE_MIN_VOX qualify, it is the CORE_MIN_VOX deepest voxels (ties broken by the plain mm EDT).

Reference side
    global: hole-filled envelope minus (all lesions + 5 mm) minus ducts (pancreatic + CBD).
    local:  the same pool restricted to 5-15 mm from *this* lesion.

Noise
    In-plane Laplacian residual r = x - mean(4 in-plane neighbours) over pool voxels whose
    4 neighbours are also in the pool; sigma = 1.4826 * MAD(r) / sqrt(1.25). For white noise
    this is the voxel SD; for correlated (real) CT noise it is a high-frequency noise index
    that underestimates the voxel SD (quantified by the phantom).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from scipy.stats import trim_mean

# Frozen from the phantom (src/attenuation-v3/phantom.py) before any real-data run.
# Rule: smallest N whose RMSE beats the v2 whole-mask median at r = 4 mm on every slice
# thickness (0.75/2.5/5 mm); D = 0.5 mm then just means "not on the outer voxel layer"
# (the smallest non-zero eroded depth is one in-plane voxel, >= 0.5 mm). N = 80 re-admits
# PV voxels (bias +7.1 HU at r=4/5 mm vs v2's +8.1); N = 20 is noisier than v2 at r=4/2.5 mm.
CORE_DEPTH_MM = 0.5
CORE_MIN_VOX = 40
TRIM = 0.2                 # trimmed mean cuts 20% from each tail
SHELL_MM = 5.0             # lesion exclusion shell for the reference pools
RING_MM = (5.0, 15.0)      # local ring, distance from this lesion
MIN_REF_VOX = 50           # below this a reference is NaN
CUTOFF_HU = 10.0

STRUCT26 = generate_binary_structure(3, 3)


def classify(dhu, cutoff: float = CUTOFF_HU):
    """hypo < -cutoff <= iso <= +cutoff < hyper (v2's convention: |dHU| == cutoff is iso)."""
    d = np.asarray(dhu, dtype=np.float64)
    out = np.where(d < -cutoff, "hypo", np.where(d > cutoff, "hyper", "iso")).astype(object)
    out[np.isnan(d)] = "unknown"
    return out if out.ndim else out.item()


def lesion_depth(comp: np.ndarray, spacing) -> tuple[np.ndarray, np.ndarray]:
    """(depth_pv, edt_mm) inside `comp` (zeros outside).

    depth_pv: mm EDT of the 26-eroded mask (0 on the outer voxel layer).
    edt_mm:   plain mm EDT of the mask (tie-breaker for lesions that are all outer layer).
    `comp` should be a crop padded by >= 1 voxel so the erosion sees background at the edge.
    """
    sp = np.asarray(spacing, dtype=np.float64)
    edt_mm = distance_transform_edt(comp, sampling=sp)
    inner = binary_erosion(comp, structure=STRUCT26)
    depth = distance_transform_edt(inner, sampling=sp) if inner.any() else np.zeros(comp.shape)
    return depth, edt_mm


def core_mask(comp: np.ndarray, spacing, depth_mm: float = CORE_DEPTH_MM,
              min_vox: int = CORE_MIN_VOX) -> tuple[np.ndarray, dict]:
    """Core voxels of one lesion component and a small info dict (n_core, fallback, min depth)."""
    depth, edt_mm = lesion_depth(comp, spacing)
    core = comp & (depth >= depth_mm) & (depth > 0)
    fallback = False
    if core.sum() < min_vox:
        fallback = True
        idx = np.flatnonzero(comp)
        if len(idx) <= min_vox:
            core = comp.copy()
        else:
            # lexsort: last key is primary -> sort by depth desc, then edt_mm desc
            order = np.lexsort((-edt_mm.ravel()[idx], -depth.ravel()[idx]))
            core = np.zeros_like(comp)
            core.ravel()[idx[order[:min_vox]]] = True
    info = {
        "n_core": int(core.sum()),
        "core_fallback": fallback,
        "core_min_depth_mm": float(depth[core].min()) if core.any() else np.nan,
        "max_depth_mm": float(depth[comp].max()) if comp.any() else np.nan,
    }
    return core, info


def robust_stats(values: np.ndarray) -> tuple[float, float]:
    """(median, 20%-trimmed mean) of a 1D array, NaN if empty."""
    if values.size == 0:
        return np.nan, np.nan
    return float(np.median(values)), float(trim_mean(values, TRIM))


def noise_sigma(ct: np.ndarray, pool: np.ndarray, slice_axis: int) -> tuple[float, int]:
    """(sigma_HU, n_residuals) from the in-plane Laplacian residual over `pool`."""
    ax = [a for a in range(3) if a != slice_axis]
    x = ct.astype(np.float32, copy=False)
    acc = np.zeros_like(x)
    ok = pool.copy()
    for a in ax:
        for shift in (1, -1):
            acc += np.roll(x, shift, axis=a)
            ok &= np.roll(pool, shift, axis=a)
    # np.roll wraps around: drop the first/last plane of each in-plane axis
    for a in ax:
        sl = [slice(None)] * 3
        sl[a] = slice(0, 1)
        ok[tuple(sl)] = False
        sl[a] = slice(-1, None)
        ok[tuple(sl)] = False
    r = (x - acc / 4.0)[ok]
    if r.size < MIN_REF_VOX:
        return np.nan, int(r.size)
    mad = np.median(np.abs(r - np.median(r)))
    return float(1.4826 * mad / np.sqrt(1.25)), int(r.size)
