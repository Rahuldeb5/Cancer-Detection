"""Crop-then-resample to isotropic mm spacing (default 1 mm), for masks and CT.

Always crop first: a full CT grid at 1 mm isotropic is hundreds of MB per case, and
caching those once exhausted the disk. Ported from src/duct-cutoff/loader.py
(resample_mask_to_isotropic), plus a CT version.

Why affine_transform with an explicit scale rather than scipy's zoom(): zoom() aligns
the first and last voxel *centers* and rounds the output shape, so the effective
spacing drifts (~1% on real PanTS ducts) and differs between two crops of one grid.
Here output index o maps to input index o / factor: the spacing is exactly target_mm
and output voxel 0 is crop voxel 0, so

    p_native = crop_offset + p_iso * (target_mm / native_spacing)     # per axis
    world_mm = affine @ [*p_native, 1]
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import affine_transform

from tumorlib.lesions import bbox_slices

CROP_PAD_VOX = 5  # native voxels of padding kept around a mask's bbox


def _resample(block: np.ndarray, spacing, target_mm: float, mode: str, cval: float = 0.0) -> np.ndarray:
    factors = np.asarray(spacing, dtype=np.float64) / target_mm
    out_shape = tuple(int(np.floor((n - 1) * f)) + 1 for n, f in zip(block.shape, factors))
    return affine_transform(block, np.diag(1.0 / factors), output_shape=out_shape,
                            order=1, mode=mode, cval=cval)


def resample_mask_to_isotropic(mask: np.ndarray, spacing, target_mm: float = 1.0,
                               pad: int = CROP_PAD_VOX):
    """Bbox-crop, then resample a binary mask to exactly target_mm isotropic.

    Linear interpolation of the 0/1 mask + threshold at 0.5, not nearest neighbour:
    nearest neighbour would just replicate the native blocky boundary, which matters
    for thin structures. Safe for a single binary label (nothing to blend with).

    Returns (uint8 mask, (target_mm,)*3, crop_offset in native voxel indices).
    An empty input comes back as a zero-size array.
    """
    iso = (float(target_mm),) * 3
    sl = bbox_slices(mask, pad)
    if sl is None:
        return np.zeros((0, 0, 0), dtype=np.uint8), iso, (0, 0, 0)
    res = _resample(mask[sl].astype(np.float32), spacing, target_mm, mode="constant")
    return (res > 0.5).astype(np.uint8), iso, tuple(s.start for s in sl)


def resample_ct_to_isotropic(ct: np.ndarray, spacing, crop: tuple[slice, ...] | None,
                             target_mm: float = 1.0):
    """Trilinear (order=1) resample of a CT block to exactly target_mm isotropic.

    `crop` is required on purpose: pass the region you need (e.g. bbox_slices of a
    search region, padded), or None only if `ct` is already a crop (e.g. from
    io.load_ct(case_id, crop)); then the returned offset is (0, 0, 0) and you add
    the crop's own start. Edges use mode="nearest" so no fake 0-HU (water) rim
    appears.

    Returns (float32 HU, (target_mm,)*3, crop_offset in native voxel indices).
    """
    iso = (float(target_mm),) * 3
    block = ct if crop is None else ct[crop]
    offset = (0, 0, 0) if crop is None else tuple(s.start or 0 for s in crop)
    res = _resample(np.asarray(block, dtype=np.float32), spacing, target_mm, mode="nearest")
    return res, iso, offset
