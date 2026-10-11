"""Step 1: the gland outline from Dataset502's predicted pancreas region (never a GT mask).

predicted region (channel 0 > 0.5) -> largest 26-connected component (native grid) -> 3D hole fill
-> crop to bbox + 10 mm -> resample to 1 mm isotropic (tumorlib) -> largest component + hole fill again
(a thin bridge can break or a pinhole appear at 1 mm).

Coordinates. Iso index i maps to native index  offset + i * (ISO_MM / spacing)  (tumorlib.resample convention),
and native index n maps to world mm  affine @ [n, 1]. With a non-sheared CT affine, iso -> world is
x_w = R @ i + t with R orthonormal (det +-1), so distances in iso index units are mm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_fill_holes, label

from glandshape import core as K
from tumorlib.lesions import CC_STRUCT
from tumorlib.resample import resample_mask_to_isotropic


@dataclass
class Outline:
    mask: np.ndarray            # uint8 (X, Y, Z) at ISO_MM, the hole-filled largest component
    offset: np.ndarray          # (3,) native index of iso voxel 0
    spacing: np.ndarray         # (3,) native spacing mm
    affine: np.ndarray          # (4, 4) CT affine (native index -> world mm, RAS+)
    n_comp: int                 # components of the raw predicted region (native grid)
    size1: int                  # voxels of the largest component (native)
    size2: int                  # voxels of the second component (native), 0 if none
    vol_mm3: float              # iso outline volume (mm^3)

    @property
    def frac2(self) -> float:
        return self.size2 / self.size1 if self.size1 else float("nan")

    @property
    def fragmented(self) -> bool:
        return self.frac2 > K.FRAG_RATIO

    @property
    def R(self) -> np.ndarray:
        """iso index -> world rotation (orthonormal for a non-sheared affine)."""
        return self.affine[:3, :3] @ np.diag(K.ISO_MM / self.spacing)

    @property
    def t(self) -> np.ndarray:
        return self.affine[:3, :3] @ self.offset + self.affine[:3, 3]

    def iso_to_world(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p, float) @ self.R.T + self.t

    def iso_to_native(self, p: np.ndarray) -> np.ndarray:
        return self.offset + np.asarray(p, float) * (K.ISO_MM / self.spacing)

    def dir_world_to_iso(self, v: np.ndarray) -> np.ndarray:
        """A world direction expressed in iso index axes (R is orthonormal, so R^-1 = R^T)."""
        return np.linalg.solve(self.R, np.asarray(v, float))

    @property
    def handedness(self) -> float:
        """det(R): -1 when the iso index axes are a mirror image of world RAS+ axes."""
        return float(np.sign(np.linalg.det(self.R)))


def _largest(mask: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    lab, n = label(mask, structure=CC_STRUCT)
    if n == 0:
        return mask, 0, 0, 0
    sizes = np.bincount(lab.ravel())[1:]
    order = np.argsort(sizes)[::-1]
    keep = lab == (order[0] + 1)
    s2 = int(sizes[order[1]]) if n > 1 else 0
    return keep, n, int(sizes[order[0]]), s2


def outline_from_coords(xyz: np.ndarray, shape, spacing, affine) -> Outline | None:
    """xyz: (N, 3) native voxel indices of the predicted region. None if empty."""
    xyz = np.asarray(xyz, dtype=np.int64)
    if len(xyz) == 0:
        return None
    sp = np.asarray(spacing, float)
    shape = np.asarray(shape, int)
    lo, hi = xyz.min(0), xyz.max(0) + 1
    crop = np.zeros(tuple(hi - lo), bool)
    crop[tuple((xyz - lo).T)] = True
    keep, n, s1, s2 = _largest(crop)
    del crop
    keep = binary_fill_holes(keep)

    # bbox of the kept component, + 10 mm (native voxels, clipped to the volume), THEN resample
    kx = np.argwhere(keep) + lo
    pad = np.ceil(K.CROP_MARGIN_MM / sp).astype(int)
    blo = np.maximum(kx.min(0) - pad, 0)
    bhi = np.minimum(kx.max(0) + 1 + pad, shape)
    block = np.zeros(tuple(bhi - blo), np.uint8)
    block[tuple((kx - blo).T)] = 1
    del kx, keep
    iso, _, off = resample_mask_to_isotropic(block, sp, K.ISO_MM, pad=pad)
    del block
    iso_b, _, _, _ = _largest(iso.astype(bool))
    iso_b = binary_fill_holes(iso_b)
    return Outline(mask=iso_b.astype(np.uint8), offset=(blo + np.asarray(off)).astype(float), spacing=sp,
                   affine=np.asarray(affine, float), n_comp=n, size1=s1, size2=s2,
                   vol_mm3=float(iso_b.sum()) * K.ISO_MM ** 3)


def outline_from_flat(flat_idx: np.ndarray, shape, spacing, affine) -> Outline | None:
    """From flat C-order indices into the (x, y, z) volume (the S13 sparse cache's panc_idx)."""
    xyz = np.stack(np.unravel_index(np.asarray(flat_idx, np.int64), tuple(int(s) for s in shape)), 1)
    return outline_from_coords(xyz, shape, spacing, affine)


def outline_from_mask(mask: np.ndarray, spacing, affine) -> Outline | None:
    return outline_from_coords(np.argwhere(np.asarray(mask).astype(bool)), mask.shape, spacing, affine)
