"""Lesion components: the ONE definition of lesion_id, and the Feret diameter.

lesion_id = scipy.ndimage.label numbering with full 26-connectivity on the mask in
nibabel's (x, y, z) array order. The connected *components* are the same in any axis
order, but label()'s numbering follows raster-scan order, so a SimpleITK (z, y, x)
array numbers them differently (this permuted ~21% of join keys once). Every script
that keys on (case_id, lesion_id) must get its labels from label_lesions().
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import generate_binary_structure, label
from scipy.spatial import ConvexHull, QhullError

CC_STRUCT = generate_binary_structure(3, 3)  # 26-connectivity
BIN_EDGES_MM = (5.0, 10.0, 20.0, 40.0)      # -> <5, 5-10, 10-20, 20-40, >=40
BIN_LABELS = ("<5", "5-10", "10-20", "20-40", ">=40")


def bbox_slices(mask: np.ndarray, pad=0) -> tuple[slice, ...] | None:
    """Slices of the mask's bounding box grown by `pad` voxels (an int, or one per axis),
    clipped to the array; None if the mask is empty."""
    pads = np.broadcast_to(np.asarray(pad, dtype=int), (3,))
    coords = [np.nonzero(mask.any(axis=tuple(a for a in range(3) if a != ax)))[0] for ax in range(3)]
    if any(c.size == 0 for c in coords):
        return None
    return tuple(slice(max(int(c[0]) - p, 0), min(int(c[-1]) + 1 + p, n))
                 for c, p, n in zip(coords, pads, mask.shape))


def label_lesions(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """(labels, n): uint16 array of mask.shape with lesion ids 1..n, 0 = background.

    Labels the bounding-box crop, not the full volume (label() scratch scales with its
    input). Cropping does not change the numbering: raster order is translation-invariant.
    The mask must be in nibabel (x, y, z) order -- see the module docstring.
    """
    labels = np.zeros(mask.shape, dtype=np.uint16)
    sl = bbox_slices(mask)
    if sl is None:
        return labels, 0
    crop, n = label(mask[sl], structure=CC_STRUCT)
    if n > np.iinfo(np.uint16).max:
        raise ValueError(f"{n} components do not fit in uint16")
    labels[sl] = crop
    return labels, int(n)


def _max_pairwise_dist(pts: np.ndarray, max_block: int = 4_000_000) -> float:
    """Exact max Euclidean distance over all point pairs, ~max_block floats of scratch."""
    pts = pts - pts.mean(axis=0)  # centering keeps the |a|^2+|b|^2-2ab expansion accurate
    chunk = max(1, max_block // len(pts))
    best = 0.0
    for i in range(0, len(pts), chunk):
        a = pts[i:i + chunk]
        d2 = (a * a).sum(1)[:, None] + (pts * pts).sum(1)[None, :] - 2.0 * a @ pts.T
        best = max(best, float(d2.max()))
    return float(np.sqrt(max(best, 0.0)))


def feret_mm(coords_vox: np.ndarray, spacing) -> float:
    """3D max caliper (Feret) diameter in mm: the largest distance between two voxel
    centers of the component. coords_vox is (N, 3) array indices in the same axis
    order as `spacing`.

    The farthest pair always lies on the convex hull, so the hull vertices give the exact
    answer cheaply. If the hull is degenerate (all points coplanar or collinear, e.g. a
    single-slice lesion), fall back to exact brute force over every point -- never a
    random subsample, which underestimates thin lesions.
    """
    pts = np.asarray(coords_vox, dtype=np.float64) * np.asarray(spacing, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    if len(pts) > 200:
        try:
            pts = pts[ConvexHull(pts).vertices]
        except (QhullError, ValueError):  # coplanar / collinear -> degenerate hull
            pass
    return _max_pairwise_dist(pts)


def size_bin(diam_mm: float) -> str:
    """Bin label for a Feret diameter, edges <5, 5-10, 10-20, 20-40, >=40 mm."""
    return BIN_LABELS[int(np.digitize([diam_mm], BIN_EDGES_MM)[0])]
