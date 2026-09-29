"""Pancreas envelope and the candidate search region derived from it.

PanTS pancreas masks are not one consistent family: pancreas.nii.gz often has the
lesion carved out, while pancreas_head/body/tail still cover it (and elsewhere the two
disagree). So the envelope is the union of all four, and the search region additionally
closes and hole-fills it so a carved-out lesion cavity is searched too.

The lesion mask is never read here. A search region that knew where the lesion was
would leak the answer into every detector built on it.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt

from tumorlib import io
from tumorlib.lesions import bbox_slices

ENVELOPE_MASKS = ("pancreas", "pancreas_head", "pancreas_body", "pancreas_tail")


def envelope(case_id: str) -> np.ndarray | None:
    """uint8 union of pancreas | head | body | tail on the CT grid, or None if any mask fails."""
    out = None
    for name in ENVELOPE_MASKS:
        m = io.load_mask(case_id, name)
        if m is None:
            return None
        out = m if out is None else np.bitwise_or(out, m, out=out)
    return out


def fill_search_region(env: np.ndarray, spacing, close_mm: float = 8.0, dilate_mm: float = 3.0) -> np.ndarray:
    """Closing (radius close_mm) -> 3D hole fill -> dilation (dilate_mm), all in mm.

    Morphology is done with Euclidean distance transforms in mm instead of voxel
    structuring elements, so the radii mean the same thing on 0.7 mm and 5 mm axes:
      dilate(A, r) = {x : dist(x, A) <= r}
      erode(A, r)  = {x : dist(x, outside A) > r}
    Closing fills notches narrower than ~2*close_mm (a lesion carved at the gland
    surface); the hole fill catches cavities fully enclosed by the gland, whatever
    their size; the final dilation adds a margin around the gland edge.

    Works on the envelope's bbox crop, padded so the dilations never hit the crop
    edge, and pastes the result back into a full-size uint8 array.
    """
    out = np.zeros(env.shape, dtype=np.uint8)
    sp = np.asarray(spacing, dtype=np.float64)
    pad = np.ceil((close_mm + dilate_mm) / sp).astype(int) + 2
    sl = bbox_slices(env, pad)
    if sl is None:
        return out

    crop = env[sl].astype(bool)
    grown = distance_transform_edt(~crop, sampling=sp) <= close_mm
    closed = (distance_transform_edt(grown, sampling=sp) > close_mm) | crop
    del grown
    filled = binary_fill_holes(closed)
    del closed
    region = distance_transform_edt(~filled, sampling=sp) <= dilate_mm
    out[sl] = region
    return out


def search_region(case_id: str, close_mm: float = 8.0, dilate_mm: float = 3.0) -> np.ndarray | None:
    """Hole-filled, dilated pancreas envelope for candidate search (uint8, CT grid)."""
    env = envelope(case_id)
    sp = io.spacing(case_id)
    if env is None or sp is None:
        return None
    if not env.any():
        print(f"{case_id}: empty pancreas envelope -> empty search region")
    return fill_search_region(env, sp, close_mm, dilate_mm)
