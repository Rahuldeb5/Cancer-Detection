"""Step 4 prerequisite: the aorta-posterior test (CONTEXT gotcha 4; S10 found y-mirrored glands).

The aorta lies posterior to the pancreas. On the CT voxel grid (aorta via tumorlib load_mask, never the aorta's own
affine), mapped to world with the CT affine: d_y = (aorta centroid - gland centroid) . world +y. Normal anatomy gives
d_y < 0 (S10: median -35.8 mm, carried over). d_y > +10 mm => the scan's world y is mirrored => "anterior" = -y for
that case. |d_y| <= 10 mm => inconclusive (v2, Rahul), treated as not flipped and reported. Only aorta voxels within
the gland's world-z range are used (the thoracic aorta and arch are excluded).
The gland is the 502 outline, not a GT mask.
"""
from __future__ import annotations

import numpy as np

from glandshape import core as K
from glandshape.outline import Outline


def aorta_test(outline: Outline, aorta_mask: np.ndarray | None) -> dict:
    out = {"aorta_dy_mm": np.nan, "aorta_n_slab": 0, "y_flipped": False, "aorta_status": "missing",
           "aorta_call": "missing"}
    if aorta_mask is None or not np.asarray(aorta_mask).any():
        return out
    gw = outline.iso_to_world(np.argwhere(outline.mask))
    zlo, zhi = gw[:, 2].min() - K.AORTA_SLAB_MARGIN_MM, gw[:, 2].max() + K.AORTA_SLAB_MARGIN_MM
    aw = np.argwhere(np.asarray(aorta_mask).astype(bool)) @ outline.affine[:3, :3].T + outline.affine[:3, 3]
    slab = aw[(aw[:, 2] >= zlo) & (aw[:, 2] <= zhi)]
    out["aorta_status"] = "slab" if len(slab) else "whole"
    use = slab if len(slab) else aw
    out["aorta_n_slab"] = int(len(slab))
    dy = float(use.mean(0)[1] - gw.mean(0)[1])
    out["aorta_dy_mm"] = dy
    if abs(dy) <= K.AORTA_INCONCLUSIVE_MM:
        out["aorta_call"] = "inconclusive"
    else:
        out["aorta_call"] = "flipped" if dy > 0 else "normal"
    out["y_flipped"] = out["aorta_call"] == "flipped"
    return out


def anterior_world(test: dict) -> np.ndarray:
    return -K.WORLD_ANTERIOR if test.get("y_flipped") else K.WORLD_ANTERIOR
