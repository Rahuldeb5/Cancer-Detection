"""The Dataset502_PanTSPancLesion label definition, as one pure function.

Why a separate module: the label map is the only thing S7 invents, so it is defined
once, unit-tested on phantoms (tests/test_ds502_labels.py) and imported by both the
builder and the oracle. Nothing here touches disk.

Label map (fixed by CONTEXT.md gotcha 9; do not change):
    0  background
    1  pancreas = union(pancreas, pancreas_head, pancreas_body, pancreas_tail, lesion)
       -- the lesion voxels are included on purpose: PanTS carves the lesion out of
          pancreas.nii.gz, so without the union the gland would have a hole exactly
          where the tumor is, and "pancreas" would mean "gland minus tumor".
    2  lesion  (overrides 1; voxel-identical to the Dataset501 binary label)

No morphological operations: no closing, no hole filling, no dilation. Whatever the
four gland masks cover is the gland; nothing is invented.

nnU-Net reads this as a *region-based* target via
    labels = {"background": 0, "pancreas": [1, 2], "lesion": 2}
    regions_class_order = [1, 2]
so "pancreas" is the nested region {1,2} (whole gland incl. tumor) and "lesion" is {2}.
"""
from __future__ import annotations

import numpy as np

BACKGROUND = 0
PANCREAS = 1
LESION = 2

# Mask basenames under <case>/segmentations/. The gland envelope is the union of all
# four because pancreas.nii.gz and head/body/tail disagree (CONTEXT.md gotcha 5).
ENVELOPE_MASKS = ("pancreas", "pancreas_head", "pancreas_body", "pancreas_tail")
LESION_MASK = "pancreatic_lesion"

DATASET_NAME = "Dataset502_PanTSPancLesion"

# Physical plausibility ceilings, checked per case (fractions of the whole scan).
MAX_PANCREAS_FRACTION = 0.10
MAX_LESION_FRACTION = 0.30


def compose_label(envelope: np.ndarray, lesion: np.ndarray) -> np.ndarray:
    """(envelope, lesion) binary masks -> uint8 {0,1,2} multi-class label.

    `envelope` is the union of the four gland masks (NOT including the lesion);
    the union with `lesion` is taken here so the carved-out hole is filled by the
    lesion itself and by nothing else.
    """
    if envelope.shape != lesion.shape:
        raise ValueError(f"shape mismatch envelope{envelope.shape} vs lesion{lesion.shape}")
    env = np.asarray(envelope) > 0
    les = np.asarray(lesion) > 0
    out = np.zeros(env.shape, dtype=np.uint8)
    out[env | les] = PANCREAS
    out[les] = LESION           # lesion overrides, applied last
    return out


def dataset_json(n_training: int) -> dict:
    """dataset.json for the region-based multi-class dataset."""
    return {
        "channel_names": {"0": "CT"},
        "labels": {"background": BACKGROUND, "pancreas": [PANCREAS, LESION], "lesion": LESION},
        "regions_class_order": [PANCREAS, LESION],
        "numTraining": n_training,
        "file_ending": ".nii.gz",
        "dataset_name": DATASET_NAME,
        "description": (
            "PanTS-Mini multi-class pancreas + lesion segmentation; same 1308-case 5-fold CV "
            "subset and same CT grids as Dataset501_PanTSTumor. Region-based target: "
            "pancreas = {1,2} (gland including tumor), lesion = {2} (identical to the "
            "Dataset501 binary label). Pancreas is a training label only, never an input channel."
        ),
    }
