"""Phantom / known-answer tests for the Dataset502 multi-class label definition.

Rule 1 of CONTEXT.md: these run before the builder ever touches a real scan.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "nnunet-multiclass"))
from labels502 import BACKGROUND, LESION, PANCREAS, compose_label, dataset_json  # noqa: E402


def _phantom():
    """A 20^3 cube with a 10^3 gland and a 4^3 lesion carved out of it.

    gland_carved mimics PanTS pancreas.nii.gz (lesion removed); head/body/tail-style
    masks that still cover the lesion are simulated by `tail_part`.
    """
    gland = np.zeros((20, 20, 20), np.uint8)
    gland[5:15, 5:15, 5:15] = 1
    lesion = np.zeros_like(gland)
    lesion[6:10, 6:10, 6:10] = 1
    gland_carved = gland & ~lesion
    return gland, gland_carved, lesion


def test_carved_hole_is_filled_by_the_lesion():
    gland, gland_carved, lesion = _phantom()
    lab = compose_label(gland_carved, lesion)
    # whole gland (including the tumor) is foreground
    assert np.array_equal(lab > 0, gland > 0)
    # the lesion is exactly class 2
    assert np.array_equal(lab == LESION, lesion > 0)
    # class 1 is the gland minus the lesion
    assert np.array_equal(lab == PANCREAS, (gland > 0) & ~(lesion > 0))
    assert set(np.unique(lab).tolist()) == {BACKGROUND, PANCREAS, LESION}
    assert lab.dtype == np.uint8


def test_lesion_outside_the_gland_still_becomes_pancreas_region():
    """The 70 `separated` lesions: nesting makes them part of the pancreas region too.
    That is a consequence of the fixed region definition, not a bug to fix."""
    _, gland_carved, _ = _phantom()
    far = np.zeros_like(gland_carved)
    far[17:19, 17:19, 17:19] = 1          # disjoint from the gland
    lab = compose_label(gland_carved, far)
    assert np.array_equal(lab == LESION, far > 0)
    assert (lab[17:19, 17:19, 17:19] == LESION).all()
    # region "pancreas" = {1,2} therefore contains the separated lesion
    assert np.isin(lab[17:19, 17:19, 17:19], [PANCREAS, LESION]).all()


def test_empty_envelope_and_empty_lesion():
    z = np.zeros((4, 4, 4), np.uint8)
    assert compose_label(z, z).max() == 0                       # negative, no gland
    les = z.copy(); les[1, 1, 1] = 1
    lab = compose_label(z, les)                                 # lesion, no gland masks
    assert lab[1, 1, 1] == LESION and lab.sum() == LESION
    gl = z.copy(); gl[2, 2, 2] = 1
    assert compose_label(gl, z)[2, 2, 2] == PANCREAS            # gland, no lesion


def test_non_binary_input_is_treated_as_binary():
    """The two on-disk int8 encodings mean a mask may arrive as {0,1} or {0,255}."""
    gland, gland_carved, lesion = _phantom()
    lab_a = compose_label(gland_carved, lesion)
    lab_b = compose_label(gland_carved * 255, lesion * 127)
    assert np.array_equal(lab_a, lab_b)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compose_label(np.zeros((2, 2, 2), np.uint8), np.zeros((2, 2, 3), np.uint8))


def test_dataset_json_is_region_based():
    d = dataset_json(1308)
    assert d["labels"] == {"background": 0, "pancreas": [1, 2], "lesion": 2}
    assert d["regions_class_order"] == [1, 2]
    assert d["channel_names"] == {"0": "CT"}          # CT only: no pancreas input channel
    assert d["file_ending"] == ".nii.gz"
    assert d["numTraining"] == 1308
    json.dumps(d)                                     # must be serialisable as-is
