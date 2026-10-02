"""Known-answer oracle for the region channel order, using nnU-Net's OWN LabelManager.

The 1-epoch rehearsal cannot prove the channel mapping: an untrained net puts every
probability below 0.5, so the saved segmentation is empty and
"(channel 1 > 0.5) == (seg == 2)" is satisfied trivially (0 == 0). This script instead
feeds hand-made probability arrays with known answers through the real
LabelManager.convert_probabilities_to_segmentation() built from Dataset502's own
dataset.json, so the mapping is established without waiting for a trained model.

    source ~/research/nnunet_env/env.sh
    python src/nnunet-multiclass/oracle_regions.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PRE = Path(os.environ["nnUNet_preprocessed"]) / "Dataset502_PanTSPancLesion"
from nnunetv2.utilities.label_handling.label_handling import LabelManager  # noqa: E402

dj = json.loads((PRE / "dataset.json").read_text())
lm = LabelManager(dj["labels"], regions_class_order=dj.get("regions_class_order"))

print(f"dataset.json labels          : {dj['labels']}")
print(f"dataset.json regions_class_order: {dj.get('regions_class_order')}")
print(f"LabelManager.has_regions     : {lm.has_regions}")
print(f"LabelManager.foreground_regions : {lm.foreground_regions}")
print(f"LabelManager.all_labels      : {lm.all_labels}")
print(f"num_segmentation_heads       : {lm.num_segmentation_heads}")
print(f"inference_nonlin             : {lm.inference_nonlin}")
assert lm.has_regions, "Dataset502 must be region-based"
assert lm.num_segmentation_heads == 2, lm.num_segmentation_heads
assert list(lm.foreground_regions) == [(1, 2), 2], lm.foreground_regions
print("\n-> channel 0 = region (1,2) = pancreas incl. tumor;  channel 1 = region 2 = lesion\n")

# Four voxels, four known answers. prob[0] = pancreas region, prob[1] = lesion.
#   voxel 0: neither    -> 0
#   voxel 1: pancreas only                 -> 1
#   voxel 2: pancreas AND lesion           -> 2  (lesion assigned last, overrides)
#   voxel 3: lesion only, pancreas BELOW .5-> 2  (the nesting quirk: channel 0 need not fire)
prob = np.zeros((2, 2, 2, 1), dtype=np.float32)
flat = prob.reshape(2, -1)
flat[0] = [0.10, 0.90, 0.90, 0.10]
flat[1] = [0.10, 0.10, 0.90, 0.90]
expected = np.array([0, 1, 2, 2], dtype=np.int64)

seg = np.asarray(lm.convert_probabilities_to_segmentation(prob)).reshape(-1)
print(f"{'voxel':>5} {'p_pancreas':>11} {'p_lesion':>9} {'seg':>4} {'expected':>9}")
for i in range(4):
    print(f"{i:>5} {flat[0][i]:>11.2f} {flat[1][i]:>9.2f} {seg[i]:>4} {expected[i]:>9}")
assert np.array_equal(seg, expected), f"got {seg.tolist()}, expected {expected.tolist()}"

# The two documented equalities, now on a non-empty segmentation
p_panc, p_les = flat[0], flat[1]
assert np.array_equal(p_les > 0.5, seg == 2), "(channel1 > 0.5) != (seg == 2)"
assert np.array_equal((p_panc > 0.5) | (p_les > 0.5), seg > 0), "channel union != (seg > 0)"
print("\n(channel1 > 0.5) == (seg == 2)                 : OK, on a NON-EMPTY segmentation")
print("(channel0>0.5 | channel1>0.5) == (seg > 0)     : OK")

# sigmoid, not softmax: channels are independent and need not sum to 1
import torch  # noqa: E402
logits = torch.tensor([[2.0], [2.0]])
p = lm.inference_nonlin(logits)
print(f"\ninference_nonlin([2,2]) = {p.flatten().tolist()} -> sum {float(p.sum()):.4f} "
      f"(sigmoid: NOT 1.0; a softmax would give 0.5/0.5)")
assert abs(float(p.sum()) - 1.0) > 0.1, "nonlinearity looks like a softmax, not a sigmoid"

print("\nPASS: region channel order and the two equalities hold under nnU-Net's own LabelManager.")
