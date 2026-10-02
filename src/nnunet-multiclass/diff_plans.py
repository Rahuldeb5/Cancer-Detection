"""Diff Dataset502's nnUNetPlans.json against Dataset501's for one configuration.

Compares the fields that decide whether the two baselines are comparable at all --
target spacing, patch size, batch size, network architecture, normalization (and its
CT clip/mean/std from the fingerprint) -- plus the dataset.json label blocks.

Exits 2 if spacing, patch_size or batch_size differ (the "STOP and report" cases).

    source ~/research/nnunet_env/env.sh
    .venv/bin/python src/nnunet-multiclass/diff_plans.py [-c 3d_fullres]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PRE = Path(os.environ["nnUNet_preprocessed"])
A_NAME, B_NAME = "Dataset501_PanTSTumor", "Dataset502_PanTSPancLesion"
MATERIAL = ("spacing", "patch_size", "batch_size")


def flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--configuration", default="3d_fullres")
    a = ap.parse_args()
    cfg = a.configuration

    pa = json.loads((PRE / A_NAME / "nnUNetPlans.json").read_text())
    pb = json.loads((PRE / B_NAME / "nnUNetPlans.json").read_text())
    ca, cb = pa["configurations"][cfg], pb["configurations"][cfg]

    print(f"=== plans diff, configuration {cfg} ===")
    print(f"A = {A_NAME}\nB = {B_NAME}\n")

    print("-- top level --")
    for k in sorted(set(pa) | set(pb)):
        if k == "configurations":
            continue
        va, vb = pa.get(k, "<absent>"), pb.get(k, "<absent>")
        if va != vb:
            print(f"  {k}:\n    A {va}\n    B {vb}")

    fa, fb = flat(ca), flat(cb)
    material_diff = []
    print(f"\n-- configurations.{cfg} --")
    for k in sorted(set(fa) | set(fb)):
        va, vb = fa.get(k, "<absent>"), fb.get(k, "<absent>")
        if va != vb:
            print(f"  {k}:\n    A {va}\n    B {vb}")
            if k in MATERIAL or k.split(".")[0] in MATERIAL:
                material_diff.append(k)
    if not material_diff:
        print(f"  spacing / patch_size / batch_size: IDENTICAL "
              f"(spacing {ca['spacing']}, patch {ca['patch_size']}, batch {ca['batch_size']})")

    print("\n-- CT normalization (from dataset_fingerprint.json, channel 0) --")
    for name in (A_NAME, B_NAME):
        fp = json.loads((PRE / name / "dataset_fingerprint.json").read_text())
        ip = fp["foreground_intensity_properties_per_channel"]["0"]
        print(f"  {name}: mean {ip['mean']:.4f} std {ip['std']:.4f} "
              f"p0.5 {ip['percentile_00_5']:.2f} p99.5 {ip['percentile_99_5']:.2f} "
              f"median {ip['median']:.2f}")
    print("  (CTNormalization clips to [p0.5, p99.5] then z-scores with these mean/std; the "
          "foreground now includes the whole pancreas, not just tumor, so these SHOULD differ)")

    print("\n-- dataset.json labels --")
    for name in (A_NAME, B_NAME):
        dj = json.loads((PRE / name / "dataset.json").read_text())
        print(f"  {name}: labels={dj['labels']} "
              f"regions_class_order={dj.get('regions_class_order', '<none>')} "
              f"numTraining={dj['numTraining']}")

    if material_diff:
        print(f"\nSTOP: material differences in {material_diff}")
        sys.exit(2)
    print("\nOK: no material difference in spacing / patch size / batch size.")


if __name__ == "__main__":
    main()
