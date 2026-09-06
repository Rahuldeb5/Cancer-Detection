"""Inject the custom stratified 5-fold CV split into nnU-Net.

Maps the project's 1-indexed fold files to nnU-Net's 0-indexed folds:
    nnU-Net fold 0  <-  val = fold_1_ids.txt
    nnU-Net fold 1  <-  val = fold_2_ids.txt
    ...
    nnU-Net fold 4  <-  val = fold_5_ids.txt
train = union of the other four folds.

Writes ``$nnUNet_preprocessed/Dataset501_PanTSTumor/splits_final.json`` so
``nnUNetv2_train`` uses these exact splits instead of generating its own.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

FOLD_DIR = Path("src/data")
PREPROC_DIR = Path(os.environ["nnUNet_preprocessed"]) / "Dataset501_PanTSTumor"


def main() -> None:
    folds: list[list[str]] = []
    for i in range(1, 6):
        ids = sorted(
            ln.strip()
            for ln in (FOLD_DIR / f"fold_{i}_ids.txt").read_text().splitlines()
            if ln.strip()
        )
        folds.append(ids)

    # sanity: folds must be disjoint
    seen: set[str] = set()
    for i, f in enumerate(folds, 1):
        dup = seen & set(f)
        assert not dup, f"fold {i} overlaps earlier folds: {sorted(dup)[:5]}"
        seen |= set(f)

    splits = []
    for val_idx in range(5):
        val = folds[val_idx]
        train = sorted(x for j in range(5) if j != val_idx for x in folds[j])
        splits.append({"train": train, "val": val})
        print(f"nnU-Net fold {val_idx}: train={len(train)}  val={len(val)}")

    PREPROC_DIR.mkdir(parents=True, exist_ok=True)
    out = PREPROC_DIR / "splits_final.json"
    out.write_text(json.dumps(splits, indent=4))
    print(f"wrote {out}  ({len(seen)} unique cases total)")


if __name__ == "__main__":
    main()
