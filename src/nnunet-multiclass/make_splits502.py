"""Write Dataset502's splits_final.json: Dataset501's splits, minus the empty-envelope
cases in the TRAIN lists only.

Why: a case whose four gland masks are all empty has no pancreas label at all. Keeping it
in training would teach the net "this abdomen contains no pancreas", which is a labelling
gap, not a fact. Keeping it in VALIDATION is required so the 502 validation sets stay
identical to 501's and the two baselines are compared on exactly the same cases.

The empty-envelope list comes from work/nnunet_ds502/oracle_summary.json (or --exclude-file,
one id per line). Asserts: folds disjoint, every val list identical to Dataset501's,
every excluded case absent from every train list.

    source ~/research/nnunet_env/env.sh
    .venv/bin/python src/nnunet-multiclass/make_splits502.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "nnunet-multiclass"))
from labels502 import DATASET_NAME  # noqa: E402

PRE = Path(os.environ["nnUNet_preprocessed"])
SRC = PRE / "Dataset501_PanTSTumor" / "splits_final.json"
DST_DIR = PRE / DATASET_NAME
ORACLE = REPO / "work" / "nnunet_ds502" / "oracle_summary.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-file", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    splits501 = json.loads(SRC.read_text())
    assert len(splits501) == 5, f"expected 5 folds in {SRC}, got {len(splits501)}"

    if a.exclude_file:
        excl = sorted({ln.strip() for ln in a.exclude_file.read_text().splitlines() if ln.strip()})
        where = str(a.exclude_file)
    else:
        excl = sorted(json.loads(ORACLE.read_text())["empty_envelope_cases"]["ids"])
        where = str(ORACLE)
    print(f"excluding {len(excl)} empty-envelope cases from TRAIN only (from {where}):")
    for c in excl:
        print(f"  {c}")
    eset = set(excl)

    # folds must be disjoint (val lists partition the cohort)
    seen: set[str] = set()
    for k, s in enumerate(splits501):
        dup = seen & set(s["val"])
        assert not dup, f"fold {k} val overlaps an earlier fold: {sorted(dup)[:5]}"
        seen |= set(s["val"])

    out = []
    print(f"\n{'fold':>4} {'train501':>9} {'train502':>9} {'dropped':>8} {'val':>5}")
    for k, s in enumerate(splits501):
        train = [c for c in s["train"] if c not in eset]
        val = list(s["val"])
        assert val == s["val"], "validation list must be byte-identical to Dataset501's"
        assert not (set(train) & eset), "an excluded case survived in train"
        assert not (set(train) & set(val)), f"fold {k}: train/val overlap"
        out.append({"train": train, "val": val})
        print(f"{k:>4} {len(s['train']):>9} {len(train):>9} "
              f"{len(s['train']) - len(train):>8} {len(val):>5}")

    # an excluded case is still validated exactly once, in its own fold
    for c in excl:
        n = sum(1 for s in out if c in s["val"])
        assert n == 1, f"{c} appears in {n} validation lists, expected 1"

    print(f"\nunique cases over the val lists: {len(seen)}")
    if a.dry_run:
        print("--dry-run: nothing written")
        return
    DST_DIR.mkdir(parents=True, exist_ok=True)
    dst = DST_DIR / "splits_final.json"
    dst.write_text(json.dumps(out, indent=4))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
