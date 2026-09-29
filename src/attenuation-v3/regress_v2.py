"""S3 step 0: rerun the committed v2 attenuation method unchanged and check it
reproduces results/attenuation_results/attenuation_labels.csv before anything new.

v2's own code (src/attenuation-labeling/main.py) is imported, not copied, so this
tests exactly what produced the published labels. Only tumor+ cases are run: v2
returns no rows for an empty lesion mask, so the negatives cannot change the output.

    .venv/bin/python src/attenuation-v3/regress_v2.py      # ~30-40 min, 2 workers
"""
import importlib.util
import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
V2_PATH = REPO / "src" / "attenuation-labeling" / "main.py"
V2_CSV = REPO / "results" / "attenuation_results" / "attenuation_labels.csv"
MASTER = REPO / "results" / "master_lesion_table" / "master_lesions.csv"
OUT = REPO / "work" / "attenuation_v3" / "v2_rerun.csv"
NUM_WORKERS = 2

_v2 = None


def v2_module():
    # loaded lazily inside each worker (forkserver re-imports this file, not main.py)
    global _v2
    if _v2 is None:
        spec = importlib.util.spec_from_file_location("attenuation_v2_main", V2_PATH)
        _v2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_v2)
    return _v2


def run_case(case_id: str) -> list[dict]:
    return v2_module().process_case(case_id)


def compare(new: pd.DataFrame, old: pd.DataFrame) -> bool:
    key = ["case_id", "lesion_id"]
    j = old.merge(new, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)
    ok = True
    print(f"rows: committed {len(old)}, rerun {len(new)}, both {(j._merge == 'both').sum()}")
    if (j._merge != "both").any():
        ok = False
        print(j[j._merge != "both"][key + ["_merge"]])
    b = j[j._merge == "both"]
    for col in ["n_vox", "n_panc_vox"]:
        bad = (b[f"{col}_old"] != b[f"{col}_new"]).sum()
        print(f"{col}: {bad} mismatches")
        ok &= bad == 0
    for col in ["delta_hu", "tumor_hu_median", "panc_hu_median"]:
        o, n = b[f"{col}_old"], b[f"{col}_new"]
        bad = ~(((o - n).abs() <= 1e-6) | (o.isna() & n.isna()))
        print(f"{col}: {bad.sum()} mismatches (max |diff| {(o - n).abs().max():.3g})")
        ok &= bad.sum() == 0
    bad = (b.attenuation_old != b.attenuation_new).sum()
    print(f"attenuation: {bad} mismatches")
    ok &= bad == 0
    print("counts rerun:", new.attenuation.value_counts().to_dict())
    return bool(ok)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    master = pd.read_csv(MASTER)
    ids = sorted(master.case_id.unique())
    print(f"{len(ids)} tumor+ cases, {NUM_WORKERS} workers")
    with mp.Pool(NUM_WORKERS) as pool:
        rows = [r for rs in pool.imap(run_case, ids, chunksize=4) for r in rs]
    new = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    ok = compare(new, pd.read_csv(V2_CSV))
    print("REGRESSION", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
