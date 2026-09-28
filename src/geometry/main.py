import multiprocessing as mp
from pathlib import Path

import pandas as pd

from loader import case_ids, load_ducts
from geometry import max_caliber_mm

RESULTS_DIR = Path("results/geometry_results")
NUM_WORKERS = 4


def process_case(case: dict) -> dict:
    cbd_stats = max_caliber_mm(case["cbd"], case["spacing"])
    mpd_stats = max_caliber_mm(case["mpd"], case["spacing"])
    return {
        "case_id": case["case_id"],
        **{f"cbd_{k}": v for k, v in cbd_stats.items()},
        **{f"mpd_{k}": v for k, v in mpd_stats.items()},
    }


def process_case_id(case_id: str) -> dict | None:
    case = load_ducts(case_id)
    if case is None:
        return None
    row = process_case(case)
    print(
        f"{case_id}: cbd_p99={row['cbd_p99_mm']} (n_skel={row['cbd_n_skel']})  "
        f"mpd_p99={row['mpd_p99_mm']} (n_skel={row['mpd_n_skel']})"
    )
    return row


def main() -> None:
    ids = case_ids()
    print(f"{len(ids)} case ids to load, {NUM_WORKERS} workers")

    with mp.Pool(NUM_WORKERS) as pool:
        results = pool.map(process_case_id, ids)

    rows = [r for r in results if r is not None]
    print(f"{len(rows)} cases processed")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "duct_caliber.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
