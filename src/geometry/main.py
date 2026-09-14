from loader import case_ids, load_ducts
from geometry import max_caliber_mm


def process_case(case: dict) -> dict:
    cbd_stats = max_caliber_mm(case["cbd"], case["spacing"])
    mpd_stats = max_caliber_mm(case["mpd"], case["spacing"])
    return {
        "case_id": case["case_id"],
        **{f"cbd_{k}": v for k, v in cbd_stats.items()},
        **{f"mpd_{k}": v for k, v in mpd_stats.items()},
    }


def main() -> None:
    ids = case_ids()
    print(f"{len(ids)} case ids to load")
    rows: list[dict] = []
    for case_id in ids:
        case = load_ducts(case_id)
        if case is None:
            continue
        row = process_case(case)
        rows.append(row)
        print(
            f"{case_id}: cbd_p99={row['cbd_p99_mm']} (n_skel={row['cbd_n_skel']})  "
            f"mpd_p99={row['mpd_p99_mm']} (n_skel={row['mpd_n_skel']})"
        )
    print(f"{len(rows)} cases processed")


if __name__ == "__main__":
    main()
