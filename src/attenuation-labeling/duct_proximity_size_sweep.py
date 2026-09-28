"""For lesions <20mm, sweep the duct-adjacency distance threshold x from 3mm
to 20mm and plot how many lesions of each small-diameter bin fall within
x mm of the MPD at each step. Answers: as you relax the "duct-adjacent"
cutoff, where do the small (hardest to segment) tumors show up?

Inputs: results/attenuation_results/attenuation_labels.csv (dist_to_mpd_mm,
mpd_present) joined with results/attenuation_results/lesion_dice_attenuation.csv
(diam_mm) on (case_id, lesion_id).

Usage (from repo root): .venv/bin/python3 src/attenuation-labeling/duct_proximity_size_sweep.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ATTEN_CSV = Path("results/attenuation_results/attenuation_labels.csv")
DICE_CSV = Path("results/attenuation_results/lesion_dice_attenuation.csv")
OUT_DIR = Path("results/attenuation_results")

DIAM_BINS = [0, 5, 10, 15, 20]
DIAM_LABELS = ["<5mm", "5-10mm", "10-15mm", "15-20mm"]
X_RANGE_MM = range(3, 21)  # 3mm..20mm inclusive


def load_data() -> pd.DataFrame:
    atten = pd.read_csv(ATTEN_CSV)
    dice = pd.read_csv(DICE_CSV)
    df = atten.merge(
        dice[["case", "lesion_id", "diam_mm"]],
        left_on=["case_id", "lesion_id"],
        right_on=["case", "lesion_id"],
        how="left",
    )
    df = df[df["mpd_present"] == True]  # dist_to_mpd_mm only defined when MPD exists
    df = df.dropna(subset=["dist_to_mpd_mm", "diam_mm"])
    df["diam_bin"] = pd.cut(df["diam_mm"], bins=DIAM_BINS, labels=DIAM_LABELS, right=False)
    return df[df["diam_bin"].notna()]  # drop >=20mm -- out of scope for this sweep


def sweep(df: pd.DataFrame) -> pd.DataFrame:
    totals = df.groupby("diam_bin").size()
    rows = []
    for x in X_RANGE_MM:
        within = df[df["dist_to_mpd_mm"] <= x]
        counts = within.groupby("diam_bin").size().reindex(DIAM_LABELS, fill_value=0)
        for label in DIAM_LABELS:
            rows.append({
                "threshold_mm": x,
                "diam_bin": label,
                "n_lesions": int(counts[label]),
                "pct_of_bin_total": counts[label] / totals.get(label, 1) * 100,
            })
    return pd.DataFrame(rows)


def plot(sweep_df: pd.DataFrame, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for label in DIAM_LABELS:
        sub = sweep_df[sweep_df["diam_bin"] == label]
        ax1.plot(sub["threshold_mm"], sub["n_lesions"], marker="o", label=label)
        ax2.plot(sub["threshold_mm"], sub["pct_of_bin_total"], marker="o", label=label)

    ax1.set_xlabel("duct-adjacency threshold x (mm)")
    ax1.set_ylabel("n lesions with dist_to_mpd_mm <= x")
    ax1.set_title("Cumulative count within x mm of MPD")
    ax1.legend(title="diameter bin")
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("duct-adjacency threshold x (mm)")
    ax2.set_ylabel("% of that bin's lesions with dist_to_mpd_mm <= x")
    ax2.set_title("Cumulative % (within-bin) within x mm of MPD")
    ax2.legend(title="diameter bin")
    ax2.grid(alpha=0.3)

    fig.suptitle("Small-lesion proximity to the main pancreatic duct")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    df = load_data()
    print(f"lesions in scope (<20mm, mpd present): {len(df)}")
    print(df.groupby("diam_bin", observed=True).size().reindex(DIAM_LABELS))

    sweep_df = sweep(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(OUT_DIR / "duct_proximity_size_sweep.csv", index=False)
    print(f"wrote {OUT_DIR / 'duct_proximity_size_sweep.csv'}")

    plot(sweep_df, OUT_DIR / "duct_proximity_size_sweep.png")


if __name__ == "__main__":
    main()
