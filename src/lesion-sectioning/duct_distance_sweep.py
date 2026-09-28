"""Cumulative share of lesions within x mm of the main pancreatic duct, per diameter bin.

Same construction as src/attenuation-labeling/duct_proximity_size_sweep.py (which covered only
<20 mm), extended to all six bins and with the excluded lesions removed. Distance is
dist_surface_to_duct_mm (nearest lesion voxel to nearest duct voxel; agrees with the original
dist_to_mpd_mm). Like the original, every lesion whose case has a duct mask is counted -- NOT
just the "duct_valid" subset used for the position maps, which by construction over-selects
lesions that sit near a duct segment.

Usage (from repo root): .venv/bin/python3 src/lesion-sectioning/duct_distance_sweep.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IN_CSV = Path("results/lesion_duct_position/lesion_duct_position.csv")
OUT_PNG = Path("results/lesion_duct_position/duct_distance_sweep_by_size.png")

BINS = ["<5mm", "5-10mm", "10-15mm", "15-20mm", "20-40mm", "40mm+"]
X_RANGE_MM = range(3, 21)  # same range as the original sweep
STYLE = {  # Okabe-Ito colors, fixed order; marker shape is a second, color-independent cue
    "<5mm": ("#0072B2", "o"), "5-10mm": ("#E69F00", "s"), "10-15mm": ("#009E73", "^"),
    "15-20mm": ("#CC79A7", "D"), "20-40mm": ("#D55E00", "v"), "40mm+": ("#56B4E9", "P"),
}


def load() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)
    df = df[df["duct_present"] & df["dist_surface_to_duct_mm"].notna()].copy()
    df["diam_bin"] = pd.Categorical(df["diam_bin"], categories=BINS, ordered=True)
    return df


def sweep(df: pd.DataFrame) -> pd.DataFrame:
    totals = df.groupby("diam_bin", observed=False).size()
    rows = []
    for x in X_RANGE_MM:
        counts = df[df["dist_surface_to_duct_mm"] <= x].groupby("diam_bin", observed=False).size()
        for b in BINS:
            rows.append({"threshold_mm": x, "diam_bin": b, "n_lesions": int(counts[b]),
                         "pct_of_bin": counts[b] / totals[b] * 100, "bin_total": int(totals[b])})
    return pd.DataFrame(rows)


def plot(sw: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for b in BINS:
        s = sw[sw["diam_bin"] == b]
        color, marker = STYLE[b]
        label = f"{b} (n={s['bin_total'].iloc[0]})"
        kw = dict(color=color, marker=marker, ms=5, lw=1.8, markeredgecolor="white", markeredgewidth=0.6, label=label)
        ax1.plot(s["threshold_mm"], s["n_lesions"], **kw)
        ax2.plot(s["threshold_mm"], s["pct_of_bin"], **kw)

    ax1.set_ylabel("n lesions with surface distance to duct <= x")
    ax2.set_ylabel("% of that bin's lesions with surface distance to duct <= x")
    ax1.set_title("Cumulative count within x mm of the duct")
    ax2.set_title("Cumulative % (within-bin) within x mm of the duct")
    for ax in (ax1, ax2):
        ax.set_xlabel("duct-adjacency threshold x (mm)")
        ax.grid(alpha=0.3)
    ax1.legend(title="diameter bin", loc="upper left", frameon=False)
    ax2.set_ylim(0, 100)

    fig.suptitle("Lesion proximity to the main pancreatic duct, all size bins (excluded lesions removed)")
    fig.text(0.5, 0.01, "lesions in cases with a duct mask only; distance = nearest lesion voxel to nearest duct voxel",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = load()
    print(f"{len(df)} lesions with a duct mask")
    sw = sweep(df)
    plot(sw, OUT_PNG)
    print(f"wrote {OUT_PNG}")
    tab = sw[sw["threshold_mm"].isin([3, 5, 10, 15, 20])].pivot(index="diam_bin", columns="threshold_mm", values="pct_of_bin")
    tab.insert(0, "n", sw.drop_duplicates("diam_bin").set_index("diam_bin")["bin_total"])
    print("\n% of each bin within x mm of the duct:")
    print(tab.round(1).to_string())


if __name__ == "__main__":
    main()
