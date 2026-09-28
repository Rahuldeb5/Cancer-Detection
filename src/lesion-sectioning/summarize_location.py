"""Turn lesion_location.csv into the final size-binned location report.

Splits the `extrapancreatic` bucket from main.py by dist_to_pancreas_mm:
most of it is lesions touching/just past the pancreas surface (boundary call,
or a tumor that has locally extended beyond the capsule); a smaller "distant"
remainder sits mm-to-cm away from any pancreas voxel and is a much better
candidate for a mislabeled metastasis/lymph node/annotation artifact riding
along under the pancreatic_lesion class.

Usage (from repo root): .venv/bin/python3 src/lesion-sectioning/summarize_location.py
"""
from pathlib import Path

import pandas as pd

IN_CSV = Path("src/lesion-sectioning/work/lesion_location.csv")
OUT_DIR = Path("src/lesion-sectioning/work")

DIAM_LABELS = ["<5mm", "5-10mm", "10-15mm", "15-20mm", "20-40mm", "40mm+"]
JUXTA_THRESHOLD_MM = 2.0  # ~1 voxel of typical in-plane spacing; "touching" the pancreas surface


def refine_location(row) -> str:
    if row["location"] == "extrapancreatic":
        return "juxtapancreatic (touching)" if row["dist_to_pancreas_mm"] <= JUXTA_THRESHOLD_MM else "distant (likely met/artifact)"
    if row["location"] == "pancreas_junction":
        return "subregion junction"
    return f"{row['location']} ({row['certainty']})"


def main() -> None:
    df = pd.read_csv(IN_CSV)
    df["diam_bin"] = pd.Categorical(df["diam_bin"], categories=DIAM_LABELS, ordered=True)
    df["location_refined"] = df.apply(refine_location, axis=1)

    pivot = pd.crosstab(df["diam_bin"], df["location_refined"])
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    print("=== counts by diameter bin x refined location ===")
    print(pivot.to_string())
    print()
    print("=== row-normalized % by diameter bin x refined location ===")
    print(pivot_pct.round(1).to_string())
    print()

    intrapancreatic = df[df["location"].isin(["head", "body", "tail"])]
    print("=== among lesions cleanly assigned to one subregion (head/body/tail), by diam bin ===")
    print(pd.crosstab(intrapancreatic["diam_bin"], intrapancreatic["location"]).to_string())
    print()
    print("=== ...as row %  ===")
    intra_pivot = pd.crosstab(intrapancreatic["diam_bin"], intrapancreatic["location"])
    print((intra_pivot.div(intra_pivot.sum(axis=1), axis=0) * 100).round(1).to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(OUT_DIR / "location_by_diam_bin_counts.csv")
    pivot_pct.round(2).to_csv(OUT_DIR / "location_by_diam_bin_pct.csv")
    intra_pivot.to_csv(OUT_DIR / "intrapancreatic_location_by_diam_bin_counts.csv")
    df.to_csv(OUT_DIR / "lesion_location_refined.csv", index=False)
    print(f"\nwrote {OUT_DIR}/location_by_diam_bin_counts.csv, location_by_diam_bin_pct.csv, "
          f"intrapancreatic_location_by_diam_bin_counts.csv, lesion_location_refined.csv")


if __name__ == "__main__":
    main()
