"""Join PCA-frame lesion position with location + nnU-Net Dice, and ask:
does WHERE a lesion sits in the gland (not just its size) predict whether
nnU-Net finds it? Motivation: duct-dilation proximity (results/attenuation_results)
already explains a chunk of the 20-40mm bin, but <20mm lesions (near-zero Dice)
don't show that pattern -- this checks whether gland position does instead.

Usage (from repo root):
  .venv/bin/python3 src/lesion-sectioning/hitrate_dice_report.py                      # all lesions
  .venv/bin/python3 src/lesion-sectioning/hitrate_dice_report.py --exclude separated  # drop lesions detached from the pancreas
  .venv/bin/python3 src/lesion-sectioning/hitrate_dice_report.py --exclude lt50       # drop everything <50% inside the pancreas envelope
(flags come from outside_pancreas_report.py; filtered runs write to src/lesion-sectioning/work/exclude_<mode>/)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("src/lesion-sectioning/work")
PCA_CSV = RESULTS_DIR / "lesion_pca_position.csv"
LOC_CSV = RESULTS_DIR / "lesion_location_refined.csv"
DICE_CSV = Path("results/attenuation_results/lesion_dice_attenuation.csv")
INCLUSION_CSV = RESULTS_DIR / "lesion_inclusion.csv"
EXCLUDE_FLAG = {"separated": "outside_separated", "lt50": "outside_lt50"}

DIAM_LABELS = ["<5mm", "5-10mm", "10-15mm", "15-20mm", "20-40mm", "40mm+"]
PC1_BIN_EDGES = np.linspace(0, 1, 6)  # 5 bins spanning head(0)->tail(1); values outside [0,1] get clipped in
PC1_BIN_LABELS = ["head 0-20%", "20-40%", "mid 40-60%", "60-80%", "tail 80-100%"]


def load_merged(exclude: str = "none") -> pd.DataFrame:
    pca = pd.read_csv(PCA_CSV)
    loc = pd.read_csv(LOC_CSV)[["case_id", "lesion_id", "diam_mm", "diam_bin", "location", "location_refined"]]
    dice = pd.read_csv(DICE_CSV).rename(columns={"case": "case_id"})[["case_id", "lesion_id", "lesion_dice"]]
    # a lesion can appear in multiple nnU-Net folds only if it's mis-keyed; dedupe defensively, keep first
    dice = dice.drop_duplicates(subset=["case_id", "lesion_id"])

    df = pca.merge(loc, on=["case_id", "lesion_id"], how="left")
    df = df.merge(dice, on=["case_id", "lesion_id"], how="left")
    df["diam_bin"] = pd.Categorical(df["diam_bin"], categories=DIAM_LABELS, ordered=True)

    if exclude != "none":
        flag = EXCLUDE_FLAG[exclude]
        inc = pd.read_csv(INCLUSION_CSV)[["case_id", "lesion_id", flag]]
        df = df.merge(inc, on=["case_id", "lesion_id"], how="left")
        n_before = len(df)
        df = df[~df[flag].fillna(False).astype(bool)].drop(columns=flag)
        print(f"exclude={exclude}: dropped {n_before - len(df)} of {n_before} lesions with PCA position")

    n_no_dice = df["lesion_dice"].isna().sum()
    print(f"{len(df)} lesions with PCA position; {n_no_dice} have no matched Dice "
          f"(not in nnU-Net's 5-fold validation output -- e.g. all-background GT slabs are skipped there)")
    return df


def pc1_bin(x: float) -> str:
    x = min(max(x, 0.0), 0.999999)
    idx = int(np.digitize([x], PC1_BIN_EDGES[1:-1])[0])
    return PC1_BIN_LABELS[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", choices=["none", *EXCLUDE_FLAG], default="none")
    args = ap.parse_args()
    out_dir = RESULTS_DIR if args.exclude == "none" else RESULTS_DIR / f"exclude_{args.exclude}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_merged(args.exclude)
    df["pc1_bin"] = df["pc1_norm"].apply(pc1_bin)
    df["pc1_bin"] = pd.Categorical(df["pc1_bin"], categories=PC1_BIN_LABELS, ordered=True)

    print("\n=== subregion (head/body/tail from mask overlap) x diameter bin, kept lesions ===")
    loc_x = pd.crosstab(df["diam_bin"], df["location"])
    print(loc_x.to_string())
    print((loc_x.div(loc_x.sum(axis=1), axis=0) * 100).round(1).to_string())
    loc_x.to_csv(out_dir / "location_by_diambin_counts.csv")

    print("\n=== hit-rate: lesion COUNT by long-axis position (head->tail) x diameter bin ===")
    hit_counts = pd.crosstab(df["diam_bin"], df["pc1_bin"])
    print(hit_counts.to_string())
    hit_counts.to_csv(out_dir / "hitrate_counts_pc1_x_diambin.csv")

    print("\n=== hit-rate: row-normalized % (within each diameter bin, where do lesions sit?) ===")
    hit_pct = hit_counts.div(hit_counts.sum(axis=1), axis=0) * 100
    print(hit_pct.round(1).to_string())
    hit_pct.round(2).to_csv(out_dir / "hitrate_pct_pc1_x_diambin.csv")

    scored = df.dropna(subset=["lesion_dice"])
    print(f"\n=== mean nnU-Net Dice by long-axis position x diameter bin (n={len(scored)} scored lesions) ===")
    dice_mean = scored.pivot_table(index="diam_bin", columns="pc1_bin", values="lesion_dice", aggfunc="mean", observed=True)
    print(dice_mean.round(3).to_string())
    dice_mean.round(4).to_csv(out_dir / "dice_mean_pc1_x_diambin.csv")

    dice_n = scored.pivot_table(index="diam_bin", columns="pc1_bin", values="lesion_dice", aggfunc="count", observed=True)
    print("\n=== n scored lesions per cell (context for the means above -- small cells are noisy) ===")
    print(dice_n.to_string())
    dice_n.to_csv(out_dir / "dice_n_pc1_x_diambin.csv")

    print("\n=== Spearman corr(dice, pc1/pc2/pc3 position) within each diameter bin ===")
    corr_rows = []
    for b in DIAM_LABELS:
        sub = scored[scored["diam_bin"] == b]
        if len(sub) < 5:
            continue
        row = {"diam_bin": b, "n": len(sub)}
        for axis in ("pc1_norm", "pc2_norm", "pc3_norm"):
            row[f"spearman_{axis}"] = sub["lesion_dice"].corr(sub[axis], method="spearman")
        corr_rows.append(row)
    corr_df = pd.DataFrame(corr_rows)
    print(corr_df.round(3).to_string(index=False))
    corr_df.to_csv(out_dir / "dice_position_spearman_by_diambin.csv", index=False)

    print("\n=== zero-Dice rate by long-axis position, small lesions only (<20mm) ===")
    small = scored[scored["diam_mm"] < 20]
    small_zero = small.assign(is_zero=small["lesion_dice"] == 0.0)
    zero_rate = small_zero.groupby("pc1_bin", observed=True)["is_zero"].agg(["mean", "count"])
    zero_rate.columns = ["zero_dice_rate", "n"]
    print(zero_rate.round(3).to_string())
    zero_rate.to_csv(out_dir / "small_lesion_zero_dice_rate_by_pc1.csv")

    print(f"\nwrote hitrate/dice CSVs to {out_dir}/")


if __name__ == "__main__":
    main()
