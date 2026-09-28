"""Which lesions sit outside the pancreas, how many, how big -- and the flags
the re-evaluation scripts use to exclude them.

Tiers (from lesion_pancreas_contact.csv; see pancreas_contact.py for why plain
overlap with pancreas.nii.gz is not enough):
  inside     >= 50% of lesion voxels inside the pancreas envelope
             (pancreas | head | body | tail)
  embedded   < 50% inside, touching the envelope, 3-voxel ring >= 30% pancreas
             (lesion carved out of / sitting in a hole in the organ mask)
  abutting   < 50% inside, touching the envelope, ring < 30% pancreas (overlaps the
             edge or sits against it; includes big masses growing out of the gland)
  separated  < 50% inside AND a gap of > 1.5 voxels to the nearest envelope voxel

"Gap" is dist_to_envelope_mm divided by the case's coarsest voxel spacing (5-7.5mm
slices exist), so a lesion one thick slice away is "touching", not "separated".
The ring-contact fraction alone is size-biased (a 130mm mass touching the gland at
one edge has a mostly non-pancreas ring), so it only splits embedded vs abutting.

Flags written to lesion_inclusion.csv:
  outside_separated = separated          (primary exclusion rule)
  outside_lt50      = not inside         (literal "<50% in the mask" rule)

Usage (from repo root): .venv/bin/python3 src/lesion-sectioning/outside_pancreas_report.py
"""
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

RESULTS_DIR = Path("src/lesion-sectioning/work")  # intermediates (git-ignored)
FINAL_DIR = Path("results/lesion_location_results")  # only the excluded-lesion list + summary go here
DICE_CSV = Path("results/attenuation_results/lesion_dice_attenuation.csv")

DIAM_LABELS = ["<5mm", "5-10mm", "10-15mm", "15-20mm", "20-40mm", "40mm+"]
TIERS = ["inside", "embedded", "abutting", "separated"]
MASK_ROOT = Path("/home/rahuldeb5/research/datasets/pants/masks/mask_only")

INSIDE_MIN_FRAC = 0.5
EMBEDDED_MIN_CONTACT = 0.30
SEPARATED_MIN_GAP_VOX = 1.5


def tier(row) -> str:
    if row["frac_envelope"] >= INSIDE_MIN_FRAC:
        return "inside"
    if row["gap_vox"] > SEPARATED_MIN_GAP_VOX:
        return "separated"
    if row["contact_3vox"] >= EMBEDDED_MIN_CONTACT:
        return "embedded"
    return "abutting"


def load() -> pd.DataFrame:
    loc = pd.read_csv(RESULTS_DIR / "lesion_location_refined.csv")
    contact = pd.read_csv(RESULTS_DIR / "lesion_pancreas_contact.csv")
    dice = pd.read_csv(DICE_CSV).rename(columns={"case": "case_id"})[["case_id", "lesion_id", "lesion_dice", "fold"]]
    dice = dice.drop_duplicates(subset=["case_id", "lesion_id"])

    df = loc.merge(contact, on=["case_id", "lesion_id"], how="left").merge(dice, on=["case_id", "lesion_id"], how="left")
    max_zoom = {c: max(nib.load(MASK_ROOT / c / "segmentations" / "pancreatic_lesion.nii.gz").header.get_zooms()[:3])
                for c in df["case_id"].unique()}
    df["gap_vox"] = df["dist_to_envelope_mm"] / df["case_id"].map(max_zoom)
    df["diam_bin"] = pd.Categorical(df["diam_bin"], categories=DIAM_LABELS, ordered=True)
    df["tier"] = pd.Categorical(df.apply(tier, axis=1), categories=TIERS, ordered=True)
    df["outside_separated"] = df["tier"] == "separated"
    df["outside_lt50"] = df["tier"] != "inside"
    return df


def export_excluded(df: pd.DataFrame) -> None:
    """The two deliverables: a plain list of excluded lesions and a per-lesion summary."""
    ex = df[df["outside_separated"]].sort_values(["case_id", "lesion_id"])
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame({
        "fold": ex["fold"].astype(int),
        "case_id": ex["case_id"],
        "lesion_id": ex["lesion_id"],
        "n_components_in_case": ex["n_components"],
        "n_vox": ex["n_vox"],
        "vol_mm3": ex["vol_mm3"].round(1),
        "diam_mm": ex["diam_mm"].round(2),
        "diam_bin": ex["diam_bin"].astype(str),
        "frac_inside_pancreas": ex["frac_envelope"].round(3),
        "dist_to_pancreas_mm": ex["dist_to_envelope_mm"].round(2),
        "gap_voxels": ex["gap_vox"].round(2),
        "ring_contact_3vox": ex["contact_3vox"].round(3),
        "nnunet_lesion_dice": ex["lesion_dice"].round(3),
    })
    summary.to_csv(FINAL_DIR / "excluded_lesions.csv", index=False)

    header = [
        "# Lesions excluded from the nnU-Net re-evaluation: detached from the pancreas in the PanTS masks.",
        "# Rule: < 50% of the lesion's voxels inside the union of pancreas / pancreas_head / pancreas_body /",
        "#   pancreas_tail, AND a gap of > 1.5 voxels (measured on the case's coarsest axis) between the",
        "#   lesion and the nearest voxel of that union.",
        "# lesion_id = component index from a 26-connectivity labeling of pancreatic_lesion.nii.gz in",
        "#   nibabel (x,y,z) axis order (same numbering as results/attenuation_results/lesion_dice_attenuation.csv).",
        f"# {len(ex)} lesions in {ex['case_id'].nunique()} cases (1308-case nnU-Net 5-fold cohort). Per-lesion details: excluded_lesions.csv",
        "# format: case_id lesion_id",
    ]
    lines = header + [f"{c} {l}" for c, l in zip(ex["case_id"], ex["lesion_id"])]
    (FINAL_DIR / "excluded_lesions.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {FINAL_DIR}/excluded_lesions.txt and excluded_lesions.csv ({len(ex)} lesions)")


def main() -> None:
    df = load()
    n = len(df)
    print(f"{n} lesions in {df.case_id.nunique()} cases; Dice matched for {df.lesion_dice.notna().sum()}\n")

    print("=== tier counts ===")
    tc = df["tier"].value_counts().reindex(TIERS)
    print(pd.DataFrame({"n": tc, "pct": (tc / n * 100).round(1)}).to_string())

    print("\n=== tier x diameter bin (counts) ===")
    xt = pd.crosstab(df["diam_bin"], df["tier"])[TIERS]
    print(xt.to_string())
    print("\n=== row % (share of each size bin in each tier) ===")
    print((xt.div(xt.sum(axis=1), axis=0) * 100).round(1).to_string())

    print("\n=== size of lesions by tier ===")
    print(df.groupby("tier", observed=True)[["diam_mm", "vol_mm3"]].agg(["median", "mean", "max"]).round(1).to_string())

    print("\n=== separated lesions: primary (lesion_id==1) vs secondary component ===")
    det = df[df["tier"] == "separated"]
    print(det["lesion_id"].eq(1).map({True: "lesion_id==1", False: "lesion_id>1"}).value_counts().to_string())
    print(f"distance to envelope, separated (mm): median={det.dist_to_envelope_mm.median():.1f}, "
          f"p75={det.dist_to_envelope_mm.quantile(.75):.1f}, max={det.dist_to_envelope_mm.max():.1f}")

    print("\n=== case-level ===")
    per_case = df.groupby("case_id").agg(n_lesions=("lesion_id", "size"),
                                         n_sep=("outside_separated", "sum"),
                                         n_lt50=("outside_lt50", "sum"))
    print(f"cases with >=1 separated lesion: {(per_case.n_sep > 0).sum()}")
    print(f"cases left with NO lesion if separated ones are dropped: {(per_case.n_sep == per_case.n_lesions).sum()}")
    print(f"cases with >=1 lesion <50% inside: {(per_case.n_lt50 > 0).sum()}")
    print(f"cases left with NO lesion if all <50%-inside are dropped: {(per_case.n_lt50 == per_case.n_lesions).sum()}")

    print("\n=== nnU-Net lesion Dice by tier x size (mean | zero-Dice rate | n) ===")
    scored = df.dropna(subset=["lesion_dice"])
    g = scored.groupby(["tier", "diam_bin"], observed=True)["lesion_dice"]
    summary = pd.DataFrame({"mean_dice": g.mean().round(3),
                            "zero_rate": g.apply(lambda s: (s == 0).mean()).round(2),
                            "n": g.size()})
    print(summary.to_string())

    print("\n=== 10 largest separated lesions (worth eyeballing in a viewer) ===")
    print(det.nlargest(10, "diam_mm")[["case_id", "lesion_id", "diam_mm", "frac_envelope",
                                       "contact_3vox", "dist_to_envelope_mm", "gap_vox", "lesion_dice"]].round(2).to_string(index=False))

    df.to_csv(RESULTS_DIR / "lesion_inclusion.csv", index=False)
    xt.to_csv(RESULTS_DIR / "tier_by_diam_bin_counts.csv")
    summary.to_csv(RESULTS_DIR / "tier_dice_by_diam_bin.csv")
    print(f"\nwrote {RESULTS_DIR}/lesion_inclusion.csv, tier_by_diam_bin_counts.csv, tier_dice_by_diam_bin.csv")
    export_excluded(df)


if __name__ == "__main__":
    main()
