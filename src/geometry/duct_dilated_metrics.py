"""nnU-Net metrics on duct-dilated, tumor-positive CT scans.

Outputs (results/geometry_results/, or .../without_excluded_lesions/ with --exclude):
  duct_mask_quality.csv          per-case CBD/MPD mask status: absent / degenerate / proper
  duct_dilated_group_metrics.csv dice + detection sensitivity, dilated vs not, tumor?==1 only
  duct_dilated_dice_by_size.csv  lesion dice by diameter bin per dilation group
  duct_flag_sens_spec.csv        sens/spec of each dilation flag vs tumor?, all cases and evaluable-only

"degenerate" = mask has voxels but the skeleton is empty (n_skel == 0), i.e. the
segmentation is too thin/fragmentary to measure a caliber. Flags in
duct_dilation_flags.csv treat both absent and degenerate as "not dilated".

--exclude FILE (lines of "case_id lesion_id", '#' comments ok) drops those lesions
from all lesion-level stats. Cases whose every lesion is excluded are dropped from
all case-level stats too (no tumor left). Cases with only some lesions excluded keep
their stored per_case_metrics Dice/detection, which still include the excluded lesion
(not recomputable without the masks).

Usage (from repo root):
  .venv/bin/python3 src/geometry/duct_dilated_metrics.py
  .venv/bin/python3 src/geometry/duct_dilated_metrics.py --exclude results/lesion_location_results/excluded_lesions.txt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GEOM = Path("results/geometry_results")
ATTEN = Path("results/attenuation_results")
FLAGS = ["mpd_dilated", "cbd_dilated", "double_duct_dilated", "either_duct_dilated"]
BINS = [0, 5, 10, 15, 20, 40, np.inf]
LABELS = ["<5mm", "5-10mm", "10-15mm", "15-20mm", "20-40mm", "40+mm"]


def load_exclusions(path: Path) -> set[tuple[str, int]]:
    pairs = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            case, lesion_id = line.split()
            pairs.add((case, int(lesion_id)))
    return pairs


def mask_status(present: pd.Series, n_skel: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(~present, "absent", np.where(n_skel == 0, "degenerate", "proper")),
        index=present.index,
    )


def build_quality() -> pd.DataFrame:
    duct = pd.read_csv(GEOM / "duct_caliber.csv")
    flags = pd.read_csv(GEOM / "duct_dilation_flags.csv")
    q = duct[["case_id", "cbd_n_skel", "mpd_n_skel"]].copy()
    q["cbd_status"] = mask_status(duct["cbd_present"], duct["cbd_n_skel"])
    q["mpd_status"] = mask_status(duct["mpd_present"], duct["mpd_n_skel"])
    q = q.merge(flags, on="case_id")
    q["both_proper"] = (q["cbd_status"] == "proper") & (q["mpd_status"] == "proper")
    return q


def evaluable_mask(q: pd.DataFrame, flag: str) -> pd.Series:
    cbd, mpd = q["cbd_status"] == "proper", q["mpd_status"] == "proper"
    return {"cbd_dilated": cbd, "mpd_dilated": mpd}.get(flag, cbd & mpd)


def sens_spec(df: pd.DataFrame, flag: str) -> dict:
    pred, y = df[flag], df["tumor_present"] == 1
    tp, fn = (pred & y).sum(), (~pred & y).sum()
    fp, tn = (pred & ~y).sum(), (~pred & ~y).sum()
    return {"n": len(df), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp)}


def group_summary(cases: pd.DataFrame, lesions: pd.DataFrame) -> dict:
    return {
        "n_cases": len(cases),
        "n_lesions": len(lesions),
        "case_dice_mean": cases["dice"].mean(),
        "case_dice_median": cases["dice"].median(),
        "case_detect_sens": 1 - cases["missed"].mean(),
        "lesion_dice_mean": lesions["lesion_dice"].mean(),
        "lesion_dice_median": lesions["lesion_dice"].median(),
        "lesion_pct_dice0": (lesions["lesion_dice"] == 0).mean() * 100,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", type=Path, default=None)
    args = ap.parse_args()

    dice = pd.read_csv(ATTEN / "lesion_dice_attenuation.csv")[["case", "lesion_id", "diam_mm", "lesion_dice"]]
    dice["diam_bin"] = pd.cut(dice["diam_mm"], bins=BINS, labels=LABELS, right=False)

    out_dir = GEOM
    fully_excluded: set[str] = set()
    if args.exclude is not None:
        out_dir = GEOM / "without_excluded_lesions"
        out_dir.mkdir(parents=True, exist_ok=True)
        excluded = load_exclusions(args.exclude)
        is_excl = pd.Series([(c, l) in excluded for c, l in zip(dice["case"], dice["lesion_id"])])
        assert is_excl.sum() == len(excluded), "some excluded lesions not found in lesion_dice_attenuation.csv"
        n_left = (~is_excl).groupby(dice["case"]).sum()
        fully_excluded = set(n_left[n_left == 0].index)
        dice = dice[~is_excl]
        print(f"excluded {len(excluded)} lesions; {len(fully_excluded)} cases lose every lesion "
              f"and are dropped from case-level stats\n")

    q_all = build_quality()
    q_all["all_lesions_excluded"] = q_all["case_id"].isin(fully_excluded)
    q_all.to_csv(out_dir / "duct_mask_quality.csv", index=False)
    q = q_all[~q_all["all_lesions_excluded"]]

    tumor_q = q[q["tumor_present"] == 1]
    print(f"== Duct mask quality, tumor-positive cases (n={len(tumor_q)}) ==")
    print(pd.crosstab(tumor_q["cbd_status"], tumor_q["mpd_status"], margins=True,
                      rownames=["cbd"], colnames=["mpd"]))
    print(f"\nboth ducts proper: {tumor_q['both_proper'].sum()} / {len(tumor_q)}")
    print(f"tumor-positive with no proper duct at all: "
          f"{((tumor_q['cbd_status'] != 'proper') & (tumor_q['mpd_status'] != 'proper')).sum()}")
    dil = tumor_q[tumor_q["either_duct_dilated"] & ~tumor_q["both_proper"]]
    print(f"\neither_duct_dilated tumor cases with an improper mask on either duct: {len(dil)}")
    print(dil[["case_id", "cbd_status", "mpd_status"]].to_string(index=False))

    # --- flag sens/spec, all cases vs evaluable-only ---
    rows = []
    for flag in FLAGS:
        for scope, sub in (("all_cases", q), ("evaluable_only", q[evaluable_mask(q, flag)])):
            rows.append({"flag": flag, "scope": scope, **sens_spec(sub, flag)})
    ss = pd.DataFrame(rows)
    ss.to_csv(out_dir / "duct_flag_sens_spec.csv", index=False)
    print("\n== Dilation flag sens/spec vs tumor? ==")
    print(ss.round(3).to_string(index=False))

    # --- nnU-Net metrics on tumor-positive cases ---
    pc = pd.concat([pd.read_csv(f"results/nnunet_fold{i}/per_case_metrics.csv") for i in range(5)])
    pc = pc[(pc["gt_tumor"] == 1) & ~pc["case"].isin(fully_excluded)][["case", "dice", "missed"]]

    cases = pc.merge(q[["case_id"] + FLAGS], left_on="case", right_on="case_id")
    lesions = dice.merge(q[["case_id"] + FLAGS], left_on="case", right_on="case_id")

    groups = {"all_tumor_positive": (cases, lesions)}
    for flag in FLAGS:
        groups[f"{flag}=True"] = (cases[cases[flag]], lesions[lesions[flag]])
        groups[f"{flag}=False"] = (cases[~cases[flag]], lesions[~lesions[flag]])

    gm = pd.DataFrame({name: group_summary(c, l) for name, (c, l) in groups.items()}).T
    gm.index.name = "group"
    gm.to_csv(out_dir / "duct_dilated_group_metrics.csv")
    print("\n== nnU-Net metrics, tumor?==1 only ==")
    print(gm.round(3).to_string())

    # --- lesion dice by size bin ---
    rows = []
    for name, (_, l) in groups.items():
        for label in LABELS:
            s = l.loc[l["diam_bin"] == label, "lesion_dice"]
            rows.append({"group": name, "diam_bin": label, "n_lesions": len(s),
                         "dice_mean": s.mean(), "dice_median": s.median(),
                         "pct_dice0": (s == 0).mean() * 100 if len(s) else np.nan})
    by_size = pd.DataFrame(rows)
    by_size.to_csv(out_dir / "duct_dilated_dice_by_size.csv", index=False)

    view = by_size[by_size["group"].isin(["all_tumor_positive"] + [f"{f}=True" for f in FLAGS])]
    cell = view.apply(lambda r: f"{r['dice_mean']:.3f} (n={r['n_lesions']})" if r["n_lesions"] else "- (n=0)", axis=1)
    table = view.assign(cell=cell).pivot(index="diam_bin", columns="group", values="cell").reindex(LABELS)
    print("\n== Lesion dice (mean, n) by size bin ==")
    print(table.to_string())


if __name__ == "__main__":
    main()
