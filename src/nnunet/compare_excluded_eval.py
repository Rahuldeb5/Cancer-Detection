"""Baseline vs excluded-lesion nnU-Net validation metrics, per fold and pooled.

Reads results/nnunet_fold{0..4}/ (baseline evaluate.py output) and
results/nnunet_excl_outside_pancreas/fold{0..4}/ (same evaluate.py on the eval set built by
prepare_excluded_eval.py) and writes one tidy CSV:
  results/nnunet_excl_outside_pancreas/summary_vs_baseline.csv
    columns: section, metric, fold, baseline, excluded, delta
Per-lesion size-bin rows use the 6 bins from the lesion-location analysis
(<5, 5-10, 10-15, 15-20, 20-40, 40+ mm), pooled over folds.

Usage (from repo root): .venv/bin/python3 src/nnunet/compare_excluded_eval.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path("results")
EXCL = Path("results/nnunet_excl_outside_pancreas")
BINS = [0, 5, 10, 15, 20, 40, np.inf]
BIN_LABELS = ["<5mm", "5-10mm", "10-15mm", "15-20mm", "20-40mm", "40mm+"]


def load(folder: Path, name: str, folds=range(5)) -> pd.DataFrame:
    parts = []
    for k in folds:
        d = pd.read_csv(folder / f"fold{k}" / name) if folder != BASE else pd.read_csv(BASE / f"nnunet_fold{k}" / name)
        d["fold"] = k
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def case_metrics(pc: pd.DataFrame) -> dict:
    pos = pc[pc["gt_tumor"] == 1]
    out = {
        "n_cases": len(pc),
        "n_ref_positive": len(pos),
        "n_missed_entirely": int(pos["missed"].sum()),
        "dice_mean": pos["dice"].mean(),
        "dice_median": pos["dice"].median(),
        "hd95_mean_mm": pos["hd95"].mean(),
        "nsd_2mm_mean": pos["nsd_2.0mm"].mean(),
    }
    y = pc["gt_tumor"].to_numpy().astype(bool)
    for name, det in (("vol>0", pc["pred_vol_mm3"].to_numpy() > 0), ("maxprob>0.5", pc["max_tumor_prob"].to_numpy() > 0.5)):
        tp, fp = int((det & y).sum()), int((det & ~y).sum())
        tn, fn = int((~det & ~y).sum()), int((~det & y).sum())
        out[f"detect[{name}]_sensitivity"] = tp / (tp + fn) if tp + fn else np.nan
        out[f"detect[{name}]_specificity"] = tn / (tn + fp) if tn + fp else np.nan
        out[f"detect[{name}]_precision"] = tp / (tp + fp) if tp + fp else np.nan
        out[f"detect[{name}]_f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    if y.any() and (~y).any() and pc["max_tumor_prob"].notna().all():
        out["auroc_maxprob"] = roc_auc_score(y, pc["max_tumor_prob"])
    return out


def main() -> None:
    base_pc, excl_pc = load(BASE, "per_case_metrics.csv"), load(EXCL, "per_case_metrics.csv")
    base_pl, excl_pl = load(BASE, "per_lesion_metrics.csv"), load(EXCL, "per_lesion_metrics.csv")

    rows = []

    def add(section, metric, fold, b, e):
        rows.append({"section": section, "metric": metric, "fold": fold, "baseline": b, "excluded": e,
                     "delta": (e - b) if pd.notna(b) and pd.notna(e) else np.nan})

    for fold in [*range(5), "pooled"]:
        b = base_pc if fold == "pooled" else base_pc[base_pc["fold"] == fold]
        e = excl_pc if fold == "pooled" else excl_pc[excl_pc["fold"] == fold]
        bm, em = case_metrics(b), case_metrics(e)
        for k in bm:
            add("case_level", k, fold, bm[k], em.get(k, np.nan))

    for pl in (base_pl, excl_pl):
        pl["diam_bin"] = pd.cut(pl["diam_mm"], bins=BINS, labels=BIN_LABELS, right=False)
    for lab in BIN_LABELS:
        b, e = base_pl[base_pl["diam_bin"] == lab], excl_pl[excl_pl["diam_bin"] == lab]
        add("lesion_by_size", f"{lab}: n_lesions", "pooled", len(b), len(e))
        add("lesion_by_size", f"{lab}: mean_lesion_dice", "pooled", b["lesion_dice"].mean(), e["lesion_dice"].mean())
        add("lesion_by_size", f"{lab}: detected_any_overlap", "pooled", b["detected_any"].mean(), e["detected_any"].mean())
        add("lesion_by_size", f"{lab}: detected_iou>=0.1", "pooled", b["detected_iou10"].mean(), e["detected_iou10"].mean())
    add("lesion_all", "n_lesions", "pooled", len(base_pl), len(excl_pl))
    add("lesion_all", "mean_lesion_dice", "pooled", base_pl["lesion_dice"].mean(), excl_pl["lesion_dice"].mean())
    add("lesion_all", "detected_any_overlap", "pooled", base_pl["detected_any"].mean(), excl_pl["detected_any"].mean())

    out = pd.DataFrame(rows)
    out_path = EXCL / "summary_vs_baseline.csv"
    out.round(4).to_csv(out_path, index=False)

    # sanity: lesions that are kept should score (almost) identically to baseline
    # (case, gt_vox) is the join key (per_lesion_metrics has no lesion_id); drop keys that are
    # not unique on either side so two same-sized lesions in one case can't cross-match
    uniq = lambda d: d[~d.duplicated(["case", "gt_vox"], keep=False)]  # noqa: E731
    m = uniq(base_pl).merge(uniq(excl_pl), on=["case", "gt_vox"], suffixes=("_b", "_e"))
    changed = (m["lesion_dice_b"] - m["lesion_dice_e"]).abs() > 1e-6
    print(f"kept lesions with a unique (case, gt_vox) key matched baseline<->excluded: {len(m)} of {len(excl_pl)}; "
          f"lesion Dice changed for {int(changed.sum())} (expected ~0: only a predicted component "
          f"spanning a kept and an excluded lesion can shift one)")

    pd.set_option("display.width", 200)
    show = out[out["fold"].astype(str) == "pooled"].drop(columns="fold")
    print("\n=== pooled over 5 folds (baseline vs excluded) ===")
    print(show.round(4).to_string(index=False))
    print("\n=== per-fold case Dice (mean over ref-positive cases) ===")
    pf = out[(out["metric"] == "dice_mean") & (out["fold"].astype(str) != "pooled")]
    print(pf[["fold", "baseline", "excluded", "delta"]].round(4).to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
