"""Does duct geometry help rank nnU-Net's predicted components beyond nnU-Net's own
scores and plain pancreas location?

Reads results/component_features/components.csv (component_features.py). Trains a
component classifier (is_tp) with leave-one-nnU-Net-fold-out CV on three nested
feature sets:
    A  nnunet             -- nnU-Net's own confidence/size
    B  A + pancreas       -- overlap with / distance to pancreas, head/body/tail
    C  B + duct           -- distances to MPD/CBD, head end, dilated segment, cutoff; dilation flags
The duct question is C vs B (not C vs A: B alone removes out-of-pancreas FPs).

Reports, per feature set:
  - component AUROC / average precision
  - case-level operating points: keep components scoring >= t; at the t giving a
    target case sensitivity (tumor+ case keeps >=1 TP component), the case
    specificity (tumor- case keeps nothing) and FP components per case
  - top-1 localization: tumor+ cases whose highest-scored component is a TP
  - the same for tumor+ cases whose largest lesion is < 20 mm
  - paired case-bootstrap 95% CI for C - B
All anatomy here is GROUND TRUTH (pancreas, ducts), so this is an upper bound on
what predicted anatomy could give.

Usage (repo root on the server):
  ~/nnunet_setup/.venv/bin/python src/duct-cutoff/component_prior_eval.py
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

IN_DIR = Path("results/component_features")
CASE_METRICS_GLOB = "results/nnunet_fold[0-4]/per_case_metrics.csv"
# fraction of RECOVERABLE tumor+ cases (those with >=1 TP component) kept; absolute
# case sensitivity is capped by nnU-Net itself (~64% here), so absolute targets like 0.9 are unreachable
KEEP_TARGETS = (0.95, 0.90, 0.80)
N_BOOT = 1000

NNUNET = ["log_vol", "max_prob", "mean_prob", "p90_prob", "rank_in_case", "n_comp_in_case"]
PANCREAS = ["frac_in_pancreas", "dist_to_pancreas_mm", "frac_head", "frac_body", "frac_tail"]
DUCT = [
    "dist_to_mpd_mm", "dist_to_cbd_mm",
    "dist_to_mpd_head_end_mm", "dist_to_mpd_dilated_seg_mm", "dist_to_mpd_cutoff_mm",
    "dist_to_cbd_head_end_mm", "dist_to_cbd_dilated_seg_mm", "dist_to_cbd_cutoff_mm",
    "mpd_dilated", "cbd_dilated", "double_duct_dilated", "mpd_head_end_dilated", "cbd_head_end_dilated",
    "mpd_p95_mm", "cbd_p99_mm",
]
FEATURE_SETS = {"A_nnunet": NNUNET, "B_+pancreas": NNUNET + PANCREAS, "C_+duct": NNUNET + PANCREAS + DUCT}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_vol"] = np.log10(df.vol_mm3)
    for c in DUCT + PANCREAS:
        if c not in df:
            df[c] = np.nan
    for c in ("mpd_dilated", "cbd_dilated", "double_duct_dilated", "mpd_head_end_dilated", "cbd_head_end_dilated"):
        df[c] = df[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    return df


def oof_scores(df: pd.DataFrame, feats: list[str], seed: int = 0) -> np.ndarray:
    """Leave-one-fold-out; HistGB handles NaN (missing duct) natively."""
    out = np.full(len(df), np.nan)
    for f in sorted(df.fold.unique()):
        tr, te = df.fold != f, df.fold == f
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           min_samples_leaf=20, l2_regularization=1.0, random_state=seed)
        m.fit(df.loc[tr, feats], df.loc[tr, "is_tp"])
        out[te.to_numpy()] = m.predict_proba(df.loc[te, feats])[:, 1]
    return out


def case_tables(df: pd.DataFrame, score: str, cases: pd.DataFrame):
    """per-case: best TP score (tumor+) and best overall score (FP-relevant)."""
    g = df.groupby("case_id")
    best_tp = df[df.is_tp == 1].groupby("case_id")[score].max()
    best_any = g[score].max()
    top1_tp = df.loc[g[score].idxmax()].set_index("case_id").is_tp
    c = cases.copy()
    c["best_tp"] = c.case_id.map(best_tp)          # NaN = no TP component at all (unrecoverable)
    c["best_any"] = c.case_id.map(best_any)        # NaN = no component at all
    c["top1_tp"] = c.case_id.map(top1_tp)
    return c


def operating_points(c: pd.DataFrame, fp_scores: pd.DataFrame, score: str) -> dict:
    """For each KEEP_TARGET k: threshold t keeping fraction k of recoverable tumor+
    cases. Returns {k: (t, abs_case_sens, case_spec, fp_components_per_case)}."""
    pos, neg = c[c.gt_tumor == 1], c[c.gt_tumor == 0]
    rec = pos.best_tp.dropna().to_numpy()
    res = {}
    for k in KEEP_TARGETS:
        t = np.quantile(rec, 1 - k, method="lower")
        sens = float((pos.best_tp.fillna(-np.inf) >= t).mean())
        spec = float((neg.best_any.fillna(-np.inf) < t).mean())
        fp_per_case = float((fp_scores[score] >= t).sum() / len(c))
        res[k] = (t, sens, spec, fp_per_case)
    return res


def top1(c: pd.DataFrame, small: bool | None = None) -> float:
    p = c[(c.gt_tumor == 1) & c.best_any.notna()]
    if small is not None:
        p = p[p.small == small]
    return float(p.top1_tp.fillna(0).mean()), len(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, default=IN_DIR)
    ap.add_argument("--case_metrics_glob", default=CASE_METRICS_GLOB,
                    help="nnU-Net per_case_metrics.csv files (for gt_tumor + largest lesion size)")
    args = ap.parse_args()

    df = prepare(pd.read_csv(args.in_dir / "components.csv"))
    df = df[df.masks_ok == True].reset_index(drop=True)  # noqa: E712
    skipped = pd.read_csv(args.in_dir / "skipped.csv")

    cm = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(args.case_metrics_glob))]).rename(columns={"case": "case_id"})
    cases = cm[["case_id", "gt_tumor", "largest_lesion_diam_mm"]].copy()
    cases["small"] = cases.largest_lesion_diam_mm < 20
    print(f"components {len(df)} from {df.case_id.nunique()} cases; case list {len(cases)} "
          f"(tumor+ {int(cases.gt_tumor.sum())}, tumor- {int((cases.gt_tumor == 0).sum())}); "
          f"skipped {len(skipped)} (no prediction -> counted as misses / true negatives)")
    print(f"TP components {int(df.is_tp.sum())} / {len(df)}; tumor+ cases with >=1 TP component: "
          f"{df[df.is_tp == 1].case_id.nunique()} / {int(cases.gt_tumor.sum())}  (ceiling for any re-ranker)")

    for name, feats in FEATURE_SETS.items():
        df[name] = oof_scores(df, feats)
    df["nnunet_raw_max_prob"] = df.max_prob   # no learning: nnU-Net's own score
    scores = ["nnunet_raw_max_prob"] + list(FEATURE_SETS)

    print("\n=== component level (out-of-fold) ===")
    for s in scores:
        print(f"{s:22s} AUROC {roc_auc_score(df.is_tp, df[s]):.3f}   AP {average_precision_score(df.is_tp, df[s]):.3f}")

    print("\n=== case level: keep components with score >= t (t keeps X% of recoverable tumor+ cases) ===")
    print("baseline nnU-Net (keep all):  case sens "
          f"{cases[cases.gt_tumor == 1].case_id.isin(df[df.is_tp == 1].case_id).mean():.3f}  case spec "
          f"{1 - cases[cases.gt_tumor == 0].case_id.isin(df.case_id).mean():.3f}  "
          f"FP comps/case {(df.is_tp == 0).sum() / len(cases):.2f}")
    fp = df[df.is_tp == 0]
    tables = {}
    for s in scores:
        c = case_tables(df, s, cases)
        tables[s] = c
        ops = operating_points(c, fp, s)
        line = "  ".join(f"keep{int(k * 100)}%: sens {v[1]:.3f} spec {v[2]:.3f} FP/case {v[3]:.2f}"
                         for k, v in ops.items())
        print(f"{s:22s} {line}")

    print("\n=== top-1 localization (tumor+ cases with any component) ===")
    for s in scores:
        a, n = top1(tables[s]); sm, ns = top1(tables[s], True); lg, nl = top1(tables[s], False)
        print(f"{s:22s} all {a:.3f} (n={n})   <20mm {sm:.3f} (n={ns})   >=20mm {lg:.3f} (n={nl})")

    print("\n=== paired case bootstrap: C_+duct minus B_+pancreas ===")
    cb, cc = tables["B_+pancreas"], tables["C_+duct"]
    ids = cases.case_id.to_numpy()
    rng = np.random.default_rng(0)
    fpB, fpC = fp[["case_id", "B_+pancreas"]], fp[["case_id", "C_+duct"]]
    d_top1, d_spec90 = [], []
    for _ in range(N_BOOT):
        samp = rng.choice(ids, len(ids), replace=True)
        bB = cb.set_index("case_id").loc[samp].reset_index()
        bC = cc.set_index("case_id").loc[samp].reset_index()
        d_top1.append(top1(bC)[0] - top1(bB)[0])
        oB = operating_points(bB, fpB, "B_+pancreas")[0.90][2]
        oC = operating_points(bC, fpC, "C_+duct")[0.90][2]
        d_spec90.append(oC - oB)
    for name, arr in (("top-1 localization", d_top1), ("case spec @ keep 90%", d_spec90)):
        arr = np.asarray(arr)
        print(f"  {name:24s} delta {np.mean(arr):+.3f}  95% CI [{np.percentile(arr, 2.5):+.3f}, {np.percentile(arr, 97.5):+.3f}]")
    print("  (FP/case not bootstrapped; specificity uses the per-sample threshold)")

    out = args.in_dir / "oof_scores.csv"
    df[["case_id", "fold", "comp_id", "is_tp", "gt_tumor"] + scores].to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
