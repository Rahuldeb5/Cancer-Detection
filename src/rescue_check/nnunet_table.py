"""S11 step 1: the only nnU-Net-dependent step. One prediction source -> one per-lesion table and one
per-case FP table in work/rescue_check/. analyze.py reads only these two files, so rerunning S11 on a
different model (e.g. Dataset502) is: produce its component tables, then run this with --pred_source.

A prediction source is a directory with
  components/components_t{t:g}.csv   26-connected components of p > t over the whole volume, one table per t
                                      in {0.5} + core.NN_THRESHOLDS; columns case_id, t, comp_id, peak_p,
                                      touch_ids (';'-joined lesion_ids the component overlaps; S2 schema)
  per_lesion.csv (optional)           case_id, lesion_id, gt_vox, peak_p_dil2 (peak p in lesion + 2 mm)
The default source `ds501_s2` points at S2's outputs instead (PC export ~/research/nnunet_probs/components/
and results/nnunet_subthreshold/per_lesion_subthreshold.csv) and adds regression asserts against S2.

Outputs (work/rescue_check/):
  nnunet_<tag>_lesions.csv   1235 rows: detected_t05, case_pred_t05, peak_p_dil2, and per t in NN_THRESHOLDS
                             hit_t, rank_t (pessimistic tie rank of the best touching component among the
                             case's components by peak_p), rank_min_t, comp_peak_p_t, n_comps_case_t
  nnunet_<tag>_cases.csv     1308 rows: n_comps_t, n_fp_t (touch rule) per t, n_fp_top3_t0.01

Usage (repo root, PC): PYTHONPATH=src .venv/bin/python -m rescue_check.nnunet_table [--pred_source DIR --tag NAME]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rescue_check import core as K
from tumorlib import io as tio

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work" / "rescue_check"
MASTER = REPO / "results/master_lesion_table/master_lesions.csv"
S2_COMPONENTS = Path.home() / "research/nnunet_probs/components"
S2_PER_LESION = REPO / "results/nnunet_subthreshold/per_lesion_subthreshold.csv"
S2_FROC = REPO / "results/nnunet_subthreshold/nnunet_froc_points.csv"
TS = (K.DETECT_T, *K.NN_THRESHOLDS)


def tcol(t: float) -> str:
    return f"t{t:g}"


def load_components(comp_dir: Path, t: float) -> pd.DataFrame:
    c = pd.read_csv(comp_dir / f"components_t{t:g}.csv", dtype={"touch_ids": str})
    assert np.allclose(c.t, t), f"{comp_dir}: wrong threshold in components_t{t:g}.csv"
    assert not c.duplicated(["case_id", "comp_id"]).any()
    return c[["case_id", "comp_id", "peak_p", "touch_ids"]]


def build(comp_dir: Path, per_lesion: Path | None, master: pd.DataFrame, cases: list[str]):
    les = master[["case_id", "lesion_id", "gt_vox"]].copy()
    out = les.copy()
    case_tab = pd.DataFrame({"case_id": cases})
    case_tab["is_negative"] = ~case_tab.case_id.isin(set(master.case_id))
    comps = {t: load_components(comp_dir, t) for t in TS}
    for t, c in comps.items():
        extra = set(c.case_id) - set(cases)
        assert not extra, f"components for cases outside the cohort: {sorted(extra)[:5]}"
        h = K.lesion_hits(c, les)
        if t == K.DETECT_T:
            out["detected_t05"] = h.hit.to_numpy()
            out["case_pred_t05"] = K.case_prediction_category(c, les).to_numpy()
        else:
            for col in ("hit", "rank", "rank_min", "comp_peak_p", "n_comps_case"):
                out[f"{col}_{tcol(t)}"] = h[col].to_numpy()
        case_tab[f"n_comps_{tcol(t)}"] = case_tab.case_id.map(c.groupby("case_id").size()).fillna(0).astype(int)
        case_tab[f"n_fp_{tcol(t)}"] = K.case_fp(c, cases).to_numpy()
    case_tab[f"n_fp_top{K.NN_OP_RANK}_{tcol(K.NN_OP_T)}"] = K.case_fp(comps[K.NN_OP_T], cases, K.NN_OP_RANK).to_numpy()
    # touch is monotone in t (a component at a higher t lies inside one at a lower t)
    assert (out.detected_t05 <= out[f"hit_{tcol(0.1)}"]).all()
    assert (out[f"hit_{tcol(0.1)}"] <= out[f"hit_{tcol(0.01)}"]).all() and \
           (out[f"hit_{tcol(0.01)}"] <= out[f"hit_{tcol(0.001)}"]).all(), "touch hits not monotone in t"
    if per_lesion is not None and per_lesion.exists():
        p = pd.read_csv(per_lesion)
        cols = ["case_id", "lesion_id", "gt_vox", "peak_p_dil2"] + (["max_t_hit_touch"] if "max_t_hit_touch" in p else [])
        out = out.merge(p[cols], on=["case_id", "lesion_id"], how="left", suffixes=("", "_pl"), validate="1:1")
        assert out.gt_vox_pl.notna().all() and (out.gt_vox == out.gt_vox_pl).all(), "per-lesion gt_vox mismatch"
        out = out.drop(columns="gt_vox_pl")
    else:
        out["peak_p_dil2"] = np.nan
    return out, case_tab, comps


def s2_regression(out: pd.DataFrame, case_tab: pd.DataFrame, master: pd.DataFrame) -> list[str]:
    """Default source only: every number must equal what S2 published."""
    msg = []
    m = master.merge(out, on=["case_id", "lesion_id", "gt_vox"], validate="1:1")
    assert len(m) == 1235 and (m.detected_t05 == m.detected_any).all(), "t=0.5 touch != master detected_any"
    msg.append("detected_t05 from components == master detected_any for 1235/1235 lesions")
    for t in K.NN_THRESHOLDS:
        exp = m.max_t_hit_touch.fillna(-1) >= t - 1e-12
        assert (m[f"hit_{tcol(t)}"] == exp).all(), f"hit at t={t} != S2 max_t_hit_touch"
    msg.append("hit_t (t = 0.1, 0.01, 0.001) == (S2 max_t_hit_touch >= t) for all 1235 lesions")
    fr = pd.read_csv(S2_FROC)
    fr = fr[(fr.hit_rule == "touch") & (fr.lesion_set == "all") & (fr.size_bin == "all")].set_index("t")
    neg = case_tab.is_negative
    for t in TS:
        fa, fn = case_tab[f"n_fp_{tcol(t)}"].mean(), case_tab.loc[neg, f"n_fp_{tcol(t)}"].mean()
        assert abs(fa - fr.at[t, "fp_per_case_all"]) < 1e-4 and abs(fn - fr.at[t, "fp_per_case_neg"]) < 1e-4, t
        msg.append(f"FP/case at t={t:g}: all {fa:.4f}, neg {fn:.4f} == S2 nnunet_froc_points.csv")
    return msg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_source", default="ds501_s2",
                    help="'ds501_s2' (default: S2 Dataset501 OOF outputs) or a directory as described above")
    ap.add_argument("--tag", default=None, help="output name; defaults to the source name / directory name")
    args = ap.parse_args()
    if args.pred_source == "ds501_s2":
        comp_dir, per_lesion, tag = S2_COMPONENTS, S2_PER_LESION, args.tag or "ds501_s2"
    else:
        d = Path(args.pred_source)
        comp_dir, per_lesion, tag = d / "components", d / "per_lesion.csv", args.tag or d.name

    master = pd.read_csv(MASTER)
    cases = sorted(c for k in range(1, 6) for c in tio.fold_ids(k))
    assert len(cases) == 1308 and len(master) == 1235
    out, case_tab, _ = build(comp_dir, per_lesion, master, cases)
    log = [f"pred_source {args.pred_source}: components {comp_dir}, per-lesion {per_lesion}"]
    if args.pred_source == "ds501_s2":
        log += s2_regression(out, case_tab, master)
    nties = {t: int((out[f"rank_{tcol(t)}"] != out[f"rank_min_{tcol(t)}"]).where(out[f"hit_{tcol(t)}"], False).sum())
             for t in K.NN_THRESHOLDS}
    log.append(f"lesions whose best rank is affected by a peak_p tie (pessimistic != optimistic): {nties}")
    WORK.mkdir(parents=True, exist_ok=True)
    out.to_csv(WORK / f"nnunet_{tag}_lesions.csv", index=False, float_format="%.6g")
    case_tab.to_csv(WORK / f"nnunet_{tag}_cases.csv", index=False)
    log.append(f"wrote {WORK / f'nnunet_{tag}_lesions.csv'} ({len(out)} rows), nnunet_{tag}_cases.csv ({len(case_tab)} rows)")
    (WORK / f"nnunet_{tag}_log.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
