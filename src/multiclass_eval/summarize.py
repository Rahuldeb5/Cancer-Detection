"""S8 summary (PC): the per-case pickles from case_pass.py -> regression asserts -> deliverables.

  results/nnunet_multiclass/per_lesion_comparison.csv   1235 rows, 501 and 502 side by side + strata
  results/nnunet_multiclass/summary_vs_baseline.csv     long table: section, metric, lesion_set, stratum, t,
                                                         baseline (501), multiclass (502), delta, bootstrap CI
  work/nnunet_multiclass/summary.txt                    every check and table quoted in the README
  work/nnunet_multiclass/evaluation_{model}_{set}.json  evaluate.py-format aggregates, pooled over the 5 folds

Regression (asserted; 501 must reproduce what is already on record before any 502 number is trusted):
  - 1308 cases per model; saved seg == (p > 0.5) everywhere (501 also argmax == seg; 502 also the gland region)
  - 1235 lesions join master on (case_id, lesion_id) with equal gt_vox, from S2's tumorlib labels AND from
    evaluate.lesion_level's numbering; 501 lesion_dice / detected_any == master
  - per case TP / n_ref / n_pred == nnU-Net's own summary.json (501 region 1; 502 regions 2 and (1, 2))
  - 501 localization 631 / 288 / 62 and detect@any by bin 2.0 / 0.0 / 25.8 / 59.8 / 83.0 %
  - 501 evaluate.py aggregates == results/nnunet_fold{k}/evaluation.json (all), and
    == results/lesion_data/nnunet_excl_outside_pancreas/fold{k}/evaluation.json (excl_separated)
  - 501 threshold ladder == results/nnunet_subthreshold/nnunet_froc_points.csv, every row

Usage (PC, repo root, after rsyncing work/nnunet_multiclass/cases and summary_json from the server):
  PYTHONPATH=src .venv/bin/python -m multiclass_eval.summarize
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from multiclass_eval import stats as S
from multiclass_eval.core import EV, C, MODELS, REPO, localization

WORK = REPO / "work" / "nnunet_multiclass"
OUT = REPO / "results" / "nnunet_multiclass"
BINS = ("<5", "5-10", "10-20", "20-40", ">=40")
LSETS = ("all", "excl_separated")
NAME = {"501": "baseline", "502": "multiclass"}
LINES: list[str] = []
ROWS: list[dict] = []


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


def add(section, metric, lesion_set="all", stratum_var="", stratum="", t=np.nan, n=np.nan, n_cases=np.nan,
        baseline=np.nan, multiclass=np.nan, delta=None, ci_lo=np.nan, ci_hi=np.nan, p_boot=np.nan, note=""):
    if delta is None:
        delta = multiclass - baseline if np.isfinite(baseline) and np.isfinite(multiclass) else np.nan
    ROWS.append(dict(section=section, metric=metric, lesion_set=lesion_set, stratum_var=stratum_var, stratum=stratum,
                     t=t, n=n, n_cases=n_cases, baseline=baseline, multiclass=multiclass, delta=delta,
                     ci_lo=ci_lo, ci_hi=ci_hi, p_boot=p_boot, note=note))


# ------------------------------------------------------------------ load
def load(model: str, only: set[str] | None = None) -> dict:
    cases, les, comps, evc, evl, excl, gland = [], [], [], [], [], [], []
    for f in sorted((WORK / "cases" / model).glob("*.pkl")):
        if only is not None and f.stem not in only:
            continue
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        cid = r["case"]["case_id"]
        cases.append(r["case"])
        les.extend(r["lesions"])
        comps.extend(r["comps"])
        evc.append({"case": cid, "fold": r["case"]["fold"], **r["ev_case"]})
        evl.extend({"case_id": cid, **x} for x in r["ev_lesions"])
        e = r["excl"]
        excl.append({"case": cid, "fold": r["case"]["fold"], "action": e["action"],
                     **({f"x_{k}": v for k, v in e.items() if k not in ("action", "ev_case", "ev_lesions")})})
        if e["action"] == "modified":
            excl[-1]["ev_case"] = {"case": cid, "fold": r["case"]["fold"], **e["ev_case"]}
            excl[-1]["ev_lesions"] = [{"case": cid, **x} for x in e["ev_lesions"]]
        if "gland" in r:
            gland.append({"case_id": cid, "fold": r["case"]["fold"], **r["gland"]})
    return dict(cases=pd.DataFrame(cases), les=pd.DataFrame(les), comps=pd.DataFrame(comps),
                evc=pd.DataFrame(evc), evl=pd.DataFrame(evl), excl=excl, gland=pd.DataFrame(gland))


# ------------------------------------------------------------------ evaluate.py aggregation (its main(), verbatim logic)
def ev_aggregate(per_case: list[dict], lesion_rows: list[dict]) -> dict:
    pos = [r for r in per_case if r["gt_tumor"]]
    seg = {"n_ref_positive": len(pos), "n_missed_entirely": sum(r["missed"] for r in pos),
           "dice": EV._stats([r["dice"] for r in pos]), "iou": EV._stats([r["iou"] for r in pos]),
           "hd95": EV._stats([r["hd95"] for r in pos]), "assd": EV._stats([r["assd"] for r in pos]),
           **{f"nsd_{t}mm": EV._stats([r[f"nsd_{t}mm"] for r in pos]) for t in EV.NSD_TOL_MM},
           "rel_vol_diff": EV._stats([r["rel_vol_diff"] for r in pos])}
    gt_pos = np.array([bool(r["gt_tumor"]) for r in per_case])
    vol = np.array([r["pred_vol_mm3"] for r in per_case])
    pr = np.array([r["max_tumor_prob"] for r in per_case], dtype=float)
    det = {"n_cases": len(per_case), "n_positive": int(gt_pos.sum()), "by_volume_threshold": {}}
    for thr in EV.VOL_THRESH_MM3:
        det["by_volume_threshold"][f">{thr:g}mm3"] = EV.detection_block(gt_pos, vol, vol > thr)
    det["score=pred_volume"] = EV.detection_block(gt_pos, vol, vol > 0)
    det["score=max_prob"] = EV.detection_block(gt_pos, pr, pr > 0.5)
    by_size = {}
    for bi, bl in enumerate(EV.bin_labels()):
        grp = [x for x in lesion_rows if x["bin"] == bi]
        if not grp:
            by_size[bl] = {"n_lesions": 0}
            continue
        d = np.array([x["lesion_dice"] for x in grp])
        by_size[bl] = {
            "n_lesions": len(grp), "diam_mm_median": float(np.median([x["diam_mm"] for x in grp])),
            "lesion_dice_mean": float(d.mean()), "lesion_dice_median": float(np.median(d)),
            "lesion_dice_mean_detected_only": float(np.mean([x["lesion_dice"] for x in grp if x["detected_any"]])
                                                    if any(x["detected_any"] for x in grp) else 0.0),
            "detection_rate_any_overlap": float(np.mean([x["detected_any"] for x in grp])),
            "detection_rate_iou>=0.1": float(np.mean([x["detected_iou10"] for x in grp]))}
    by_scan = {}
    for bi, bl in enumerate(EV.bin_labels()):
        cids = {x["case"] for x in lesion_rows if EV.bin_index(x["scan_largest_diam_mm"]) == bi}
        dv = [r["dice"] for r in pos if r["case"] in cids and not np.isnan(r["dice"])]
        by_scan[bl] = {"n_scans": len(cids), **EV._stats(dv)} if cids else {"n_scans": 0}
    return {"segmentation_tumor": seg, "detection_tumor": det,
            "size_stratified": {"total_gt_lesions": len(lesion_rows), "per_lesion_by_size": by_size,
                                "per_scan_dice_by_largest_lesion_size": by_scan}}


def ev_inputs(D: dict, lset: str, folds=None) -> tuple[list[dict], list[dict]]:
    """Per-case rows + lesion rows in evaluate.py's shape for one lesion set (excl: prepare_excluded_eval rule)."""
    evc = D["evc"].set_index("case")
    les_by_case = {c: g.drop(columns="case_id").assign(case=c).to_dict("records") for c, g in D["evl"].groupby("case_id")}
    pc, lr = [], []
    for e in D["excl"]:
        if folds is not None and e["fold"] not in folds:
            continue
        c = e["case"]
        if lset == "excl_separated" and e["action"] == "dropped":
            continue
        if lset == "excl_separated" and e["action"] == "modified":
            pc.append(e["ev_case"])
            lr.extend(e["ev_lesions"])
        else:
            pc.append({"case": c, **evc.loc[c].to_dict()})
            lr.extend(les_by_case.get(c, []))
    return pc, lr


def compare_json(mine, ref, path="", rtol=1e-6, bad=None) -> list[str]:
    """Recursive numeric comparison; returns mismatching paths."""
    bad = [] if bad is None else bad
    if isinstance(ref, dict):
        for k, v in ref.items():
            if k in ("bin_edges_mm", "size_metric"):
                continue
            if k not in mine:
                bad.append(f"{path}/{k} missing")
                continue
            compare_json(mine[k], v, f"{path}/{k}", rtol, bad)
    elif isinstance(ref, (int, float)) and ref is not None:
        m = mine
        if m is None or not np.isclose(float(m), float(ref), rtol=rtol, atol=1e-9, equal_nan=True):
            bad.append(f"{path}: {m} vs {ref}")
    return bad


# ------------------------------------------------------------------ ladder (S2 summarize.py step 3, same logic)
def split_ids(s: pd.Series) -> list[set[int]]:
    return [set(map(int, x.split(";"))) if isinstance(x, str) and x else set() for x in s]


def ladder(D: dict, per_lesion: pd.DataFrame, neg_ids: set[str]) -> pd.DataFrame:
    comps = D["comps"].copy()
    comps["touch_set"] = split_ids(comps.touch_ids)
    comps["peak_set"] = split_ids(comps.peakhit_ids)
    n_all, n_neg = len(D["cases"]), len(neg_ids)
    rows = []
    for t in C.THRESHOLDS:
        ct = comps[comps.t == t]
        for rule, col in (("peak_in_dil2mm", "peak_set"), ("touch", "touch_set")):
            fp = ct[[len(s) == 0 for s in ct[col]]]
            fp_all, fp_neg = len(fp), int(fp.case_id.isin(neg_ids).sum())
            hk = {(c, j) for c, s in zip(ct.case_id, ct[col]) for j in s}
            for lset in LSETS:
                sub = per_lesion if lset == "all" else per_lesion[~per_lesion.excluded]
                for b in ("all", *BINS):
                    s = sub if b == "all" else sub[sub.diam_bin == b]
                    nh = sum((c, l) in hk for c, l in zip(s.case_id, s.lesion_id))
                    rows.append(dict(t=t, hit_rule=rule, lesion_set=lset, size_bin=b, n_lesions=len(s), n_hit=nh,
                                     sensitivity=nh / len(s) if len(s) else np.nan, n_components=len(ct),
                                     n_fp_components=fp_all, fp_per_case_all=fp_all / n_all,
                                     fp_per_case_neg=fp_neg / n_neg, n_cases_all=n_all, n_cases_neg=n_neg))
    return pd.DataFrame(rows)


def max_t_hit(D: dict, per_lesion: pd.DataFrame, rule: str) -> list[float]:
    comps = D["comps"]
    col = "touch_ids" if rule == "touch" else "peakhit_ids"
    hits = {}
    for t, c, s in zip(comps.t, comps.case_id, split_ids(comps[col])):
        for j in s:
            hits[(c, j)] = max(hits.get((c, j), -1.0), t)
    return [hits.get((c, l), np.nan) for c, l in zip(per_lesion.case_id, per_lesion.lesion_id)]


# ------------------------------------------------------------------ main
def main() -> None:
    master = pd.read_csv(REPO / "results/master_lesion_table/master_lesions.csv")
    fold_of = C.fold_of_case()
    D = {m: load(m) for m in MODELS}
    if os.environ.get("S8_PARTIAL"):  # smoke test on an unfinished pass (run with python -O); never for deliverables
        global WORK, OUT
        have = set(D["501"]["cases"].case_id) & set(D["502"]["cases"].case_id)
        D = {m: load(m, have) for m in MODELS}
        master = master[master.case_id.isin(have)]
        fold_of = {c: f for c, f in fold_of.items() if c in have}
        OUT = WORK / "partial_test"

    say("== Integrity")
    for m in MODELS:
        cs = D[m]["cases"]
        assert len(cs) == 1308 and set(cs.case_id) == set(fold_of), f"{m}: {len(cs)} cases"
        assert (cs.fold == cs.case_id.map(fold_of)).all()
        assert cs.argmax_mismatch.sum() == 0 and cs.p05_vs_seg_mismatch.sum() == 0, m
        labs = set(";".join(cs.seg_labels).split(";"))
        extra = f", gland region mismatch {int(cs.gland_mismatch.sum())}" if m == "502" else ""
        if m == "502":
            assert cs.gland_mismatch.sum() == 0
        say(f"{m}: 1308 cases, OOF fold == fold_k_ids; label values {sorted(labs)}; seg vs p>0.5 mismatch 0"
            f"{' (and argmax == seg)' if m == '501' else ''}{extra}; npz dtype {cs.npz_dtype.unique().tolist()}; "
            f"seg/gt affines close in {int(cs.seg_gt_affine_close.sum())}/1308")
        ev = D[m]["evl"][["case_id", "lesion_id", "gt_vox", "lesion_dice", "detected_any"]].rename(
            columns={"gt_vox": "gt_vox_ev", "lesion_dice": "lesion_dice_ev", "detected_any": "detected_any_ev"})
        les = D[m]["les"][["case_id", "lesion_id", "gt_vox", "seg_touches"]].rename(columns={"gt_vox": "gt_vox_s2"})
        les = les.merge(ev, on=["case_id", "lesion_id"], how="outer", validate="1:1")
        j = master.merge(les, on=["case_id", "lesion_id"], how="outer", indicator=True)
        assert (j._merge == "both").all() and len(j) == 1235, j._merge.value_counts()
        assert (j.gt_vox == j.gt_vox_s2).all() and (j.gt_vox == j.gt_vox_ev).all()
        assert (j.seg_touches == j.detected_any_ev.astype(bool)).all()
        say(f"{m}: 1235/1235 lesions join master; gt_vox equal (S2 labels and evaluate.lesion_level numbering); "
            f"S2 touch == evaluate detected_any")
        if m == "501":
            assert np.allclose(j.lesion_dice_ev, j.lesion_dice, atol=1e-6) and (j.detected_any_ev.astype(bool) == j.detected_any).all()
            say("501: lesion_dice and detected_any == master for all 1235 (max |diff| "
                f"{np.abs(j.lesion_dice_ev - j.lesion_dice).max():.1e})")

    # nnU-Net summary.json per-case counts
    sj = WORK / "summary_json"
    for m, regions in (("501", ("1",)), ("502", ("2", "(1, 2)"))):
        evc = D[m]["evc"].set_index("case")
        gl = D[m]["gland"].set_index("case_id") if m == "502" else None
        nmis, means, nchk = 0, [], 0
        for f in range(5):
            js = json.loads((sj / f"{m}_fold{f}.json").read_text())
            for rc in js["metric_per_case"]:
                cid = Path(rc["prediction_file"]).name[:-7]
                if cid not in evc.index:  # only in S8_PARTIAL; the nchk assert below catches it otherwise
                    continue
                nchk += 1
                for reg in regions:
                    ref = rc["metrics"][reg]
                    src = (evc.loc[cid, "nn_tp"], evc.loc[cid, "nn_n_ref"], evc.loc[cid, "nn_n_pred"]) if reg in ("1", "2") \
                        else (gl.loc[cid, "tp"], gl.loc[cid, "n_ref"], gl.loc[cid, "n_pred"])
                    nmis += int((ref["TP"], ref["n_ref"], ref["n_pred"]) != tuple(int(x) for x in src))
            for reg in regions:
                col = evc[evc.fold == f]["nn_dice"] if reg in ("1", "2") else gl[gl.fold == f]["dice"]
                means.append((f, reg, float(np.nanmean(col)), js["mean"][reg]["Dice"]))
        assert nmis == 0 and nchk == 1308, f"{m}: {nmis} summary.json count mismatches, {nchk} cases checked"
        assert all(np.isclose(a, b, rtol=1e-9) for *_, a, b in means), means
        say(f"{m}: per-case TP / n_ref / n_pred == nnU-Net summary.json for all 1308 cases, regions {regions}; "
            f"per-fold mean Dice reproduced: " + ", ".join(f"f{f} {reg} {a:.4f}" for f, reg, a, _ in means))

    # ------------------------------------------------------------ per-lesion table
    neg_ids = set(D["501"]["cases"].query("gt_tumor == 0").case_id)
    assert len(neg_ids) == 327
    att = pd.read_csv(REPO / "results/attenuation_v3/attenuation_labels_v3.csv")
    ph = pd.read_csv(REPO / "results/phase_estimate/phase_estimate.csv")[["case_id", "est_phase"]]
    pl = master.merge(att[["case_id", "lesion_id", "gt_vox", "attenuation_v3"]], on=["case_id", "lesion_id"],
                      how="left", suffixes=("", "_att"), validate="1:1")
    assert (pl.gt_vox == pl.gt_vox_att).all()
    pl = pl.drop(columns="gt_vox_att").merge(ph, on="case_id", how="left", validate="m:1")
    pl["thickness_bin"] = pd.cut(pl.slice_thickness_mm, [0, 2.5, 5.0, np.inf], right=False,
                                 labels=["<2.5", "2.5-5", ">=5"]).astype(str)
    pl["attenuation_v3"] = pl.attenuation_v3.fillna("unknown")
    pl["region"] = pl.region.fillna("none")
    pl["ct_phase"] = pl.ct_phase.fillna("missing")
    pl["est_phase"] = pl.est_phase.fillna("missing")
    for m in MODELS:
        e = D[m]["evl"].set_index(["case_id", "lesion_id"])
        s2 = D[m]["les"].set_index(["case_id", "lesion_id"])
        k = list(zip(pl.case_id, pl.lesion_id))
        tag = NAME[m]
        pl[f"dice_{tag}"] = e.loc[k, "lesion_dice"].to_numpy()
        pl[f"iou_{tag}"] = e.loc[k, "lesion_iou"].to_numpy()
        pl[f"det_{tag}"] = e.loc[k, "detected_any"].to_numpy().astype(int)
        pl[f"det_iou10_{tag}"] = e.loc[k, "detected_iou10"].to_numpy().astype(int)
        pl[f"peak_p_dil2_{tag}"] = s2.loc[k, "peak_p_dil2"].to_numpy()
        pl[f"max_t_hit_touch_{tag}"] = max_t_hit(D[m], pl, "touch")
        pl[f"max_t_hit_peak_{tag}"] = max_t_hit(D[m], pl, "peak")
    pl["dice_delta"] = pl.dice_multiclass - pl.dice_baseline
    pl["change"] = np.select([(pl.det_baseline == 1) & (pl.det_multiclass == 1), (pl.det_baseline == 0) & (pl.det_multiclass == 1),
                              (pl.det_baseline == 1) & (pl.det_multiclass == 0)], ["both", "recovered", "lost"], "neither")
    # S2 regression for the baseline per-lesion ladder columns
    s2pl = pd.read_csv(REPO / "results/nnunet_subthreshold/per_lesion_subthreshold.csv")
    s2j = pl.merge(s2pl[["case_id", "lesion_id", "peak_p_dil2", "max_t_hit_touch", "max_t_hit_peak"]], on=["case_id", "lesion_id"])
    assert np.allclose(s2j.peak_p_dil2_baseline, s2j.peak_p_dil2, rtol=1e-5)
    assert np.allclose(s2j.max_t_hit_touch_baseline, s2j.max_t_hit_touch, equal_nan=True)
    assert np.allclose(s2j.max_t_hit_peak_baseline, s2j.max_t_hit_peak, equal_nan=True)
    say("501: peak_p_dil2, max_t_hit_touch, max_t_hit_peak == S2 per_lesion_subthreshold.csv for all 1235")

    # ------------------------------------------------------------ step 2: evaluate.py metrics
    say("\n== Step 2: evaluate.py metrics on the lesion mask (OOF, pooled over the 5 folds)")
    agg = {}
    for m in MODELS:
        for lset in LSETS:
            pc, lr = ev_inputs(D[m], lset)
            agg[(m, lset)] = ev_aggregate(pc, lr)
            (WORK / f"evaluation_{m}_{lset}.json").write_text(json.dumps(agg[(m, lset)], indent=2, default=float))
    # regression: per fold vs the files on record
    for lset, refdir in (("all", "results/nnunet_fold{f}"), ("excl_separated", "results/lesion_data/nnunet_excl_outside_pancreas/fold{f}")):
        bad = []
        for f in range(5):
            pc, lr = ev_inputs(D["501"], lset, folds={f})
            ref = json.loads((REPO / refdir.format(f=f) / "evaluation.json").read_text())
            mine = ev_aggregate(pc, lr)
            bad += [f"fold{f}{b}" for b in compare_json(mine, {k: ref[k] for k in ("segmentation_tumor", "detection_tumor", "size_stratified")}, rtol=1e-5)]
        say(f"501 {lset}: evaluate.py aggregates vs {refdir.format(f='{0..4}')}/evaluation.json: "
            f"{'ALL EQUAL (rtol 1e-5)' if not bad else f'{len(bad)} mismatches'}")
        for b in bad[:20]:
            say(f"   {b}")
        # Known, explained exception: NSD at exactly 3 mm on 1.5 mm isotropic grids (PanTS_00007130, PanTS_00001751).
        # A 2-voxel surface distance is exactly 3.000 mm, on the tolerance; evaluate.py's float32 header spacing and
        # tumorlib's affine spacing (1.5 +- 1e-7) put it on opposite sides of <= 3. Both S8 models share the spacing.
        other = [b for b in bad if "nsd_3.0mm" not in b]
        assert not other, f"501 {lset} does not reproduce {refdir}: {other[:5]}"
        if bad:
            say("   (only NSD@3mm differs: 1.5 mm isotropic grids, a 2-voxel distance sits exactly on the 3 mm tolerance; "
                "float32 header vs affine spacing; per case PanTS_00007130 0.690 vs 0.713, PanTS_00001751 0.293 vs 0.299)")

    def segrow(a, key, stat="mean"):
        return a["segmentation_tumor"][key][stat]

    for lset in LSETS:
        a, b = agg[("501", lset)], agg[("502", lset)]
        say(f"-- lesion set {lset}: ref-positive cases {a['segmentation_tumor']['n_ref_positive']}, "
            f"cases {a['detection_tumor']['n_cases']}")
        say(f"{'metric':<34}{'501':>10}{'502':>10}{'delta':>10}")
        for key, stat in (("dice", "mean"), ("dice", "median"), ("iou", "mean"), ("hd95", "mean"), ("hd95", "median"),
                          ("assd", "mean"), ("nsd_2.0mm", "mean"), ("rel_vol_diff", "median")):
            x, y = segrow(a, key, stat), segrow(b, key, stat)
            say(f"{key + ' ' + stat:<34}{x:>10.4f}{y:>10.4f}{y - x:>+10.4f}")
            add("case_seg", f"{key}_{stat}", lset, n=segrow(a, key, "n"), baseline=x, multiclass=y,
                note=f"502 n={segrow(b, key, 'n')}" if segrow(a, key, "n") != segrow(b, key, "n") else "")
        x, y = a["segmentation_tumor"]["n_missed_entirely"], b["segmentation_tumor"]["n_missed_entirely"]
        say(f"{'ref-positive with empty prediction':<34}{x:>10}{y:>10}{y - x:>+10}")
        add("case_seg", "n_missed_entirely", lset, baseline=x, multiclass=y)
        for sc in ("score=pred_volume", "score=max_prob"):
            for k in ("sensitivity", "specificity", "precision_ppv", "npv", "balanced_accuracy", "auroc", "auprc"):
                x, y = a["detection_tumor"][sc][k], b["detection_tumor"][sc][k]
                say(f"{sc + ' ' + k:<34}{x:>10.4f}{y:>10.4f}{y - x:>+10.4f}")
                add("case_detect_any_voxel", f"{sc}_{k}", lset, n=a["detection_tumor"]["n_cases"], baseline=x, multiclass=y)
        for thr in ("50", "100"):
            for k in ("sensitivity", "specificity"):
                x, y = (a["detection_tumor"]["by_volume_threshold"][f">{thr}mm3"][k], b["detection_tumor"]["by_volume_threshold"][f">{thr}mm3"][k])
                add("case_detect_any_voxel", f"vol>{thr}mm3_{k}", lset, baseline=x, multiclass=y)
        say(f"{'bin':>7} {'n':>5} | {'Dice 501':>8} {'502':>7} | {'Dice|det 501':>12} {'502':>7} | {'det@any 501':>11} {'502':>7} | {'det@IoU.1 501':>13} {'502':>7}")
        for bl in EV.bin_labels():
            x, y = a["size_stratified"]["per_lesion_by_size"][bl], b["size_stratified"]["per_lesion_by_size"][bl]
            say(f"{bl:>7} {x['n_lesions']:>5} | {x['lesion_dice_mean']:>8.3f} {y['lesion_dice_mean']:>7.3f} | "
                f"{x['lesion_dice_mean_detected_only']:>12.3f} {y['lesion_dice_mean_detected_only']:>7.3f} | "
                f"{x['detection_rate_any_overlap']:>11.1%} {y['detection_rate_any_overlap']:>7.1%} | "
                f"{x['detection_rate_iou>=0.1']:>13.1%} {y['detection_rate_iou>=0.1']:>7.1%}")
            for k in ("lesion_dice_mean", "lesion_dice_median", "lesion_dice_mean_detected_only",
                      "detection_rate_any_overlap", "detection_rate_iou>=0.1"):
                add("lesion_by_size_evaluate", k, lset, "diam_bin_evaluate", bl, n=x["n_lesions"], baseline=x[k], multiclass=y[k])
        say("per-scan Dice by the scan's largest lesion: " + "; ".join(
            f"{bl} n={a['size_stratified']['per_scan_dice_by_largest_lesion_size'][bl]['n_scans']} "
            f"{a['size_stratified']['per_scan_dice_by_largest_lesion_size'][bl].get('mean') or float('nan'):.3f} -> "
            f"{b['size_stratified']['per_scan_dice_by_largest_lesion_size'][bl].get('mean') or float('nan'):.3f}"
            for bl in EV.bin_labels()))

    # regression line for the README: 501 detect@any by master bin
    say("\n501 detect@any by master diam_bin: " + ", ".join(
        f"{b} {pl[pl.diam_bin == b].det_baseline.mean():.1%} (n={int((pl.diam_bin == b).sum())})" for b in BINS)
        + "   [carried over 2.0 / 0.0 / 25.8 / 59.8 / 83.0%]")
    exp = {"<5": 1 / 49, "5-10": 0.0, "10-20": 71 / 275, "20-40": 357 / 597, ">=40": 230 / 277}
    for b in BINS:
        assert round(pl[pl.diam_bin == b].det_baseline.mean(), 3) == round(exp[b], 3), b

    # ------------------------------------------------------------ pancreas channel
    say("\n== Pancreas region (1,2) Dice, 502 only (GT = 502 label > 0 = union of the 4 gland masks + lesion)")
    gl = D["502"]["gland"].merge(D["502"]["cases"][["case_id", "gt_tumor"]], on="case_id")
    ne = gl[gl.n_ref > 0]
    say(f"cases with a non-empty gland GT: {len(ne)}; per-case Dice mean {ne.dice.mean():.4f}, median {ne.dice.median():.4f}, "
        f"p10 {ne.dice.quantile(.1):.4f}; tumor+ {ne[ne.gt_tumor == 1].dice.mean():.4f} (n={int((ne.gt_tumor == 1).sum())}), "
        f"tumor- {ne[ne.gt_tumor == 0].dice.mean():.4f} (n={int((ne.gt_tumor == 0).sum())}); pooled voxel Dice "
        f"{2 * ne.tp.sum() / (ne.n_ref.sum() + ne.n_pred.sum()):.4f}")
    em = gl[gl.n_ref == 0]
    say(f"empty-envelope cases: {len(em)}; with a predicted gland {int((em.n_pred > 0).sum())}; predicted voxels median {em.n_pred.median():.0f}")
    lowd = ne.nsmallest(8, "dice")[["case_id", "gt_tumor", "dice", "n_ref", "n_pred"]]
    say("lowest gland Dice:\n" + lowd.to_string(index=False))
    add("pancreas", "gland_dice_mean_nonempty_gt", n=len(ne), multiclass=ne.dice.mean(), note="502 only; region (1,2) vs 502 GT label>0")
    add("pancreas", "gland_dice_median_nonempty_gt", n=len(ne), multiclass=ne.dice.median(), note="502 only")
    add("pancreas", "gland_dice_mean_tumor_pos", n=int((ne.gt_tumor == 1).sum()), multiclass=ne[ne.gt_tumor == 1].dice.mean(), note="502 only")
    add("pancreas", "gland_dice_mean_tumor_neg", n=int((ne.gt_tumor == 0).sum()), multiclass=ne[ne.gt_tumor == 0].dice.mean(), note="502 only")
    add("pancreas", "empty_gt_cases_with_pred", n=len(em), multiclass=int((em.n_pred > 0).sum()), note="502 only")
    # does the gland prediction track lesion success? (descriptive)
    lg = pl.groupby("case_id").agg(any_det=("det_multiclass", "max")).reset_index().merge(gl[["case_id", "dice"]], on="case_id")
    say(f"tumor+ gland Dice when 502 detects >=1 lesion {lg[lg.any_det == 1].dice.mean():.4f} (n={int((lg.any_det == 1).sum())}) "
        f"vs none {lg[lg.any_det == 0].dice.mean():.4f} (n={int((lg.any_det == 0).sum())})")

    # ------------------------------------------------------------ step 3: localization
    say("\n== Step 3: localization-aware detection (saved segmentation = ladder t 0.5)")
    big = master.copy()
    for lset in LSETS:
        if lset == "excl_separated":
            big = master[~master.excluded].copy()
        largest = big.sort_values(["gt_vox", "diam_mm"], ascending=False).drop_duplicates("case_id")[["case_id", "diam_bin"]]
        largest["lbin"] = largest.diam_bin.replace({"<5": "<10", "5-10": "<10"})
        loc = {}
        for m in MODELS:
            cs = D[m]["cases"].set_index("case_id")
            ex = {e["case"]: e for e in D[m]["excl"]}
            rows = []
            for c in largest.case_id:
                e = ex[c]
                if lset == "excl_separated" and e["action"] == "modified":
                    rows.append((c, localization(e["x_seg_any"], e["x_seg_touches_gt"])))
                else:
                    rows.append((c, localization(cs.at[c, "seg_any"], cs.at[c, "seg_touches_gt"])))
            loc[m] = pd.DataFrame(rows, columns=["case_id", "cat"]).merge(largest, on="case_id")
        a, b = loc["501"], loc["502"]
        n = len(a)
        say(f"-- lesion set {lset}: tumor+ cases {n}")
        for cat in ("localized", "wrong_place", "nothing"):
            x, y = int((a.cat == cat).sum()), int((b.cat == cat).sum())
            say(f"   {cat:<12} 501 {x:>4} ({x / n:.1%})   502 {y:>4} ({y / n:.1%})")
            add("localization", f"cases_{cat}", lset, n_cases=n, baseline=x, multiclass=y)
        if lset == "all":
            assert tuple(int((a.cat == c).sum()) for c in ("localized", "wrong_place", "nothing")) == (631, 288, 62)
            say("   501 reproduces 631 / 288 / 62 (carried over)")
        ab = a[["case_id", "cat", "lbin"]].merge(b[["case_id", "cat"]], on="case_id", suffixes=("_501", "_502"))
        ab["l501"], ab["l502"] = (ab.cat_501 == "localized").astype(float), (ab.cat_502 == "localized").astype(float)
        W = S.case_weights(n)
        r = S.paired_ratio(ab, "case_id", "l501", "l502", ab.case_id.to_numpy(), W)
        add("localization", "localized_rate", lset, "largest_lesion_bin", "all", n_cases=n, baseline=r["a"], multiclass=r["b"],
            ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p_boot=r["p_boot"])
        say(f"   localized rate {r['a']:.1%} -> {r['b']:.1%}, delta {r['delta']:+.1%} [{r['ci_lo']:+.1%}, {r['ci_hi']:+.1%}]")
        trans = pd.crosstab(ab.cat_501, ab.cat_502)
        say("   transitions (rows 501, cols 502):\n" + "\n".join("     " + s for s in trans.to_string().splitlines()))
        for lb in ("<10", "10-20", "20-40", ">=40"):
            g = ab[ab.lbin == lb]
            r = S.paired_ratio(g, "case_id", "l501", "l502", ab.case_id.to_numpy(), W)
            say(f"   largest lesion {lb:>6}: n={len(g):>4}  {int(g.l501.sum()):>4} ({r['a']:.1%}) -> {int(g.l502.sum()):>4} ({r['b']:.1%})  "
                f"delta {r['delta']:+.1%} [{r['ci_lo']:+.1%}, {r['ci_hi']:+.1%}]")
            add("localization", "localized_rate", lset, "largest_lesion_bin", lb, n_cases=len(g), baseline=r["a"], multiclass=r["b"],
                ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p_boot=r["p_boot"])

    # FP components (t = 0.5, touch rule: a component touching no GT lesion; touching an excluded one is not FP)
    say("-- FP lesion components at the saved segmentation (26-connected, touch rule)")
    fp = {}
    for m in MODELS:
        c5 = D[m]["comps"][D[m]["comps"].t == 0.5]
        per = pd.DataFrame({"case_id": sorted(fold_of)})
        per["n_comp"] = per.case_id.map(c5.groupby("case_id").size()).fillna(0)
        per["n_fp"] = per.case_id.map(c5[c5.touch_ids.isna() | (c5.touch_ids == "")].groupby("case_id").size()).fillna(0)
        per["neg"] = per.case_id.isin(neg_ids)
        fp[m] = per
    j = fp["501"].merge(fp["502"], on=["case_id", "neg"], suffixes=("_501", "_502"))
    for grp, sub in (("negatives", j[j.neg]), ("tumor+", j[~j.neg]), ("all cases", j)):
        Wg = S.case_weights(len(sub))
        r = S.paired_ratio(sub, "case_id", "n_fp_501", "n_fp_502", sub.case_id.to_numpy(), Wg)
        sub = sub.assign(a1=(sub.n_fp_501 > 0).astype(float), a2=(sub.n_fp_502 > 0).astype(float))
        r2 = S.paired_ratio(sub, "case_id", "a1", "a2", sub.case_id.to_numpy(), Wg)
        say(f"   {grp:<10} n={len(sub)}: FP comps/case {r['a']:.3f} -> {r['b']:.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]; "
            f"cases with >=1 FP comp {r2['a']:.1%} -> {r2['b']:.1%}")
        add("false_positives", "fp_components_per_case", "all", "case_group", grp, n_cases=len(sub), baseline=r["a"], multiclass=r["b"],
            ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p_boot=r["p_boot"])
        add("false_positives", "frac_cases_with_fp_component", "all", "case_group", grp, n_cases=len(sub), baseline=r2["a"],
            multiclass=r2["b"], ci_lo=r2["ci_lo"], ci_hi=r2["ci_hi"], p_boot=r2["p_boot"])

    # ------------------------------------------------------------ step 4: ladder
    say("\n== Step 4: threshold ladder on the lesion probability (whole-volume components of p > t)")
    lad = {m: ladder(D[m], pl, neg_ids) for m in MODELS}
    s2 = pd.read_csv(REPO / "results/nnunet_subthreshold/nnunet_froc_points.csv")
    k = ["t", "hit_rule", "lesion_set", "size_bin"]
    jj = s2.merge(lad["501"], on=k, suffixes=("_s2", ""), validate="1:1")
    assert len(jj) == len(s2) == len(lad["501"])
    for col in ("n_lesions", "n_hit", "n_components", "n_fp_components"):
        assert (jj[col] == jj[f"{col}_s2"]).all(), col
    say(f"501 ladder == S2 nnunet_froc_points.csv: all {len(s2)} rows (n_hit, n_components, n_fp_components)")
    lad["502"].to_csv(WORK / "ladder_502.csv", index=False)
    for rule in ("peak_in_dil2mm", "touch"):
        say(f"-- rule {rule}, lesion set all: sensitivity 501 -> 502 | FP comps/negative case 501 -> 502")
        say(f"{'t':>6} " + " ".join(f"{b:>15}" for b in ("all", "10-20", "20-40", ">=40")) + f" {'FP/neg':>14}")
        for t in C.THRESHOLDS:
            fa = lad["501"][(lad["501"].t == t) & (lad["501"].hit_rule == rule) & (lad["501"].lesion_set == "all")].set_index("size_bin")
            fb = lad["502"][(lad["502"].t == t) & (lad["502"].hit_rule == rule) & (lad["502"].lesion_set == "all")].set_index("size_bin")
            say(f"{t:>6g} " + " ".join(f"{fa.at[b, 'sensitivity']:>6.1%} -> {fb.at[b, 'sensitivity']:>5.1%}" for b in ("all", "10-20", "20-40", ">=40"))
                + f" {fa.at['all', 'fp_per_case_neg']:>5.2f} -> {fb.at['all', 'fp_per_case_neg']:>4.2f}")
    for (t, rule, lset, b), g in lad["501"].groupby(k):
        h = lad["502"].set_index(k).loc[(t, rule, lset, b)]
        add("ladder", f"sensitivity_{rule}", lset, "diam_bin", b, t=t, n=g.n_lesions.iloc[0],
            baseline=g.sensitivity.iloc[0], multiclass=h.sensitivity)
        if b == "all":
            add("ladder", f"fp_per_negative_case_{rule}", lset, t=t, n_cases=327, baseline=g.fp_per_case_neg.iloc[0], multiclass=h.fp_per_case_neg)
            add("ladder", f"fp_per_case_all_{rule}", lset, t=t, n_cases=1308, baseline=g.fp_per_case_all.iloc[0], multiclass=h.fp_per_case_all)
    # FP-matched: for each 501 rung, 502 at the LOWEST t whose FP/neg does not exceed 501's (no interpolation)
    say("-- FP-matched (touch rule, lesion set all): 502 at the lowest rung with FP/neg <= 501's FP/neg at t")
    for t in C.THRESHOLDS:
        a = lad["501"][(lad["501"].t == t) & (lad["501"].hit_rule == "touch") & (lad["501"].lesion_set == "all")].set_index("size_bin")
        cand = lad["502"][(lad["502"].hit_rule == "touch") & (lad["502"].lesion_set == "all") & (lad["502"].size_bin == "all")
                          & (lad["502"].fp_per_case_neg <= a.at["all", "fp_per_case_neg"])]
        if cand.empty:
            say(f"   t={t:g}: no 502 rung at or below FP/neg {a.at['all', 'fp_per_case_neg']:.2f}")
            continue
        t2 = cand.t.min()
        bb = lad["502"][(lad["502"].t == t2) & (lad["502"].hit_rule == "touch") & (lad["502"].lesion_set == "all")].set_index("size_bin")
        say(f"   501 t={t:<6g} FP/neg {a.at['all', 'fp_per_case_neg']:.2f}: all {a.at['all', 'sensitivity']:.1%}, 10-20 {a.at['10-20', 'sensitivity']:.1%}, "
            f"20-40 {a.at['20-40', 'sensitivity']:.1%} | 502 t={t2:<6g} FP/neg {bb.at['all', 'fp_per_case_neg']:.2f}: all {bb.at['all', 'sensitivity']:.1%}, "
            f"10-20 {bb.at['10-20', 'sensitivity']:.1%}, 20-40 {bb.at['20-40', 'sensitivity']:.1%}")
        for b in ("all", "10-20", "20-40", ">=40"):
            add("ladder_fp_matched", "sensitivity_touch", "all", "diam_bin", b, t=t, n=a.at[b, "n_lesions"],
                baseline=a.at[b, "sensitivity"], multiclass=bb.at[b, "sensitivity"],
                note=f"502 at t={t2:g} (FP/neg {bb.at['all', 'fp_per_case_neg']:.3f} <= 501's {a.at['all', 'fp_per_case_neg']:.3f})")

    # ------------------------------------------------------------ step 5: paired bootstrap
    say("\n== Step 5: paired per-lesion comparison, case-level bootstrap (B = 1000, seed 0, percentile 95% CI)")
    strata = [("all", None), ("diam_bin", None), ("attenuation_v3", None), ("region", None), ("ct_phase", None),
              ("est_phase", None), ("thickness_bin", None), ("tier", None)]
    focus_vars = ("attenuation_v3", "region", "ct_phase", "est_phase", "thickness_bin", "tier")
    for lset in LSETS:
        sub = pl if lset == "all" else pl[~pl.excluded]
        cases = np.array(sorted(sub.case_id.unique()))
        W = S.case_weights(len(cases))
        say(f"-- lesion set {lset}: {len(sub)} lesions in {len(cases)} cases")

        def one(g, var, val, focus="", focus_bin=""):
            rd = S.paired_ratio(g, "case_id", "dice_baseline", "dice_multiclass", cases, W)
            rt = S.paired_ratio(g, "case_id", "det_baseline", "det_multiclass", cases, W)
            ch = g.change.value_counts()
            note = (f"recovered {ch.get('recovered', 0)}, lost {ch.get('lost', 0)}, both {ch.get('both', 0)}, "
                    f"neither {ch.get('neither', 0)}")
            sv = f"{focus} x {var}" if focus else var
            stratum = f"{val}" if not focus else f"{focus_bin} | {val}"
            add("paired_lesion", "mean_lesion_dice", lset, sv, stratum, n=rd["n"], n_cases=rd["n_cases"], baseline=rd["a"],
                multiclass=rd["b"], ci_lo=rd["ci_lo"], ci_hi=rd["ci_hi"], p_boot=rd["p_boot"], note=note)
            add("paired_lesion", "detect_any_rate", lset, sv, stratum, n=rt["n"], n_cases=rt["n_cases"], baseline=rt["a"],
                multiclass=rt["b"], ci_lo=rt["ci_lo"], ci_hi=rt["ci_hi"], p_boot=rt["p_boot"], note=note)
            return rd, rt, ch

        for var, _ in strata:
            vals = ["all"] if var == "all" else (list(BINS) if var == "diam_bin" else sorted(sub[var].astype(str).unique()))
            for val in vals:
                g = sub if var == "all" else sub[sub[var].astype(str) == val]
                if g.empty:
                    continue
                rd, rt, ch = one(g, var, val)
                dd_a = g[g.det_baseline == 1].dice_baseline.mean()
                dd_b = g[g.det_multiclass == 1].dice_multiclass.mean()
                add("paired_lesion", "dice_among_detected", lset, var, val, n=len(g), baseline=dd_a, multiclass=dd_b,
                    note=f"unpaired: 501 n={int(g.det_baseline.sum())}, 502 n={int(g.det_multiclass.sum())}")
                if var in ("all", "diam_bin") or val in ("hypo", "iso", "hyper"):
                    say(f"   {var:>14} {val:<14} n={rd['n']:>4}  Dice {rd['a']:.3f} -> {rd['b']:.3f} {rd['delta']:+.3f} "
                        f"[{rd['ci_lo']:+.3f}, {rd['ci_hi']:+.3f}]  det {rt['a']:.1%} -> {rt['b']:.1%} {rt['delta']:+.1%} "
                        f"[{rt['ci_lo']:+.1%}, {rt['ci_hi']:+.1%}]  rec {ch.get('recovered', 0)} / lost {ch.get('lost', 0)}")
        for focus_bin in ("10-20", "20-40"):
            fb = sub[sub.diam_bin == focus_bin]
            say(f"   focus {focus_bin} (n={len(fb)}):")
            for var in focus_vars:
                for val in sorted(fb[var].astype(str).unique()):
                    g = fb[fb[var].astype(str) == val]
                    rd, rt, ch = one(g, var, val, focus=f"diam_bin={focus_bin}", focus_bin=focus_bin)
                    say(f"     {var:>14} {val:<14} n={rd['n']:>4}  Dice {rd['a']:.3f} -> {rd['b']:.3f} {rd['delta']:+.3f} "
                        f"[{rd['ci_lo']:+.3f}, {rd['ci_hi']:+.3f}]  det {rt['a']:.1%} -> {rt['b']:.1%} {rt['delta']:+.1%} "
                        f"[{rt['ci_lo']:+.1%}, {rt['ci_hi']:+.1%}]  rec {ch.get('recovered', 0)} / lost {ch.get('lost', 0)}")

    # per fold: each fold is an independently trained pair of models, so this is the only look at
    # training-run variability (the bootstrap resamples cases, not training runs)
    say("-- per nnU-Net fold (lesion set all): mean lesion Dice and detect@any, 501 -> 502")
    for f in range(5):
        g = pl[pl.fold == f]
        parts = []
        for b in ("all", "10-20", "20-40"):
            h = g if b == "all" else g[g.diam_bin == b]
            parts.append(f"{b} n={len(h)} Dice {h.dice_baseline.mean():.3f}->{h.dice_multiclass.mean():.3f} "
                         f"det {h.det_baseline.mean():.1%}->{h.det_multiclass.mean():.1%}")
            for met, a, bb in (("mean_lesion_dice", h.dice_baseline.mean(), h.dice_multiclass.mean()),
                               ("detect_any_rate", h.det_baseline.mean(), h.det_multiclass.mean())):
                add("per_fold", met, "all", "fold | diam_bin", f"{f} | {b}", n=len(h), baseline=a, multiclass=bb)
        nf = fp["501"].merge(fp["502"], on=["case_id", "neg"], suffixes=("_501", "_502"))
        nf = nf[nf.neg & (nf.case_id.map(fold_of) == f)]
        add("per_fold", "fp_components_per_negative_case", "all", "fold", str(f), n_cases=len(nf),
            baseline=nf.n_fp_501.mean(), multiclass=nf.n_fp_502.mean())
        say(f"   fold {f}: " + "; ".join(parts) + f"; FP/neg {nf.n_fp_501.mean():.2f}->{nf.n_fp_502.mean():.2f}")

    # case-level Dice, paired
    for lset in LSETS:
        pa, _ = ev_inputs(D["501"], lset)
        pb, _ = ev_inputs(D["502"], lset)
        a = pd.DataFrame(pa)[["case", "gt_tumor", "dice"]]
        b = pd.DataFrame(pb)[["case", "dice"]]
        ab = a.merge(b, on="case", suffixes=("_501", "_502"))
        ab = ab[ab.gt_tumor == 1]
        r = S.paired_ratio(ab, "case", "dice_501", "dice_502", ab.case.to_numpy(), S.case_weights(len(ab)))
        say(f"case Dice ({lset}, ref-positive n={len(ab)}): {r['a']:.4f} -> {r['b']:.4f} {r['delta']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")
        add("case_seg", "dice_mean_paired_bootstrap", lset, n_cases=len(ab), baseline=r["a"], multiclass=r["b"],
            ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p_boot=r["p_boot"])

    # ------------------------------------------------------------ the 20-40 mm lesions the baseline missed
    say("\n== 20-40 mm lesions the baseline missed completely (501 detected_any == 0)")
    for lset in LSETS:
        sub = pl if lset == "all" else pl[~pl.excluded]
        mm = sub[(sub.diam_bin == "20-40") & (sub.det_baseline == 0)]
        rec = mm[mm.det_multiclass == 1]
        hit = sub[(sub.diam_bin == "20-40") & (sub.det_baseline == 1)]
        lost = hit[hit.det_multiclass == 0]
        say(f"-- {lset}: missed by 501 n={len(mm)}; detected by 502 {len(rec)} ({len(rec) / len(mm):.1%}); "
            f"their 502 Dice mean {rec.dice_multiclass.mean():.3f} / median {rec.dice_multiclass.median():.3f}; "
            f"502 Dice >= 0.5 in {int((rec.dice_multiclass >= 0.5).sum())}; all {len(mm)} incl. still-missed: mean Dice {mm.dice_multiclass.mean():.3f}. "
            f"Lost (501 hit, 502 miss): {len(lost)}/{len(hit)}, their 501 Dice mean {lost.dice_baseline.mean():.3f}")
        add("missed_20_40", "n_missed_by_baseline", lset, n=len(mm), baseline=len(mm), multiclass=np.nan)
        add("missed_20_40", "recovered_by_multiclass", lset, n=len(mm), baseline=0, multiclass=len(rec))
        add("missed_20_40", "recovered_fraction", lset, n=len(mm), baseline=0.0, multiclass=len(rec) / len(mm))
        add("missed_20_40", "dice_mean_recovered", lset, n=len(rec), baseline=0.0, multiclass=rec.dice_multiclass.mean())
        add("missed_20_40", "dice_mean_all_missed", lset, n=len(mm), baseline=0.0, multiclass=mm.dice_multiclass.mean())
        add("missed_20_40", "lost_by_multiclass", lset, n=len(hit), baseline=len(hit), multiclass=len(hit) - len(lost),
            note=f"{len(lost)} of the {len(hit)} 501-detected 20-40 mm lesions are missed by 502")
        for var in ("attenuation_v3", "tier", "est_phase", "thickness_bin", "region"):
            t = mm.groupby(var).agg(n=("det_multiclass", "size"), rec=("det_multiclass", "sum"))
            say(f"   by {var}: " + ", ".join(f"{i} {int(r.rec)}/{int(r.n)}" for i, r in t.iterrows()))
            for i, r in t.iterrows():
                add("missed_20_40", "recovered_fraction", lset, var, str(i), n=int(r.n), baseline=0.0, multiclass=r.rec / r.n)
        # touch at a lower threshold for the still-missed ones
        still = mm[mm.det_multiclass == 0]
        say(f"   still missed by 502 ({len(still)}): touched by 502 at t>=0.1 {int((still.max_t_hit_touch_multiclass >= 0.1).sum())}, "
            f"t>=0.01 {int((still.max_t_hit_touch_multiclass >= 0.01).sum())}, t>=0.001 {int((still.max_t_hit_touch_multiclass >= 0.001).sum())}; "
            f"median 502 peak p in lesion+2 mm {still.peak_p_dil2_multiclass.median():.3g} (501: {still.peak_p_dil2_baseline.median():.3g})")
    # S11 cross-check: fold-0 502 detection from the saved seg read on the PC
    s11 = REPO / "work/rescue_check/ds502_fold0_lesions.csv"
    if s11.exists():
        f0 = pd.read_csv(s11).merge(pl[["case_id", "lesion_id", "det_multiclass", "det_baseline", "diam_bin"]], on=["case_id", "lesion_id"])
        assert len(f0) == int((pl.fold == 0).sum()) and (f0.detected_502.astype(int) == f0.det_multiclass).all()
        g = f0[(f0.diam_bin == "20-40") & (f0.det_baseline == 0)]
        say(f"S11 cross-check: fold-0 502 detections identical for all {len(f0)} fold-0 lesions; fold-0 missed 20-40 mm "
            f"recovered {int(g.det_multiclass.sum())}/{len(g)} [carried over 25/36]")

    # ------------------------------------------------------------ write
    OUT.mkdir(parents=True, exist_ok=True)
    cols = (["case_id", "lesion_id", "fold", "diam_mm", "diam_bin", "gt_vox", "vol_mm3", "tiny_flag", "largest_in_case",
             "n_lesions_in_case", "slice_thickness_mm", "thickness_bin", "ct_phase", "est_phase", "attenuation_v3", "region",
             "tier", "excluded"]
            + [f"{c}_{t}" for t in ("baseline", "multiclass") for c in ("dice", "iou", "det", "det_iou10", "peak_p_dil2",
                                                                         "max_t_hit_touch", "max_t_hit_peak")]
            + ["dice_delta", "change"])
    pl[cols].sort_values(["case_id", "lesion_id"]).to_csv(OUT / "per_lesion_comparison.csv", index=False, float_format="%.6g")
    pd.DataFrame(ROWS).to_csv(OUT / "summary_vs_baseline.csv", index=False, float_format="%.6g")
    say(f"\nwrote {OUT / 'per_lesion_comparison.csv'} ({len(pl)} rows), {OUT / 'summary_vs_baseline.csv'} ({len(ROWS)} rows)")
    (OUT / "summary.txt" if os.environ.get("S8_PARTIAL") else WORK / "summary.txt").write_text("\n".join(LINES) + "\n")



if __name__ == "__main__":
    sys.exit(main())
