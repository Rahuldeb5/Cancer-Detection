"""S11 step 2: join everything per lesion, evaluate the three signals at their fixed operating points,
and write the deliverables.

Inputs (all existing outputs; nothing is recomputed from images):
  results/master_lesion_table/master_lesions.csv            population, size bins, region, phase, MPD distance
  work/rescue_check/nnunet_<tag>_{lesions,cases}.csv         from rescue_check.nnunet_table (--nnunet_tag)
  results/normative_pilot/lesion_ranks.csv, froc_points.csv  S9 score A percentile / candidate rank / FP
  results/attenuation_v3/attenuation_labels_v3.csv           v3 attenuation class
  results/geometry_results/duct_dilation_flags.csv           case-level duct flags
  results/geometry_results/duct_mask_quality.csv             MPD / CBD mask status
  results/geometry_results/duct_cutoff/cutoff_per_duct.csv   MPD head-end caliber (caliber_head_mm)

Outputs:
  results/rescue_check/rescue_by_lesion.csv   1235 rows
  results/rescue_check/summary.csv            long table: section x lesion_set x size_bin x group x signal x metric
  work/rescue_check/summary_<tag>.txt         readable log of every number

Usage (repo root, PC): PYTHONPATH=src .venv/bin/python -m rescue_check.analyze [--nnunet_tag ds501_s2]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rescue_check import core as K

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work" / "rescue_check"
OUT = REPO / "results" / "rescue_check"
R = REPO / "results"
LESION_SETS = ("all", "excl_separated")
REPORT_BINS = ("10-20", "20-40", ">=40")
N_REVIEW = 15
STRATA = ("attenuation_v3", "ct_phase", "region", "thick_bin", "case_pred_t05")

LINES: list[str] = []
ROWS: list[dict] = []


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


def add(section, lesion_set="", size_bin="", group="", signal="", metric="", n=np.nan, k=np.nan,
        value=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_not_evaluable=np.nan, note="") -> None:
    ROWS.append(dict(section=section, lesion_set=lesion_set, size_bin=size_bin, group=group, signal=signal,
                     metric=metric, n=n, k=k, value=value, ci_lo=ci_lo, ci_hi=ci_hi,
                     n_not_evaluable=n_not_evaluable, note=note))


def fmt(r: dict) -> str:
    if not r["n"]:
        return f"  -/0 (NE {r['n_not_evaluable']})"
    return (f"{r['k']:>4}/{r['n']:<4} {r['rate']:6.1%} [{r['ci_lo']:5.1%}-{r['ci_hi']:5.1%}]"
            + (f" NE {r['n_not_evaluable']}" if r["n_not_evaluable"] else ""))


# ------------------------------------------------------------------ join
def assert_join(left: pd.DataFrame, right: pd.DataFrame, name: str, cols) -> pd.DataFrame:
    m = left.merge(right[["case_id", "lesion_id", "gt_vox", *cols]], on=["case_id", "lesion_id"], how="left",
                   suffixes=("", f"_{name}"), validate="1:1")
    assert m[f"gt_vox_{name}"].notna().all() and (m.gt_vox == m[f"gt_vox_{name}"]).all(), f"gt_vox mismatch: {name}"
    say(f"join {name}: {len(m)} rows, gt_vox equal for all")
    return m.drop(columns=f"gt_vox_{name}")


def build_table(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = pd.read_csv(R / "master_lesion_table/master_lesions.csv")
    nn = pd.read_csv(WORK / f"nnunet_{tag}_lesions.csv")
    cases_nn = pd.read_csv(WORK / f"nnunet_{tag}_cases.csv")
    s9 = pd.read_csv(R / "normative_pilot/lesion_ranks.csv")
    v3 = pd.read_csv(R / "attenuation_v3/attenuation_labels_v3.csv")
    say(f"== Joins (nnU-Net source tag: {tag})")
    nn_cols = [c for c in nn.columns if c not in ("case_id", "lesion_id", "gt_vox")]
    d = assert_join(master, nn, "nnunet", nn_cols)
    d = assert_join(d, s9.rename(columns={"attenuation_v3": "attenuation_v3_s9"}), "s9",
                    ["pct_A", "cand_rank_A", "matched_radius_mm", "n_centres_dil2", "n_far_centres", "attenuation_v3_s9"])
    d = assert_join(d, v3, "v3", ["attenuation_v3"])
    assert (d.attenuation_v3.fillna("NA") == d.attenuation_v3_s9.fillna("NA")).all()
    d = d.drop(columns="attenuation_v3_s9")

    flags = pd.read_csv(R / "geometry_results/duct_dilation_flags.csv")
    qual = pd.read_csv(R / "geometry_results/duct_mask_quality.csv")
    cut = pd.read_csv(R / "geometry_results/duct_cutoff/cutoff_per_duct.csv")
    mpd = cut[cut.duct == "mpd"][["case_id", "status", "caliber_head_mm"]].rename(columns={"status": "mpd_centerline_status"})
    case = (flags[["case_id", "tumor_present", "mpd_dilated", "cbd_dilated", "either_duct_dilated", "double_duct_dilated"]]
            .merge(qual[["case_id", "mpd_status", "cbd_status"]], on="case_id", validate="1:1")
            .merge(mpd, on="case_id", validate="1:1")
            .merge(cases_nn, on="case_id", validate="1:1"))
    assert len(case) == 1308 and case.case_id.is_unique
    assert ((case.tumor_present == 0) == case.is_negative).all()
    case["mpd_mask"] = case.mpd_status != "absent"
    case["cbd_mask"] = case.cbd_status != "absent"
    case["head_end_dilated"] = (case.caliber_head_mm >= K.MPD_HEAD_DILATE_MM).astype("boolean").where(
        case.caliber_head_mm.notna())
    case["duct_case_flag"] = K.duct_case_flag(case.mpd_dilated, case.caliber_head_mm).to_numpy()
    say(f"case-level duct join: 1308 cases; MPD mask {int(case.mpd_mask.sum())}, head caliber available "
        f"{int(case.caliber_head_mm.notna().sum())} (MPD centerline status ok)")
    keep = ["case_id", "mpd_dilated", "cbd_dilated", "either_duct_dilated", "double_duct_dilated", "mpd_status",
            "cbd_status", "mpd_mask", "cbd_mask", "mpd_centerline_status", "caliber_head_mm", "head_end_dilated",
            "duct_case_flag"]
    d = d.merge(case[keep], on="case_id", how="left", validate="m:1")
    assert (d.mpd_mask == d.mpd_present).all(), "MPD mask status disagrees with master mpd_present"

    d["missed"] = ~d.detected_t05.astype(bool)
    d["group"] = np.where(d.missed, "missed", "detected")
    d["thick_bin"] = K.thick_bin(d.slice_thickness_mm)
    d["sig_nnunet"] = K.nnunet_signal(d[f"hit_t{K.NN_OP_T:g}"], d[f"rank_t{K.NN_OP_T:g}"]).to_numpy()
    d["sig_normative"] = K.normative_signal(d.pct_A, d.cand_rank_A).to_numpy()
    d["sig_duct"] = K.duct_signal(d.duct_case_flag, d.dist_to_mpd_mm, d.mpd_present).to_numpy()
    d["s9_not_evaluable_reason"] = np.select(
        [d.pct_A.notna(), d.n_centres_dil2 == 0], ["", "no_lattice_centre_in_lesion_2mm"], "footprint_too_large")
    d["n_signals"] = K.caught(d).sum(axis=1)
    return d, case


# ------------------------------------------------------------------ oracles
def oracles(d: pd.DataFrame, tag: str) -> None:
    say("\n== Oracles")
    assert len(d) == 1235 and not d.duplicated(["case_id", "lesion_id"]).any()
    n2040 = int((d.diam_bin == "20-40").sum())
    assert n2040 == 597, n2040
    det = d[d.diam_bin == "20-40"].detected_t05.mean()
    say(f"total lesions 1235; 20-40 mm n = {n2040}; detected at t=0.5 in 20-40 mm {det:.1%} (S2: 59.8%, carried over)")
    if tag == "ds501_s2":
        assert round(det, 3) == 0.598
        assert (d.detected_t05 == d.detected_any).all()
        say("missed set == master detected_any == 0 for all 1235 lesions")
    for b in K.BINS:
        s = d[d.diam_bin == b]
        say(f"  {b:>6}: n {len(s):>4}, detected {int(s.detected_t05.sum()):>4} ({s.detected_t05.mean():5.1%}), "
            f"missed {int(s.missed.sum()):>4}; excl. separated: n {int((~s.excluded).sum())}, "
            f"missed {int((s.missed & ~s.excluded).sum())}")
        for ls in LESION_SETS:
            ss = s if ls == "all" else s[~s.excluded]
            for g in ("missed", "detected"):
                add("population", ls, b, g, metric="n_lesions", n=int((ss.group == g).sum()))


# ------------------------------------------------------------------ per-signal hit rates
def component_metrics(d: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    """(signal, metric, nullable-boolean hit series) for every reported component of every signal."""
    out = []
    for t in K.NN_THRESHOLDS:
        h = d[f"hit_t{t:g}"].astype(bool)
        out.append(("nnunet", f"touch_t{t:g}", pd.Series(h.to_numpy(), dtype="boolean")))
        out.append(("nnunet", f"touch_t{t:g}_rank<={K.NN_OP_RANK}",
                    pd.Series((h & (d[f"rank_t{t:g}"] <= K.NN_OP_RANK)).to_numpy(), dtype="boolean")))
    out.append(("nnunet", "OP", d.sig_nnunet))
    s9_ev = d.pct_A.notna().to_numpy()
    out.append(("normative", "pct_A>=90", K._nullable((d.pct_A >= 90).to_numpy(), s9_ev)))
    out.append(("normative", "OP", d.sig_normative))
    ev = {"mpd_dilated": d.mpd_mask, "cbd_dilated": d.cbd_mask, "either_duct_dilated": d.mpd_mask | d.cbd_mask,
          "double_duct_dilated": d.mpd_mask & d.cbd_mask}
    for f, e in ev.items():
        out.append(("duct", f"case_{f}", K._nullable(d[f].astype(bool).to_numpy(), e.to_numpy())))
    out.append(("duct", "case_head_end_dilated", pd.Series(d.head_end_dilated.to_numpy(), dtype="boolean")))
    out.append(("duct", "case_mpd_or_head_end", K._nullable(d.duct_case_flag.to_numpy(), d.mpd_mask.to_numpy())))
    out.append(("duct", f"dist_to_mpd<={K.DUCT_DIST_MM:g}mm",
                K._nullable((d.dist_to_mpd_mm <= K.DUCT_DIST_MM).to_numpy(), d.mpd_mask.to_numpy())))
    out.append(("duct", "OP", d.sig_duct))
    return out


def signal_rates(d: pd.DataFrame) -> None:
    say("\n== Signal hit rates, MISSED vs DETECTED (Wilson 95% CI; NE = not evaluable, excluded from n)")
    d = d.reset_index(drop=True)
    metrics = component_metrics(d)
    for ls in LESION_SETS:
        for b in REPORT_BINS:
            base = (d.diam_bin == b) & (True if ls == "all" else ~d.excluded)
            say(f"-- {b} mm, lesion set {ls}: missed n={int((base & d.missed).sum())}, "
                f"detected n={int((base & ~d.missed).sum())}")
            for sig, met, s in metrics:
                cells = []
                for g in ("missed", "detected"):
                    r = K.rate(s[base & (d.group == g)])
                    add("signal_hit", ls, b, g, sig, met, r["n"], r["k"], r["rate"], r["ci_lo"], r["ci_hi"],
                        r["n_not_evaluable"])
                    cells.append(fmt(r))
                say(f"   {sig:<9} {met:<26} missed {cells[0]:<44} detected {cells[1]}")
            # continuous summaries
            for g in ("missed", "detected"):
                s = d[base & (d.group == g)]
                for t in K.NN_THRESHOLDS:
                    rk = s.loc[s[f"hit_t{t:g}"].astype(bool), f"rank_t{t:g}"]
                    add("signal_hit", ls, b, g, "nnunet", f"median_rank_among_touch_t{t:g}", n=len(rk),
                        value=float(rk.median()) if len(rk) else np.nan)
                add("signal_hit", ls, b, g, "nnunet", "median_peak_p_dil2", n=int(s.peak_p_dil2.notna().sum()),
                    value=float(s.peak_p_dil2.median()))
                add("signal_hit", ls, b, g, "normative", "median_pct_A", n=int(s.pct_A.notna().sum()),
                    value=float(s.pct_A.median()), n_not_evaluable=int(s.pct_A.isna().sum()))
                for reason in ("no_lattice_centre_in_lesion_2mm", "footprint_too_large"):
                    add("not_evaluable", ls, b, g, "normative", reason, n=len(s),
                        k=int((s.s9_not_evaluable_reason == reason).sum()))
                add("not_evaluable", ls, b, g, "duct", "no_mpd_mask", n=len(s), k=int((~s.mpd_mask).sum()))
                add("not_evaluable", ls, b, g, "duct", "no_head_caliber", n=len(s), k=int(s.caliber_head_mm.isna().sum()))
            for g in ("missed", "detected"):
                s = d[base & (d.group == g)]
                rk = {t: s.loc[s[f"hit_t{t:g}"].astype(bool), f"rank_t{t:g}"].median() for t in K.NN_THRESHOLDS}
                say(f"   {g:>8}: median rank among touched {', '.join(f't{t:g} {v:.0f}' for t, v in rk.items())}; "
                    f"median peak p (lesion+2mm) {s.peak_p_dil2.median():.3g}; median pct_A {s.pct_A.median():.1f}; "
                    f"S9 NE: no centre {int((s.s9_not_evaluable_reason == 'no_lattice_centre_in_lesion_2mm').sum())}, "
                    f"footprint too large {int((s.s9_not_evaluable_reason == 'footprint_too_large').sum())}; "
                    f"duct NE (no MPD mask) {int((~s.mpd_mask).sum())}")


# ------------------------------------------------------------------ costs
def costs(d: pd.DataFrame, case: pd.DataFrame) -> None:
    say("\n== Costs")
    neg = case[case.is_negative]
    for t in K.NN_THRESHOLDS:
        c = f"n_fp_t{t:g}"
        add("cost", "", "", "all_1308", "nnunet", f"fp_per_case_t{t:g}", n=len(case), value=case[c].mean())
        add("cost", "", "", "negatives_327", "nnunet", f"fp_per_case_t{t:g}", n=len(neg), value=neg[c].mean())
        say(f"nnU-Net FP components/case at t={t:g}: all {case[c].mean():.2f}, negatives {neg[c].mean():.2f}")
    c = f"n_fp_top{K.NN_OP_RANK}_t{K.NN_OP_T:g}"
    add("cost", "", "", "all_1308", "nnunet", "fp_per_case_OP", n=len(case), value=case[c].mean())
    add("cost", "", "", "negatives_327", "nnunet", "fp_per_case_OP", n=len(neg), value=neg[c].mean())
    say(f"nnU-Net OP (t={K.NN_OP_T:g}, top {K.NN_OP_RANK}): FP/case all {case[c].mean():.2f}, negatives {neg[c].mean():.2f}")

    fr = pd.read_csv(R / "normative_pilot/froc_points.csv")
    fr = fr[(fr.score == "A") & (fr.op_type == "top_n") & (fr.op_value == K.S9_TOP_N) & (fr.stratum == fr.stratum.iloc[0])]
    add("cost", "", "", "", "normative", "candidates_per_case_OP", value=float(K.S9_TOP_N), note="fixed by design")
    for r in fr.itertuples():
        add("cost", "", "", r.fp_source, "normative", f"fp_per_case_OP_r{r.radius_mm:g}", n=r.n_fp_cases,
            value=r.fp_per_case, note="read from S9 froc_points.csv")
        say(f"S9 score A top-{K.S9_TOP_N}, r {r.radius_mm:g} mm, FP source {r.fp_source}: {r.fp_per_case:.2f} FP/case "
            f"(n cases {r.n_fp_cases}; read from S9 froc_points.csv)")

    pos_cases = d.groupby("case_id").agg(any_det=("detected_t05", "any"), any_miss=("missed", "any"))
    sets = {"negatives_327": neg,
            "negatives_with_mpd_mask": neg[neg.mpd_mask],
            "detected_lesion_cases": case[case.case_id.isin(pos_cases.index[pos_cases.any_det])],
            "detected_lesion_cases_with_mpd_mask": case[case.case_id.isin(pos_cases.index[pos_cases.any_det]) & case.mpd_mask],
            "missed_lesion_cases": case[case.case_id.isin(pos_cases.index[pos_cases.any_miss])]}
    for name, s in sets.items():
        for f in ("mpd_dilated", "head_end_dilated", "duct_case_flag"):
            v = s[f].fillna(False).astype(bool)
            lo, hi = K.wilson(int(v.sum()), len(s))
            add("cost", "", "", name, "duct", f"case_flag_{f}", n=len(s), k=int(v.sum()), value=v.mean(), ci_lo=lo, ci_hi=hi)
        v = s.duct_case_flag.astype(bool)
        say(f"duct case flag (mpd_dilated OR head-end) in {name}: {int(v.sum())}/{len(s)} = {v.mean():.1%}; "
            f"mpd_dilated {int(s.mpd_dilated.sum())}, head-end {int(s.head_end_dilated.fillna(False).sum())}")


# ------------------------------------------------------------------ overlap + gate
def overlap(d: pd.DataFrame) -> dict:
    say("\n== Overlap among MISSED lesions (not evaluable = not caught)")
    gates = {}
    for ls in LESION_SETS:
        for b in REPORT_BINS:
            for g in ("missed", "detected"):
                s = d[(d.diam_bin == b) & (d.group == g) & (True if ls == "all" else ~d.excluded)]
                variants = [("", s)]
                if g == "missed" and b == K.PRIMARY_BIN:
                    full = s[s.sig_normative.notna() & s.sig_duct.notna()]
                    variants.append(("fully_evaluable", full))
                for note, ss in variants:
                    ov = K.overlap_counts(ss)
                    tag = f"{ls}|{b}|{g}" + (f"|{note}" if note else "")
                    for cell, k in ov["cells"].items():
                        add("overlap", ls, b, g, cell, "venn_cell", ov["n"], k, k / ov["n"] if ov["n"] else np.nan, note=note)
                    for nsig, k in ov["by_n_signals"].items():
                        add("overlap", ls, b, g, "", f"caught_by_{nsig}", ov["n"], k, k / ov["n"] if ov["n"] else np.nan, note=note)
                    add("overlap", ls, b, g, "union", "union", ov["n"], ov["union_k"], ov["union_rate"], *ov["union_ci"], note=note)
                    for sig in K.SIGNALS:
                        add("overlap", ls, b, g, sig, "unique", ov["n"], ov["unique"][sig],
                            ov["unique"][sig] / ov["n"] if ov["n"] else np.nan, note=note)
                    if g == "missed":
                        say(f"-- {tag}: n {ov['n']}; caught by 0/1/2/3 = {list(ov['by_n_signals'].values())}; "
                            f"union {ov['union_k']} = {ov['union_rate']:.1%} [{ov['union_ci'][0]:.1%}-{ov['union_ci'][1]:.1%}]; "
                            f"unique {ov['unique']}; cells {ov['cells']}")
                    else:
                        say(f"   ({tag}: union {ov['union_k']}/{ov['n']} = {ov['union_rate']:.1%})")
                    if g == "missed" and b == K.PRIMARY_BIN and not note:
                        gates[ls] = (K.gate(ov), ov)
    say("\n== Pairwise unions, missed 20-40 mm (reported only; not the gate)")
    for ls in LESION_SETS:
        c = K.caught(d[(d.diam_bin == K.PRIMARY_BIN) & d.missed & (True if ls == "all" else ~d.excluded)])
        for a, b in (("nnunet", "duct"), ("nnunet", "normative"), ("normative", "duct")):
            k, n = int((c[a] | c[b]).sum()), len(c)
            lo, hi = K.wilson(k, n)
            add("overlap", ls, K.PRIMARY_BIN, "missed", f"{a}|{b}", "pair_union", n, k, k / n, lo, hi,
                note="reported only; not the gate")
            say(f"   {ls}: {a} OR {b}: {k}/{n} = {k / n:.1%} [{lo:.1%}-{hi:.1%}]")
    say("\n== GATE (missed 20-40 mm; fixed a priori: union >= 30% with CI lo >= 20%, AND one signal's unique "
        "catches >= 10% of the missed lesions)")
    for ls, (g, ov) in gates.items():
        say(f"   lesion set {ls}: union {g['union_rate']:.1%} (CI lo {g['union_ci_lo']:.1%}) -> "
            f"{'ok' if g['union_ok'] else 'NOT met'}; unique fractions "
            + ", ".join(f"{s} {v:.1%}" for s, v in g["unique_frac"].items())
            + f" -> {'ok' if g['unique_ok'] else 'NOT met'};  GATE {'PASS' if g['passed'] else 'FAIL'}")
        add("gate", ls, K.PRIMARY_BIN, "missed", "union", "union_rate>=0.30_and_ci_lo>=0.20", ov["n"], ov["union_k"],
            g["union_rate"], g["union_ci_lo"], ov["union_ci"][1], note="ok" if g["union_ok"] else "not met")
        for s, v in g["unique_frac"].items():
            add("gate", ls, K.PRIMARY_BIN, "missed", s, "unique_frac>=0.10", ov["n"], ov["unique"][s], v,
                note="ok" if v >= K.GATE_UNIQUE_MIN else "not met")
        add("gate", ls, K.PRIMARY_BIN, "missed", "", "VERDICT", ov["n"], note="PASS" if g["passed"] else "FAIL")
        # secondary reading, reported only: unique catches / misses not caught by the other two
        c = K.caught(d[(d.diam_bin == K.PRIMARY_BIN) & d.missed & (True if ls == "all" else ~d.excluded)])
        alt = {}
        for s in K.SIGNALS:
            others = [o for o in K.SIGNALS if o != s]
            pool = ~c[others].any(axis=1)
            alt[s] = (int((c[s] & pool).sum()), int(pool.sum()))
            add("gate", ls, K.PRIMARY_BIN, "missed", s, "secondary_unique_over_not_caught_by_others",
                alt[s][1], alt[s][0], alt[s][0] / alt[s][1] if alt[s][1] else np.nan, note="reported only; not the gate")
        say("   (secondary reading, not the gate: unique / misses not caught by the other two = "
            + ", ".join(f"{s} {k}/{n} = {k / n:.1%}" for s, (k, n) in alt.items()) + ")")
    return gates


# ------------------------------------------------------------------ chance reference for S9 top-N
def s9_chance(d: pd.DataFrame) -> None:
    """Reported only. If S9's N candidates were N lattice centres drawn uniformly (with replacement) from
    the case's search region, a lesion with footprint fraction f = n_centres_dil2 / n_centres would be hit
    with probability 1 - (1 - f)^N. Local-maximum suppression spreads real candidates apart, which makes
    the true chance rate slightly higher than this. n_centres comes from S9's work/normative/case_table.csv."""
    ct = pd.read_csv(REPO / "work/normative/case_table.csv")[["case_id", "n_centres"]]
    x = d.merge(ct, on="case_id", how="left", validate="m:1")
    f = x.n_centres_dil2 / x.n_centres
    x["chance"] = 1.0 - (1.0 - f) ** K.S9_TOP_N
    d["s9_chance_topN"] = x["chance"].to_numpy()
    say(f"\n== S9 top-{K.S9_TOP_N} vs a uniform-placement chance rate (reported only)")
    for ls in LESION_SETS:
        for b in REPORT_BINS:
            for g in ("missed", "detected"):
                s = x[(x.diam_bin == b) & (x.group == g) & x.pct_A.notna() & (True if ls == "all" else ~x.excluded)]
                obs = s.sig_normative.astype(bool).mean()
                add("signal_hit", ls, b, g, "normative", f"chance_top{K.S9_TOP_N}_uniform", n=len(s),
                    value=s.chance.mean(), note=f"observed OP {obs:.4f}; median footprint fraction {f[s.index].median():.4f}")
                say(f"   {ls:<14} {b:>6} {g:>8}: n {len(s):>3}, observed {obs:6.1%}, chance {s.chance.mean():6.1%}, "
                    f"median footprint fraction {f[s.index].median():.3f}")


# ------------------------------------------------------------------ nothing caught
def nothing_caught(d: pd.DataFrame) -> None:
    say("\n== Missed lesions caught by NOTHING, stratified (share of the stratum's missed lesions)")
    for ls in LESION_SETS:
        for b in REPORT_BINS:
            s = d[(d.diam_bin == b) & d.missed & (True if ls == "all" else ~d.excluded)]
            none = s.n_signals == 0
            say(f"-- {ls} | {b}: {int(none.sum())}/{len(s)} caught by nothing")
            for var in STRATA:
                parts = []
                for lvl, g in s.groupby(s[var].fillna("NA").astype(str)):
                    k = int((g.n_signals == 0).sum())
                    lo, hi = K.wilson(k, len(g))
                    add("nothing_caught", ls, b, "missed", var, str(lvl), len(g), k, k / len(g), lo, hi)
                    parts.append(f"{lvl} {k}/{len(g)}")
                if b == K.PRIMARY_BIN:
                    say(f"   {var:<15} " + "; ".join(parts))


# ------------------------------------------------------------------ optional: Dataset502 fold 0
def ds502_fold0(d: pd.DataFrame) -> None:
    """501 vs 502 missed sets on nnU-Net fold 0 (the only 502 fold trained). Reads the table written by
    rescue_check.ds502_fold0; skipped if it is absent."""
    f = WORK / "ds502_fold0_lesions.csv"
    d["detected_502_fold0"] = pd.array([pd.NA] * len(d), dtype="boolean")
    if not f.exists():
        say("\n== Dataset502 fold 0: skipped (no work/rescue_check/ds502_fold0_lesions.csv)")
        return
    t = pd.read_csv(f)
    j = d.reset_index().merge(t, on=["case_id", "lesion_id"], how="inner", suffixes=("", "_502"), validate="1:1")
    assert len(j) == int((d.fold == 0).sum()) and (j.gt_vox == j.gt_vox_502).all(), "502 join / gt_vox mismatch"
    d.loc[j["index"].to_numpy(), "detected_502_fold0"] = j.detected_502.astype(bool).to_numpy()
    say(f"\n== Dataset502 vs Dataset501, nnU-Net FOLD 0 ONLY ({len(j)} lesions, gt_vox equal); touch rule at the "
        "saved segmentation (502: label == 2)")
    cz = pd.read_csv(WORK / "ds502_fold0_cases.csv")
    nn = pd.read_csv(WORK / "nnunet_ds501_s2_cases.csv")
    cz = cz.merge(nn[["case_id", "is_negative", "n_comps_t0.5", "n_fp_t0.5"]], on="case_id", validate="1:1")
    assert len(cz) == 264
    for name, s in (("all_fold0", cz), ("negatives_fold0", cz[cz.is_negative])):
        vals = dict(fp_per_case_501=s["n_fp_t0.5"].mean(), fp_per_case_502=s.n_fp_502.mean(),
                    cases_any_pred_501=(s["n_comps_t0.5"] > 0).mean(), cases_any_pred_502=s.any_label2.mean())
        for k, v in vals.items():
            add("ds502_fold0", "", "", name, "", k, n=len(s), value=v, note="fold 0 only; touch-rule FP components")
        say(f"   cost, {name} (n {len(s)}): FP components/case 501 {vals['fp_per_case_501']:.2f} vs 502 "
            f"{vals['fp_per_case_502']:.2f}; cases with any lesion prediction 501 {vals['cases_any_pred_501']:.1%} "
            f"vs 502 {vals['cases_any_pred_502']:.1%}")
    for ls in LESION_SETS:
        for b in ("all", *REPORT_BINS):
            sel = np.ones(len(j), bool) if b == "all" else (j.diam_bin == b).to_numpy().copy()
            if ls != "all":
                sel &= ~j.excluded.to_numpy(bool)
            s = j[sel]
            m501, m502 = s.missed, ~s.detected_502.astype(bool)
            cnt = dict(n=len(s), missed_501=int(m501.sum()), missed_502=int(m502.sum()), missed_both=int((m501 & m502).sum()),
                       recovered_by_502=int((m501 & ~m502).sum()), newly_missed_by_502=int((~m501 & m502).sum()))
            for k, v in cnt.items():
                add("ds502_fold0", ls, b, "", "", k, n=len(s), k=v, note="fold 0 only; n small")
            say(f"   {ls:<14} {b:>6}: n {cnt['n']:>3}; detected 501 {len(s) - cnt['missed_501']:>3} "
                f"({1 - m501.mean():5.1%}), 502 {len(s) - cnt['missed_502']:>3} ({1 - m502.mean():5.1%}); "
                f"missed by both {cnt['missed_both']}, recovered by 502 {cnt['recovered_by_502']}, "
                f"newly missed by 502 {cnt['newly_missed_by_502']}")
            if b == K.PRIMARY_BIN:
                r = s[m501 & ~m502]
                q = s[m501 & m502]
                say(f"      of the {len(r)} recovered, caught by >=1 signal: {int((r.n_signals > 0).sum())}; "
                    f"of the {len(q)} missed by both: {int((q.n_signals > 0).sum())}")
                add("ds502_fold0", ls, b, "recovered_by_502", "", "caught_by_>=1_signal", len(r), int((r.n_signals > 0).sum()))
                add("ds502_fold0", ls, b, "missed_by_both", "", "caught_by_>=1_signal", len(q), int((q.n_signals > 0).sum()))


# ------------------------------------------------------------------ review list
def review_list(d: pd.DataFrame) -> pd.DataFrame:
    s = d[(d.diam_bin == K.PRIMARY_BIN) & d.missed]
    o = K.evidence_order(s)
    strong = o.head(N_REVIEW)
    none = o[o.n_signals == 0].tail(N_REVIEW).iloc[::-1]   # weakest graded evidence first
    d["evidence_graded"] = np.nan
    d.loc[o.index, "evidence_graded"] = o.graded
    d["review_list"] = ""
    d.loc[strong.index, "review_list"] = "strongest"
    d.loc[none.index, "review_list"] = "none"
    d["review_rank"] = np.nan
    d.loc[strong.index, "review_rank"] = np.arange(1, len(strong) + 1)
    d.loc[none.index, "review_rank"] = np.arange(1, len(none) + 1)
    cols = ["case_id", "lesion_id", "diam_mm", "attenuation_v3", "region", "ct_phase", "excluded", "n_signals",
            "sig_nnunet", "sig_normative", "sig_duct", "peak_p_dil2", "pct_A", "cand_rank_A", "dist_to_mpd_mm"]
    for name, t in (("STRONGEST combined evidence", strong), ("NO evidence (weakest graded first)", none)):
        say(f"\n== Review list: missed 20-40 mm, {name}")
        say(t[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnunet_tag", default="ds501_s2", help="tag written by rescue_check.nnunet_table")
    ap.add_argument("--out", type=Path, default=None, help="results dir (default results/rescue_check/ for ds501_s2)")
    args = ap.parse_args()
    out = args.out or (OUT if args.nnunet_tag == "ds501_s2" else WORK / f"results_{args.nnunet_tag}")
    d, case = build_table(args.nnunet_tag)
    oracles(d, args.nnunet_tag)
    signal_rates(d)
    costs(d, case)
    s9_chance(d)
    overlap(d)
    nothing_caught(d)
    ds502_fold0(d)
    d = review_list(d)

    out.mkdir(parents=True, exist_ok=True)
    cols = ["case_id", "lesion_id", "fold", "gt_vox", "diam_mm", "diam_bin", "excluded", "tier", "attenuation_v3",
            "ct_phase", "region", "slice_thickness_mm", "thick_bin", "largest_in_case",
            "detected_t05", "missed", "case_pred_t05", "peak_p_dil2"]
    for t in K.NN_THRESHOLDS:
        cols += [f"hit_t{t:g}", f"rank_t{t:g}", f"n_comps_case_t{t:g}"]
    cols += ["sig_nnunet", "pct_A", "cand_rank_A", "matched_radius_mm", "s9_not_evaluable_reason", "sig_normative",
             "s9_chance_topN",
             "mpd_present", "mpd_dilated", "cbd_dilated", "either_duct_dilated", "double_duct_dilated",
             "caliber_head_mm", "head_end_dilated", "duct_case_flag", "dist_to_mpd_mm", "sig_duct",
             "n_signals", "evidence_graded", "review_list", "review_rank", "detected_502_fold0"]
    d[cols].sort_values(["case_id", "lesion_id"]).to_csv(out / "rescue_by_lesion.csv", index=False, float_format="%.6g")
    pd.DataFrame(ROWS).to_csv(out / "summary.csv", index=False, float_format="%.6g")
    say(f"\nwrote {out / 'rescue_by_lesion.csv'} ({len(d)} rows), {out / 'summary.csv'} ({len(ROWS)} rows)")
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / f"summary_{args.nnunet_tag}.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
