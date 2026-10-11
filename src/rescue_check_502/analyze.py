"""S14 step 2 (PC): join, regress, describe, write deliverables.

Inputs: master table; S12 phase_estimate; v3 attenuation; S11 rescue_check.analyze.build_table('ds502') (the S11
join of the 502 component table with the duct outputs, imported unchanged); S8's per-lesion comparison CSV and
S8b's phase_addon.csv (regression targets); work/rescue_check_502/cases/*.pkl (case_pass).

Outputs:
  results/rescue_check_502/missed_lesions.csv   one row per lesion RAW 502 misses (all size bins)
  work/rescue_check_502/all_lesions.csv         the same columns for all 1235 lesions
  work/rescue_check_502/summary.txt             every printed number

Usage (PC, repo root): PYTHONPATH=src .venv/bin/python -m rescue_check_502.analyze
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

from multiclass_eval import stats as S
from multiclass_eval.phase_addon import cell_weights, ess, thickness_bin
from rescue_check import analyze as A11, core as K11
from rescue_check_502 import core as K

REPO = Path(__file__).resolve().parents[2]
R = REPO / "results"
WORK = REPO / "work/rescue_check_502"
OUT = R / "rescue_check_502"
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


def pct(k, n) -> str:
    return f"{k}/{n} = {k / n:.1%}" if n else f"{k}/0"


def ci(k, n) -> str:
    lo, hi = K11.wilson(int(k), int(n))
    return f"[{lo:.1%}-{hi:.1%}]" if n else ""


# ------------------------------------------------------------------ build
def load_pass() -> tuple[pd.DataFrame, pd.DataFrame]:
    cases, les = [], []
    files = sorted((WORK / "cases").glob("PanTS_*.pkl"))
    assert len(files) == 1308, f"{len(files)} case pickles (expected 1308)"
    for f in files:
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        cases.append(r["case"])
        les.extend(r["lesions"])
    return pd.DataFrame(cases), pd.DataFrame(les)


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    say("== Build (S11 join for the ds502 tag, imported unchanged)")
    d, case11 = A11.build_table("ds502")
    for line in A11.LINES:
        say("   S11 " + line)
    cp, lp = load_pass()
    d = A11.assert_join(d, lp, "case_pass", [c for c in lp.columns if c not in ("case_id", "lesion_id", "gt_vox")])
    ph = pd.read_csv(R / "phase_estimate/phase_estimate.csv")
    ph["phase5"] = [K.phase_level(a, b) for a, b in zip(ph.est_phase, ph.early_arterial.astype(bool))]
    ph["thick"] = thickness_bin(ph.slice_thickness_mm)
    d = d.merge(ph[["case_id", "est_phase", "phase5"]], on="case_id", how="left", validate="m:1")
    case = case11.merge(cp, on="case_id", validate="1:1").merge(ph[["case_id", "est_phase", "phase5", "thick"]],
                                                                 on="case_id", validate="1:1")
    assert len(case) == 1308 and d.phase5.notna().all()
    d["missed"] = ~d.detected_t05.astype(bool)
    d["group"] = np.where(d.missed, "missed", "detected")
    d["det_dice_d"] = d.dice_d >= K.DICE_DET
    d["det_dice_raw"] = d.dice_raw >= K.DICE_DET
    d["satellite"] = ~d.largest_in_case.astype(bool)
    d["iso"] = d.attenuation_v3 == "iso"
    d["thick_bin"] = thickness_bin(d.slice_thickness_mm)
    d["non_contrast"] = d.phase5 == "non-contrast"
    d["outside_gland_gt"] = d.tier == "separated"
    d["outside_pred_panc"] = d.panc_frac_in < K.FRAC_IN
    # duct chance = case flag x placement fraction g; NA without an MPD mask (as the signal)
    d["duct_chance"] = np.where(d.mpd_present.astype(bool), d.duct_case_flag.astype(float) * d.duct_near_g, np.nan)
    d["nn_hit"] = d.sig_nnunet.astype(float)
    d["duct_hit"] = d.sig_duct.astype("Float64").astype(float)
    return d, case


# ------------------------------------------------------------------ regression
def regress(d: pd.DataFrame, case: pd.DataFrame) -> None:
    say("\n== Regression checks (asserted before any S14 number is read)")
    s8 = pd.read_csv(R / "nnunet_multiclass/per_lesion_comparison.csv")
    m = d.merge(s8[["case_id", "lesion_id", "gt_vox", "det_multiclass", "dice_multiclass", "peak_p_dil2_multiclass"]],
                on=["case_id", "lesion_id"], validate="1:1", suffixes=("", "_s8"))
    assert (m.gt_vox == m.gt_vox_s8).all()
    assert (m.touch_raw == m.det_multiclass).all() and (m.detected_t05.astype(int) == m.det_multiclass).all()
    assert np.allclose(m.dice_raw, m.dice_multiclass, rtol=1e-5, atol=1e-6)
    assert np.allclose(m.peak_p_dil2, m.peak_p_dil2_multiclass, rtol=1e-5)
    say("raw 502 touch / Dice / peak p (lesion+2 mm) == S8 per_lesion_comparison for 1235/1235 "
        "(touch from S13 atoms AND from the S11 table)")
    n2040 = d[d.diam_bin == "20-40"]
    assert int(n2040.missed.sum()) == 116 and len(n2040) == 597
    say(f"missed 20-40 mm = {int(n2040.missed.sum())}/597 (S13: 116, carried over)")
    assert (d.nn_op_hit_obs.astype(bool) == d.sig_nnunet.astype(bool)).all()
    say("nnU-Net OP at the identity placement (case_pass, sparse cache) == S11 sig_nnunet (S8 tables) for 1235/1235")
    assert (np.round(d.panc_frac_in * d.gt_vox).astype(int) == d.les_in_panc_atoms).all()
    say("lesion voxels in predicted pancreas == S13 atoms les_in_panc for 1235/1235")
    pc = case[case.gt_tumor == 1]   # the pass reads the MPD only for tumor+ cases
    agree_m = pc.mpd_present.astype(bool) == pc.mpd_mask.astype(bool)
    assert agree_m.all(), pc.loc[~agree_m, "case_id"].tolist()
    say(f"MPD mask non-empty (this pass, tumorlib) == S11 mpd_mask (mpd_status != absent) for {len(pc)}/{len(pc)} "
        "tumor+ cases")
    assert d.duct_near_obs.isna().equals(~d.mpd_present.astype(bool)), "lesion-level MPD presence != master"
    ev = d[d.mpd_present.astype(bool)]
    agree = (ev.duct_near_obs.astype(bool) == (ev.dist_to_mpd_mm <= K.DUCT_MM))
    say(f"duct 'within 10 mm' at the identity placement vs master dist_to_mpd_mm <= 10: {int(agree.sum())}/{len(ev)} agree")
    for r in ev[~agree].itertuples():
        say(f"   disagree: {r.case_id} lesion {r.lesion_id}: master dist {r.dist_to_mpd_mm:.3f} mm, mine "
            f"{'<=' if r.duct_near_obs else '>'} 10 mm")
    # signal used for the rates = S11's sig_duct (master distance); chance uses this pass's geometry
    neg = case[case.gt_tumor == 0]
    sp_raw = float((neg.n_fp_raw == 0).mean())
    assert round(sp_raw, 4) == 0.7217 and round(float(neg.n_fp_raw.mean()), 4) == 0.3303
    w = weights(case)
    assert abs(float(w @ (neg.n_fp_raw == 0)) - 0.70339) < 5e-6, float(w @ (neg.n_fp_raw == 0))
    say(f"502 raw specificity {sp_raw:.4f}, FP/negative {neg.n_fp_raw.mean():.4f} == S8; S8b est_phase x thickness "
        f"weighted specificity {float(w @ (neg.n_fp_raw == 0)):.5f} == S8b 0.70339 (ESS {ess(w):.1f})")


def weights(case: pd.DataFrame) -> np.ndarray:
    """S8b primary weights (est_phase x thickness), aligned with case[case.gt_tumor == 0]."""
    pos, neg = case[case.gt_tumor == 1], case[case.gt_tumor == 0]
    w, _ = cell_weights(pos.est_phase.astype(str) + " | " + pos.thick, neg.est_phase.astype(str) + " | " + neg.thick)
    return w


# ------------------------------------------------------------------ population + model
def population(d: pd.DataFrame, case: pd.DataFrame) -> None:
    say("\n== Model: 502 + D (fixed rule t_low 0.1 / t_seg 0.1, core-only, all folds). Missed = raw 502 touch == 0")
    say(f"{'bin':>6} {'n':>4} {'missed':>6} | touch raw 502 | Dice>=0.1 502+D | touch 502+D | Dice>=0.1 raw | "
        "mean Dice raw -> D")
    for b in ("<5", "5-10", *K.REPORT_BINS):
        s = d[d.diam_bin == b]
        say(f"{b:>6} {len(s):>4} {int(s.missed.sum()):>6} | {s.touch_raw.mean():12.1%} | {s.det_dice_d.mean():14.1%} | "
            f"{s.touch_d.mean():10.1%} | {s.det_dice_raw.mean():12.1%} | {s.dice_raw.mean():.3f} -> {s.dice_d.mean():.3f}")
    say(f"all 1235: mean lesion Dice raw {d.dice_raw.mean():.3f} -> D {d.dice_d.mean():.3f} "
        "(S13 LOFO-pooled D 0.418, carried over; S13 fold 3 used the t_low 1e-4 variant)")
    for b in K.FOCUS_BINS:
        s = d[(d.diam_bin == b) & d.missed]
        say(f"{b} mm misses: D touches {int(s.touch_d.sum())}/{len(s)}, Dice >= 0.1 under D {int(s.det_dice_d.sum())}, "
            f"max Dice under D {s.dice_d.max():.3f}")
        s = d[(d.diam_bin == b) & ~d.missed]
        say(f"{b} mm detected (touch): Dice >= 0.1 under D {pct(int(s.det_dice_d.sum()), len(s))}; "
            f"raw {pct(int(s.det_dice_raw.sum()), len(s))}")
    neg = case[case.gt_tumor == 0]
    w = weights(case)
    for tag in ("raw", "d"):
        spec = (neg[f"n_fp_{tag}"] == 0).astype(float)
        say(f"specificity 502 {'+ D' if tag == 'd' else 'raw'}: unweighted {spec.mean():.4f}, S8b-weighted {w @ spec:.4f}; "
            f"FP comps/negative {neg[f'n_fp_{tag}'].mean():.3f} (weighted {w @ neg[f'n_fp_{tag}']:.3f})")


# ------------------------------------------------------------------ step 2: missed vs detected
CATS = [("phase (S12, S8b strata)", "phase5", K.PHASE_LEVELS),
        ("attenuation v3", "attenuation_v3", ("hypo", "iso", "hyper")),
        ("satellite (not largest in case)", "satellite", (True, False)),
        ("tier (GT envelope)", "tier", ("inside", "embedded", "abutting", "separated")),
        ("region", "region", ("head", "body", "tail")),
        ("slice thickness", "thick_bin", K.THICK_LABELS),
        ("502 predicted pancreas category", "panc_category", K.PANC_CATS)]
CONTS = [("slice thickness (mm)", "slice_thickness_mm"), ("diameter (mm)", "diam_mm"),
         ("dist to predicted pancreas (mm)", "panc_dist_mm"), ("frac of lesion in predicted pancreas", "panc_frac_in"),
         ("frac in filled predicted pancreas", "panc_frac_filled"), ("shell (3 mm) frac in predicted pancreas", "panc_shell_frac"),
         ("502 peak p in lesion+2 mm", "peak_p_dil2")]


def compare(d: pd.DataFrame) -> None:
    for b in K.FOCUS_BINS:
        s = d[d.diam_bin == b]
        mi, de = s[s.missed], s[~s.missed]
        say(f"\n== Step 2, {b} mm: missed n {len(mi)} vs detected (touch) n {len(de)}")
        say(f"   {'level':<16} {'missed':>16} {'detected':>16}  Fisher p | detection in level: touch / Dice>=0.1 (D)")
        for title, col, levels in CATS:
            say(f" {title}")
            vals = s[col].fillna("NA")
            extra = [v for v in vals.unique() if v not in levels]
            for lv in (*levels, *sorted(map(str, extra))):
                km, kd = int((mi[col].fillna("NA") == lv).sum()), int((de[col].fillna("NA") == lv).sum())
                if km + kd == 0:
                    continue
                p = fisher_exact([[km, len(mi) - km], [kd, len(de) - kd]])[1]
                g = s[vals == lv]
                say(f"   {str(lv):<16} {km:>4} ({km / len(mi):5.1%})   {kd:>4} ({kd / len(de):5.1%})   {p:8.2g} | "
                    f"n {len(g):>3}: {g.touch_raw.mean():5.1%} / {g.det_dice_d.mean():5.1%}")
        say(" continuous: median [IQR] missed vs detected, Mann-Whitney p")
        for title, col in CONTS:
            a, c = mi[col].dropna(), de[col].dropna()
            q = lambda x: f"{x.median():.3g} [{x.quantile(.25):.3g}-{x.quantile(.75):.3g}]"
            p = mannwhitneyu(a, c).pvalue if len(a) and len(c) else np.nan
            say(f"   {title:<40} {q(a):>24} vs {q(c):>24}  p {p:.2g} (n {len(a)} / {len(c)})")


def plain_table(d: pd.DataFrame) -> None:
    say("\n== Plain table: share of misses (and of detected, same bin) with each property")
    flags = [("non-contrast (est. phase)", "non_contrast"), ("iso (v3)", "iso"), ("satellite", "satellite"),
             ("outside gland (GT tier separated)", "outside_gland_gt"),
             ("< 50% inside 502's predicted pancreas", "outside_pred_panc")]
    for b in K.FOCUS_BINS:
        s = d[d.diam_bin == b]
        mi, de = s[s.missed], s[~s.missed]
        say(f"-- {b} mm (missed n {len(mi)}, detected n {len(de)})")
        for name, col in flags:
            km, kd = int(mi[col].sum()), int(de[col].sum())
            say(f"   {name:<40} missed {pct(km, len(mi)):>16} {ci(km, len(mi)):<14} detected {pct(kd, len(de)):>16}")
        anyf = mi[[c for _, c in flags]].any(axis=1)
        say(f"   {'at least one of the five':<40} missed {pct(int(anyf.sum()), len(mi)):>16}   "
            f"detected {pct(int(de[[c for _, c in flags]].any(axis=1).sum()), len(de))}")
        none = mi[~anyf]
        say(f"   none of the five (enhanced, not iso, index lesion, inside both glands): {len(none)} misses; "
            f"attenuation {none.attenuation_v3.value_counts().to_dict()}, phase {none.phase5.value_counts().to_dict()}")


def panc_section(d: pd.DataFrame) -> None:
    say("\n== 502 predicted pancreas around the lesion (bulge vs dent question)")
    for b in (*K.FOCUS_BINS, ">=40"):
        s = d[d.diam_bin == b]
        for g in ("missed", "detected"):
            x = s[s.group == g]
            cats = ", ".join(f"{c} {pct(int((x.panc_category == c).sum()), len(x))}" for c in K.PANC_CATS)
            say(f"   {b:>6} {g:<8} n {len(x):>3}: {cats}; median frac_in {x.panc_frac_in.median():.2f}, "
                f"shell {x.panc_shell_frac.median():.2f}")
    for b in K.FOCUS_BINS:
        x = d[(d.diam_bin == b) & d.missed]
        say(f"-- {b} mm misses, category x GT tier:")
        say("   " + pd.crosstab(x.panc_category, x.tier).reindex(index=list(K.PANC_CATS), fill_value=0)
            .to_string().replace("\n", "\n   "))
        inc = x[x.panc_category == "included"]
        say(f"   included misses: frac_in == 1.0 for {int((inc.panc_frac_in >= 0.999).sum())}/{len(inc)}; "
            f"phase {inc.phase5.value_counts().to_dict()}; attenuation {inc.attenuation_v3.value_counts().to_dict()}; "
            f"satellite {int(inc.satellite.sum())}")
        exc = x[x.panc_category != "included"]
        say(f"   not-included misses: shell frac median {exc.panc_shell_frac.median():.2f}; "
            f"dist median {exc.panc_dist_mm.median():.1f} mm; tier {exc.tier.value_counts().to_dict()}")


# ------------------------------------------------------------------ step 3: signals vs chance
def signals(d: pd.DataFrame, case: pd.DataFrame) -> None:
    say("\n== Step 3: signals vs uniform-placement chance (S9 dropped). diff = mean(hit - chance), case bootstrap CI")
    for b in (*K.FOCUS_BINS, ">=40"):
        for g in ("missed", "detected"):
            x = d[(d.diam_bin == b) & (d.group == g)]
            for sig, hit, ch in (("nnU-Net OP", "nn_hit", "nn_op_chance"), ("duct OP", "duct_hit", "duct_chance")):
                e = x[x[hit].notna() & x[ch].notna()]
                cases = np.array(sorted(e.case_id.unique()))
                if len(e) == 0:
                    say(f"   {b:>6} {g:<8} {sig:<10} n 0")
                    continue
                r = S.paired_ratio(e, "case_id", ch, hit, cases, S.case_weights(len(cases)))
                k = int(e[hit].sum())
                ne = len(x) - len(e)
                say(f"   {b:>6} {g:<8} {sig:<10} {pct(k, len(e)):>16} {ci(k, len(e)):<14} chance {r['a']:6.1%}  "
                    f"diff {r['delta']:+6.1%} [{r['ci_lo']:+.1%}, {r['ci_hi']:+.1%}]" + (f"  (NE {ne})" if ne else ""))
    say(" duct pieces (evaluable = MPD mask present): case flag rate, and 'within 10 mm' observed vs placement g")
    for b in K.FOCUS_BINS:
        for g in ("missed", "detected"):
            x = d[(d.diam_bin == b) & (d.group == g) & d.mpd_present.astype(bool)]
            say(f"   {b:>6} {g:<8} n {len(x):>3}: flag {x.duct_case_flag.mean():5.1%}; within 10 mm observed "
                f"{(x.dist_to_mpd_mm <= K.DUCT_MM).mean():5.1%} vs chance g {x.duct_near_g.mean():5.1%}")
    neg = case[case.gt_tumor == 0]
    w = weights(case)
    say(" costs on the 327 negatives (S8b weights in brackets)")
    fp = neg[f"n_fp_top{K11.NN_OP_RANK}_t{K11.NN_OP_T:g}"]
    say(f"   nnU-Net OP: FP components/negative {fp.mean():.3f} [{w @ fp:.3f}]; negatives with >= 1 {(fp > 0).mean():.1%} "
        f"[{w @ (fp > 0):.1%}]")
    fl = neg.duct_case_flag.astype(float)
    say(f"   duct case flag: {fl.mean():.1%} [{w @ fl:.1%}] of negatives; MPD mask in {neg.mpd_mask.mean():.1%}")
    say(" faint candidates, descriptive only (no rescue claim): Dice of the touching OP component(s), p > 0.01")
    for b in K.FOCUS_BINS:
        x = d[(d.diam_bin == b) & d.missed & d.nn_hit.astype(bool)]
        say(f"   {b} mm misses touched by the OP: {len(x)}; Dice median {x.nn_op_dice.median():.3f}, "
            f"max {x.nn_op_dice.max():.3f}, >= 0.1: {int((x.nn_op_dice >= 0.1).sum())}")
    for b in K.FOCUS_BINS:
        x = d[(d.diam_bin == b) & d.missed]
        both = x.nn_hit.astype(bool) | x.duct_hit.fillna(0).astype(bool)
        say(f"   {b} mm misses flagged by either signal: {pct(int(both.sum()), len(x))} (NE duct counted as not flagged)")


# ------------------------------------------------------------------ S15 target sizing
def s15_sizes(d: pd.DataFrame) -> None:
    say("\n== S15 sizing (missed by raw 502; counts by subgroup)")
    for b in K.FOCUS_BINS:
        x = d[(d.diam_bin == b) & d.missed]
        inc = x.panc_category == "included"
        enh = ~x.non_contrast & (x.phase5 != "unmeasurable")
        say(f"-- {b}: missed {len(x)}; included in predicted pancreas {int(inc.sum())}; not included "
            f"{int((~inc).sum())} (dent {int((x.panc_category == 'dent').sum())}, adjacent "
            f"{int((x.panc_category == 'adjacent').sum())}, away {int((x.panc_category == 'away').sum())})")
        say(f"   included & index lesion: {int((inc & ~x.satellite).sum())}; included & index & not separated: "
            f"{int((inc & ~x.satellite & ~x.outside_gland_gt).sum())}; ... & enhanced: "
            f"{int((inc & ~x.satellite & ~x.outside_gland_gt & enh).sum())}; ... & enhanced & not iso: "
            f"{int((inc & ~x.satellite & ~x.outside_gland_gt & enh & ~x.iso).sum())}")
        y = d[(d.diam_bin == b) & ~d.missed & (d.panc_category == "included")]
        say(f"   detected & included (comparison pool for the same bin): {len(y)}")


# ------------------------------------------------------------------ write
COLS = ["case_id", "lesion_id", "fold", "gt_vox", "diam_mm", "diam_bin", "largest_in_case", "satellite",
        "n_lesions_in_case", "tier", "excluded", "region", "attenuation_v3", "est_phase", "phase5", "ct_phase",
        "slice_thickness_mm", "thick_bin", "peak_p_dil2", "touch_raw", "dice_raw", "touch_d", "dice_d", "det_dice_d",
        "vol_ratio_d", "panc_frac_in", "panc_frac_filled", "panc_shell_frac", "panc_dist_mm", "panc_category",
        "case_pred_t05", "hit_t0.01", "rank_t0.01", "n_comps_case_t0.01", "sig_nnunet", "nn_op_n_placed",
        "nn_op_chance", "nn_op_dice", "mpd_present", "dist_to_mpd_mm", "mpd_dilated", "head_end_dilated",
        "duct_case_flag", "sig_duct", "duct_near_g", "duct_chance"]


def main() -> None:
    d, case = build()
    regress(d, case)
    population(d, case)
    plain_table(d)
    compare(d)
    panc_section(d)
    signals(d, case)
    s15_sizes(d)
    OUT.mkdir(parents=True, exist_ok=True)
    t = d[COLS].sort_values(["case_id", "lesion_id"])
    t.to_csv(WORK / "all_lesions.csv", index=False, float_format="%.6g")
    t[~t.touch_raw.astype(bool)].to_csv(OUT / "missed_lesions.csv", index=False, float_format="%.6g")
    say(f"\nwrote {OUT / 'missed_lesions.csv'} ({int((~t.touch_raw.astype(bool)).sum())} rows), "
        f"{WORK / 'all_lesions.csv'} ({len(t)} rows)")
    (WORK / "summary.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
