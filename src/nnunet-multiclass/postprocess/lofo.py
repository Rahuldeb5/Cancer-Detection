"""S13 step 2 (PC): leave-one-fold-out tuning of the post-processing families, then the held-out report.

Objective (fixed before results): mean over all 1308 cases of the per-case lesion Dice (tumor+: Dice of the lesion
mask vs all GT lesion voxels, 0 if nothing is predicted; tumor-: 1 if nothing is predicted, else 0).
Tuning: for held-out nnU-Net fold k, the setting with the highest objective on the other 4 folds (first one on exact
ties; the no-op is first in every family) is applied to fold k; the 5 held-out folds are pooled. Pooled results never
feed back into a choice. Families (core.family): A filter, B faint-candidate rescue, D regrowth of the saved
segmentation's own objects, C = A + B + D; AB and BD are reported as attribution only.

Inputs: work/nnunet_postprocess/atoms/*.pkl (atoms_pass.py), master lesion table, S1 case spacing, S8's
summary_vs_baseline.csv (regression). Outputs: results/nnunet_postprocess/lofo_results.csv,
work/nnunet_postprocess/summary.txt (every number quoted in the README / notes).

Usage (PC, repo root):  PYTHONPATH=src .venv/bin/python src/nnunet-multiclass/postprocess/lofo.py
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as K  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from multiclass_eval import stats as S  # noqa: E402

WORK = REPO / "work/nnunet_postprocess"
OUT = REPO / "results/nnunet_postprocess"
BINS = ("<5", "5-10", "10-20", "20-40", ">=40")
TIERS = ("inside", "embedded", "abutting", "separated")
THICK = ("<2.5", "2.5-5", ">=5")
MAIN = ("A", "B", "D", "ABD")
EXTRA = ("AB", "BD", "ABDc")
FAM_NAME = {"baseline": "S8 502 (no post-processing)", "A": "A filter", "B": "B rescue", "D": "D regrow",
            "ABD": "C = A+B+D", "AB": "A+B", "BD": "B+D", "ABDc": "C+ (post hoc)"}
LINES: list[str] = []
ROWS: list[dict] = []


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


def add(section, family, metric, value, stratum="", fold="", n=np.nan, ci_lo=np.nan, ci_hi=np.nan, setting="", note=""):
    ROWS.append(dict(section=section, family=family, metric=metric, stratum=stratum, fold=fold, value=value, n=n,
                     ci_lo=ci_lo, ci_hi=ci_hi, setting=setting, note=note))


# ------------------------------------------------------------------ load
def load():
    cases = {}
    for f in sorted((WORK / "atoms").glob("PanTS_*.pkl")):
        with open(f, "rb") as fh:
            c = pickle.load(fh)
        cases[c["case_id"]] = c
    ids = sorted(cases)
    assert len(ids) == 1308, len(ids)
    bad = [(c, cases[c]["s8_bad"]) for c in ids if not cases[c]["s8_ok"]]
    assert not bad, f"{len(bad)} cases fail the per-case S8 regression: {bad[:5]}"
    settings = K.all_settings()
    assert all(cases[c]["n_settings"] == len(settings) for c in ids)
    G = np.stack([cases[c]["grid"] for c in ids])
    meta = pd.DataFrame({"case_id": ids, "fold": [cases[c]["fold"] for c in ids],
                         "pos": [cases[c]["gt_vox"] > 0 for c in ids]})
    sp = pd.read_csv(REPO / "work/master_lesion_table/case_spacing.csv")[["case_id", "slice_thickness_mm", "fold"]]
    meta = meta.merge(sp.rename(columns={"fold": "fold_s1"}), on="case_id", how="left", validate="1:1")
    assert (meta.fold == meta.fold_s1).all(), "S1 case_spacing stores the nnU-Net fold (0-4)"
    meta["thick"] = pd.cut(meta.slice_thickness_mm, [0, 2.5, 5.0, np.inf], right=False, labels=list(THICK)).astype(str)
    master = pd.read_csv(REPO / "results/master_lesion_table/master_lesions.csv")
    return cases, ids, settings, G, meta, master


# ------------------------------------------------------------------ evaluate a policy (case -> setting)
def run_policy(cases, ids, policy: dict, master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    crow, lrow = [], []
    for c in ids:
        s = policy[c]
        r, les = K.evaluate(cases[c], s)
        crow.append(dict(case_id=c, setting=s.label(), tie_at_cut=K.tie_at_cut(cases[c], s), **r))
        lrow.extend(dict(case_id=c, **x) for x in les)
    cd, ld = pd.DataFrame(crow), pd.DataFrame(lrow)
    j = master[["case_id", "lesion_id", "gt_vox", "diam_bin", "tier", "excluded"]].merge(
        ld, on=["case_id", "lesion_id"], how="outer", suffixes=("", "_pp"), validate="1:1", indicator=True)
    assert (j._merge == "both").all() and len(j) == 1235 and (j.gt_vox == j.gt_vox_pp).all()
    return cd, j.drop(columns=["_merge", "gt_vox_pp"])


def metrics(cd: pd.DataFrame, ld: pd.DataFrame, meta: pd.DataFrame) -> dict:
    cd = cd.merge(meta, on="case_id", validate="1:1")
    pos, neg = cd[cd.pos], cd[~cd.pos]
    spec_bin = neg.groupby("thick").seg_any.apply(lambda s: 1 - s.mean())
    mix = pos.thick.value_counts(normalize=True)
    cover = float(mix.reindex(spec_bin.index).fillna(0).sum())
    m = dict(objective=cd.objective.mean(), case_dice_pos=pos.dice.mean(),
             spec_raw=1 - neg.seg_any.mean(),
             spec_reweighted=float((spec_bin * mix.reindex(spec_bin.index).fillna(0)).sum() / cover),
             spec_reweight_cover=cover,
             fp_per_neg=neg.n_fp_cc.mean(), fp_per_pos=pos.n_fp_cc.mean(),
             neg_any_fp=(neg.n_fp_cc > 0).mean(),
             localized=int(pos.localized.sum()), wrong_place=int((pos.seg_any & ~pos.localized).sum()),
             nothing=int((~pos.seg_any).sum()),
             rel_vol_diff_median=float(((pos.pred_vol_mm3 - pos.ref_vol_mm3) / pos.ref_vol_mm3).median()),
             lesion_dice=ld.lesion_dice.mean(), det_all=ld.detected_any.mean(),
             n_tie_at_cut=int(cd.tie_at_cut.sum()))
    for b in THICK:
        m[f"spec_thick_{b}"] = float(spec_bin.get(b, np.nan))
        m[f"n_neg_thick_{b}"] = int((neg.thick == b).sum())
    for b in BINS:
        g = ld[ld.diam_bin == b]
        dt = g[g.detected_any == 1]
        m[f"det_{b}"], m[f"dice_{b}"] = g.detected_any.mean(), g.lesion_dice.mean()
        m[f"dice_det_{b}"] = dt.lesion_dice.mean() if len(dt) else np.nan
        m[f"volratio_det_{b}"] = dt.pred_vol_ratio.median() if len(dt) else np.nan
        m[f"n_det_{b}"] = len(dt)
    for t in TIERS:
        g = ld[ld.tier == t]
        m[f"det_tier_{t}"], m[f"dice_tier_{t}"] = g.detected_any.mean(), g.lesion_dice.mean()
    return m


# ------------------------------------------------------------------ S8 regression on the aggregate level
def check_s8(m: dict) -> None:
    s8 = pd.read_csv(REPO / "results/nnunet_multiclass/summary_vs_baseline.csv")
    s8 = s8[s8.lesion_set == "all"]

    def ref(section, metric, stratum=None, svar=None):
        q = s8[(s8.section == section) & (s8.metric == metric)]
        if stratum is not None:
            q = q[q.stratum.astype(str) == stratum]
        if svar is not None:
            q = q[q.stratum_var == svar]
        assert len(q) == 1, (section, metric, stratum, len(q))
        return float(q.multiclass.iloc[0])

    pairs = [("case Dice (981)", m["case_dice_pos"], ref("case_seg", "dice_mean")),
             ("specificity", m["spec_raw"], ref("case_detect_any_voxel", "score=pred_volume_specificity")),
             ("rel_vol_diff median", m["rel_vol_diff_median"], ref("case_seg", "rel_vol_diff_median")),
             ("FP/neg", m["fp_per_neg"], ref("false_positives", "fp_components_per_case", "negatives")),
             ("FP/pos", m["fp_per_pos"], ref("false_positives", "fp_components_per_case", "tumor+")),
             ("localized", m["localized"], ref("localization", "cases_localized")),
             ("wrong_place", m["wrong_place"], ref("localization", "cases_wrong_place")),
             ("nothing", m["nothing"], ref("localization", "cases_nothing")),
             ("lesion Dice (1235)", m["lesion_dice"], ref("paired_lesion", "mean_lesion_dice", "all", "all"))]
    for b in BINS:
        pairs += [(f"det {b}", m[f"det_{b}"], ref("paired_lesion", "detect_any_rate", b, "diam_bin")),
                  (f"Dice {b}", m[f"dice_{b}"], ref("paired_lesion", "mean_lesion_dice", b, "diam_bin")),
                  (f"Dice|det {b}", m[f"dice_det_{b}"], ref("paired_lesion", "dice_among_detected", b, "diam_bin"))]
    for t in TIERS:
        pairs.append((f"det tier {t}", m[f"det_tier_{t}"], ref("paired_lesion", "detect_any_rate", t, "tier")))
    for name, a, b in pairs:
        assert float(f"{a:.6g}") == b or (np.isnan(a) and np.isnan(b)), f"S8 regression: {name} {a} vs {b}"  # CSV holds %.6g
    say(f"S8 regression (no filter, t=0.5, keep all, t_seg=0.5): {len(pairs)} aggregate numbers == "
        f"results/nnunet_multiclass/summary_vs_baseline.csv to its stored 6 significant digits; per case / per lesion "
        f"checked exactly in atoms_pass")


# ------------------------------------------------------------------ main
def main() -> None:
    cases, ids, settings, G, meta, master = load()
    SI = {s: i for i, s in enumerate(settings)}
    folds = meta.fold.to_numpy()
    say("== Inputs")
    say(f"1308 cases ({int(meta.pos.sum())} tumor+, {int((~meta.pos).sum())} tumor-); {len(settings)} distinct settings; "
        f"every case passed the per-case S8 regression in atoms_pass")
    nsp = np.array([cases[c]["n_sparse"] for c in ids])
    say(f"sparse cache: voxels with p > 1e-4 per case median {np.median(nsp):.0f}, max {nsp.max()}; predicted pancreas "
        f"empty in {sum(cases[c]['n_panc'] == 0 for c in ids)} cases "
        f"({sum(cases[c]['n_panc'] == 0 and cases[c]['gt_vox'] > 0 for c in ids)} tumor+)")
    ncomp = {t: np.array([cases[c]["comps"][t]["n_vox"].size for c in ids]) for t in K.T_LOWS}
    say("components per case (whole volume) by t_low: " + ", ".join(
        f"{t:g}: median {np.median(v):.0f}, mean {v.mean():.1f}, max {v.max()}" for t, v in ncomp.items()))
    neg_ids = meta[~meta.pos].case_id
    say("components per NEGATIVE case by t_low (mean): " + ", ".join(
        f"{t:g}: {np.mean([cases[c]['comps'][t]['n_vox'].size for c in neg_ids]):.2f}" for t in K.T_LOWS))

    base_policy = {c: K.BASELINE for c in ids}
    cd0, ld0 = run_policy(cases, ids, base_policy, master)
    m0 = metrics(cd0, ld0, meta)
    say("\n== Regression")
    check_s8(m0)
    assert np.allclose(G[:, SI[K.BASELINE]], cd0.objective.to_numpy())
    say(f"baseline objective {m0['objective']:.4f} = (981 x case Dice {m0['case_dice_pos']:.4f} + 327 x spec "
        f"{m0['spec_raw']:.4f}) / 1308")

    # ---------------------------------------------------------------- LOFO
    say("\n== Leave-one-fold-out picks (setting chosen on the other 4 folds, applied to the held-out fold)")
    W = S.case_weights(len(ids))
    results = {"baseline": (cd0, ld0, m0, {})}
    for fam in MAIN + EXTRA:
        fidx = np.array([SI[s] for s in K.family(fam)])
        picks, policy = {}, {}
        for k in range(5):
            tr, te = folds != k, folds == k
            mean_tr = G[tr][:, fidx].mean(0)
            j = int(np.argmax(mean_tr))
            s = settings[fidx[j]]
            srt = np.sort(mean_tr)[::-1]
            picks[k] = dict(setting=s, train=mean_tr[j], train_base=G[tr, SI[K.BASELINE]].mean(),
                            test=G[te, fidx[j]].mean(), test_base=G[te, SI[K.BASELINE]].mean(),
                            n_ties=int((mean_tr == mean_tr[j]).sum()), gap2=srt[0] - srt[1] if srt.size > 1 else np.nan)
            for c in meta.case_id[te]:
                policy[c] = s
        cd, ld = run_policy(cases, ids, policy, master)
        m = metrics(cd, ld, meta)
        results[fam] = (cd, ld, m, picks)
        labels = [p["setting"].label() for p in picks.values()]
        say(f"-- {FAM_NAME[fam]} ({len(fidx)} settings): {len(set(labels))} distinct pick(s) over 5 folds")
        for k, p in picks.items():
            say(f"   fold {k}: {p['setting'].label():<52} train {p['train_base']:.4f} -> {p['train']:.4f} | held-out "
                f"{p['test_base']:.4f} -> {p['test']:.4f} ({p['test'] - p['test_base']:+.4f}); ties {p['n_ties']}, "
                f"gap to 2nd {p['gap2']:.5f}")
            for met, v in (("train_objective_baseline", p["train_base"]), ("train_objective", p["train"]),
                           ("heldout_objective_baseline", p["test_base"]), ("heldout_objective", p["test"])):
                add("picks", fam, met, v, fold=k, setting=p["setting"].label(),
                    note=f"exact ties at the max: {p['n_ties']}; margin to 2nd best {p['gap2']:.6f}")

    # ---------------------------------------------------------------- scoreboard
    say("\n== Pooled held-out results (5 folds, each with its own pick)")
    obj0 = cd0[["case_id", "objective"]].rename(columns={"objective": "o0"})
    cols = ("baseline",) + MAIN + EXTRA
    cis = {"baseline": dict(ci_lo=np.nan, ci_hi=np.nan)}
    for f in MAIN + EXTRA:
        j = obj0.merge(results[f][0][["case_id", "objective"]], on="case_id", validate="1:1")
        cis[f] = S.paired_ratio(j, "case_id", "o0", "objective", np.array(ids), W)
    for fam in cols:
        m, ci = results[fam][2], cis[fam]
        for key, v in m.items():
            add("pooled", fam, key, v, n=1308 if key == "objective" else np.nan,
                ci_lo=ci["ci_lo"] if key == "objective" else np.nan, ci_hi=ci["ci_hi"] if key == "objective" else np.nan,
                note="CI = paired case bootstrap of the delta vs baseline (B=1000, seed 0)" if key == "objective" and fam != "baseline" else "")
    hdr = f"{'':<28}" + "".join(f"{FAM_NAME[f][:13]:>14}" for f in cols)
    say(hdr)

    def line(name, key, fmt):
        say(f"{name:<28}" + "".join(f"{format(results[f][2][key], fmt):>14}" for f in cols))

    line("objective (1308)", "objective", ".4f")
    say(f"{'  delta vs S8 [95% CI]':<28}{'':>14}" + "".join(
        f"{'%+.4f' % (results[f][2]['objective'] - m0['objective']):>14}" for f in MAIN + EXTRA))
    say(f"{'':<28}{'':>14}" + "".join(f"{'[%+.3f,%+.3f]' % (cis[f]['ci_lo'], cis[f]['ci_hi']):>14}" for f in MAIN + EXTRA))
    line("case Dice, tumor+ (981)", "case_dice_pos", ".4f")
    line("case specificity raw", "spec_raw", ".4f")
    line("  reweighted to + thickness", "spec_reweighted", ".4f")
    line("FP CCs / negative", "fp_per_neg", ".3f")
    line("FP CCs / tumor+ (non-tumor)", "fp_per_pos", ".3f")
    line("negatives with any FP", "neg_any_fp", ".3f")
    line("localized tumor+ cases", "localized", "d")
    line("wrong place only", "wrong_place", "d")
    line("nothing predicted", "nothing", "d")
    line("lesion Dice (1235)", "lesion_dice", ".4f")
    for b in BINS:
        line(f"detect@touch {b}", f"det_{b}", ".3f")
    for b in BINS:
        line(f"lesion Dice {b}", f"dice_{b}", ".3f")
    for b in ("10-20", "20-40", ">=40"):
        line(f"Dice | detected {b}", f"dice_det_{b}", ".3f")
    for b in ("10-20", "20-40", ">=40"):
        line(f"median vol ratio|det {b}", f"volratio_det_{b}", ".3f")
    line("case rel_vol_diff median", "rel_vol_diff_median", "+.3f")
    for t in TIERS:
        line(f"detect tier {t}", f"det_tier_{t}", ".3f")
    line("tie at top-k cut (cases)", "n_tie_at_cut", "d")
    say(f"thickness reweighting: negatives per bin " + ", ".join(f"{b} {m0[f'n_neg_thick_{b}']}" for b in THICK)
        + f"; tumor+ mix " + ", ".join(f"{b} {(meta[meta.pos].thick == b).mean():.3f}" for b in THICK)
        + f"; coverage {m0['spec_reweight_cover']:.3f}")
    for f in ("baseline",) + MAIN:
        say(f"   specificity by thickness, {FAM_NAME[f]}: " + ", ".join(
            f"{b} {results[f][2][f'spec_thick_{b}']:.3f}" for b in THICK))

    # transitions per lesion vs baseline (held-out)
    say("\n-- per-lesion detection changes vs S8 (held-out policies)")
    for fam in MAIN + EXTRA:
        ld = results[fam][1].merge(ld0[["case_id", "lesion_id", "detected_any", "lesion_dice"]],
                                   on=["case_id", "lesion_id"], suffixes=("", "_0"))
        rec = ld[(ld.detected_any == 1) & (ld.detected_any_0 == 0)]
        lost = ld[(ld.detected_any == 0) & (ld.detected_any_0 == 1)]
        say(f"   {FAM_NAME[fam]:<12} recovered {len(rec)} ({', '.join(f'{b} {int((rec.diam_bin == b).sum())}' for b in BINS)}); "
            f"lost {len(lost)} ({', '.join(f'{t} {int((lost.tier == t).sum())}' for t in TIERS)})")
        for b in BINS:
            add("lesion_changes", fam, "recovered", int((rec.diam_bin == b).sum()), stratum=f"diam_bin={b}")
            add("lesion_changes", fam, "lost", int((lost.diam_bin == b).sum()), stratum=f"diam_bin={b}")
        for t in TIERS:
            add("lesion_changes", fam, "lost", int((lost.tier == t).sum()), stratum=f"tier={t}")
            add("lesion_changes", fam, "recovered", int((rec.tier == t).sum()), stratum=f"tier={t}")

    # ---------------------------------------------------------------- A: losses by tier for every filter setting
    say("\n== Filter A on the saved segmentation, every setting on all 1308 (descriptive; not used to choose)")
    det0 = ld0.set_index(["case_id", "lesion_id"])
    at_risk = ld0[ld0.detected_any == 1]
    say(f"lesions detected by S8 502 (at risk): {len(at_risk)}; by tier " +
        ", ".join(f"{t} {int((at_risk.tier == t).sum())}/{int((ld0.tier == t).sum())}" for t in TIERS))
    say(f"{'setting':<34}{'objective':>10}{'spec':>8}{'FP/neg':>8}{'FP/pos':>8}  lost (inside/embedded/abutting/separated)  "
        f"lost with a single touching CC")
    for f in K.FILTERS:
        s = K.Setting(filt=f)
        cd, ld = run_policy(cases, ids, {c: s for c in ids}, master)
        m = metrics(cd, ld, meta)
        x = ld.merge(ld0[["case_id", "lesion_id", "detected_any", "n_touching_cc"]], on=["case_id", "lesion_id"],
                     suffixes=("", "_0"))
        lost = x[(x.detected_any_0 == 1) & (x.detected_any == 0)]
        single = lost[lost.n_touching_cc_0 == 1]
        say(f"{s.label():<34}{m['objective']:>10.4f}{m['spec_raw']:>8.3f}{m['fp_per_neg']:>8.3f}{m['fp_per_pos']:>8.3f}  "
            f"{len(lost):>3} = " + "/".join(str(int((lost.tier == t).sum())) for t in TIERS)
            + f"   {len(single)}  ({', '.join(f'{b} {int((lost.diam_bin == b).sum())}' for b in BINS if (lost.diam_bin == b).any())})")
        for t in TIERS:
            add("filterA_losses", "A", "lesions_lost", int((lost.tier == t).sum()), stratum=f"tier={t}",
                n=int(((ld0.tier == t) & (ld0.detected_any == 1)).sum()), setting=s.label(),
                note="detected by S8 502, no touching component after the filter; n = detected by S8 in that tier")
            add("filterA_losses", "A", "lesions_lost_single_touching_cc", int((single.tier == t).sum()), stratum=f"tier={t}",
                setting=s.label())
        for key in ("objective", "spec_raw", "fp_per_neg", "fp_per_pos", "case_dice_pos"):
            add("filterA_all_settings", "A", key, m[key], setting=s.label(), note="descriptive, all 1308, not used to choose")

    # the same filter applied to a faint footprint (C+'s pick) bites much harder: separate it from the regrowth
    ref_s = K.Setting(t_low=0.001, t_seg=0.1, require_core=True)
    cdr, ldr = run_policy(cases, ids, {c: ref_s for c in ids}, master)
    mr = metrics(cdr, ldr, meta)
    say(f"-- the filter on a faint footprint: D at t_low 0.001 / t_seg 0.1 (core-only), each filter vs none; reference "
        f"objective {mr['objective']:.4f}, spec {mr['spec_raw']:.3f}, FP/neg {mr['fp_per_neg']:.3f}, detected {int(ldr.detected_any.sum())}")
    for f in K.FILTERS[1:]:
        s2 = replace(ref_s, filt=f)
        cd, ld = run_policy(cases, ids, {c: s2 for c in ids}, master)
        m = metrics(cd, ld, meta)
        x = ld.merge(ldr[["case_id", "lesion_id", "detected_any"]], on=["case_id", "lesion_id"], suffixes=("", "_r"))
        lost = x[(x.detected_any_r == 1) & (x.detected_any == 0)]
        say(f"   {s2.label():<44} objective {m['objective']:.4f} spec {m['spec_raw']:.3f} FP/neg {m['fp_per_neg']:.3f} "
            f"FP/pos {m['fp_per_pos']:.3f}; lost {len(lost)} = " + "/".join(str(int((lost.tier == t).sum())) for t in TIERS)
            + " (inside/embedded/abutting/separated); by size " + ", ".join(f"{b} {int((lost.diam_bin == b).sum())}" for b in BINS))
        for t in TIERS:
            add("filterA_losses_faint_footprint", "A", "lesions_lost", int((lost.tier == t).sum()), stratum=f"tier={t}",
                n=int(((ldr.tier == t) & (ldr.detected_any == 1)).sum()), setting=s2.label(),
                note="vs the same setting without the filter; descriptive, all 1308")
        add("filterA_losses_faint_footprint", "A", "objective", m["objective"], setting=s2.label(), note="descriptive, all 1308")
        add("filterA_losses_faint_footprint", "A", "spec_raw", m["spec_raw"], setting=s2.label(), note="descriptive, all 1308")

    # ---------------------------------------------------------------- B/D: where the components and gains are
    say("\n== B: 20-40 mm lesions S8 misses (detected_any == 0): rescued under each held-out policy")
    miss = ld0[(ld0.diam_bin == "20-40") & (ld0.detected_any == 0)]
    m501 = pd.read_csv(REPO / "results/nnunet_multiclass/per_lesion_comparison.csv")
    m501 = m501[(m501.diam_bin == "20-40")]
    both = int(((m501.det_baseline == 0) & (m501.det_multiclass == 0)).sum())
    assert int((m501.det_multiclass == 0).sum()) == len(miss)
    say(f"S8 502 misses {len(miss)} of {int((ld0.diam_bin == '20-40').sum())} 20-40 mm lesions; {both} of them were also "
        f"missed by 501 (the '100' in S8's notes and in the S13 brief is this subset)")
    for fam in MAIN + EXTRA:
        ld = results[fam][1].set_index(["case_id", "lesion_id"])
        r = ld.loc[list(zip(miss.case_id, miss.lesion_id))]
        say(f"   {FAM_NAME[fam]:<12} rescued {int(r.detected_any.sum())}/{len(miss)}; their mean Dice "
            f"{r[r.detected_any == 1].lesion_dice.mean() if r.detected_any.any() else float('nan'):.3f}")
        add("missed_20_40", fam, "rescued", int(r.detected_any.sum()), n=len(miss))

    say("\n== B at fixed settings on all 1308 (descriptive, never used to choose): what rescue costs")
    say(f"{'setting':<46}{'objective':>10}{'spec':>7}{'FP/neg':>8}{'FP/pos':>8}{'rescued/116':>12}{' Dice>=0.1':>10}{'lost':>6}{'det 10-20':>10}")
    for t in K.T_LOWS[1:]:
        for k in (1, 3):
            s = K.Setting(t_low=t, k=k)
            cd, ld = run_policy(cases, ids, {c: s for c in ids}, master)
            m = metrics(cd, ld, meta)
            x = ld.set_index(["case_id", "lesion_id"])
            r = x.loc[list(zip(miss.case_id, miss.lesion_id))]
            lost = int(((ld0.set_index(["case_id", "lesion_id"]).detected_any == 1) & (x.detected_any == 0)).sum())
            say(f"{s.label():<46}{m['objective']:>10.4f}{m['spec_raw']:>7.3f}{m['fp_per_neg']:>8.3f}{m['fp_per_pos']:>8.3f}"
                f"{int(r.detected_any.sum()):>12}{int((r.lesion_dice >= 0.1).sum()):>10}{lost:>6}{m['det_10-20']:>10.3f}")
            for key, v in (("objective", m["objective"]), ("spec_raw", m["spec_raw"]), ("fp_per_neg", m["fp_per_neg"]),
                           ("missed_20_40_rescued", int(r.detected_any.sum())), ("lesions_lost", lost)):
                add("B_fixed_settings", "B", key, v, setting=s.label(), note="descriptive, all 1308, not used to choose")

    say("\n== D: volume ratio (pred voxels of touching CCs / lesion voxels) and Dice among detected lesions")
    for fam in ("baseline", "D", "BD", "ABD"):
        m = results[fam][2]
        say(f"   {FAM_NAME[fam]:<28} " + "; ".join(
            f"{b}: n {m[f'n_det_{b}']} ratio {m[f'volratio_det_{b}']:.3f} Dice|det {m[f'dice_det_{b}']:.3f}"
            for b in BINS if m[f"n_det_{b}"]))

    # what D's extra detections are: contact by a regrown mask, or real overlap?
    for fam in ("D", "ABDc"):
        ld = results[fam][1].merge(ld0[["case_id", "lesion_id", "detected_any"]], on=["case_id", "lesion_id"], suffixes=("", "_0"))
        rec = ld[(ld.detected_any == 1) & (ld.detected_any_0 == 0)]
        say(f"   {FAM_NAME[fam]}: {len(rec)} newly touched lesions; Dice >= 0.1 in {int((rec.lesion_dice >= 0.1).sum())}, "
            f"median Dice {rec.lesion_dice.median():.3f}, median volume ratio {rec.pred_vol_ratio.median():.1f}")
        add("D_new_detections", fam, "n_newly_touched", len(rec), note=f"Dice>=0.1 in {int((rec.lesion_dice >= 0.1).sum())}")

    # ---------------------------------------------------------------- why A is nearly a no-op
    say("\n== A diagnostics: how much of each lesion component already lies in the model's own pancreas (channel 0 > 0.5)")
    for t in (0.5, 0.01, 1e-4):
        fr = np.concatenate([cases[c]["comps"][t]["panc_n"][0.0] / np.maximum(cases[c]["comps"][t]["n_vox"], 1) for c in ids])
        fr5 = np.concatenate([cases[c]["comps"][t]["panc_n"][5.0] / np.maximum(cases[c]["comps"][t]["n_vox"], 1) for c in ids])
        say(f"   components of p > {t:g} (n={fr.size}): inside channel 0 median {np.median(fr):.3f}; < 50% inside: "
            f"{int((fr < 0.5).sum())}; none inside: {int((fr == 0).sum())}; < 50% within 5 mm: {int((fr5 < 0.5).sum())}")
        add("A_diagnostics", "A", "components_below_half_in_channel0", int((fr < 0.5).sum()), stratum=f"t={t:g}", n=fr.size)
    lp = pd.DataFrame([dict(case_id=c, lesion_id=j + 1, in_panc=int(v), gt_vox_a=int(cases[c]["les_vox"][j]))
                       for c in ids for j, v in enumerate(cases[c]["les_in_panc"])])
    lp = lp.merge(ld0[["case_id", "lesion_id", "gt_vox", "tier", "detected_any"]], on=["case_id", "lesion_id"], validate="1:1")
    assert (lp.gt_vox == lp.gt_vox_a).all()
    lp["frac"] = lp.in_panc / lp.gt_vox
    for t in TIERS:
        g = lp[lp.tier == t]
        say(f"   GT lesions, tier {t:<9} n={len(g):>4}: median fraction inside the predicted pancreas {g.frac.median():.3f}; "
            f"zero overlap {int((g.in_panc == 0).sum())}; among S8-detected (n={int(g.detected_any.sum())}) median "
            f"{g[g.detected_any == 1].frac.median():.3f}")
        add("A_diagnostics", "A", "gt_lesion_median_frac_in_channel0", g.frac.median(), stratum=f"tier={t}", n=len(g))

    # ---------------------------------------------------------------- per-fold held-out deltas
    say("\n== Per held-out fold: objective delta vs S8 (training-run variance is visible only here)")
    for fam in MAIN + EXTRA:
        p = results[fam][3]
        say(f"   {FAM_NAME[fam]:<12} " + "  ".join(f"f{k} {p[k]['test'] - p[k]['test_base']:+.4f}" for k in range(5)))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ROWS).to_csv(OUT / "lofo_results.csv", index=False, float_format="%.6g")
    say(f"\nwrote {OUT / 'lofo_results.csv'} ({len(ROWS)} rows)")
    (WORK / "summary.txt").write_text("\n".join(LINES) + "\n")
    with open(WORK / "policies.pkl", "wb") as f:
        pickle.dump({fam: (r[0], r[1], r[3]) for fam, r in results.items()}, f)


if __name__ == "__main__":
    sys.exit(main())
