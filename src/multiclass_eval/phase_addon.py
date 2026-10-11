"""S8b: phase and negative-matching add-on to S8. Reads only S8's existing per-case pickles
(work/nnunet_multiclass/cases/{501,502}/), S8's CSVs, results/phase_estimate/phase_estimate.csv,
results/attenuation_v3/attenuation_labels_v3.csv and the master table. The case pass is NOT rerun and no npz is read.

One extra input: the GT lesion mask (tumorlib.load_mask, PC, read-only), used for the item-3 test "component
peak > 10 mm from every GT lesion". The pickles hold each component's peak voxel and which lesions it overlaps,
but no lesion geometry.

Items (CONTEXT / S8 session brief):
  1  five phase strata, 501 vs 502 Dice and detect@any at 10-20 and 20-40 mm, S8's paired case bootstrap.
     Already in S8: non-contrast / venous / uncertain / unmeasurable and pooled arterial. They are recomputed
     only as a regression check (asserted equal to S8's rows, then labelled source = S8). New: early and late
     arterial.
  2  v3 iso fraction by estimated phase, metadata phase side by side (S3's metadata table is the regression target)
  3  negatives reweighted to the tumor+ joint distribution of estimated phase x thickness bin. Also FP components
     per case on positives' non-tumor region (component peak > 10 mm from every GT lesion).

Frozen before any result was looked at:
  phase strata  est_phase from S12; arterial split by early_arterial (pv_enh_frac < phase.core.EARLY_ART_F = 0.5,
                re-derived and asserted); "unmeasurable" (no vessel HU) is kept as its own level, not one of the five
  thickness     <2.5, 2.5-<5, >=5 mm (S8's bins), from phase_estimate.slice_thickness_mm (asserted == master)
  reweighting   cell = phase x thickness. Each negative in cell c gets weight P_pos(c | covered) / n_neg(c), so the
                weights sum to 1 and the weighted cell mix equals the tumor+ CASE mix over cells that have at
                least one negative. Tumor+ cells with no negative are reported as uncovered; their mass is dropped,
                not imputed. ESS = (sum w)^2 / sum w^2.
                primary strata = S12 est_phase (3 classes + uncertain + unmeasurable); secondary = the 5 strata
                of item 1 (+ unmeasurable)
  bootstrap     B = 1000. Item 1: S8's exact W (stats.case_weights over the lesion set's tumor+ cases, seed 0).
                Item 3: positives and negatives resampled independently (seed 0 and seed 1); weights recomputed in
                every replicate, so the CI includes weight uncertainty
  "far" FP      a 26-connected component of the saved lesion segmentation (S2/S8 t = 0.5 rung) that touches no GT
                lesion and whose peak-probability voxel lies > 10 mm (Euclidean, mm, voxel centres) from every GT
                lesion voxel. Touching-nothing components with the peak <= 10 mm are counted separately as "near"
  n < 20        strata with fewer than 20 lesions (or cells with fewer than 20 cases) are flagged and not interpreted

Usage (PC, repo root): PYTHONPATH=src .venv/bin/python -m multiclass_eval.phase_addon [--workers 4]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from multiclass_eval import stats as S

REPO = Path(__file__).resolve().parents[2]
CASES = REPO / "work" / "nnunet_multiclass" / "cases"
WORK = REPO / "work" / "nnunet_multiclass_phase"
OUT = REPO / "results" / "nnunet_multiclass_phase"
MODELS = ("501", "502")
TAG = {"501": "baseline", "502": "multiclass"}
FAR_MM = 10.0
MIN_N = 20
THICK_EDGES = [0, 2.5, 5.0, np.inf]
THICK_LABELS = ["<2.5", "2.5-5", ">=5"]
PHASE5 = ("non-contrast", "early arterial", "late arterial", "venous", "uncertain")
FOCUS = ("10-20", "20-40")
LINES: list[str] = []
ROWS: list[dict] = []


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


def add(item, section, metric, lesion_set="", stratum_var="", stratum="", n=np.nan, n_cases=np.nan, value=np.nan,
        baseline=np.nan, multiclass=np.nan, delta=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_boot=np.nan, source="S8b",
        note=""):
    ROWS.append(dict(item=item, section=section, metric=metric, lesion_set=lesion_set, stratum_var=stratum_var,
                     stratum=stratum, n=n, n_cases=n_cases, value=value, baseline=baseline, multiclass=multiclass,
                     delta=delta, ci_lo=ci_lo, ci_hi=ci_hi, p_boot=p_boot, source=source, note=note))


# ------------------------------------------------------------------ pure functions (tested)
def phase_stratum(est_phase: pd.Series, early_arterial: pd.Series) -> pd.Series:
    """S12 est_phase with arterial split into early / late; other labels unchanged."""
    out = est_phase.astype(str).copy()
    art = est_phase == "arterial"
    out[art & early_arterial.astype(bool)] = "early arterial"
    out[art & ~early_arterial.astype(bool)] = "late arterial"
    return out


def thickness_bin(mm: pd.Series) -> pd.Series:
    return pd.cut(mm, THICK_EDGES, right=False, labels=THICK_LABELS).astype(str)


def cell_weights(pos_cells: pd.Series, neg_cells: pd.Series) -> tuple[np.ndarray, pd.DataFrame]:
    """Per-negative weights (sum 1) matching the negatives' cell mix to the positives' mix over covered cells.
    Returns (weights aligned with neg_cells, cell table with n_pos, n_neg, covered, w_per_neg)."""
    npos = pos_cells.value_counts()
    nneg = neg_cells.value_counts()
    cells = sorted(set(npos.index) | set(nneg.index))
    t = pd.DataFrame({"n_pos": npos.reindex(cells, fill_value=0), "n_neg": nneg.reindex(cells, fill_value=0)})
    t["covered"] = (t.n_pos > 0) & (t.n_neg > 0)
    mass = t.n_pos.where(t.covered, 0)
    t["p_pos_covered"] = mass / mass.sum()
    t["w_per_neg"] = np.where(t.n_neg > 0, t.p_pos_covered / t.n_neg.clip(lower=1), np.nan)
    w = neg_cells.map(t.w_per_neg).fillna(0.0).to_numpy(float)
    return w, t


def ess(w: np.ndarray) -> float:
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if (w ** 2).sum() > 0 else 0.0


def weighted_bootstrap(pos_cells: np.ndarray, neg_cells: np.ndarray, y: dict[str, np.ndarray], b: int = S.B,
                       seed: int = S.SEED) -> dict[str, np.ndarray]:
    """Replicates of the reweighted negative mean for each outcome in y (arrays aligned with neg_cells).
    Positives (seed) and negatives (seed + 1) are resampled independently; the weights are recomputed per
    replicate from the resampled cell masses (cells that lose all negatives become uncovered)."""
    cells = np.array(sorted(set(pos_cells) | set(neg_cells)))
    P = (pos_cells[:, None] == cells[None, :]).astype(float)          # n_pos x C
    N = (neg_cells[:, None] == cells[None, :]).astype(float)          # n_neg x C
    Wp = S.case_weights(len(pos_cells), b, seed)                        # b x n_pos
    Wn = S.case_weights(len(neg_cells), b, seed + 1)                    # b x n_neg
    pos_mass = Wp @ P                                                   # b x C
    neg_cnt = Wn @ N                                                    # b x C
    cov = (pos_mass > 0) & (neg_cnt > 0)
    pm = np.where(cov, pos_mass, 0.0)
    pm /= pm.sum(1, keepdims=True)
    per = np.divide(pm, neg_cnt, out=np.zeros_like(pm), where=neg_cnt > 0)   # b x C, weight per drawn negative
    wneg = Wn * (per @ N.T)                                             # b x n_neg
    return {k: wneg @ np.asarray(v, float) for k, v in y.items()}


def peak_distances_mm(peaks_vox: np.ndarray, lesion_mask: np.ndarray, spacing) -> np.ndarray:
    """Euclidean distance (mm, voxel centres) from each peak voxel to the nearest lesion voxel."""
    if len(peaks_vox) == 0:
        return np.zeros(0)
    sp = np.asarray(spacing, float)
    tree = cKDTree(np.argwhere(lesion_mask) * sp)
    return tree.query(np.asarray(peaks_vox, float) * sp)[0]


# ------------------------------------------------------------------ IO helpers
def load_pickles() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases, comps, les = [], [], []
    for m in MODELS:
        for f in sorted((CASES / m).glob("*.pkl")):
            with open(f, "rb") as fh:
                r = pickle.load(fh)
            c = r["case"]
            cases.append(dict(model=m, case_id=c["case_id"], fold=c["fold"], gt_tumor=c["gt_tumor"],
                              pred_vol_mm3=r["ev_case"]["pred_vol_mm3"]))
            comps.extend({"model": m, **x} for x in r["comps"] if x["t"] == 0.5)
            les.extend({"model": m, "case_id": c["case_id"], "lesion_id": x["lesion_id"], "gt_vox": x["gt_vox"],
                        "lesion_dice": x["lesion_dice"], "detected_any": x["detected_any"]} for x in r["ev_lesions"])
    return pd.DataFrame(cases), pd.DataFrame(comps), pd.DataFrame(les)


def _dist_job(job):
    case_id, peaks = job
    from tumorlib import io as tio
    from tumorlib.lesions import label_lesions
    les = tio.load_mask(case_id, "pancreatic_lesion")
    sp = tio.spacing(case_id)
    if les is None or sp is None:
        return case_id, None, None, None
    n = label_lesions(les)[1]
    return case_id, peak_distances_mm(np.asarray(peaks), les.astype(bool), sp), int(les.sum()), n


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    from phase.core import EARLY_ART_F
    from rescue_check.core import wilson

    master = pd.read_csv(REPO / "results/master_lesion_table/master_lesions.csv")
    ph = pd.read_csv(REPO / "results/phase_estimate/phase_estimate.csv")
    plc = pd.read_csv(REPO / "results/nnunet_multiclass/per_lesion_comparison.csv")
    s8 = pd.read_csv(REPO / "results/nnunet_multiclass/summary_vs_baseline.csv")
    att = pd.read_csv(REPO / "results/attenuation_v3/attenuation_labels_v3.csv")
    cases, comps, les = load_pickles()

    say("== Inputs and checks")
    assert len(ph) == 1308 and ph.case_id.is_unique
    early = (ph.est_phase == "arterial") & (ph.pv_enh_frac < EARLY_ART_F)
    assert (early == ph.early_arterial.astype(bool)).all(), "early_arterial != (arterial & pv_enh_frac < 0.5)"
    ph["phase5"] = phase_stratum(ph.est_phase, ph.early_arterial)
    ph["thick"] = thickness_bin(ph.slice_thickness_mm)
    mt = master.drop_duplicates("case_id").set_index("case_id").slice_thickness_mm
    pt = ph.set_index("case_id").loc[mt.index, "slice_thickness_mm"]
    assert np.allclose(pt, mt, rtol=0, atol=1e-3), "thickness != master"          # CSV rounding only (max 2.4e-4 mm)
    assert (thickness_bin(pt).to_numpy() == thickness_bin(mt).to_numpy()).all(), "thickness bin != master's"
    say(f"phase_estimate: 1308 cases; early_arterial == (arterial & pv_enh_frac < {EARLY_ART_F}) for all; "
        f"thickness == master for the 981 tumor+; strata {ph.phase5.value_counts().to_dict()}")
    for m in MODELS:
        cm = cases[cases.model == m]
        assert len(cm) == 1308
        lm = les[les.model == m].merge(plc[["case_id", "lesion_id", "gt_vox", f"dice_{TAG[m]}", f"det_{TAG[m]}"]],
                                       on=["case_id", "lesion_id"], validate="1:1", suffixes=("", "_s8"))
        assert len(lm) == 1235 and (lm.gt_vox == lm.gt_vox_s8).all()
        assert np.allclose(lm.lesion_dice, lm[f"dice_{TAG[m]}"], atol=1e-5) and (lm.detected_any == lm[f"det_{TAG[m]}"]).all()
    say("pickles: 1308 cases x 2 models; per-lesion Dice / detect@any == S8 per_lesion_comparison.csv for all 1235")
    pl = plc.merge(ph[["case_id", "phase5"]], on="case_id", validate="m:1")
    assert (pl.est_phase == ph.set_index("case_id").loc[pl.case_id, "est_phase"].to_numpy()).all()

    # ------------------------------------------------------------ item 1
    say("\n== Item 1: phase strata x 10-20 / 20-40 mm, paired case bootstrap (S8 W, seed 0)")
    for lset in ("all", "excl_separated"):
        sub = pl if lset == "all" else pl[~pl.excluded]
        cids = np.array(sorted(sub.case_id.unique()))
        W = S.case_weights(len(cids))                 # identical to S8's W for this lesion set
        say(f"-- lesion set {lset}")
        for fb in FOCUS:
            g0 = sub[sub.diam_bin == fb]
            # regression: S8's est_phase strata reproduce exactly
            for val in sorted(g0.est_phase.astype(str).unique()):
                g = g0[g0.est_phase == val]
                for metric, a, b in (("mean_lesion_dice", "dice_baseline", "dice_multiclass"),
                                     ("detect_any_rate", "det_baseline", "det_multiclass")):
                    r = S.paired_ratio(g, "case_id", a, b, cids, W)
                    ref = s8[(s8.section == "paired_lesion") & (s8.lesion_set == lset) & (s8.metric == metric)
                             & (s8.stratum_var == f"diam_bin={fb} x est_phase") & (s8.stratum == f"{fb} | {val}")]
                    assert len(ref) == 1, (lset, fb, val, metric)
                    for k in ("a", "b", "ci_lo", "ci_hi"):
                        col = {"a": "baseline", "b": "multiclass"}.get(k, k)
                        assert np.isclose(r[k], ref[col].iloc[0], rtol=0, atol=1e-5), (lset, fb, val, metric, k)  # S8 CSV is %.6g
            say(f"   {fb}: S8's est_phase rows reproduced exactly (same W)")
            say(f"   {fb:>5} {'stratum':<15} {'n':>4} {'cases':>5} | {'Dice 501->502':>15} {'delta [CI]':>24} | "
                f"{'det 501->502':>15} {'delta [CI]':>24} | rec/lost")
            for val in (*PHASE5, "unmeasurable"):
                g = g0[g0.phase5 == val]
                if g.empty:
                    say(f"   {fb:>5} {val:<15}    0")
                    continue
                src = "S8" if val in ("non-contrast", "venous", "uncertain", "unmeasurable") else "S8b"
                flag = f"n<{MIN_N}: not interpreted" if len(g) < MIN_N else ""
                if val == "unmeasurable":
                    flag = ("not one of the five strata; " + flag).strip("; ")
                rd = S.paired_ratio(g, "case_id", "dice_baseline", "dice_multiclass", cids, W)
                rt = S.paired_ratio(g, "case_id", "det_baseline", "det_multiclass", cids, W)
                ch = g.change.value_counts()
                note = "; ".join(x for x in (flag, f"recovered {ch.get('recovered', 0)}, lost {ch.get('lost', 0)}") if x)
                for metric, r in (("mean_lesion_dice", rd), ("detect_any_rate", rt)):
                    add(1, "phase_strata", metric, lset, f"diam_bin={fb} x phase5", f"{fb} | {val}", n=r["n"],
                        n_cases=r["n_cases"], baseline=r["a"], multiclass=r["b"], delta=r["delta"], ci_lo=r["ci_lo"],
                        ci_hi=r["ci_hi"], p_boot=r["p_boot"], source=src, note=note)
                say(f"   {fb:>5} {val:<15} {rd['n']:>4} {rd['n_cases']:>5} | {rd['a']:.3f} -> {rd['b']:.3f}   "
                    f"{rd['delta']:+.3f} [{rd['ci_lo']:+.3f}, {rd['ci_hi']:+.3f}] | {rt['a']:6.1%} -> {rt['b']:6.1%}  "
                    f"{rt['delta']:+6.1%} [{rt['ci_lo']:+6.1%}, {rt['ci_hi']:+6.1%}] | {ch.get('recovered', 0)}/{ch.get('lost', 0)}"
                    f"{'   (' + flag + ')' if flag else ''}  [{src}]")

    # ------------------------------------------------------------ item 2
    say("\n== Item 2: v3 iso fraction by phase (Wilson 95% CI)")
    a2 = master.merge(att[["case_id", "lesion_id", "gt_vox", "attenuation_v3"]], on=["case_id", "lesion_id"],
                      validate="1:1", suffixes=("", "_att")).merge(ph[["case_id", "est_phase", "phase5"]], on="case_id")
    assert len(a2) == 1235 and (a2.gt_vox == a2.gt_vox_att).all() and a2.attenuation_v3.notna().all()
    a2["iso"] = a2.attenuation_v3 == "iso"
    a2["ct_phase"] = a2.ct_phase.fillna("missing")
    s3 = {("Venous", "all"): (557, 0.303), ("Arterial", "all"): (260, 0.342), ("Non-contrast", "all"): (413, 0.487),
          ("Delay", "all"): (4, 0.5), ("Venous", ">=10"): (543, 0.308), ("Arterial", ">=10"): (246, 0.337),
          ("Non-contrast", ">=10"): (355, 0.524)}
    for size, sub in (("all", a2), (">=10", a2[a2.diam_mm >= 10])):
        say(f"-- lesions {size} mm (n={len(sub)}), iso overall {sub.iso.mean():.1%}")
        for var, order in (("metadata ct_phase", ("Non-contrast", "Arterial", "Venous", "Delay", "missing")),
                           ("est_phase (S12)", ("non-contrast", "arterial", "venous", "uncertain", "unmeasurable")),
                           ("phase5", (*PHASE5, "unmeasurable"))):
            col = {"metadata ct_phase": "ct_phase", "est_phase (S12)": "est_phase", "phase5": "phase5"}[var]
            parts = []
            for val in order:
                g = sub[sub[col] == val]
                if g.empty:
                    continue
                k, n = int(g.iso.sum()), len(g)
                lo, hi = wilson(k, n)
                if col == "ct_phase" and (val, size) in s3:      # S3 regression
                    en, ef = s3[(val, size)]
                    assert n == en and round(k / n, 3) == ef, (val, size, n, k / n)
                flag = f"n<{MIN_N}: not interpreted" if n < MIN_N else ""
                add(2, "iso_by_phase", "iso_fraction", f"diam {size} mm", var, val, n=n, value=k / n, ci_lo=lo, ci_hi=hi,
                    source="S3 (reproduced)" if col == "ct_phase" else "S8b",
                    note="; ".join(x for x in (f"{n / len(sub):.1%} of lesions", flag) if x))
                parts.append(f"{val} {k}/{n} = {k / n:.1%} [{lo:.1%}, {hi:.1%}] ({n / len(sub):.0%} of lesions)")
            say(f"   {var:<18}: " + "; ".join(parts))
        if size == "all":
            say("   (metadata rows reproduce S3: Venous 557 30.3%, Arterial 260 34.2%, Non-contrast 413 48.7%, Delay 4 50%)")
        else:
            say("   (metadata rows reproduce S3: Venous 543 30.8%, Arterial 246 33.7%, Non-contrast 355 52.4%)")
    # where the metadata non-contrast lesions actually sit
    x = a2[a2.ct_phase == "Non-contrast"]
    t = x.groupby("est_phase").iso.agg(["size", "sum"])
    say("   metadata 'Non-contrast' lesions by est_phase: " + ", ".join(f"{i} {int(r['sum'])}/{int(r['size'])} iso" for i, r in t.iterrows()))
    for i, r in t.iterrows():
        add(2, "iso_by_phase", "iso_fraction", "diam all mm", "metadata Non-contrast x est_phase", str(i), n=int(r["size"]),
            value=r["sum"] / r["size"], note=f"n<{MIN_N}: not interpreted" if r["size"] < MIN_N else "")

    # ------------------------------------------------------------ item 3
    say("\n== Item 3: negatives reweighted to the tumor+ case mix of phase x thickness")
    cs = cases[cases.model == "501"][["case_id", "gt_tumor"]].merge(ph[["case_id", "est_phase", "phase5", "thick"]], on="case_id")
    assert len(cs) == 1308 and int((cs.gt_tumor == 0).sum()) == 327
    for m in MODELS:
        c5 = comps[comps.model == m]
        cs[f"ncomp_{m}"] = cs.case_id.map(c5.groupby("case_id").size()).fillna(0).astype(int)
        pv = cases[cases.model == m].set_index("case_id").pred_vol_mm3
        assert ((cs.case_id.map(pv) > 0) == (cs[f"ncomp_{m}"] > 0)).all(), "pred volume > 0 != has a component"
    neg = cs[cs.gt_tumor == 0].reset_index(drop=True)
    pos = cs[cs.gt_tumor == 1].reset_index(drop=True)
    # unweighted regression vs S8
    for m, (spec, fpn) in (("501", (0.2232, 2.0642)), ("502", (0.7217, 0.3303))):
        assert round((neg[f"ncomp_{m}"] == 0).mean(), 4) == spec and round(neg[f"ncomp_{m}"].mean(), 4) == fpn
    say("unweighted Sp / FP per negative reproduce S8 (501 0.2232 / 2.0642, 502 0.7217 / 0.3303)")

    y = {}
    for m in MODELS:
        y[f"spec_{m}"] = (neg[f"ncomp_{m}"] == 0).astype(float).to_numpy()
        y[f"anyvox_{m}"] = (neg[f"ncomp_{m}"] > 0).astype(float).to_numpy()
        y[f"fp_{m}"] = neg[f"ncomp_{m}"].astype(float).to_numpy()
    for scheme, pcol in (("est_phase x thickness", "est_phase"), ("phase5 x thickness", "phase5")):
        pc = pos[pcol].astype(str) + " | " + pos.thick
        nc = neg[pcol].astype(str) + " | " + neg.thick
        w, tab = cell_weights(pc, nc)
        e = ess(w)
        unc = tab[(tab.n_pos > 0) & (tab.n_neg == 0)]
        negonly = tab[(tab.n_pos == 0) & (tab.n_neg > 0)]
        say(f"-- strata {scheme}: {len(tab)} cells; ESS {e:.1f} of 327 negatives; "
            f"tumor+ cells with NO negative (cannot be reweighted, mass dropped): "
            + (", ".join(f"{i} ({int(r.n_pos)} tumor+)" for i, r in unc.iterrows()) if len(unc) else "none")
            + f" = {int(unc.n_pos.sum())}/981 tumor+ cases ({unc.n_pos.sum() / 981:.1%}); "
            f"negative-only cells (weight 0): {int(negonly.n_neg.sum())} negatives in {len(negonly)} cells")
        add(3, "reweighting", "ess", "", scheme, "", n_cases=327, value=e)
        add(3, "reweighting", "uncovered_tumor_pos_cases", "", scheme, "; ".join(unc.index), n_cases=981,
            value=int(unc.n_pos.sum()), note="tumor+ cells with zero negatives: cannot be reweighted, mass dropped")
        add(3, "reweighting", "zero_weight_negatives", "", scheme, "; ".join(negonly.index), n_cases=327,
            value=int(negonly.n_neg.sum()), note="negatives in cells with no tumor+ case")
        say(f"   {'cell':<28} {'n_pos':>5} {'n_neg':>5} {'w/neg x327':>10}")
        for i, r in tab.iterrows():
            say(f"   {i:<28} {int(r.n_pos):>5} {int(r.n_neg):>5} {r.w_per_neg * 327 if r.n_neg else float('nan'):>10.2f}")
            add(3, "reweighting_cells", "cell_counts", "", scheme, i, n=int(r.n_pos), n_cases=int(r.n_neg),
                value=r.w_per_neg * 327 if r.n_neg else np.nan, note="n = tumor+ cases, n_cases = negatives, value = weight per negative x 327")
        reps = weighted_bootstrap(pc.to_numpy(), nc.to_numpy(), y)
        for metric, key in (("specificity", "spec"), ("frac_negatives_any_lesion_voxel", "anyvox"),
                            ("fp_components_per_negative", "fp")):
            uw = {m: y[f"{key}_{m}"].mean() for m in MODELS}
            wt = {m: float(w @ y[f"{key}_{m}"]) for m in MODELS}
            ci = {m: np.percentile(reps[f"{key}_{m}"], [2.5, 97.5]) for m in MODELS}
            d = reps[f"{key}_502"] - reps[f"{key}_501"]
            dci = np.percentile(d, [2.5, 97.5])
            say(f"   {metric:<32} unweighted 501 {uw['501']:.3f} 502 {uw['502']:.3f} | weighted 501 {wt['501']:.3f} "
                f"[{ci['501'][0]:.3f}, {ci['501'][1]:.3f}]  502 {wt['502']:.3f} [{ci['502'][0]:.3f}, {ci['502'][1]:.3f}]  "
                f"delta {wt['502'] - wt['501']:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]")
            add(3, "negatives", metric, "", scheme, "unweighted", n_cases=327, baseline=uw["501"], multiclass=uw["502"],
                delta=uw["502"] - uw["501"], source="S8")
            add(3, "negatives", metric, "", scheme, "weighted to tumor+ mix", n_cases=327, baseline=wt["501"],
                multiclass=wt["502"], delta=wt["502"] - wt["501"], ci_lo=dci[0], ci_hi=dci[1],
                note=f"ESS {e:.1f}; per-model CI 501 [{ci['501'][0]:.4f}, {ci['501'][1]:.4f}], 502 [{ci['502'][0]:.4f}, {ci['502'][1]:.4f}]; "
                     f"CI on delta; uncovered tumor+ {int(unc.n_pos.sum())}")

    # ---- positives' non-tumor region
    say("-- FP components on positives' non-tumor region (peak > 10 mm from every GT lesion), saved segmentation")
    pc5 = comps[(comps.case_id.isin(set(pos.case_id))) & ((comps.touch_ids == "") | comps.touch_ids.isna())].copy()
    jobs = [(c, g[["peak_x", "peak_y", "peak_z"]].drop_duplicates().to_numpy().tolist()) for c, g in pc5.groupby("case_id")]
    with mp.Pool(args.workers) as pool:
        res = pool.map(_dist_job, jobs, chunksize=4)
    vox = master.groupby("case_id").gt_vox.sum()
    nles = master.groupby("case_id").size()
    dist = {}
    for (c, peaks), (c2, d, nv, nl) in zip(jobs, res):
        assert c == c2 and d is not None, c
        assert nv == vox[c] and nl == nles[c], f"{c}: lesion mask {nv} vox / {nl} lesions vs master {vox[c]} / {nles[c]}"
        for p, dd in zip(peaks, d):
            dist[(c, *p)] = dd
    pc5["dist_mm"] = [dist[(c, x, y_, z)] for c, x, y_, z in zip(pc5.case_id, pc5.peak_x, pc5.peak_y, pc5.peak_z)]
    assert (pc5.dist_mm > 0).all(), "a non-touching component's peak sits on a lesion voxel"
    say(f"   lesion masks read for {len(jobs)} tumor+ cases with a non-touching component; voxel and lesion counts == master")
    for m in MODELS:
        g = pc5[pc5.model == m]
        pos[f"far_{m}"] = pos.case_id.map(g[g.dist_mm > FAR_MM].groupby("case_id").size()).fillna(0).astype(float)
        pos[f"near_{m}"] = pos.case_id.map(g[g.dist_mm <= FAR_MM].groupby("case_id").size()).fillna(0).astype(float)
        assert int((pos[f"far_{m}"] + pos[f"near_{m}"]).sum()) == len(g)
    Wp = S.case_weights(len(pos))
    cid = pos.case_id.to_numpy()
    for metric, a, b in (("far_fp_components_per_tumor_pos_case", "far_501", "far_502"),
                         ("near_fp_components_per_tumor_pos_case", "near_501", "near_502")):
        r = S.paired_ratio(pos, "case_id", a, b, cid, Wp)
        say(f"   {metric}: 501 {r['a']:.3f} -> 502 {r['b']:.3f}  delta {r['delta']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] (981 cases)")
        add(3, "positives_non_tumor", metric, "", "all", "all", n_cases=981, baseline=r["a"], multiclass=r["b"],
            delta=r["delta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p_boot=r["p_boot"],
            note="peak > 10 mm from every GT lesion" if metric.startswith("far") else "touches no lesion, peak <= 10 mm")
    pos["any_far_501"], pos["any_far_502"] = (pos.far_501 > 0).astype(float), (pos.far_502 > 0).astype(float)
    r = S.paired_ratio(pos, "case_id", "any_far_501", "any_far_502", cid, Wp)
    say(f"   tumor+ cases with >=1 far FP component: 501 {r['a']:.1%} -> 502 {r['b']:.1%}")
    add(3, "positives_non_tumor", "frac_tumor_pos_cases_with_far_fp", "", "all", "all", n_cases=981, baseline=r["a"],
        multiclass=r["b"], delta=r["delta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p_boot=r["p_boot"])
    say(f"   by stratum (FP comps per case; far = positives' non-tumor region, neg = negatives, unweighted within cell):")
    say(f"   {'cell':<28} {'n_pos':>5} {'far 501':>8} {'far 502':>8} | {'n_neg':>5} {'neg 501':>8} {'neg 502':>8}")
    for scheme, pcol in (("est_phase x thickness", "est_phase"), ("phase5 x thickness", "phase5")):
        pos["cell"] = pos[pcol].astype(str) + " | " + pos.thick
        neg["cell"] = neg[pcol].astype(str) + " | " + neg.thick
        if scheme == "phase5 x thickness":
            say(f"   ({scheme})")
        for cell in sorted(set(pos.cell) | set(neg.cell)):
            gp, gn = pos[pos.cell == cell], neg[neg.cell == cell]
            flag = "" if len(gp) >= MIN_N and len(gn) >= MIN_N else f"a side has <{MIN_N} cases: not interpreted"
            rp = S.paired_ratio(gp, "case_id", "far_501", "far_502", cid, Wp) if len(gp) else None
            if rp is not None:
                add(3, "positives_non_tumor_by_cell", "far_fp_components_per_case", "", scheme, cell, n_cases=len(gp),
                    baseline=rp["a"], multiclass=rp["b"], delta=rp["delta"], ci_lo=rp["ci_lo"], ci_hi=rp["ci_hi"],
                    p_boot=rp["p_boot"], note=flag)
            if len(gn):
                add(3, "negatives_by_cell", "fp_components_per_negative", "", scheme, cell, n_cases=len(gn),
                    baseline=gn.ncomp_501.mean(), multiclass=gn.ncomp_502.mean(),
                    delta=gn.ncomp_502.mean() - gn.ncomp_501.mean(), note=flag)
                add(3, "negatives_by_cell", "specificity", "", scheme, cell, n_cases=len(gn),
                    baseline=(gn.ncomp_501 == 0).mean(), multiclass=(gn.ncomp_502 == 0).mean(),
                    delta=(gn.ncomp_502 == 0).mean() - (gn.ncomp_501 == 0).mean(), note=flag)
            say(f"   {cell:<28} {len(gp):>5} {gp.far_501.mean() if len(gp) else float('nan'):>8.2f} "
                f"{gp.far_502.mean() if len(gp) else float('nan'):>8.2f} | {len(gn):>5} "
                f"{gn.ncomp_501.mean() if len(gn) else float('nan'):>8.2f} {gn.ncomp_502.mean() if len(gn) else float('nan'):>8.2f}"
                f"{'   *' if flag else ''}")
    say("   (* = a side has fewer than 20 cases: not interpreted)")

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ROWS).to_csv(OUT / "phase_addon.csv", index=False, float_format="%.6g")
    say(f"\nwrote {OUT / 'phase_addon.csv'} ({len(ROWS)} rows)")
    (WORK / "summary.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
