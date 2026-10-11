"""S15 Part C (PC): gland shape profile on the 1308 cohort cases, scored against the GT lesions.

STATUS: EXPLORATORY ("tool failed its insertion check"). Round 5's outcome rule (SESSION_15.md): the v5 core rerun
failed the iso noise-free bend gate and the 20 mm insertion median was 87.8 (< 95), so these results cannot pass a
gate and are not used for the case-level classifier. Every definition below was fixed before this run.

Per case (out-of-fold 502 outline from the S13 sparse cache; GT used only for evaluation):
  gland QC (glandshape.qc), aorta test, centerline, profile with HU -> ~/research/gland_profiles/cohort/{case}.npz;
  scores: v5 PRIMARY = multi-scale bulge_local on raw width, extrapolated tip stops excluded; SECONDARY = residual
  (width minus the bank reference median); case-level max primary score; case atrophy (whole-side, z vs the bank
  reference; reported only); top-3 peaks (partc.top_peaks).
  Echo map: per (stop, axis) cell, the max 502 lesion probability (sparse cache, p > 1e-4, else 0) along either half
  of the axis's in-gland diameter (0.25 mm steps, nearest native voxel); per-case Spearman rho with the primary score.
Per GT lesion (tumorlib labels from the sparse cache; gt_vox asserted equal to S14's table):
  footprint (footprint.py; v5 tip-excluded, plus the pre-v5 version for the round-5 item-4 counts); not evaluable if
  no footprint cell; within-scan percentile (partc a); top-3 hit (partc b); chance = max(uniform exact, empirical
  from the 49 held-out bank glands' v5 top-3 peaks) (partc c); fold-heavy, censored footprint, fragmented gland.

Groups (S14 definitions, read from work/rescue_check_502/all_lesions.csv, not redefined):
  target    = missed (touch_raw 0) 20-40 mm index lesions, panc_category included, tier not separated (48)
  secondary = the same at 10-20 mm (46)
  reference = detected (touch_raw 1) 20-40 mm lesions, panc_category included (461)
  control   = detected >= 40 mm lesions (253)
Gates (brief; evaluated for the record only, EXPLORATORY): control median percentile >= 90; target median >= 80 AND
top-3 hit >= 2 x chance AND >= 25%.

Usage: PYTHONPATH=src .venv/bin/python -m glandshape.partc_run [--workers 4] [--skip_pass]
"""
from __future__ import annotations

import argparse
import contextlib
import io as sio
import json
import multiprocessing as mp
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from glandshape import centerline as C
from glandshape import core as K
from glandshape import footprint as FP
from glandshape import orient as OR
from glandshape import partc as PC
from glandshape import profile as P
from glandshape import qc as QC
from glandshape import qc_cohort as QCC
from glandshape import scores as S
from tumorlib import io as tio

REPO = Path(__file__).resolve().parents[2]
OUTW = REPO / "work/gland_shape/partC"
RES = REPO / "results/gland_shape"
PROF_C = Path.home() / "research/gland_profiles/cohort"
REF_FILE = Path.home() / "research/gland_profiles/bank_reference.npz"
PEAKS = REPO / "work/gland_shape/partB/heldout_peaks_v5.json"
S14 = REPO / "work/rescue_check_502/all_lesions.csv"
LABEL = "EXPLORATORY (tool failed its insertion check)"
SEG = np.arange(0.0, K.RAY_MAX_MM + 1e-9, 0.25)
B_BOOT, SEED = 2000, 0


def load_reference() -> S.Reference:
    with np.load(REF_FILE) as d:
        return S.Reference(d["w_med"], d["w_mad"], d["atro_med"], d["atro_mad"], int(d["n_glands"]),
                           d["atw_med"], d["atw_mad"]), float(d["atrophy_thr"])


def cell_lesion_prob(ol, cl, pr, p_idx, p_val, shape) -> np.ndarray:
    """(n, 12) max 502 lesion probability along either half of each axis's in-gland diameter."""
    out = np.full((len(cl.s), P.N_AXES), np.nan)
    vi = np.nonzero(pr.valid)[0]
    if not len(vi):
        return out
    order = np.argsort(p_idx)
    pi, pv = p_idx[order].astype(np.int64), p_val[order]
    dirs = P.ray_dirs(cl)[vi]
    pts = pr.origin[vi, None, None, :] + SEG[None, None, :, None] * dirs[:, :, None, :]
    within = SEG[None, None, :] <= np.nan_to_num(pr.r[vi], nan=-1.0)[:, :, None]
    nat = np.rint(ol.iso_to_native(pts.reshape(-1, 3))).astype(np.int64)
    ok = np.all((nat >= 0) & (nat < np.array(shape)), axis=1) & within.reshape(-1)
    val = np.zeros(len(nat))
    if ok.any() and len(pi):
        flat = np.ravel_multi_index(nat[ok].T, shape)
        j = np.clip(np.searchsorted(pi, flat), 0, len(pi) - 1)
        val[ok] = np.where(pi[j] == flat, pv[j], 0.0)
    m = np.where(within.reshape(-1), val, 0.0).reshape(len(vi), K.N_DIRS, len(SEG)).max(-1)
    out[vi] = np.maximum(m[:, :P.N_AXES], m[:, P.N_AXES:])
    return out


def one(cid: str, peaks: list, s14_vox: dict) -> dict:
    t0 = time.time()
    ref, thr = load_reference()
    row = {"case_id": cid}
    les_rows = []
    with np.load(QCC.SPARSE / f"{cid}.npz") as d:
        shape = tuple(int(v) for v in d["shape"])
        gt_idx, gt_lab, p_idx, p_val = d["gt_idx"], d["gt_lab"], d["p_idx"], d["p_val"]
    lids = sorted(int(v) for v in np.unique(gt_lab))
    lesions = {}
    for lid in lids:
        sel = gt_idx[gt_lab == lid]
        assert len(sel) == s14_vox[(cid, lid)], (cid, lid)          # join check (gotcha 2)
        lesions[lid] = np.stack(np.unravel_index(sel.astype(np.int64), shape), 1)
    try:
        ol, g = QCC.load_outline(cid)
        drop = QC.outline_drop(ol)
        row.update(fragmented=bool(ol.fragmented) if ol else False, vol_cm3=ol.vol_mm3 / 1000 if ol else np.nan)
        cl = pr = None
        if drop is None:
            with contextlib.redirect_stdout(sio.StringIO()):
                aorta = tio.load_mask(cid, "aorta")
            at = OR.aorta_test(ol, aorta)
            del aorta
            row.update(aorta_call=at["aorta_call"])
            try:
                cl = C.build_centerline(ol, OR.anterior_world(at))
                drop = QC.centerline_drop(ol, cl.L)
            except Exception as e:
                drop = f"centerline_error: {type(e).__name__}"
        if drop is None:
            try:
                ct = P.make_ct_sampler(ol, g[0], lambda sl: tio.load_ct(cid, sl))
                pr = P.compute_profile(ol, cl, ct_sampler=ct)
            except Exception as e:
                drop = f"profile_error: {type(e).__name__}"
        row["gland_drop"] = drop or ""
        if drop:
            for lid in lids:
                les_rows.append(dict(case_id=cid, lesion_id=lid, evaluable=False, not_evaluable=f"gland dropped: {drop}"))
            return dict(row=row, lesions=les_rows)

        PROF_C.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(PROF_C / f"{cid}.npz", s=pr.s, u=pr.u, r=pr.r.astype(np.float32), w=pr.w.astype(np.float32),
                            A=pr.A, valid=pr.valid, fold=pr.fold, censored=pr.censored, w_censored=pr.w_censored,
                            tip=pr.tip, origin=pr.origin, origin_fallback=pr.origin_fallback, hu=pr.hu, L=pr.L,
                            P=cl.P, T=cl.T, N=cl.N, B=cl.B, offset=ol.offset)
        sc = S.primary_scores(pr, ref)
        prim, resid = sc["primary"], sc["residual"]
        z = S.atrophy_z(pr.A, pr.u, ref)
        amag, asgn, _ = S.case_atrophy(z)
        lp = cell_lesion_prob(ol, cl, pr, p_idx, p_val, shape)
        both = np.isfinite(prim) & np.isfinite(lp)
        rho = spearmanr(prim[both], lp[both])[0] if both.sum() > 10 and np.ptp(lp[both]) > 0 else np.nan
        pk_p, pk_r = PC.top_peaks(prim, pr.s), PC.top_peaks(resid, pr.s)
        row.update(L_mm=pr.L, n_stops=len(pr.s), n_tip=int(pr.tip.sum()), n_fold=int(pr.fold.sum()),
                   max_primary=float(np.nanmax(prim)), max_residual=float(np.nanmax(resid)),
                   atrophy_mag=amag, atrophy_signed=asgn, atrophy_flag=bool(amag > thr and asgn < 0),
                   echo_rho=rho, top_u=float(pr.u[pk_p[0][0]]) if pk_p else np.nan)
        scoreable = FP.scoreable(pr)
        for lid, xyz in lesions.items():
            fp_full = FP.footprint(ol, cl, pr, xyz, exclude_tip=False)
            fp = fp_full & scoreable
            r = dict(case_id=cid, lesion_id=lid, n_fp_cells_full=int(fp_full.sum()), n_fp_cells=int(fp.sum()),
                     evaluable=bool(fp.any()), not_evaluable="" if fp.any() else
                     ("no footprint cell" if not fp_full.any() else "footprint only on tip stops"))
            if fp.any():
                st = FP.footprint_stops(fp)
                a, b = PC.within_scan_percentile(prim, fp), PC.within_scan_percentile(resid, fp)
                cu = PC.chance_top3_uniform(fp, scoreable, pr.s)
                ce = PC.chance_top3_empirical(fp, pr.u, peaks)
                r.update(n_fp_stops=len(st), u_lo=float(pr.u[st].min()), u_hi=float(pr.u[st].max()),
                         fold_heavy=FP.fold_heavy(fp, pr), censored_fp=FP.censored_footprint(fp, pr),
                         pct_primary=a["percentile"], n_windows=a["n_windows"], win_stops=a["win_stops"],
                         win_axes=a["win_axes"], lesion_score=a["lesion_score"], pct_residual=b["percentile"],
                         top3_primary=any(fp[i, j] for i, j in pk_p), top3_residual=any(fp[i, j] for i, j in pk_r),
                         chance_uniform=cu, chance_empirical=ce, chance_gate=PC.gate_chance(cu, ce),
                         fp_lesion_prob_max=float(np.nanmax(lp[fp])) if np.isfinite(lp[fp]).any() else np.nan)
            les_rows.append(r)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        for lid in lids:
            les_rows.append(dict(case_id=cid, lesion_id=lid, evaluable=False, not_evaluable=f"error: {e}"))
    row["secs"] = time.time() - t0
    return dict(row=row, lesions=les_rows)


# ------------------------------------------------------------------ report helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot_median(x, rng):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    m = np.median(rng.choice(x, (B_BOOT, len(x)), replace=True), axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def gstats(d: pd.DataFrame, name: str, rng, score="primary") -> dict:
    ev = d[d.evaluable.astype(bool)]
    pct = ev[f"pct_{score}"]
    hit = ev[f"top3_{score}"].astype(bool)
    k, n = int(hit.sum()), len(ev)
    lo, hi = wilson(k, n)
    ch = ev.chance_gate.mean()
    blo, bhi = boot_median(pct, rng)
    return dict(group=name, score=score, n=len(d), n_eval=n, pct_median=float(pct.median()) if n else np.nan,
                pct_ci=f"{blo:.1f}-{bhi:.1f}", top3=k / n if n else np.nan, top3_ci=f"{lo:.0%}-{hi:.0%}",
                chance=ch, chance_uniform=ev.chance_uniform.mean(), chance_emp=ev.chance_empirical.mean(),
                ratio=(k / n) / ch if n and ch > 0 else np.nan)


def fmt(s: dict) -> str:
    return (f"| {s['group']} | {s['n']} | {s['n_eval']} | {s['pct_median']:.1f} ({s['pct_ci']}) | "
            f"{s['top3']:.0%} ({s['top3_ci']}) | {s['chance']:.0%} (unif {s['chance_uniform']:.0%}, emp "
            f"{s['chance_emp']:.0%}) | {s['ratio']:.2f} |")


def auc_boot(y, x, rng):
    from sklearn.metrics import roc_auc_score
    y, x = np.asarray(y), np.asarray(x, float)
    ok = np.isfinite(x)
    y, x = y[ok], x[ok]
    a = roc_auc_score(y, x)
    bs = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            bs.append(roc_auc_score(y[i], x[i]))
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip_pass", action="store_true")
    a = ap.parse_args()
    OUTW.mkdir(parents=True, exist_ok=True)
    s14 = pd.read_csv(S14)
    vox = {(r.case_id, int(r.lesion_id)): int(r.gt_vox) for r in s14.itertuples()}
    peaks = list(json.loads(PEAKS.read_text()).values())
    assert len(peaks) == 49, len(peaks)
    t0 = time.time()
    if not a.skip_pass:
        ids = tio.case_ids()
        with mp.Pool(a.workers) as pool:
            res = pool.map(partial(one, peaks=peaks, s14_vox=vox), ids, chunksize=2)
        pd.DataFrame([r["row"] for r in res]).to_csv(OUTW / "cases.csv", index=False)
        pd.DataFrame([x for r in res for x in r["lesions"]]).to_csv(OUTW / "lesions.csv", index=False)
    cases = pd.read_csv(OUTW / "cases.csv")
    les = pd.read_csv(OUTW / "lesions.csv")
    report(cases, les, s14, time.time() - t0)


def report(cases, les, s14, secs):
    rng = np.random.default_rng(SEED)
    d = s14.merge(les, on=["case_id", "lesion_id"], how="left", validate="1:1")
    assert len(d) == 1235 and d.evaluable.notna().all()
    d["evaluable"] = d.evaluable.astype(bool)
    d = d.merge(cases[["case_id", "fragmented", "gland_drop"]], on="case_id", how="left")
    d["fragmented"] = d.fragmented.fillna(False).astype(bool)
    for c in ("fold_heavy", "censored_fp", "top3_primary", "top3_residual"):
        d[c] = d[c].fillna(False).astype(bool)
    d["enhanced"] = d.est_phase != "non-contrast"
    d["iso"] = d.attenuation_v3 == "iso"
    d["target"] = ((d.diam_bin == "20-40") & (d.touch_raw == 0) & d.largest_in_case.astype(bool)
                   & (d.panc_category == "included") & (d.tier != "separated"))
    d["secondary"] = ((d.diam_bin == "10-20") & (d.touch_raw == 0) & d.largest_in_case.astype(bool)
                      & (d.panc_category == "included") & (d.tier != "separated"))
    d["reference"] = (d.diam_bin == "20-40") & (d.touch_raw == 1) & (d.panc_category == "included")
    d["control"] = (d.diam_bin == ">=40") & (d.touch_raw == 1)
    assert (d.target.sum(), d.secondary.sum(), d.reference.sum(), d.control.sum()) == (48, 46, 461, 253)
    d["status"] = LABEL
    keep = ["case_id", "lesion_id", "fold", "gt_vox", "diam_mm", "diam_bin", "tier", "satellite", "touch_raw",
            "est_phase", "phase5", "enhanced", "attenuation_v3", "iso", "target", "secondary", "reference", "control",
            "gland_drop", "fragmented", "evaluable", "not_evaluable", "n_fp_cells", "n_fp_cells_full", "n_fp_stops",
            "u_lo", "u_hi", "fold_heavy", "censored_fp", "pct_primary", "top3_primary", "chance_uniform",
            "chance_empirical", "chance_gate", "lesion_score", "n_windows", "pct_residual", "top3_residual",
            "fp_lesion_prob_max", "status"]
    RES.mkdir(parents=True, exist_ok=True)
    d[keep].to_csv(RES / "per_lesion_shape.csv", index=False)
    d.to_csv(OUTW / "lesions_joined.csv", index=False)

    L = [f"S15 Part C — {LABEL}", f"run {secs:.0f} s; cases {len(cases)}; errors {int(cases.get('error', pd.Series(dtype=str)).notna().sum())}",
         f"gland drops: {cases.gland_drop.fillna('').replace('', 'kept').value_counts().to_dict()}"]
    kept = cases[cases.gland_drop.fillna("") == ""]
    L.append(f"kept glands {len(kept)}: tip stops median {kept.n_tip.median():.0f}, fold stops median {kept.n_fold.median():.0f}")
    # round-5 item 4 counts
    for g in ("target", "secondary"):
        x = d[d[g]]
        was = x[x.n_fp_cells_full.fillna(0) > 0]
        lost = was[was.n_fp_cells < was.n_fp_cells_full]
        allg = was[was.n_fp_cells == 0]
        L.append(f"round-5 item 4, {g} (n={len(x)}): evaluable before tip exclusion {len(was)}; lose >= 1 footprint "
                 f"cell to the tip exclusion {len(lost)}; lose all (become not evaluable) {len(allg)}; evaluable now "
                 f"{int(x.evaluable.sum())}")
    # not evaluable by tier
    ne = d[np.logical_not(d.evaluable)]
    L.append("not evaluable by tier (all 1235 lesions): " + str(pd.crosstab(ne.not_evaluable.str.slice(0, 40), ne.tier).to_dict()))
    L.append("evaluable share by tier: " + str((d.groupby("tier").evaluable.mean().round(3)).to_dict()))

    rows = []
    T = d[d.target]
    groups = [("TARGET 20-40 missed, included (S14)", T),
              ("  target, no fold-heavy", T[np.logical_not(T.fold_heavy)]),
              ("  target, no fragmented gland", T[np.logical_not(T.fragmented)]),
              ("  target, no censored footprint", T[np.logical_not(T.censored_fp)]),
              ("  target, no flag at all", T[np.logical_not(T.fold_heavy | T.fragmented | T.censored_fp)]),
              ("  target, enhanced", T[T.enhanced]), ("  target, non-contrast", T[np.logical_not(T.enhanced)]),
              ("  target, iso", T[T.iso]), ("  target, not iso", T[np.logical_not(T.iso)]),
              ("  target, enhanced and not iso", T[T.enhanced & np.logical_not(T.iso)]),
              ("SECONDARY 10-20 missed, included", d[d.secondary]),
              ("REFERENCE 20-40 detected, included (461)", d[d.reference]),
              ("CONTROL >= 40 detected (253)", d[d.control])]
    for name, g in groups:
        rows.append(gstats(g, name, rng))
    for name, g in (groups[0], groups[10], groups[11], groups[12]):
        rows.append(gstats(g, name + " [residual, secondary]", rng, score="residual"))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTW / "group_table.csv", index=False)
    L.append("| group | n | evaluable | median percentile (95% CI) | top-3 hit (Wilson CI) | chance used (larger) | "
             "hit / chance |\n|---|---|---|---|---|---|---|")
    L += [fmt(r) for r in rows]
    t0, c0 = rows[0], rows[12]
    L.append(f"GATES as written (for the record; {LABEL} -> cannot pass): control median {c0['pct_median']:.1f} vs >= 90 "
             f"[{'met' if c0['pct_median'] >= 90 else 'not met'}]; target median {t0['pct_median']:.1f} vs >= 80 "
             f"[{'met' if t0['pct_median'] >= 80 else 'not met'}], top-3 {t0['top3']:.0%} vs >= 2 x chance "
             f"{2 * t0['chance']:.0%} [{'met' if t0['top3'] >= 2 * t0['chance'] else 'not met'}] and >= 25% "
             f"[{'met' if t0['top3'] >= 0.25 else 'not met'}]")
    # echo
    pos = cases[cases.case_id.isin(d.case_id)]
    L.append(f"echo: per-case Spearman rho (primary score vs 502 lesion probability, all scoreable cells), tumor+ kept "
             f"glands median {pos.echo_rho.median():.3f} (IQR {pos.echo_rho.quantile(.25):.3f}-{pos.echo_rho.quantile(.75):.3f}, "
             f"n={int(pos.echo_rho.notna().sum())})")
    for gname in ("target", "reference", "control"):
        x = d[d[gname] & d.evaluable]
        r = spearmanr(x.lesion_score, x.fp_lesion_prob_max, nan_policy="omit")[0]
        L.append(f"  {gname}: lesion score vs max 502 lesion probability in the footprint, Spearman {r:.3f} (n={len(x)}); "
                 f"footprint lesion prob median {x.fp_lesion_prob_max.median():.3g}")
    # case level
    meta = pd.read_csv(REPO / "results/PanTS_metadata_new.csv", usecols=["PanTS ID", "tumor?"]).rename(
        columns={"PanTS ID": "case_id", "tumor?": "tumor"})
    ph = pd.read_csv(REPO / "results/phase_estimate/phase_estimate.csv", usecols=["case_id", "slice_thickness_mm"])
    k = kept.merge(meta, on="case_id").merge(ph, on="case_id", how="left")
    L.append(f"case level (reported only; kept glands tumor+ {int(k.tumor.sum())}, tumor- {int((k.tumor == 0).sum())}; "
             f"slice median {k[k.tumor == 1].slice_thickness_mm.median():.2f} vs {k[k.tumor == 0].slice_thickness_mm.median():.2f} mm):")
    for col in ("max_primary", "max_residual", "atrophy_mag"):
        a_, lo, hi, n = auc_boot(k.tumor, k[col], rng)
        line = f"  AUC tumor+ vs tumor- by {col}: {a_:.3f} (95% CI {lo:.3f}-{hi:.3f}, n={n})"
        for lab, m in (("<2.5 mm", k.slice_thickness_mm < 2.5), (">=2.5 mm", k.slice_thickness_mm >= 2.5)):
            kk = k[m]
            if kk.tumor.nunique() == 2:
                a2, l2, h2, n2 = auc_boot(kk.tumor, kk[col], rng)
                line += f"; {lab}: {a2:.3f} ({l2:.3f}-{h2:.3f}, n={n2}, tumor- {int((kk.tumor == 0).sum())})"
        L.append(line)
    L.append(f"  atrophy flagged (|z| > threshold, z < 0): tumor+ {k[k.tumor == 1].atrophy_flag.mean():.1%}, "
             f"tumor- {k[k.tumor == 0].atrophy_flag.mean():.1%}")
    txt = "\n".join(L)
    (OUTW / "partc_summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
