"""PHASES 2-4 driver: fit on the bank, score the cohort, oracles, evaluation, deliverables.

Subcommands (run in this order; each caches its output in work/normative/):
  fit         normal models (r = 6, 10 mm) on the 300-case bank, and on nested 100 / 200-case subsets
  score       score every cohort case (A, B, C_A, C_B per radius) with the 300-case model
  insertion   PHASE 3b: score the planted-sphere runs, compare with S4's LoG on the same CTs
  report      PHASE 3a/3c + PHASE 4 + gates -> results/normative_pilot/ and work/normative/summary.txt

Design (frozen 2026-10-02, before any cohort lesion was scored):
  * Lesion percentile: lesion score = max patch score over lattice centres in the lesion dilated 2 mm,
    at the patch radius closest to the lesion's equivalent-sphere radius (tie -> 6 mm). Compared with
    N_RANDOM_SETS random same-footprint sets from the same search region, all centres > 10 mm from
    every lesion. A same-footprint set = the n far centres nearest a random far anchor, n = the
    lesion's own centre count (features.footprint_sets). percentile = 100 * P(random set < lesion);
    S4's "beaten by" = 1 - percentile/100.
  * Candidates: lattice local maxima of the per-radius anomaly map, 5 mm apart, top 50 per case.
    Hit = candidate centre inside the lesion dilated 2 mm. False = > 10 mm from every lesion.
    FP denominator headline = source (b), positives' search region > 10 mm from any lesion.
  * Headline map for the 10-20 mm group = r 6 mm (equivalent radius ~5-8 mm); >= 20 mm uses r 10 mm.
  * Score C (within-case z) is (to within leave-one-out rounding) a monotone per-case transform of A
    or B, so within-scan percentiles and top-N-per-case lists cannot differ from A / B. It only
    changes operating points defined by a GLOBAL threshold. Reported, not hidden.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from normative import common as C
from normative import features as FT
from normative import model as M
from normative.case_pass import OUT as FEAT_DIR, bank_ids, load
from tumorlib import io as tio

MODELS = C.WORK / "models"
SCORES = C.WORK / "scores"
SCORE_KEYS = ("A", "B", "CA", "CB")
LINES: list[str] = []
_MODELS: dict = {}
_META: dict = {}


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1.0 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def matched_radius(r_eq: float) -> float:
    return min(C.RADII_MM, key=lambda r: (abs(r - r_eq), r))


def phase_of() -> dict[str, str]:
    m = C.metadata()
    return dict(zip(m.case_id, m["ct phase"]))


# ------------------------------------------------------------------ fit
def bank_table(ids: list[str]) -> dict[float, pd.DataFrame]:
    ph = phase_of()
    out = {r: [] for r in C.RADII_MM}
    for c in ids:
        rec = load(FEAT_DIR["bank"] / f"{c}.pkl.gz")
        if not rec["case"].get("sr_ok") or "F" not in rec:
            continue
        for r in C.RADII_MM:
            df = pd.DataFrame(rec["F"][r], columns=FT.feature_names(r))
            df["case_id"] = c
            df["phase_group"] = C.phase_group(ph[c])
            df["thick_group"] = C.thick_group(rec["case"]["thick"])
            df["u"] = rec["u"]
            out[r].append(df)
    return {r: pd.concat(v, ignore_index=True) for r, v in out.items()}


def bank_subsets() -> dict[int, list[str]]:
    ids = sorted(bank_ids())
    perm = list(np.random.default_rng(C.SEED).permutation(ids))
    return {n: sorted(perm[:n]) for n in (*C.BANK_SIZE_CURVE, len(ids))}


def cmd_fit() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    subsets = bank_subsets()
    full = bank_table(subsets[max(subsets)])
    for n, ids in subsets.items():
        for r in C.RADII_MM:
            bank = full[r][full[r].case_id.isin(set(ids))]
            t0 = time.time()
            mdl = M.fit(bank, FT.feature_names(r), r)
            with open(MODELS / f"model_n{n}_r{r:g}.pkl", "wb") as f:
                pickle.dump(mdl, f, protocol=pickle.HIGHEST_PROTOCOL)
            rep = mdl.report
            say(f"model n={n} r={r:g}: {rep['n_fit']} bank patches from {bank.case_id.nunique()} cases "
                f"({rep['n_dropped_unknown']} without u/phase dropped), PCA {rep['n_components']} comps "
                f"({rep['explained']:.3f} var), u tercile edges {tuple(round(e, 3) for e in mdl.u_edges)}, "
                f"{len(mdl.strata)} strata, short {rep['short_strata']}  [{time.time() - t0:.0f}s]")
            if n == max(subsets):
                say("   strata sizes: " + ", ".join(f"{k} {v}" for k, v in rep["strata_sizes"].items()))
                say("   fine cells : " + ", ".join(f"{k} {v}" for k, v in rep["fine_counts"].items()))
    (C.WORK / "fit_log.txt").write_text("\n".join(LINES) + "\n")


def load_models(n: int) -> dict[float, M.NormalModel]:
    out = {}
    for r in C.RADII_MM:
        with open(MODELS / f"model_n{n}_r{r:g}.pkl", "rb") as f:
            out[r] = pickle.load(f)
    return out


# ------------------------------------------------------------------ score
def _init(n: int) -> None:
    _MODELS.update(load_models(n))
    _META["phase"] = phase_of()


def score_record(rec: dict, centres_F: dict, u: np.ndarray, case_id: str, thick: float) -> dict:
    ph = C.phase_group(_META["phase"].get(case_id))
    th = C.thick_group(thick)
    out = {}
    for r in C.RADII_MM:
        a, b, lab = _MODELS[r].score(centres_F[r], ph, th, u)
        out[r] = dict(A=a.astype(np.float32), B=b.astype(np.float32),
                      CA=M.within_case_z(a).astype(np.float32), CB=M.within_case_z(b).astype(np.float32))
        out[f"strata_{r:g}"] = pd.Series(lab).value_counts().to_dict()
    out["phase_group"], out["thick_group"] = ph, th
    return out


def score_job(job: tuple[str, int]) -> tuple[str, str]:
    case_id, n = job
    dst = SCORES / f"n{n}" / f"{case_id}.pkl"
    if dst.exists():
        return case_id, "cached"
    rec = load(FEAT_DIR["cohort"] / f"{case_id}.pkl.gz")
    meta = rec["case"]
    res = dict(case_id=case_id, sr_ok=bool(meta.get("sr_ok")), thick=meta["thick"])
    if res["sr_ok"]:
        res.update(score_record(rec, rec["F"], rec["u"], case_id, meta["thick"]))
        res.update(centres=rec["centres"], dil_id=rec["dil_id"], dist_les=rec["dist_les"], u=rec["u"],
                   dist=rec["dist"], lesions=rec["lesions"], n_lesions=rec["n_lesions"],
                   sr_vol_cm3=meta["sr_vol_cm3"], sr_vol_far_cm3=rec["sr_vol_far_cm3"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    return case_id, "ok"


def cmd_score(n: int = C.N_BANK, ids: list[str] | None = None) -> None:
    ids = ids or tio.case_ids()
    assert not set(ids) & set(bank_ids())
    t0 = time.time()
    with mp.Pool(C.NUM_WORKERS, initializer=_init, initargs=(n,)) as p:
        for i, (c, s) in enumerate(p.imap_unordered(score_job, [(c, n) for c in ids], chunksize=4), 1):
            if i % 100 == 0 or i == len(ids):
                print(f"  scored {i}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)


def load_scores(n: int = C.N_BANK, ids=None) -> dict[str, dict]:
    out = {}
    for f in sorted((SCORES / f"n{n}").glob("*.pkl")):
        if ids is not None and f.stem not in ids:
            continue
        with open(f, "rb") as fh:
            out[f.stem] = pickle.load(fh)
    return out


# ------------------------------------------------------------------ lesion percentiles
def lesion_percentiles(sc: dict, rng_seed: int) -> list[dict]:
    """One row per lesion of a scored cohort case (all score keys, both radii, matched radius)."""
    rows = []
    cen = sc["centres"].astype(float)
    far = np.flatnonzero(sc["dist_les"] > C.FP_EXCL_MM)
    tree = cKDTree(cen[far]) if len(far) >= 10 else None
    for L in sc["lesions"]:
        j = L["lesion_id"]
        foot = np.flatnonzero(sc["dil_id"] == j)
        rm = matched_radius(L["r_eq_mm"])
        row = dict(case_id=sc["case_id"], lesion_id=j, r_eq_mm=L["r_eq_mm"], matched_radius_mm=rm,
                   n_centres_dil2=len(foot), n_far_centres=len(far), u_centroid_s9=L["u_centroid"],
                   in_sr_frac_s9=L["in_sr_frac"], gt_vox_s9=L["gt_vox"])
        rng = np.random.default_rng(rng_seed + j)
        sets = FT.footprint_sets(cen[far], len(foot), rng, tree=tree) if tree is not None and len(foot) else None
        for r in C.RADII_MM:
            for k in SCORE_KEYS:
                s = sc[r][k]
                if sets is None:
                    p, f = np.nan, np.nan
                else:
                    p, f = FT.set_percentile(float(s[foot].max()), s[far][sets].max(1))
                row[f"pct_{k}_r{r:g}"], row[f"fge_{k}_r{r:g}"] = p, f
        for k in SCORE_KEYS:
            row[f"pct_{k}"] = row[f"pct_{k}_r{rm:g}"]
            row[f"fge_{k}"] = row[f"fge_{k}_r{rm:g}"]
        rows.append(row)
    return rows


# ------------------------------------------------------------------ candidates
def candidates(sc: dict) -> list[dict]:
    rows = []
    cen = sc["centres"].astype(int)
    for r in C.RADII_MM:
        for k in SCORE_KEYS:
            s = sc[r][k].astype(np.float64)
            idx = FT.grid_local_maxima(cen, s)
            for rank, i in enumerate(idx, 1):
                rows.append(dict(case_id=sc["case_id"], radius=r, score=k, rank=rank, value=float(s[i]),
                                 dil_id=int(sc["dil_id"][i]), dist_les=float(sc["dist_les"][i]),
                                 is_false=bool(sc["dist_les"][i] > C.FP_EXCL_MM),
                                 **{f"d_{n}": float(sc["dist"][n][i]) for n in C.CONTEXT_STRUCTURES}))
    return rows


def _eval_case(path: Path) -> tuple[list, list, dict]:
    with open(path, "rb") as f:
        sc = pickle.load(f)
    case = dict(case_id=sc["case_id"], sr_ok=sc["sr_ok"], thick=sc["thick"])
    if not sc["sr_ok"]:
        return [], [], case
    case.update(n_lesions=sc["n_lesions"], sr_vol_cm3=sc["sr_vol_cm3"], sr_vol_far_cm3=sc["sr_vol_far_cm3"],
                phase_group=sc["phase_group"], thick_group=sc["thick_group"], n_centres=len(sc["centres"]))
    far = sc["dist_les"] > C.FP_EXCL_MM
    for r in C.RADII_MM:
        for k in ("A", "B"):
            case[f"mean_{k}_r{r:g}"] = float(np.mean(sc[r][k]))
            case[f"mean_far_{k}_r{r:g}"] = float(np.mean(sc[r][k][far])) if far.any() else np.nan
        case[f"strata_r{r:g}"] = ";".join(f"{a}:{b}" for a, b in sc[f"strata_{r:g}"].items())
    les = lesion_percentiles(sc, C.SEED * 1000 + int(sc["case_id"][-8:]) * 10)
    return les, candidates(sc), case


# ------------------------------------------------------------------ calibration (3a)
def _calib_case(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        sc = pickle.load(f)
    if not sc["sr_ok"] or sc["n_lesions"] > 0:
        return []
    rng = np.random.default_rng(C.SEED + int(sc["case_id"][-8:]))
    cen = sc["centres"].astype(float)
    rows = []
    # (i) the literal check: one random patch vs every other patch of the same pancreas
    i = int(rng.integers(len(cen)))
    for r in C.RADII_MM:
        for k in ("A", "B"):
            s = sc[r][k]
            rows.append(dict(case_id=sc["case_id"], test="single_patch_vs_all", n_foot=1, radius=r, score=k,
                             pct=100.0 * float((np.delete(s, i) < s[i]).mean())))
    # (ii) the full lesion protocol on a pseudo-lesion (compact random footprint)
    for n_foot in (1, 5, 20, 60):
        if len(cen) < n_foot + 50:
            continue
        les = FT.footprint_sets(cen, n_foot, rng, n_sets=1)[0]
        d = cKDTree(cen[les]).query(cen, k=1)[0]
        far = np.flatnonzero(d > C.FP_EXCL_MM)
        sets = FT.footprint_sets(cen[far], n_foot, rng)
        if sets is None:
            continue
        for r in C.RADII_MM:
            for k in SCORE_KEYS:
                s = sc[r][k]
                p, _ = FT.set_percentile(float(s[les].max()), s[far][sets].max(1))
                rows.append(dict(case_id=sc["case_id"], test="pseudo_lesion_protocol", n_foot=n_foot,
                                 radius=r, score=k, pct=p))
    return rows


def calibration_summary(cal: pd.DataFrame) -> None:
    say("\n== 3a. Calibration on held-out cohort NEGATIVES (pseudo-lesions; expected: uniform 0-100)")
    say("   'single_patch_vs_all' is uniform by exchangeability (a sanity check of the bookkeeping);")
    say("   'pseudo_lesion_protocol' runs the real protocol (exclusion zone + same-footprint random sets).")
    say(f"   {'test':<24} {'n_foot':>6} {'r':>4} {'score':>5} {'n':>4} {'mean':>6} {'p10':>6} {'p50':>6} "
        f"{'p90':>6} {'>=90':>6} {'>=95':>6} {'KS D':>6} {'KS p':>7}")
    for (t, nf, r, k), g in cal.groupby(["test", "n_foot", "radius", "score"]):
        p = g.pct.dropna().to_numpy()
        ks = stats.kstest(p / 100.0, "uniform")
        say(f"   {t:<24} {nf:>6} {r:>4g} {k:>5} {len(p):>4} {p.mean():>6.1f} {np.percentile(p, 10):>6.1f} "
            f"{np.median(p):>6.1f} {np.percentile(p, 90):>6.1f} {(p >= 90).mean():>6.1%} {(p >= 95).mean():>6.1%} "
            f"{ks.statistic:>6.3f} {ks.pvalue:>7.3f}")


# ------------------------------------------------------------------ insertion (3b)
def _insert_case(path: Path) -> list[dict]:
    rec = load(path)
    cid = rec["case"]["case_id"]
    rows = []
    for run in rec["runs"]:
        R = run["diam"] / 2.0
        assert np.array_equal(run["centres"], rec["runs"][0]["centres"])     # planting never moves the region
        sc = score_record(rec, run["F"], rec["u"], cid, rec["case"]["thick"])
        cen = run["centres"].astype(float)
        foot = np.flatnonzero(run["dist_centre"] <= R + C.DIL_MM)
        far = np.flatnonzero(run["dist_centre"] > R + C.FP_EXCL_MM)
        rng = np.random.default_rng(C.SEED + int(cid[-8:]))          # same sets for every condition
        sets = FT.footprint_sets(cen[far], len(foot), rng) if len(foot) else None
        base = dict(case_id=cid, diam_mm=run["diam"], dHU=run["dhu"], n_foot=len(foot),
                    in_sr_frac=run["in_sr_frac"], matched_radius_mm=matched_radius(R),
                    thick=rec["case"]["thick"])
        for r in C.RADII_MM:
            for k in SCORE_KEYS:
                s = sc[r][k]
                p, f = (FT.set_percentile(float(s[foot].max()), s[far][sets].max(1)) if sets is not None
                        else (np.nan, np.nan))
                rows.append(dict(base, method=f"{k}_r{r:g}", score=k, radius=r,
                                 is_matched=r == matched_radius(R), percentile=p, frac_ge=f))
        pol = "bright" if run["dhu"] > 0 else "dark"
        f = run["s4"][f"patch_frac_ge_{pol}"]
        rows.append(dict(base, method=f"S4_LoG_{pol}", score="S4", radius=np.nan, is_matched=True,
                         percentile=100.0 * (1.0 - f) if f is not None and np.isfinite(f) else np.nan,
                         frac_ge=f))
    return rows


def cmd_insertion() -> None:
    files = sorted(FEAT_DIR["insert"].glob("*.pkl.gz"))
    with mp.Pool(C.NUM_WORKERS, initializer=_init, initargs=(C.N_BANK,)) as p:
        rows = [r for rs in p.map(_insert_case, files, chunksize=1) for r in rs]
    df = pd.DataFrame(rows)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.RESULTS / "insertion_test.csv", index=False, float_format="%.6g")
    say(f"insertion: {df.case_id.nunique()} cases, {len(df)} rows -> {C.RESULTS / 'insertion_test.csv'}")
    insertion_summary(df)
    (C.WORK / "insertion_summary.txt").write_text("\n".join(LINES) + "\n")


def insertion_summary(df: pd.DataFrame) -> None:
    say("\n== 3b. Insertion test: percentile of the planted sphere (median [frac >= 90]) at the matched radius")
    say("   rows = condition; dHU 0 = no sphere planted (null control, expect ~50 / ~10%)")
    d = df[df.is_matched]
    meth = ["A", "B", "CA", "CB", "S4"]
    say(f"   {'diam':>5} {'dHU':>5} {'n':>4} " + " ".join(f"{m:>14}" for m in meth))
    for (dm, h), g in d.groupby(["diam_mm", "dHU"]):
        cells = []
        for m in meth:
            p = g[g.score == m].percentile.dropna()
            cells.append(f"{p.median():>6.1f} [{(p >= 90).mean():>4.0%}]" if len(p) else f"{'-':>14}")
        say(f"   {dm:>5g} {h:>5g} {g.case_id.nunique():>4} " + " ".join(f"{c:>14}" for c in cells))


# ------------------------------------------------------------------ strata
def lesion_table(les: pd.DataFrame) -> pd.DataFrame:
    master = pd.read_csv(C.MASTER)
    a3 = pd.read_csv(C.ATTEN3)[["case_id", "lesion_id", "gt_vox", "dHU_core_global", "attenuation_v3", "cnr"]]
    s2 = pd.read_csv(C.S2_LESIONS)[["case_id", "lesion_id", "missed"]].rename(columns={"missed": "nnunet_missed"})
    s4 = pd.read_csv(C.S4_LESIONS)[["case_id", "lesion_id", "patch_frac_ge_dark", "rank_dark"]].rename(
        columns={"patch_frac_ge_dark": "s4_frac_ge_dark", "rank_dark": "s4_rank_dark"})
    t = master.merge(les, on=["case_id", "lesion_id"], how="left", validate="1:1", indicator=True)
    say(f"lesions joining the S9 pass: {(t._merge == 'both').sum()}/{len(t)}")
    assert (t._merge == "both").all()
    t = t.drop(columns="_merge")
    assert (t.gt_vox == t.gt_vox_s9).all(), "gt_vox mismatch vs master"
    du = (t.u - t.u_centroid_s9).abs()
    say(f"u regression vs master (S1/lesion-sectioning): max |diff| {du.max():.2e} over {du.notna().sum()} lesions "
        f"(NaN in master {t.u.isna().sum()}, in S9 {t.u_centroid_s9.isna().sum()})")
    t = t.merge(a3.rename(columns={"gt_vox": "gt_vox_a3"}), on=["case_id", "lesion_id"], validate="1:1")
    assert (t.gt_vox == t.gt_vox_a3).all()
    t = t.merge(s2, on=["case_id", "lesion_id"], validate="1:1").merge(s4, on=["case_id", "lesion_id"], validate="1:1")
    t["thick_bin"] = C.thick_bin(t.slice_thickness_mm).values
    t["enhanced"] = t.ct_phase.isin(C.ENHANCED)
    hh = t.attenuation_v3.isin(["hypo", "hyper"])
    t["target"] = (t.diam_bin == "10-20") & t.enhanced & hh
    t["target_20_40"] = (t.diam_bin == "20-40") & t.enhanced & hh
    t["target_ge40"] = (t.diam_bin == ">=40") & t.enhanced & hh
    return t.drop(columns=["gt_vox_s9", "gt_vox_a3"])


def strata(t: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    out = [("TARGET 10-20mm hypo/hyper enhanced", t.target),
           ("nnU-Net missed, TARGET", t.target & t.nnunet_missed),
           ("nnU-Net detected, TARGET", t.target & ~t.nnunet_missed),
           ("20-40mm hypo/hyper enhanced", t.target_20_40),
           (">=40mm hypo/hyper enhanced", t.target_ge40),
           ("10-20mm all", t.diam_bin == "10-20"),
           ("10-20mm enhanced", (t.diam_bin == "10-20") & t.enhanced),
           ("10-20mm iso(v3)", (t.diam_bin == "10-20") & (t.attenuation_v3 == "iso")),
           ("nnU-Net missed, 10-20mm", t.nnunet_missed & (t.diam_bin == "10-20")),
           ("<10mm all", t.diam_mm < 10), (">=20mm all", t.diam_mm >= 20),
           ("all lesions", pd.Series(True, index=t.index)), ("all, not separated", ~t.excluded)]
    for b in ("<5", "5-10", "10-20", "20-40", ">=40"):
        out.append((f"size {b}", t.diam_bin == b))
    for c in ("hypo", "iso", "hyper"):
        out.append((f"v3 {c}", t.attenuation_v3 == c))
        out.append((f"10-20mm v3 {c}", (t.attenuation_v3 == c) & (t.diam_bin == "10-20")))
    for p in sorted(t.ct_phase.dropna().unique()):
        out.append((f"phase {p}", t.ct_phase == p))
        out.append((f"10-20mm phase {p}", (t.ct_phase == p) & (t.diam_bin == "10-20")))
    for r in ("head", "body", "tail"):
        out.append((f"region {r}", t.region == r))
        out.append((f"10-20mm region {r}", (t.region == r) & (t.diam_bin == "10-20")))
    for b in C.THICK_LABELS:
        out.append((f"thickness {b} mm", t.thick_bin == b))
        out.append((f"10-20mm thickness {b} mm", (t.thick_bin == b) & (t.diam_bin == "10-20")))
    for tier in sorted(t.tier.dropna().unique()):
        out.append((f"tier {tier}", t.tier == tier))
    out += [("nnU-Net detected", ~t.nnunet_missed), ("nnU-Net missed", t.nnunet_missed)]
    return out


# ------------------------------------------------------------------ FROC
def froc(t: pd.DataFrame, cand: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    ok = cases[cases.sr_ok]
    pos, neg = ok[ok.n_lesions > 0], ok[ok.n_lesions == 0]
    st = strata(t)
    keys = list(zip(t.case_id, t.lesion_id))
    rows = []
    for (r, k), cd in cand.groupby(["radius", "score"]):
        if k.startswith("C"):
            ladder = [("z", z) for z in C.Z_GRID]
        else:
            qs = (0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
            ladder = [("value_q", (q, float(np.quantile(cd.value, q)))) for q in qs]
        ops = [("top_n", n) for n in C.TOP_N_GRID] + ladder
        for kind, v in ops:
            if kind == "top_n":
                sel = cd[cd["rank"] <= v]
                opv, thr = float(v), np.nan
            elif kind == "z":
                sel = cd[cd.value >= v]
                opv, thr = float(v), float(v)
            else:
                sel = cd[cd.value >= v[1]]
                opv, thr = float(v[0]), v[1]
            h = sel[sel.dil_id > 0]
            hit = set(zip(h.case_id, h.dil_id.astype(int)))
            hv = np.array([kk in hit for kk in keys])
            f = sel[sel.is_false]
            nb = f[f.case_id.isin(set(pos.case_id))].groupby("case_id").size().reindex(pos.case_id, fill_value=0)
            na = f[f.case_id.isin(set(neg.case_id))].groupby("case_id").size().reindex(neg.case_id, fill_value=0)
            fp = {"b_positives_nontumor": (float(nb.mean()), 100.0 * nb.sum() / pos.sr_vol_far_cm3.sum(), len(pos)),
                  "a_negatives": (float(na.mean()), 100.0 * na.sum() / neg.sr_vol_cm3.sum(), len(neg))}
            for src, (pc, p100, nc) in fp.items():
                for name, m in st:
                    m = m.to_numpy()
                    n, kh = int(m.sum()), int(hv[m].sum())
                    lo, hi = wilson(kh, n)
                    rows.append(dict(score=k, radius_mm=r, op_type=kind, op_value=opv, threshold=thr,
                                     fp_source=src, stratum=name, n_lesions=n, n_hit=kh,
                                     sensitivity=kh / n if n else np.nan, sens_ci_lo=lo, sens_ci_hi=hi,
                                     fp_per_case=pc, fp_per_100cm3=p100, n_fp_cases=nc))
    return pd.DataFrame(rows)


def best_rank(t: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    """Rank of the best in-lesion candidate per lesion, per (score, radius) and at the matched radius."""
    h = cand[cand.dil_id > 0].groupby(["case_id", "dil_id", "radius", "score"])["rank"].min().reset_index()
    h = h.rename(columns={"dil_id": "lesion_id"})
    for (r, k), g in h.groupby(["radius", "score"]):
        t = t.merge(g[["case_id", "lesion_id", "rank"]].rename(columns={"rank": f"cand_rank_{k}_r{r:g}"}),
                    on=["case_id", "lesion_id"], how="left", validate="1:1")
    for k in SCORE_KEYS:
        t[f"cand_rank_{k}"] = np.where(t.matched_radius_mm == 6.0, t[f"cand_rank_{k}_r6"], t[f"cand_rank_{k}_r10"])
    return t


# ------------------------------------------------------------------ report
def report_percentiles(t: pd.DataFrame) -> None:
    say("\n== 4a. Within-pancreas percentile of each lesion (matched radius), vs S4 on the same lesions")
    say("   S4 pct = 100 * (1 - patch_frac_ge_dark), carried over from results/log_candidates (dark polarity)")
    say(f"   {'stratum':<38} {'n':>5} {'scored':>6} " + " ".join(f"{k:>11}" for k in ("A", "B", "CA", "CB"))
        + f" {'S4 pct':>7} {'A>=90':>6} {'B>=90':>6} {'A top10':>7} {'B top10':>7}")
    for name, m in strata(t):
        s = t[m]
        if not len(s):
            continue
        sc = s.dropna(subset=["pct_A"])
        cells = [f"{sc[f'pct_{k}'].median():>5.1f}" for k in ("A", "B", "CA", "CB")]
        s4 = 100 * (1 - s.s4_frac_ge_dark.dropna())
        say(f"   {name:<38} {len(s):>5} {len(sc):>6} " + " ".join(f"{c:>11}" for c in cells)
            + f" {s4.median():>7.1f} {(sc.pct_A >= 90).mean():>6.1%} {(sc.pct_B >= 90).mean():>6.1%}"
            + f" {(s.cand_rank_A <= 10).mean():>7.1%} {(s.cand_rank_B <= 10).mean():>7.1%}")


def report_radii(t: pd.DataFrame) -> None:
    say("\n   Same, both radii shown (median percentile, no radius matching):")
    for name, m in [x for x in strata(t) if x[0].startswith(("TARGET", "nnU-Net missed, TARGET", "20-40mm hypo",
                                                               ">=40mm hypo", "size "))]:
        s = t[m].dropna(subset=["pct_A_r6"])
        say(f"   {name:<38} n={len(s):>4} " + "  ".join(
            f"{k}_r{r:g} {s[f'pct_{k}_r{r:g}'].median():>5.1f}" for k in ("A", "B") for r in C.RADII_MM))


def report_froc(fr: pd.DataFrame) -> None:
    say("\n== 4b. FROC, top-N per case, FP source (b) = positives' search region > 10 mm from any lesion")
    cols = ["TARGET 10-20mm hypo/hyper enhanced", "nnU-Net missed, TARGET", "10-20mm all",
            "20-40mm hypo/hyper enhanced", ">=40mm hypo/hyper enhanced", "all lesions"]
    short = ["TARGET", "miss∩TGT", "10-20", "20-40 hh", ">=40 hh", "all"]
    for r in C.RADII_MM:
        for k in SCORE_KEYS:
            d = fr[(fr.radius_mm == r) & (fr.score == k) & (fr.fp_source == "b_positives_nontumor")
                   & (fr.op_type == "top_n")]
            da = fr[(fr.radius_mm == r) & (fr.score == k) & (fr.fp_source == "a_negatives") & (fr.op_type == "top_n")]
            say(f"-- score {k}, r {r:g} mm   {'N':>3} {'FP/case b':>9} {'FP/case a':>9} "
                + " ".join(f"{s:>10}" for s in short))
            for n in C.TOP_N_GRID:
                g = d[d.op_value == n].set_index("stratum")
                ga = da[da.op_value == n]
                say(f"   {'':<20} {n:>3} {g.iloc[0].fp_per_case:>9.2f} {ga.iloc[0].fp_per_case:>9.2f} "
                    + " ".join(f"{g.at[c, 'sensitivity']:>10.1%}" for c in cols))
    say("\n   Global-threshold operating points (where C can differ from A/B), source (b), TARGET / 10-20 / all:")
    for r in C.RADII_MM:
        for k in SCORE_KEYS:
            d = fr[(fr.radius_mm == r) & (fr.score == k) & (fr.fp_source == "b_positives_nontumor")
                   & (fr.op_type != "top_n")]
            pts = []
            for v, g in d.groupby("op_value"):
                g = g.set_index("stratum")
                pts.append(f"{v:g}: {g.iloc[0].fp_per_case:.1f}fp "
                           f"{g.at['TARGET 10-20mm hypo/hyper enhanced', 'sensitivity']:.0%}/"
                           f"{g.at['10-20mm all', 'sensitivity']:.0%}/{g.at['all lesions', 'sensitivity']:.0%}")
            say(f"   {k} r{r:g}: " + "; ".join(pts))


def gates(t: pd.DataFrame, fr: pd.DataFrame) -> dict:
    say("\n== GATES (fixed before any cohort lesion was scored)")
    out = {}
    big = t[t.diam_bin == ">=40"].dropna(subset=["pct_A"])
    meds = {k: big[f"pct_{k}"].median() for k in SCORE_KEYS}
    pc = any(v >= 90 for v in meds.values())
    out["positive_control"] = pc
    say(f"   Positive control: >=40 mm (n={len(big)} scored) median within-pancreas percentile "
        + ", ".join(f"{k} {v:.1f}" for k, v in meds.items()) + f"  ->  {'PASS' if pc else 'FAIL'} (needs >= 90 for one score)")
    tg = t[t.target]
    for k in SCORE_KEYS:
        med = tg[f"pct_{k}"].median()
        g = fr[(fr.score == k) & (fr.radius_mm == 6.0) & (fr.op_type == "top_n") & (fr.op_value == 10)
               & (fr.fp_source == "b_positives_nontumor") & (fr.stratum == "TARGET 10-20mm hypo/hyper enhanced")].iloc[0]
        ci_ok = g.sens_ci_lo > 0.080
        p = (med >= 90) and ci_ok
        out[f"target_{k}"] = p
        km = tg[tg.nnunet_missed]
        gm = fr[(fr.score == k) & (fr.radius_mm == 6.0) & (fr.op_type == "top_n") & (fr.op_value == 10)
                & (fr.fp_source == "b_positives_nontumor") & (fr.stratum == "nnU-Net missed, TARGET")].iloc[0]
        say(f"   Target, score {k}: median pct {med:.1f} (needs >= 90); top-10 (r 6 mm map) {g.n_hit}/{g.n_lesions} = "
            f"{g.sensitivity:.1%} (CI {g.sens_ci_lo:.1%}-{g.sens_ci_hi:.1%}; needs CI above S4's 1.0-8.0%) at "
            f"{g.fp_per_case:.1f} FP/case  ->  {'PASS' if p else 'FAIL'}")
        say(f"      nnU-Net-missed subset (n={len(km)}): median pct {km[f'pct_{k}'].median():.1f}, top-10 "
            f"{gm.n_hit}/{gm.n_lesions} = {gm.sensitivity:.1%} (CI {gm.sens_ci_lo:.1%}-{gm.sens_ci_hi:.1%})")
    if not pc:
        say("   POSITIVE CONTROL FAILED: per the brief the features are treated as broken; no target claim is made.")
    return out


def scan_level(cases: pd.DataFrame) -> None:
    say("\n== 3c. Scan-level confound: per-case mean anomaly score, positives vs negatives, by slice thickness")
    say("   (report only; within-scan ranking is the primary metric because this can differ)")
    ok = cases[cases.sr_ok].copy()
    sp = pd.read_csv(C.S1_CASES)[["case_id", "slice_thickness_mm"]]
    ok = ok.merge(sp, on="case_id", validate="1:1")
    ok["tbin"] = C.thick_bin(ok.slice_thickness_mm).values
    ok["grp"] = np.where(ok.n_lesions > 0, "pos", "neg")
    for r in C.RADII_MM:
        for k in ("A", "B"):
            say(f"-- score {k}, r {r:g}: median of per-case mean [pos all patches | pos far-only | neg]  (n pos / n neg)")
            for b in C.THICK_LABELS:
                g = ok[ok.tbin == b]
                p, n = g[g.grp == "pos"], g[g.grp == "neg"]
                if not len(p) or not len(n):
                    say(f"   {b:>8}: n {len(p)}/{len(n)}")
                    continue
                mw = stats.mannwhitneyu(p[f"mean_far_{k}_r{r:g}"].dropna(), n[f"mean_{k}_r{r:g}"])
                say(f"   {b:>8}: {p[f'mean_{k}_r{r:g}'].median():.3f} | {p[f'mean_far_{k}_r{r:g}'].median():.3f} | "
                    f"{n[f'mean_{k}_r{r:g}'].median():.3f}   (n {len(p)}/{len(n)}; far-pos vs neg MWU p={mw.pvalue:.2g})")


def false_structures(cand: pd.DataFrame) -> None:
    say("\n== Where the top false candidates sit (rank <= 5, source b; GT anatomy, nearest structure within 3 mm)")
    for r in C.RADII_MM:
        for k in ("A", "B"):
            f = cand[(cand.radius == r) & (cand.score == k) & cand.is_false & (cand["rank"] <= 5)]
            tp = cand[(cand.radius == r) & (cand.score == k) & (cand.dil_id > 0)]
            def frac(df):
                d = df[[f"d_{n}" for n in C.CONTEXT_STRUCTURES]]
                near = d.min(1) <= 3.0
                lab = d.idxmin(1).str[2:]
                return ", ".join(f"{n} {((lab == n) & near).mean():.1%}" for n in C.CONTEXT_STRUCTURES) + \
                    f", none {(~near).mean():.1%}"
            say(f"   {k} r{r:g} false top-5 (n={len(f)}): {frac(f)}")
            say(f"   {k} r{r:g} in-lesion candidates (n={len(tp)}): {frac(tp)}")


def bank_size_curve(t: pd.DataFrame) -> pd.DataFrame:
    say("\n== 4c. Bank-size curve (nested seed-42 subsets; everything refit on the subset)")
    ids = sorted(set(t[t.target | (t.diam_bin == ">=40")].case_id))
    rows = []
    for n in (*C.BANK_SIZE_CURVE, C.N_BANK):
        if n != C.N_BANK:
            cmd_score(n, ids)
        sc = load_scores(n, set(ids))
        les = pd.DataFrame([r for s in sc.values() if s["sr_ok"]
                            for r in lesion_percentiles(s, C.SEED * 1000 + int(s["case_id"][-8:]) * 10)])
        tt = t[["case_id", "lesion_id", "target", "diam_bin"]].merge(les, on=["case_id", "lesion_id"])
        for grp, m in (("TARGET", tt.target), (">=40", tt.diam_bin == ">=40")):
            for k in ("A", "B"):
                p = tt[m][f"pct_{k}"].dropna()
                rows.append(dict(bank_cases=n, group=grp, score=k, n=len(p), median_pct=p.median(),
                                 frac_ge90=(p >= 90).mean()))
    df = pd.DataFrame(rows)
    for (grp, k), g in df.groupby(["group", "score"]):
        say(f"   {grp:>7} score {k}: " + ", ".join(f"n_bank {r.bank_cases}: median {r.median_pct:.1f} "
                                                   f"(>=90: {r.frac_ge90:.1%})" for r in g.itertuples()))
    return df


def per_fold(t: pd.DataFrame) -> None:
    say("\n== Per nnU-Net fold (fold 0 = designated dev fold; no design change was made after cohort results)")
    for k in ("A", "B"):
        say(f"   score {k}: " + "; ".join(
            f"fold {f}: TARGET med {g[g.target][f'pct_{k}'].median():.1f} (n={int(g.target.sum())}), "
            f">=40 med {g[g.diam_bin == '>=40'][f'pct_{k}'].median():.1f}" for f, g in t.groupby("fold")))


def cmd_report() -> None:
    files = sorted((SCORES / f"n{C.N_BANK}").glob("*.pkl"))
    say(f"== S9 normative pilot report: {len(files)} scored cohort cases")
    assert len(files) == 1308, len(files)
    t0 = time.time()
    with mp.Pool(C.NUM_WORKERS) as p:
        res = p.map(_eval_case, files, chunksize=8)
        cal = pd.DataFrame([r for rs in p.map(_calib_case, files, chunksize=8) for r in rs])
    say(f"   evaluated in {time.time() - t0:.0f}s")
    les = pd.DataFrame([r for a, _, _ in res for r in a])
    cand = pd.DataFrame([r for _, b, _ in res for r in b])
    cases = pd.DataFrame([c for _, _, c in res])
    fold = {c: k - 1 for k in range(1, 6) for c in tio.fold_ids(k)}
    cases["fold"] = cases.case_id.map(fold)
    ok = cases[cases.sr_ok]
    say(f"   cases with a search region: {len(ok)} ({int((ok.n_lesions > 0).sum())} tumor+, "
        f"{int((ok.n_lesions == 0).sum())} tumor-); no search region: {sorted(cases[~cases.sr_ok].case_id)}")
    say(f"   lattice centres per case: median {ok.n_centres.median():.0f} (min {ok.n_centres.min()}, max {ok.n_centres.max()})")
    say(f"   phase groups: {ok.phase_group.value_counts().to_dict()}; thickness groups: {ok.thick_group.value_counts().to_dict()}")

    calibration_summary(cal)
    t = best_rank(lesion_table(les), cand)
    say(f"   lesions with no lattice centre in lesion + 2 mm (unscorable): {int(t.pct_A.isna().sum())} "
        f"(TARGET {int(t[t.target].pct_A.isna().sum())}, >=40 {int(t[t.diam_bin == '>=40'].pct_A.isna().sum())})")
    say(f"   matched radius in TARGET: {t[t.target].matched_radius_mm.value_counts().to_dict()}; "
        f"10-20 all: {t[t.diam_bin == '10-20'].matched_radius_mm.value_counts().to_dict()}")
    s4t = t[t.target].s4_frac_ge_dark
    say(f"   S4 baseline recomputed from its CSV: TARGET n={int(t.target.sum())}, median beaten-by "
        f"{s4t.median():.3f} (carried over: {C.S4_TARGET_FRAC_BEATEN})")
    ident = {k: float((t[f"pct_C{k}"] - t[f"pct_{k}"]).abs().max()) for k in ("A", "B")}
    say(f"   C vs A/B within-scan percentile, max |diff| over all lesions: {ident}")
    report_percentiles(t)
    report_radii(t)
    fr = froc(t, cand, cases)
    report_froc(fr)
    scan_level(cases)
    false_structures(cand)
    per_fold(t)
    g = gates(t, fr)
    bs = bank_size_curve(t)

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    keep = [c for c in t.columns if not c.startswith(("fge_",))] + [f"fge_{k}" for k in SCORE_KEYS]
    t[keep].to_csv(C.RESULTS / "lesion_ranks.csv", index=False, float_format="%.6g")
    fr.to_csv(C.RESULTS / "froc_points.csv", index=False, float_format="%.6g")
    cand.to_csv(C.WORK / "candidates.csv.gz", index=False, float_format="%.6g", compression="gzip")
    cal.to_csv(C.WORK / "calibration.csv", index=False, float_format="%.6g")
    cases.to_csv(C.WORK / "case_table.csv", index=False, float_format="%.6g")
    bs.to_csv(C.WORK / "bank_size_curve.csv", index=False, float_format="%.6g")
    say(f"\nwrote lesion_ranks.csv ({len(t)} rows), froc_points.csv ({len(fr)} rows); gates {g}")
    (C.WORK / "summary.txt").write_text("\n".join(LINES) + "\n")


def cmd_calibrate() -> None:
    """PHASE 3a on its own, so it can be read before any lesion is evaluated."""
    files = sorted((SCORES / f"n{C.N_BANK}").glob("*.pkl"))
    with mp.Pool(C.NUM_WORKERS) as p:
        cal = pd.DataFrame([r for rs in p.map(_calib_case, files, chunksize=8) for r in rs])
    say(f"calibration: {cal.case_id.nunique()} negatives")
    calibration_summary(cal)
    (C.WORK / "calibration_summary.txt").write_text("\n".join(LINES) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fit", "score", "calibrate", "insertion", "report"])
    a = ap.parse_args()
    {"fit": cmd_fit, "score": cmd_score, "calibrate": cmd_calibrate, "insertion": cmd_insertion,
     "report": cmd_report}[a.step]()


if __name__ == "__main__":
    sys.exit(main())
