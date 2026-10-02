"""S4 summary: turn the per-case pickles from case_pass.py into the deliverables.

  results/log_candidates/candidates.csv.gz          one row per candidate peak
  results/log_candidates/separability_by_lesion.csv one row per GT lesion (1235)
  results/log_candidates/log_froc_points.csv        z ladder x lesion stratum x FP source
  work/log_candidates/summary.txt                   every number printed, with the oracles

Oracles asserted here: 1308 cases accounted for, the 1235 lesions join the master table with
equal gt_vox (and volume within resampling tolerance), coverage agrees with S2's independently
computed in_sr_frac, the 11 empty-envelope negatives are the ones S2 found, and peak coordinates
sit inside the search region.

The headline FROC uses slice-matched false peaks -- source (b), non-tumor parts of positive
scans -- because the negatives are systematically thinner-sliced than the positives (median 1.5
vs 2.5 mm), and a blob filter's false-alarm rate depends on slice thickness.

Usage (repo root): .venv/bin/python src/log-candidates/summarize.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
from case_pass import CASE_DIR, FP_EXCL_MM, N_PATCH, WORK, Z_GRID  # noqa: E402
import logblob as LB  # noqa: E402

OUT = REPO / "results" / "log_candidates"
MASTER = REPO / "results" / "master_lesion_table" / "master_lesions.csv"
ATTEN3 = REPO / "results" / "attenuation_v3" / "attenuation_labels_v3.csv"
S2 = REPO / "results" / "nnunet_subthreshold" / "per_lesion_subthreshold.csv"
S1_CASES = REPO / "work" / "master_lesion_table" / "case_spacing.csv"   # all 1308, incl. negatives
BINS = ("<5", "5-10", "10-20", "20-40", ">=40")
THICK_EDGES = [0.0, 1.5001, 2.5001, 5.0001, np.inf]
THICK_LABELS = ["<=1.5", "1.5-2.5", "2.5-5", ">5"]
ENHANCED = ("Venous", "Arterial", "Delay")     # contrast-enhanced phases (non-contrast excluded)
# Frozen from phantom.py check D (r=2 mm dark ball vs r=2 mm dark tube), which separated them
# completely: ball blobness >= 0.50, tube <= 0.15. The cut is the midpoint of that gap. Chosen on
# the phantom before any evaluation lesion was scored, so it is a priori (CONTEXT rule 3).
PHANTOM_BLOB_CUT = 0.33
# phantom.py check C: over a 200 cm^3 noise-only mask the single largest z was 4.5-4.7 and z >= 5
# produced ~0 peaks per volume. So "this lesion produced a response at all" means z >= 5: above
# what pure noise reaches in a pancreas-sized region. Also fixed before any lesion was scored.
NULL_Z_FLOOR = 5.0
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


def load():
    cases, peaks, scales, lesions = [], [], [], []
    for f in sorted(CASE_DIR.glob("*.pkl")):
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        cases.append(r["case"])
        peaks.extend(r["peaks"])
        scales.extend(r["scales"])
        lesions.extend(r["lesions"])
    cases = pd.DataFrame(cases)
    sp1 = pd.read_csv(S1_CASES)[["case_id", "slice_thickness_mm", "slice_axis"]]
    cases = cases.merge(sp1, on="case_id", how="left", validate="1:1")
    les = pd.DataFrame(lesions).rename(columns={"vol_mm3": "vol_mm3_pass"})
    return cases, pd.DataFrame(peaks), pd.DataFrame(scales), les


def thick_bin(s: pd.Series) -> pd.Series:
    return pd.cut(s, THICK_EDGES, labels=THICK_LABELS, right=True)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion -- the same interval S3 used. It stays
    inside [0, 1] at k = 0 or k = n, where the normal approximation does not."""
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1.0 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def wmean(values: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, float)
    return float(np.average(np.asarray(values, float), weights=w)) if w.sum() > 0 else np.nan


# ------------------------------------------------------------------ joins
def build_lesion_table(les: pd.DataFrame, cases: pd.DataFrame, partial: bool = False) -> pd.DataFrame:
    master = pd.read_csv(MASTER)
    if partial:
        master = master[master.case_id.isin(set(cases.case_id))]
    a3 = pd.read_csv(ATTEN3)[["case_id", "lesion_id", "gt_vox", "dHU_core_global", "attenuation_v3",
                              "sigma_hu", "cnr"]].rename(columns={"gt_vox": "gt_vox_a3"})
    s2 = pd.read_csv(S2)[["case_id", "lesion_id", "in_sr_frac", "peak_p_dil2", "missed"]].rename(
        columns={"in_sr_frac": "in_sr_frac_s2", "peak_p_dil2": "nnunet_peak_p_dil2"})
    t = master.merge(les, on=["case_id", "lesion_id"], how="left", validate="1:1", indicator=True)
    say(f"lesions joining the per-case pass: {(t._merge == 'both').sum()}/{len(t)}")
    assert (t._merge == "both").all(), t.loc[t._merge != "both", ["case_id", "lesion_id"]].head()
    t = t.drop(columns="_merge")
    t = t.merge(a3, on=["case_id", "lesion_id"], how="left", validate="1:1")
    assert (t.gt_vox == t.gt_vox_a3).all(), "gt_vox mismatch vs attenuation_v3"
    t = t.merge(s2, on=["case_id", "lesion_id"], how="left", validate="1:1")
    d = (t.in_sr_frac - t.in_sr_frac_s2).abs()
    say(f"coverage vs S2's independent in_sr_frac (server masks): max |diff| {d.max():.3g}, "
        f"rows > 0.01: {int((d > 0.01).sum())}, mean |diff| {d.mean():.3g}")
    # master_lesions.csv is written with %.6g, so 6 significant figures is the tightest check available
    rel = (t.vol_mm3_pass - t.vol_mm3).abs() / t.vol_mm3
    assert rel.max() < 1e-5, f"native lesion volume disagrees with the master table (max rel {rel.max():.2e})"
    say(f"lesion volume recomputed from the masks vs the master table: max relative diff "
        f"{rel.max():.1e} (master is stored at 6 significant figures)")
    ratio = t.gt_vox_iso / t.vol_mm3
    say(f"lesion volume: 1 mm-grid voxel count / native mm3, median {np.median(ratio):.4f} "
        f"(1.0 = the nearest-neighbour label map preserves volume), 5-95 pct "
        f"{np.percentile(ratio, [5, 95]).round(3)}")
    t["thick_bin"] = thick_bin(t.slice_thickness_mm)
    t["enhanced"] = t.ct_phase.isin(ENHANCED)
    t["contrast_v3"] = t.attenuation_v3
    t["iso5"] = t.dHU_core_global.abs() <= 5.0
    t["target"] = (t.diam_bin == "10-20") & t.enhanced & t.contrast_v3.isin(["hypo", "hyper"])
    return t


# ------------------------------------------------------------------ strata
def strata(t: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """(name, boolean mask) pairs, most specific first. The target group is fixed by the session
    brief: 10-20 mm, hypo or hyper on v3, contrast-enhanced phase."""
    out = [("TARGET 10-20mm hypo/hyper enhanced", t.target),
           ("10-20mm all", t.diam_bin == "10-20"),
           ("10-20mm enhanced", (t.diam_bin == "10-20") & t.enhanced),
           ("10-20mm iso(v3)", (t.diam_bin == "10-20") & (t.contrast_v3 == "iso")),
           ("10-20mm iso within 5 HU", (t.diam_bin == "10-20") & t.iso5),
           ("<10mm all", t.diam_mm < 10),
           (">=20mm all", t.diam_mm >= 20),
           ("all lesions", pd.Series(True, index=t.index)),
           ("all, not separated", ~t.excluded)]
    for b in BINS:
        out.append((f"size {b}", t.diam_bin == b))
    for c in ("hypo", "iso", "hyper"):
        out.append((f"v3 {c}", t.contrast_v3 == c))
    for p in sorted(t.ct_phase.dropna().unique()):
        out.append((f"phase {p}", t.ct_phase == p))
    for r in ("head", "body", "tail"):
        out.append((f"region {r}", t.region == r))
    for b in THICK_LABELS:
        out.append((f"thickness {b} mm", t.thick_bin == b))
    for tier in sorted(t.tier.dropna().unique()):
        out.append((f"tier {tier}", t.tier == tier))
    out += [("nnU-Net detected", t.detected_any), ("nnU-Net missed", ~t.detected_any),
            ("nnU-Net missed, 10-20mm", ~t.detected_any & (t.diam_bin == "10-20")),
            ("nnU-Net missed, TARGET", ~t.detected_any & t.target)]
    return out


# ------------------------------------------------------------------ FROC
def froc(t: pd.DataFrame, peaks: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity (a lesion is hit if a candidate peak with z >= threshold lands in its 2 mm
    dilation) against false peaks per case and per 100 cm^3, from both FP sources."""
    ok = cases[cases.sr_ok].copy()
    ok["thick_bin"] = thick_bin(ok.slice_thickness_mm)
    pos_mix = ok[ok.gt_tumor == 1].thick_bin.value_counts(normalize=True)
    neg = ok[ok.gt_tumor == 0]
    neg_mix = neg.thick_bin.value_counts(normalize=True)
    fp = peaks[peaks.is_false]
    rows = []
    for pol in ("dark", "bright", "either", "dark_blobgate"):
        if pol == "dark_blobgate":     # supplementary: keep only blob-shaped peaks (cut from the phantom)
            pk = peaks[(peaks.polarity == "dark") & (peaks.blobness >= PHANTOM_BLOB_CUT)]
            fpp = fp[(fp.polarity == "dark") & (fp.blobness >= PHANTOM_BLOB_CUT)]
        else:
            pk = peaks if pol == "either" else peaks[peaks.polarity == pol]
            fpp = fp if pol == "either" else fp[fp.polarity == pol]
        cap = LB.TOP_K * (2 if pol == "either" else 1)   # the per-case budget this polarity has
        for zt in Z_GRID:
            h = pk.loc[(pk.z >= zt) & (pk.in_dil2 > 0), ["case_id", "in_dil2"]]
            hit = {(str(c), int(l)) for c, l in zip(h.case_id, h.in_dil2)}
            f = fpp[fpp.z >= zt]
            # (a) negatives; (b) positives' non-tumor region; (b') negatives reweighted to the
            # positives' slice-thickness mix
            src = {}
            na = f[f.gt_tumor == 0].groupby("case_id").size().reindex(neg.case_id, fill_value=0)
            src["a_negatives"] = (float(na.mean()),
                                  100.0 * na.sum() / neg.sr_vol_cm3.sum(), len(neg))
            posc = ok[ok.gt_tumor == 1]
            nb = f[f.gt_tumor == 1].groupby("case_id").size().reindex(posc.case_id, fill_value=0)
            src["b_positives_nontumor"] = (float(nb.mean()),
                                           100.0 * nb.sum() / posc.sr_vol_excl_tumor_cm3.sum(), len(posc))
            per_thick = na.groupby(neg.set_index("case_id").thick_bin.reindex(na.index),
                                   observed=True).mean()
            w = pos_mix.reindex(per_thick.index).fillna(0.0)
            src["a_neg_reweighted_to_pos"] = (wmean(per_thick.values, w.values), np.nan, len(neg))
            for sname, (per_case, per_100cm3, ncase) in src.items():
                for lname, m in strata(t):
                    s = t[m]
                    n_hit = sum((c, l) in hit for c, l in zip(s.case_id, s.lesion_id))
                    lo, hi = wilson(n_hit, len(s))
                    rows.append(dict(polarity=pol, z_thresh=zt, fp_source=sname, stratum=lname,
                                     n_lesions=len(s), n_hit=n_hit,
                                     sensitivity=n_hit / len(s) if len(s) else np.nan,
                                     sens_ci_lo=lo, sens_ci_hi=hi,
                                     fp_per_case=per_case, fp_per_100cm3=per_100cm3,
                                     n_fp_cases=ncase,
                                     frac_cases_censored=float((na >= cap).mean())
                                     if sname == "a_negatives" else
                                     float((nb >= cap).mean()) if sname == "b_positives_nontumor" else np.nan))
    cover = float(pos_mix.reindex(neg_mix.index).fillna(0.0).sum())
    say(f"\nreweighting (a)->(b) covers {cover:.1%} of the positives' slice-thickness mass: the "
        f"negatives have no case in the remaining bins, so the reweighted estimate is blind there.")
    say(f"\nslice-thickness mix (cases with a search region): positives "
        f"{ {k: round(v, 3) for k, v in pos_mix.sort_index().items()} }; negatives "
        f"{ {k: round(v, 3) for k, v in neg_mix.sort_index().items()} }")
    return pd.DataFrame(rows)


TOP_N_GRID = (1, 2, 3, 5, 10, 20, 50)


def froc_by_rank(t: pd.DataFrame, peaks: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """The other natural operating curve for a candidate generator: keep the top N peaks per case.

    Unlike a z threshold this fixes the per-case budget exactly, so sensitivity and false peaks
    move together with no censoring to worry about. N = 50 is the whole list.
    """
    ok = cases[cases.sr_ok]
    neg, pos = ok[ok.gt_tumor == 0], ok[ok.gt_tumor == 1]
    rows = []
    for pol in ("dark", "bright", "either"):
        pk = peaks if pol == "either" else peaks[peaks.polarity == pol]
        for n in TOP_N_GRID:
            sel = pk[pk["rank"] <= n]
            hh = sel.loc[sel.in_dil2 > 0, ["case_id", "in_dil2"]]
            hit = {(str(c), int(l)) for c, l in zip(hh.case_id, hh.in_dil2)}
            f = sel[sel.is_false]
            na = f[f.gt_tumor == 0].groupby("case_id").size().reindex(neg.case_id, fill_value=0)
            nb = f[f.gt_tumor == 1].groupby("case_id").size().reindex(pos.case_id, fill_value=0)
            for sname, cnt, vol, nc in (("a_negatives", na, neg.sr_vol_cm3.sum(), len(neg)),
                                        ("b_positives_nontumor", nb,
                                         pos.sr_vol_excl_tumor_cm3.sum(), len(pos))):
                for lname, m in strata(t):
                    sub = t[m]
                    k = sum((c, l) in hit for c, l in zip(sub.case_id, sub.lesion_id))
                    lo, hi = wilson(k, len(sub))
                    rows.append(dict(polarity=pol, z_thresh=np.nan, top_n=n, fp_source=sname,
                                     stratum=lname, n_lesions=len(sub), n_hit=k,
                                     sensitivity=k / len(sub) if len(sub) else np.nan,
                                     sens_ci_lo=lo, sens_ci_hi=hi,
                                     fp_per_case=float(cnt.mean()),
                                     fp_per_100cm3=100.0 * cnt.sum() / vol,
                                     n_fp_cases=nc, frac_cases_censored=np.nan))
    return pd.DataFrame(rows)


def report_froc_rank(fr: pd.DataFrame) -> None:
    say("\n== 3c. Fixed-budget FROC (top N candidates per case, dark polarity), FP source (b)")
    d = fr[(fr.polarity == "dark") & (fr.fp_source == "b_positives_nontumor") & fr.top_n.notna()]
    say(f"{'top N':>6} {'FP/case':>8} {'FP/100cm3':>10} " +
        " ".join(f"{n:>9}" for n in ("TARGET", "10-20 all", "<10", ">=20", "all")))
    for n in TOP_N_GRID:
        g = d[d.top_n == n].set_index("stratum")
        say(f"{n:>6} {g.iloc[0].fp_per_case:>8.2f} {g.iloc[0].fp_per_100cm3:>10.2f} " + " ".join(
            f"{g.at[k, 'sensitivity']:>8.1%} " for k in
            ("TARGET 10-20mm hypo/hyper enhanced", "10-20mm all", "<10mm all", ">=20mm all", "all lesions")))


def fp_by_thickness(peaks: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """False peaks per case and per 100 cm^3 by slice-thickness bin and source, per z threshold."""
    ok = cases[cases.sr_ok].copy()
    ok["thick_bin"] = thick_bin(ok.slice_thickness_mm)
    rows = []
    for src, sel, volcol in (("a_negatives", ok.gt_tumor == 0, "sr_vol_cm3"),
                             ("b_positives_nontumor", ok.gt_tumor == 1, "sr_vol_excl_tumor_cm3")):
        sub = ok[sel]
        for pol in ("dark", "bright"):
            f = peaks[peaks.is_false & (peaks.polarity == pol) & peaks.case_id.isin(set(sub.case_id))]
            for zt in Z_GRID:
                n = f[f.z >= zt].groupby("case_id").size().reindex(sub.case_id, fill_value=0)
                for b in THICK_LABELS:
                    m = (sub.thick_bin == b).values
                    if not m.any():
                        continue
                    rows.append(dict(fp_source=src, polarity=pol, z_thresh=zt, thick_bin=b,
                                     n_cases=int(m.sum()), fp_per_case=float(n[m].mean()),
                                     fp_per_100cm3=100.0 * n[m].sum() / sub[volcol].values[m].sum(),
                                     frac_censored=float((n[m] >= LB.TOP_K).mean())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ report
def report_coverage(t: pd.DataFrame) -> None:
    say("\n== 1. Search-region coverage (ceiling for any region-restricted detector)")
    say("   fraction of each lesion's voxels inside the region; a lesion with 0 cannot be found at all")
    for by in ("tier", "diam_bin"):
        g = t.groupby(by, observed=True).agg(n=("in_sr_frac", "size"), mean=("in_sr_frac", "mean"),
                                            median=("in_sr_frac", "median"),
                                            frac_zero=("in_sr_frac", lambda s: (s <= 0).mean()),
                                            frac_ge50=("in_sr_frac", lambda s: (s >= 0.5).mean()),
                                            frac_full=("in_sr_frac", lambda s: (s >= 0.999).mean()))
        say(f"-- by {by}")
        say(g.to_string(float_format=lambda v: f"{v:.3f}"))
    say(f"   whole cohort: mean {t.in_sr_frac.mean():.3f}, {int((t.in_sr_frac <= 0).sum())} lesions "
        f"entirely outside, {int((t.in_sr_frac >= 0.999).sum())} fully inside")
    say(f"   TARGET group (n={int(t.target.sum())}): mean coverage {t.loc[t.target, 'in_sr_frac'].mean():.3f}, "
        f"{int((t.loc[t.target, 'in_sr_frac'] <= 0).sum())} entirely outside")


def report_ranks(t: pd.DataFrame) -> None:
    say("\n== 2. Rank of the best in-lesion peak among the case's candidates (dark polarity)")
    say("   'in-lesion' = the peak voxel lies in the lesion dilated 2 mm. Rank 1 = the case's top peak.")
    say(f"{'stratum':<38} {'n':>5} {'hit':>6} {'top1':>6} {'top3':>6} {'top10':>6} {'med rank':>9} "
        f"{'med z':>7} {'CNR':>6}")
    for name, m in strata(t):
        s = t[m]
        if not len(s):
            continue
        r = s.rank_dark
        say(f"{name:<38} {len(s):>5} {r.notna().mean():>6.1%} {(r <= 1).mean():>6.1%} "
            f"{(r <= 3).mean():>6.1%} {(r <= 10).mean():>6.1%} "
            f"{(f'{r.median():.0f}' if r.notna().any() else '-'):>9} "
            f"{s.z_dark.median() if r.notna().any() else np.nan:>7.2f} {s.cnr.median():>6.2f}")
    sig = [c for c in t.columns if c.startswith("rank_dark_s")]
    say("\n   per scale (dark), TARGET group then all 10-20 mm: hit rate / median rank")
    for label, m in (("TARGET", t.target), ("10-20 all", t.diam_bin == "10-20")):
        s = t[m]
        say(f"   {label:<10} " + "  ".join(
            f"{c.split('_s')[1]:>4}mm {s[c].notna().mean():>5.1%}/"
            f"{(f'{s[c].median():.0f}' if s[c].notna().any() else '-'):>3}" for c in sig))
    say("\n   best single scale by hit rate, per size bin (dark):")
    for b in BINS:
        s = t[t.diam_bin == b]
        hr = {c.split("_s")[1]: s[c].notna().mean() for c in sig}
        best = max(hr, key=hr.get)
        say(f"     {b:>6} n={len(s):>4}  best sigma {best:>4} mm ({hr[best]:.1%}); "
            f"pooled over scales {s.rank_dark.notna().mean():.1%}")


def report_patches(t: pd.DataFrame) -> None:
    say(f"\n== 4. In-lesion response vs {N_PATCH} random lesion-sized patches in the same region")
    say("   patch_frac_ge = fraction of same-size patches elsewhere in this case's search region whose")
    say("   max z is at least the lesion's. 0.5 = the lesion looks exactly like a random patch; a")
    say("   detector can only work if this is small.")
    say("   CAVEAT: both sides are read off the max-over-scales z map, so this inherits the same")
    say("   bias against coarse scales that section 2b exposes. The per-scale clutter measure that")
    say("   does not is 'med rank' in 2b.")
    say(f"{'stratum':<38} {'n':>5} {'median':>8} {'<=0.05':>7} {'<=0.10':>7} {'<=0.25':>7} {'best-in-case':>13}")
    for name, m in strata(t):
        s = t[m].dropna(subset=["patch_frac_ge_dark"])
        if not len(s):
            continue
        p = s.patch_frac_ge_dark
        say(f"{name:<38} {len(s):>5} {p.median():>8.3f} {(p <= 0.05).mean():>7.1%} "
            f"{(p <= 0.10).mean():>7.1%} {(p <= 0.25).mean():>7.1%} {(p <= 0).mean():>13.1%}")


def report_verdicts(t: pd.DataFrame) -> None:
    """Three-way verdict per stratum, on two independent axes.

    resp = fraction of the stratum whose own max z inside the lesion + 2 mm reaches NULL_Z_FLOOR
           (what pure noise does not reach in a pancreas-sized region, from the phantom null).
    frac = median fraction of same-size random patches elsewhere in the same search region whose
           max z is at least the lesion's.

    Both are needed: a big lesion can have a strong response (high resp) and still be
    indistinguishable from a random equal-size patch of the same pancreas (high frac), because an
    equal-size patch also contains vessels and bowel.
    """
    say("\n== Verdict per stratum (dark polarity)")
    say(f"   featureless      : < 25% of the stratum reaches z >= {NULL_Z_FLOOR:g} anywhere in the lesion + 2 mm")
    say("                      -- there is no response above the pure-noise floor to rank")
    say("   clutter-dominated: a response exists, but > 25% of same-size patches in the same")
    say("                      pancreas match it (median patch_frac_ge > 0.25)")
    say("   signal-exists    : median patch_frac_ge <= 0.10 AND >= 50% of lesions land in the")
    say("                      case's top 10 candidates")
    say(f"{'stratum':<38} {'n':>5} {'resp>=5':>8} {'med frac':>9} {'hit':>6} {'top10':>6} {'verdict':>18}")
    for name, m in strata(t):
        s_ = t[m].dropna(subset=["patch_frac_ge_dark"])
        if len(s_) < 5:
            continue
        resp = float((s_.smax_dil2_dark >= NULL_Z_FLOOR).mean())
        frac, r = s_.patch_frac_ge_dark.median(), s_.rank_dark
        v = ("featureless" if resp < 0.25
             else "signal-exists" if frac <= 0.10 and (r <= 10).mean() >= 0.5
             else "clutter-dominated")
        say(f"{name:<38} {len(s_):>5} {resp:>8.1%} {frac:>9.3f} {r.notna().mean():>6.1%} "
            f"{(r <= 10).mean():>6.1%} {v:>18}")


def report_per_scale_budget(t: pd.DataFrame, scales: pd.DataFrame) -> None:
    """One fixed scale's own peak list vs the z-pooled scale-space list.

    The budget is NOT equal: the response gets smoother with sigma, so a coarse scale yields far
    fewer local maxima than the top_k cap allows (the "cand/case" column). A coarse scale that
    beats the pooled list therefore does so with fewer candidates, i.e. at a lower false-peak
    rate as well -- the pooled list is dominated, not merely different.

    Every scale is shown so that no scale is being selected on these lesions. PICKING the best
    row would be tuning on evaluation lesions (CONTEXT rule 3), which is why the headline FROC
    stays with the a-priori pooled list; a corrected ranking would need out-of-fold selection.

    "med rank" also reads directly as clutter: rank 8 means seven non-lesion peaks in that same
    pancreas scored higher than the lesion's best peak at that scale.
    """
    cand = scales[scales.polarity == "dark"].groupby("sigma_mm").n_peaks.mean()
    say(f"\n== 2b. One fixed sigma vs the z-pooled scale-space list (cap {LB.TOP_K}/case/polarity)")
    say("   hit = an in-lesion peak anywhere in that list; then top-1 / top-3 / top-10 rates")
    for label, m in (("TARGET (n=%d)" % int(t.target.sum()), t.target),
                     ("10-20 mm all (n=%d)" % int((t.diam_bin == "10-20").sum()), t.diam_bin == "10-20"),
                     ("all lesions (n=%d)" % len(t), pd.Series(True, index=t.index))):
        sub = t[m]
        say(f"-- {label}")
        say(f"{'ranking':>16} {'cand/case':>10} {'hit':>7} {'top1':>7} {'top3':>7} {'top10':>7} {'med rank':>9}")
        rows = [(f"rank_dark_s{sg:g}", f"sigma {sg:g} mm", cand.get(sg, np.nan)) for sg in LB.SIGMAS_MM]
        rows.append(("rank_dark", "POOLED (z)", float(LB.TOP_K)))
        for c, nm, nc in rows:
            r = sub[c]
            say(f"{nm:>16} {nc:>10.1f} {r.notna().mean():>7.1%} {(r <= 1).mean():>7.1%} "
                f"{(r <= 3).mean():>7.1%} {(r <= 10).mean():>7.1%} "
                f"{(f'{r.median():.0f}' if r.notna().any() else '-'):>9}")


def report_froc(fr: pd.DataFrame, pols=("dark",)) -> None:
    for pol in pols:
        say(f"\n== 3. LoG-only FROC, polarity = {pol}. "
            f"HEADLINE source = (b) positives' non-tumor region")
        _report_froc_pol(fr, pol)


def _report_froc_pol(fr: pd.DataFrame, pol: str) -> None:
    for src in ("b_positives_nontumor", "a_negatives", "a_neg_reweighted_to_pos"):
        d = fr[(fr.polarity == pol) & (fr.fp_source == src) & fr.z_thresh.notna()]
        say(f"-- FP source: {src}")
        say(f"{'z':>5} {'FP/case':>8} {'FP/100cm3':>10} {'censored':>9} " +
            " ".join(f"{n:>9}" for n in ("TARGET", "10-20 all", "10-20 iso", "<10", ">=20", "all")))
        for zt in Z_GRID:
            g = d[d.z_thresh == zt].set_index("stratum")
            def sv(k):
                return f"{g.at[k, 'sensitivity']:.1%}" if k in g.index else "-"
            r0 = g.iloc[0]
            say(f"{zt:>5g} {r0.fp_per_case:>8.2f} "
                f"{(f'{r0.fp_per_100cm3:.2f}' if np.isfinite(r0.fp_per_100cm3) else '-'):>10} "
                f"{(f'{r0.frac_cases_censored:.1%}' if np.isfinite(r0.frac_cases_censored) else '-'):>9} "
                + " ".join(f"{sv(k):>9}" for k in
                           ("TARGET 10-20mm hypo/hyper enhanced", "10-20mm all", "10-20mm iso(v3)",
                            "<10mm all", ">=20mm all", "all lesions")))


def main() -> None:
    partial = "--allow-partial" in sys.argv   # smoke test only; never for deliverables
    cases, peaks, scales, les = load()
    say(f"== S4 scale-space LoG candidates: {len(cases)} case pickles, {len(peaks)} candidate peaks, "
        f"{len(les)} lesion rows")
    say(f"detector (fixed a priori): sigma {LB.SIGMAS_MM} mm, HU clip {LB.HU_CLIP}, "
        f"min separation {LB.MIN_SEP_MM} mm, top K {LB.TOP_K} per case per polarity, "
        f"FP exclusion {FP_EXCL_MM} mm")
    assert partial or len(cases) == 1308, f"{len(cases)} cases, expected 1308"
    bad = cases[~cases.sr_ok]
    say(f"cases with no search region (empty pancreas envelope): {len(bad)} {sorted(bad.case_id)}")
    ok = cases[cases.sr_ok]
    say(f"cases processed: {len(ok)} ({int((ok.gt_tumor == 1).sum())} tumor+, "
        f"{int((ok.gt_tumor == 0).sum())} tumor-)")
    say(f"crop margin around the search region: min {ok.crop_margin_mm.min():.1f} mm, "
        f"cases below 4*sigma_max = {4 * max(LB.SIGMAS_MM):.0f} mm: "
        f"{int((ok.crop_margin_mm < 4 * max(LB.SIGMAS_MM)).sum())}")
    say(f"search-region volume: median {ok.sr_vol_cm3.median():.1f} cm3 "
        f"(tumor-excluded, positives: {ok.loc[ok.gt_tumor == 1, 'sr_vol_excl_tumor_cm3'].median():.1f} cm3)")
    say(f"peaks per case: {peaks.groupby(['case_id', 'polarity']).size().describe()[['min','50%','max']].to_dict()}")
    assert peaks.in_dil2.notna().all()

    mad = pd.DataFrame(list(ok.noise_mad))
    say("\nper-case response MAD by scale (the z denominator), median over cases:")
    say("  " + "  ".join(f"s={c:g}: {mad[c].median():.1f}" for c in mad.columns))
    say("  -> MAD RISES with sigma on real scans (it FALLS on a pure-noise phantom): what the z"
        " denominator measures in a real pancreas is anatomical clutter, not photon noise.")

    t = build_lesion_table(les, cases, partial)
    report_coverage(t)
    report_ranks(t)
    report_per_scale_budget(t, scales)
    report_patches(t)
    fr = pd.concat([froc(t, peaks, cases), froc_by_rank(t, peaks, cases)], ignore_index=True)
    report_froc(fr, ("dark", "dark_blobgate", "bright", "either"))
    report_froc_rank(fr)
    report_verdicts(t)

    ft = fp_by_thickness(peaks, cases)
    say("\n== 3b. False peaks by slice thickness and source (dark polarity)")
    for src in ("a_negatives", "b_positives_nontumor"):
        say(f"-- {src}")
        d = ft[(ft.fp_source == src) & (ft.polarity == "dark")]
        for zt in (4.0, 6.0, 8.0, 12.0, 20.0):
            g = d[d.z_thresh == zt].set_index("thick_bin")
            say(f"  z>={zt:<4g} " + "  ".join(
                f"{b}: {g.at[b, 'fp_per_case']:.2f}/case {g.at[b, 'fp_per_100cm3']:.2f}/100cm3 (n={int(g.at[b, 'n_cases'])})"
                for b in THICK_LABELS if b in g.index))
    a = ft[(ft.fp_source == "a_negatives") & (ft.polarity == "dark")].groupby("z_thresh").fp_per_100cm3.mean()
    b = ft[(ft.fp_source == "b_positives_nontumor") & (ft.polarity == "dark")].groupby("z_thresh").fp_per_100cm3.mean()
    say("  (a) vs (b) per 100 cm3, thickness-bin mean: "
        + ", ".join(f"z>={z:g}: {a[z]:.2f} vs {b[z]:.2f}" for z in (4.0, 6.0, 8.0, 12.0, 20.0)))

    # ---- structures
    if "struct" in peaks.columns and "missing_structures" in ok.columns:
        say("\n== 6. Nearest structure to the top false peaks (dark, z >= 6, ranks 1-5)")
        f = peaks[peaks.is_false & (peaks.polarity == "dark") & (peaks.z >= 6.0) & (peaks["rank"] <= 5)]
        say("  " + ", ".join(f"{k} {v:.1%}" for k, v in f.struct.value_counts(normalize=True).items()))
        say(f"  median distance to the named structure: {f.struct_mm.median():.1f} mm; n={len(f)}")
        g = f[f.blobness >= PHANTOM_BLOB_CUT]
        say(f"  after the phantom blobness gate (>= {PHANTOM_BLOB_CUT}), n={len(g)} "
            f"({len(g) / max(len(f), 1):.1%} kept): "
            + ", ".join(f"{k} {v:.1%}" for k, v in g.struct.value_counts(normalize=True).items()))
        tp = peaks[(peaks.in_dil2 > 0) & (peaks.polarity == "dark") & (peaks.z >= 6.0)]
        say(f"  for comparison, in-lesion dark peaks with z >= 6 (n={len(tp)}): median blobness "
            f"{tp.blobness.median():.3f} vs {f.blobness.median():.3f} for the top false peaks; "
            f"gate keeps {(tp.blobness >= PHANTOM_BLOB_CUT).mean():.1%} of them")
        say(f"  masks unavailable on this machine: "
            f"{sorted(set(';'.join(ok.missing_structures.dropna().unique()).split(';')) - {''})}")

    out_dir = WORK / "partial_test" if partial else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    keep = ["case_id", "fold", "gt_tumor", "polarity", "rank", "sigma_mm", "z", "score_raw",
            "noise_mad", "hu", "ix", "iy", "iz", "nx", "ny", "nz", "lam1", "lam2", "lam3",
            "blobness", "Ra", "Rb", "frob", "n_pos", "dist_lesion_mm", "in_lesion", "in_dil2",
            "is_false"] + ([ "struct", "struct_mm"] if "struct" in peaks.columns else [])
    OUT_ = out_dir
    peaks[keep].sort_values(["case_id", "polarity", "rank"]).to_csv(
        OUT_ / "candidates.csv.gz", index=False, float_format="%.6g", compression="gzip")
    drop = [c for c in ("gt_vox_a3", "in_sr_frac_s2", "_merge") if c in t.columns]
    t.drop(columns=drop).to_csv(OUT_ / "separability_by_lesion.csv", index=False, float_format="%.6g")
    fr.to_csv(OUT_ / "log_froc_points.csv", index=False, float_format="%.6g")
    ft.to_csv(WORK / "fp_by_thickness.csv", index=False, float_format="%.6g")
    scales.to_csv(WORK / "per_scale_peak_counts.csv", index=False, float_format="%.6g")
    ok.drop(columns=["noise_mad"]).to_csv(WORK / "case_table.csv", index=False, float_format="%.6g")
    say(f"\nwrote {OUT_ / 'candidates.csv.gz'} ({len(peaks)} rows)")
    say(f"wrote {OUT_ / 'separability_by_lesion.csv'} ({len(t)} rows, {t.shape[1]} cols)")
    say(f"wrote {OUT_ / 'log_froc_points.csv'} ({len(fr)} rows)")
    (WORK / "summary.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
