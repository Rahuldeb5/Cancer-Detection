"""Session S2 summary: turn the per-case pickles from case_pass.py into the deliverables.

  results/nnunet_subthreshold/per_lesion_subthreshold.csv   one row per GT lesion (1235)
  results/nnunet_subthreshold/nnunet_froc_points.csv        threshold ladder x hit rule x size bin
  work/nnunet_subthreshold/summary.txt                      regression, oracles, saturation, readings
  work/nnunet_subthreshold/saturation_hist.png              log-y histograms (search-region voxels)
  work/nnunet_subthreshold/components/components_t<t>.csv   one components table per threshold (exported)

Checks that must hold (asserted): 1308 cases present, argmax == saved seg everywhere,
p1 > 0.5 == saved seg everywhere, 1235 lesions joining the master table with equal
gt_vox, recomputed touch == master detected_any, touch hit at t=0.5 == saved-seg touch.

Usage (server, repo root): ~/nnunet_setup/.venv/bin/python src/nnunet-subthreshold/summarize.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from case_pass import CASE_DIR, L_EDGES, P_EDGES, SAT_HI, SAT_LO  # noqa: E402

OUT = C.REPO / "results" / "nnunet_subthreshold"
BINS = ("<5", "5-10", "10-20", "20-40", ">=40")
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    cases, les, comps = [], [], []
    hist = None
    for f in sorted(CASE_DIR.glob("*.pkl")):
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        cases.append(r["case"])
        les.extend(r["lesions"])
        comps.extend(r["comps"])
        h = {k: v.copy() for k, v in r["hist"].items()}
        h = {f"{k}_{'pos' if r['case']['gt_tumor'] else 'neg'}": v for k, v in h.items()}
        if hist is None:
            hist = {f"{k}_{s}": np.zeros_like(v) for k, v in r["hist"].items() for s in ("pos", "neg")}
        for k, v in h.items():
            hist[k] += v
    return pd.DataFrame(cases), pd.DataFrame(les), pd.DataFrame(comps), hist


def split_ids(s: pd.Series) -> list[set[int]]:
    return [set(map(int, x.split(";"))) if isinstance(x, str) and x else set() for x in s]


def main() -> None:
    partial = "--allow-partial" in sys.argv  # smoke test on an unfinished pass; never for deliverables
    cases, les, comps, hist = load()
    master = pd.read_csv(C.REPO / "results/master_lesion_table/master_lesions.csv")
    fold = C.fold_of_case()

    # ------------------------------------------------------------ oracles
    say("== Cohort-wide checks")
    assert partial or (len(cases) == 1308 and set(cases.case_id) == set(fold)), f"{len(cases)} cases"
    say(f"cases {len(cases)}: tumor+ {int(cases.gt_tumor.sum())}, tumor- {int((cases.gt_tumor == 0).sum())}; "
        f"npz dtypes {cases.npz_dtype.value_counts().to_dict()}")
    say(f"argmax != saved seg: {int(cases.argmax_mismatch.sum())} voxels over all cases; "
        f"(p1 > 0.5) != saved seg: {int(cases.p05_vs_seg_mismatch.sum())}; "
        f"max |p0+p1-1| {cases.max_sum_dev.max():.2e}")
    assert cases.argmax_mismatch.sum() == 0 and cases.p05_vs_seg_mismatch.sum() == 0
    sp1 = pd.read_csv(C.WORK / "s1_case_spacing.csv").set_index("case_id")
    d = np.abs(cases.set_index("case_id")[["sp_x", "sp_y", "sp_z"]].to_numpy()
               - sp1.loc[cases.case_id, ["spacing_x_mm", "spacing_y_mm", "spacing_z_mm"]].to_numpy())
    say(f"spacing (gt_segmentations header) vs S1 CT-affine spacing: max diff {d.max():.2e} mm over 1308 cases; "
        f"cases > 1e-4 mm: {int((d.max(1) > 1e-4).sum())}")

    if partial:
        master = master[master.case_id.isin(cases.case_id)]
    m = master.merge(les, on=["case_id", "lesion_id"], how="outer", suffixes=("", "_s2"), indicator=True)
    assert (m._merge == "both").all() and (partial or len(m) == 1235), m._merge.value_counts()
    assert (m.gt_vox == m.gt_vox_s2).all(), "gt_vox mismatch after join"
    assert (m.seg_touches == m.detected_any).all(), "touch != master detected_any"
    say("lesions: 1235/1235 join on (case_id, lesion_id), gt_vox equal, recomputed touch == master detected_any")
    say(f"search region unavailable (empty envelope): {int((~cases.sr_ok).sum())} cases "
        f"{cases.loc[~cases.sr_ok, 'case_id'].tolist()}")

    # ------------------------------------------------------------ step 1 regression
    say("\n== Step 1 regression (saved segmentation, 'a predicted component touches GT')")
    pos = cases[cases.gt_tumor == 1]
    loc, wrong, none = int(pos.seg_touches_gt.sum()), int((pos.seg_any & ~pos.seg_touches_gt).sum()), int((~pos.seg_any).sum())
    say(f"tumor+ {len(pos)}: localized {loc} ({loc / len(pos):.1%}), only wrong place {wrong}, nothing {none}"
        f"   [carried over: 631 / 288 / 62]")
    big = master[master.largest_in_case][["case_id", "diam_bin"]].copy()
    big["bin"] = big.diam_bin.replace({"<5": "<10", "5-10": "<10"})
    big = big.merge(pos[["case_id", "seg_touches_gt"]], on="case_id")
    carried = {"<10": "0%", "10-20": "31%", "20-40": "66%", ">=40": "86%"}
    for b in ("<10", "10-20", "20-40", ">=40"):
        g = big[big.bin == b]
        say(f"  largest lesion {b:>6}: {int(g.seg_touches_gt.sum())}/{len(g)} = {g.seg_touches_gt.mean():.1%} "
            f"[carried over {carried[b]}]")
    carried_l = {"<5": "2%", "5-10": "0%", "10-20": "26%", "20-40": "60%", ">=40": "83%"}
    say("  lesion-level detect@any-overlap by bin: " + ", ".join(
        f"{b} {m[m.diam_bin == b].seg_touches.mean():.1%} (n={int((m.diam_bin == b).sum())}, carried {carried_l[b]})"
        for b in BINS))

    # ------------------------------------------------------------ per-threshold hits per lesion
    comps["touch_set"] = split_ids(comps.touch_ids)
    comps["peak_set"] = split_ids(comps.peakhit_ids)
    key = {(r.case_id, r.lesion_id) for r in m.itertuples()}
    hits = {rule: {t: set() for t in C.THRESHOLDS} for rule in ("peak", "touch")}
    for rule, col in (("peak", "peak_set"), ("touch", "touch_set")):
        for t, cid, ids in zip(comps.t, comps.case_id, comps[col]):
            for j in ids:
                hits[rule][t].add((cid, j))
    assert all(k in key for rule in hits for t in hits[rule] for k in hits[rule][t])
    t05 = {(r.case_id, r.lesion_id) for r in m.itertuples() if r.seg_touches}
    assert hits["touch"][0.5] == t05, "touch hit at t=0.5 must equal saved-seg touch"
    say("touch hit at t=0.5 from probabilities == saved-seg touch for all 1235 lesions")

    for rule in ("peak", "touch"):
        first = []
        for r in m.itertuples():
            ts = [t for t in C.THRESHOLDS if (r.case_id, r.lesion_id) in hits[rule][t]]
            first.append(max(ts) if ts else np.nan)
        m[f"max_t_hit_{rule}"] = first

    # ------------------------------------------------------------ step 2 per-lesion table
    cz = cases.set_index("case_id")
    neg = cases[(cases.gt_tumor == 0) & cases.sr_ok]
    neg_sr = np.sort(neg.sr_peak_p.to_numpy())
    neg_vol = np.sort(cases[cases.gt_tumor == 0].vol_peak_p.to_numpy())

    def frac_ge(sorted_vals, x):
        return 1.0 - np.searchsorted(sorted_vals, x, "left") / len(sorted_vals)

    for tag in ("les", "dil2"):
        m[f"neg_sr_peak_frac_ge_{tag}"] = [frac_ge(neg_sr, x) for x in m[f"peak_p_{tag}"]]
        m[f"neg_vol_peak_frac_ge_{tag}"] = [frac_ge(neg_vol, x) for x in m[f"peak_p_{tag}"]]
        xyz = pd.DataFrame(m[f"peak_xyz_{tag}"].tolist(), columns=[f"peak_{a}_{tag}" for a in "xyz"])
        m = pd.concat([m.drop(columns=[f"peak_xyz_{tag}"]), xyz], axis=1)
    for c in ("sr_nonlesion_peak_p", "sr_nonlesion_p50", "sr_nonlesion_p99", "sr_peak_p", "vol_peak_p"):
        m[f"case_{c}"] = m.case_id.map(cz[c])
    for c, lc in (("case_sr_nonlesion_peak_p", "case_sr_nonlesion_peak_logit"),
                  ("case_sr_nonlesion_p50", "case_sr_nonlesion_p50_logit"),
                  ("case_sr_nonlesion_p99", "case_sr_nonlesion_p99_logit")):
        m[lc] = np.where(m[c].isna(), np.nan, C.clip_logit(m[c].fillna(0.5)))
    m["missed"] = ~m.seg_touches
    cols = (["case_id", "lesion_id", "fold", "diam_mm", "diam_bin", "gt_vox", "vol_mm3", "tiny_flag",
             "largest_in_case", "n_lesions_in_case", "slice_thickness_mm", "tier", "excluded", "attenuation_v2",
             "lesion_dice", "detected_any", "missed", "in_sr_frac",
             "peak_p_les", "peak_logit_les", "peak_raw_logit_les", "mean_p_les",
             "peak_p_dil2", "peak_logit_dil2", "peak_raw_logit_dil2", "mean_p_dil2",
             "peak_x_les", "peak_y_les", "peak_z_les", "peak_x_dil2", "peak_y_dil2", "peak_z_dil2",
             "sr_bg_frac_ge_les", "sr_bg_frac_ge_dil2",
             "case_sr_nonlesion_peak_p", "case_sr_nonlesion_peak_logit",
             "case_sr_nonlesion_p50", "case_sr_nonlesion_p50_logit",
             "case_sr_nonlesion_p99", "case_sr_nonlesion_p99_logit", "case_vol_peak_p",
             "neg_sr_peak_frac_ge_les", "neg_sr_peak_frac_ge_dil2",
             "neg_vol_peak_frac_ge_les", "neg_vol_peak_frac_ge_dil2",
             "max_t_hit_peak", "max_t_hit_touch"])
    per_lesion = m[cols].sort_values(["case_id", "lesion_id"])
    out = C.WORK / "partial_test" if partial else OUT
    out.mkdir(parents=True, exist_ok=True)
    per_lesion.to_csv(out / "per_lesion_subthreshold.csv", index=False, float_format="%.6g")
    say(f"\nwrote {out / 'per_lesion_subthreshold.csv'} ({len(per_lesion)} rows)")

    # ------------------------------------------------------------ step 3 FROC points
    n_all, n_neg = len(cases), int((cases.gt_tumor == 0).sum())
    neg_ids = set(cases.case_id[cases.gt_tumor == 0])
    rows = []
    for t in C.THRESHOLDS:
        ct = comps[comps.t == t]
        for rule, col in (("peak_in_dil2mm", "peak_set"), ("touch", "touch_set")):
            fp = ct[[len(s) == 0 for s in ct[col]]]
            fp_all, fp_neg = len(fp), int(fp.case_id.isin(neg_ids).sum())
            hk = hits["peak" if rule.startswith("peak") else "touch"][t]
            for lset in ("all", "excl_separated"):
                sub = per_lesion if lset == "all" else per_lesion[~per_lesion.excluded]
                for b in ("all", *BINS):
                    s = sub if b == "all" else sub[sub.diam_bin == b]
                    nh = sum((c, l) in hk for c, l in zip(s.case_id, s.lesion_id))
                    rows.append(dict(t=t, hit_rule=rule, lesion_set=lset, size_bin=b, n_lesions=len(s), n_hit=nh,
                                     sensitivity=nh / len(s) if len(s) else np.nan,
                                     n_components=len(ct), n_fp_components=fp_all,
                                     fp_per_case_all=fp_all / n_all, fp_per_case_neg=fp_neg / n_neg,
                                     n_cases_all=n_all, n_cases_neg=n_neg))
    froc = pd.DataFrame(rows)
    froc.to_csv(out / "nnunet_froc_points.csv", index=False, float_format="%.6g")
    say(f"wrote {out / 'nnunet_froc_points.csv'} ({len(froc)} rows)")
    say("\n== Step 3 FROC (lesion_set=all): sensitivity by bin | FP components/case (all 1308, negatives 327)")
    for rule in ("peak_in_dil2mm", "touch"):
        say(f"-- hit rule: {rule}")
        say(f"{'t':>6} " + " ".join(f"{b:>7}" for b in ("all", *BINS)) + f" {'FP/all':>8} {'FP/neg':>8}")
        for t in C.THRESHOLDS:
            f = froc[(froc.t == t) & (froc.hit_rule == rule) & (froc.lesion_set == "all")].set_index("size_bin")
            say(f"{t:>6g} " + " ".join(f"{f.at[b, 'sensitivity']:>7.1%}" for b in ("all", *BINS))
                + f" {f.at['all', 'fp_per_case_all']:>8.2f} {f.at['all', 'fp_per_case_neg']:>8.2f}")

    # ------------------------------------------------------------ step 4 saturation
    say("\n== Step 4 saturation (search-region voxels = hole-filled pancreas envelope dilated 3 mm)")
    ok = cases[cases.sr_ok]
    for name, g in (("all", ok), ("tumor+", ok[ok.gt_tumor == 1]), ("tumor-", ok[ok.gt_tumor == 0])):
        say(f"{name:>7}: {int(g.sr_vox.sum()):,} voxels; in [{SAT_LO}, {SAT_HI}] {g.sr_n_mid.sum() / g.sr_vox.sum():.3%}; "
            f"p1 >= 1-1e-7 {g.sr_n_ge_hi_clip.sum() / g.sr_vox.sum():.2e}; p1 <= 1e-7 {g.sr_n_le_lo_clip.sum() / g.sr_vox.sum():.2%}; "
            f"p1 == 1.0 exactly {int(g.sr_n_eq_1.sum())}")
    p_les = hist["p_les_pos"]
    p_bg = hist["p_bg_pos"] + hist["p_bg_neg"]
    mid = (P_EDGES[:-1] >= SAT_LO) & (P_EDGES[1:] <= SAT_HI)
    say(f"lesion voxels inside the search region (tumor+): {int(p_les.sum()):,}; in [0.01, 0.99] {p_les[mid].sum() / p_les.sum():.1%}; "
        f"p >= 0.99 {p_les[P_EDGES[:-1] >= 0.99].sum() / p_les.sum():.1%}; p < 0.01 {p_les[P_EDGES[1:] <= 0.01].sum() / p_les.sum():.1%}")
    say(f"non-lesion voxels inside the search region: {int(p_bg.sum()):,}; in [0.01, 0.99] {p_bg[mid].sum() / p_bg.sum():.2%}; "
        f"p >= 0.99 {p_bg[P_EDGES[:-1] >= 0.99].sum() / p_bg.sum():.3%}; p < 0.01 {p_bg[P_EDGES[1:] <= 0.01].sum() / p_bg.sum():.2%}")
    say("components whose peak p is in [0.01, 0.99], per threshold (whole volume | peak inside search region):")
    for t in C.THRESHOLDS:
        ct = comps[comps.t == t]
        inr = ct.peak_p.between(SAT_LO, SAT_HI)
        s = ct[ct.peak_in_sr]
        say(f"  t={t:<6g} n={len(ct):>6} mid {inr.mean():6.1%} | n={len(s):>6} mid {s.peak_p.between(SAT_LO, SAT_HI).mean():6.1%}"
            f"   peak p >= 0.99: {(ct.peak_p >= 0.99).mean():6.1%}")
    say(f"float ceiling: whole-volume max p1 {cases.vol_peak_p.max():.9f}; voxels with p1 == 1.0 exactly {int(cases.n_p1_eq_1.sum())}; "
        f"voxels with p0 == 0 {int(cases.n_p0_eq_0.sum())}; max unclipped logit {cases.vol_peak_raw_logit.max():.2f}; "
        f"clipped-logit ceiling {C.clip_logit(1.0):.2f}")
    for tag in ("les", "dil2"):
        say(f"lesion peak ({tag}): at clip floor (p <= 1e-7) {int((m[f'peak_p_{tag}'] <= C.EPS).sum())}; "
            f"at clip ceiling {int((m[f'peak_p_{tag}'] >= 1 - C.EPS).sum())}; "
            f"max |clipped - unclipped logit| where 1e-7 < p < 1-1e-7: "
            f"{np.nanmax(np.abs(m[f'peak_logit_{tag}'] - m[f'peak_raw_logit_{tag}'])[(m[f'peak_p_{tag}'] > C.EPS) & (m[f'peak_p_{tag}'] < 1 - C.EPS)]):.2e}; "
            f"min unclipped logit {m[f'peak_raw_logit_{tag}'].min():.2f}")
    exp = cases[cases.export == "ok"]
    say(f"export: {len(exp)} crops; float16 p==1.0 voxels {int(exp.export_n_f16_eq_1.sum())}; float16 p==0 voxels {int(exp.export_n_f16_eq_0.sum()):,}")
    plot_hist(hist)

    # ------------------------------------------------------------ readings (a) vs (b)
    say("\n== Readings: missed lesions (saved seg does not touch them), peak p in lesion+2 mm")
    say(f"negative-case search-region peak p: n={len(neg_sr)}, median {np.median(neg_sr):.4g}, "
        f"p10 {np.percentile(neg_sr, 10):.4g}, p90 {np.percentile(neg_sr, 90):.4g}; "
        f"whole-volume peak median {np.median(neg_vol):.4g}")
    for lset, sub in (("all", per_lesion), ("excl_separated", per_lesion[~per_lesion.excluded])):
        say(f"-- lesion set: {lset}")
        for b in ("<10", *BINS[2:]):
            s = sub[sub.missed & (sub.diam_bin.isin(["<5", "5-10"]) if b == "<10" else sub.diam_bin == b)]
            if not len(s):
                continue
            q = s.peak_p_dil2
            say(f"  {b:>6} missed n={len(s):>3}: peak p median {q.median():.3g} [IQR {q.quantile(.25):.3g}-{q.quantile(.75):.3g}], "
                f"logit median {s.peak_logit_dil2.median():.2f}; "
                f"above every negative SR peak {int((s.neg_sr_peak_frac_ge_dil2 == 0).sum())}, "
                f"above the median negative SR peak {int((s.neg_sr_peak_frac_ge_dil2 < 0.5).sum())}, "
                f"median frac of negatives >= it {s.neg_sr_peak_frac_ge_dil2.median():.2f}; "
                f"above own-case non-lesion SR peak {int((s.peak_p_dil2 > s.case_sr_nonlesion_peak_p).sum())}; "
                f"<= own-case SR p50 {int((s.peak_p_dil2 <= s.case_sr_nonlesion_p50).sum())}, "
                f"<= own-case SR p99 {int((s.peak_p_dil2 <= s.case_sr_nonlesion_p99).sum())}; "
                f"at clip floor {int((s.peak_p_dil2 <= C.EPS).sum())}; "
                f"median own-case SR voxel frac >= it {s.sr_bg_frac_ge_dil2.median():.3g}; "
                f"inside SR (in_sr_frac>0) {int((s.in_sr_frac > 0).sum())}")
    (out / "summary.txt" if partial else C.WORK / "summary.txt").write_text("\n".join(LINES) + "\n")

    # ------------------------------------------------------------ step 5 components tables
    if partial:
        return
    cdir = C.WORK / "components"
    cdir.mkdir(exist_ok=True)
    keep = ["case_id", "fold", "t", "comp_id", "n_vox", "vol_mm3", "peak_p", "peak_logit", "peak_raw_logit",
            "peak_x", "peak_y", "peak_z", "peak_in_sr", "frac_in_sr", "touch_ids", "peakhit_ids"]
    for t in C.THRESHOLDS:
        comps.loc[comps.t == t, keep].to_csv(cdir / f"components_t{t:g}.csv", index=False, float_format="%.6g")
    say(f"wrote {len(C.THRESHOLDS)} component tables to {cdir}")


def plot_hist(hist: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, ink, muted = "#fcfcfb", "#1f1f1e", "#6b6a64"
    s_bg, s_les = "#2a78d6", "#eb6834"   # categorical slots 1, 2 (validated default palette)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=surface)
    series = (("non-lesion voxels", hist["p_bg_pos"] + hist["p_bg_neg"], hist["l_bg_pos"] + hist["l_bg_neg"], s_bg),
              ("lesion voxels", hist["p_les_pos"], hist["l_les_pos"], s_les))
    lc = np.clip(L_EDGES, -16.5, 16.5)
    for ax, xlabel in ((axes[0], "nnU-Net tumor probability p (0.01-wide bins)"),
                       (axes[1], "clipped logit of p (0.25-wide bins; outer bins = clip edges)")):
        ax.set_facecolor(surface)
        for name, hp, hl, col in series:
            if ax is axes[0]:
                ax.stairs(np.where(hp > 0, hp, np.nan), P_EDGES, color=col, lw=2, label=name)
            else:
                ax.stairs(np.where(hl > 0, hl, np.nan), lc, color=col, lw=2, label=name)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, color=muted)
        ax.set_ylabel("voxels (log scale)", color=muted)
        ax.tick_params(colors=muted)
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c9c8c0")
    axes[0].axvspan(SAT_LO, SAT_HI, color="#efeee9", zorder=0)
    axes[0].legend(frameon=False, labelcolor=ink)
    fig.suptitle("nnU-Net tumor probability inside the search region, all 1308 OOF cases", color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(C.WORK / "saturation_hist.png", dpi=130, facecolor=surface)


if __name__ == "__main__":
    main()
