"""S12 steps 4-5: GMM phase rule, report tables, scatter PNGs.

Inputs: work/phase/measure.csv (phase.measure), results/PanTS_metadata_new.csv (metadata phase and
tumor status: used ONLY for the report tables, never in the fit), results/attenuation_v3 (gland-pool
regression check).
Outputs:
  results/phase_estimate/phase_estimate.csv              one row per cohort case
  work/phase/analysis.md                                 every table the README quotes
  notes/LoG_notes/SESSION_12_scatter_{metadata,estimated}.png

The rule (core.N_COMPONENTS, POST_MIN, name_clusters) is fixed in core.py. Fit features are exactly
(aorta_hu, pv_hu); cases where either is NaN are "unmeasurable" and are not in the fit.

    PYTHONPATH=src .venv/bin/python -m phase.analyze
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.mixture import GaussianMixture

from phase import core

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = core.REPO
MEAS = REPO / "work" / "phase" / "measure.csv"
META = REPO / "results" / "PanTS_metadata_new.csv"
ATTN = REPO / "results" / "attenuation_v3" / "attenuation_labels_v3.csv"
OUT_CSV = REPO / "results" / "phase_estimate" / "phase_estimate.csv"
OUT_MD = REPO / "work" / "phase" / "analysis.md"
FIG_DIR = REPO / "notes" / "LoG_notes"

EST_ORDER = ["non-contrast", "arterial", "venous", "uncertain", "unmeasurable"]
META_ORDER = ["Non-contrast", "Arterial", "Venous", "Delay", "missing"]
# dataviz reference palette, categorical slots 1-3 (the all-pairs-validated set for scatters), light mode
COLORS = {"non-contrast": "#2a78d6", "arterial": "#eb6834", "venous": "#1baf7a"}
META_COLORS = {"Non-contrast": COLORS["non-contrast"], "Arterial": COLORS["arterial"], "Venous": COLORS["venous"]}
NEUTRAL = "#8a8984"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def md(df: pd.DataFrame, floatfmt="{:.1f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([str(df.index.name or "")] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                 else ("" if pd.isna(v) else str(v)) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def fit_gmm(X: np.ndarray, seed: int = core.GMM_SEED, k: int = core.N_COMPONENTS) -> GaussianMixture:
    return GaussianMixture(n_components=k, covariance_type="full", n_init=core.GMM_N_INIT,
                           random_state=seed).fit(X)


def describe(s: pd.Series) -> dict:
    s = s.dropna()
    if not len(s):
        return {"n": 0, "median": np.nan, "q25": np.nan, "q75": np.nan}
    return {"n": len(s), "median": s.median(), "q25": s.quantile(.25), "q75": s.quantile(.75)}


# ------------------------------------------------------------------ plotting
def _style(ax, title):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("Aorta median HU (eroded, middle 80% of slices)", color=INK2, fontsize=10)
    ax.set_ylabel("Portal/splenic vein median HU", color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, loc="left")


def scatter(d: pd.DataFrame, col: str, order, colors, title, path, gmm=None, names=None):
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax, title)
    lim_x = (min(-10, d.aorta_hu.min() - 10), min(d.aorta_hu.quantile(0.995) + 40, 650))
    lim_y = (min(-10, d.pv_hu.min() - 10), d.pv_hu.max() + 20)
    ax.plot([0, 700], [0, 700], color=INK2, lw=1, ls="--", zorder=1)
    ax.text(lim_x[1] - 5, min(lim_x[1], lim_y[1]) - 8, "aorta = PV", color=INK2, fontsize=8, ha="right", va="top")
    for lvl in order:
        g = d[d[col] == lvl]
        if not len(g):
            continue
        if lvl in colors:
            ax.scatter(g.aorta_hu, g.pv_hu, s=14, c=colors[lvl], alpha=0.75, edgecolors=SURFACE,
                       linewidths=0.6, label=f"{lvl} (n={len(g)})", zorder=3)
        else:  # neutral classes: hollow gray markers, distinct shape
            ax.scatter(g.aorta_hu, g.pv_hu, s=22, facecolors="none", edgecolors=NEUTRAL, linewidths=1.0,
                       marker="D" if lvl in ("Delay",) else "o", label=f"{lvl} (n={len(g)})", zorder=4)
    off = d[(d.aorta_hu > lim_x[1]) & d.pv_hu.notna()]
    if len(off):
        ax.text(lim_x[1] - 5, lim_y[0] + 8, f"{len(off)} case(s) with aorta > {lim_x[1]:.0f} HU not shown",
                color=INK2, fontsize=8, ha="right")
    if gmm is not None:
        for i, (mu, cov) in enumerate(zip(gmm.means_, gmm.covariances_)):
            w, v = np.linalg.eigh(cov)
            t = np.linspace(0, 2 * np.pi, 200)
            ell = (v @ (np.sqrt(w)[:, None] * 2.0 * np.vstack([np.cos(t), np.sin(t)]))).T + mu
            ax.plot(ell[:, 0], ell[:, 1], color=INK, lw=1, alpha=0.6, zorder=5)
            ax.scatter([mu[0]], [mu[1]], marker="x", s=60, c=INK, linewidths=1.6, zorder=6)
            ax.annotate(f"{names[i]}\n({mu[0]:.0f}, {mu[1]:.0f})", mu, xytext=(8, 8), textcoords="offset points",
                        fontsize=8, color=INK, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))
    ax.set_xlim(*lim_x)
    ax.set_ylim(*lim_y)
    leg = ax.legend(loc="upper left", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    m = pd.read_csv(MEAS)
    meta = pd.read_csv(META)[["PanTS ID", "ct phase", "tumor?"]].rename(
        columns={"PanTS ID": "case_id", "ct phase": "metadata_phase", "tumor?": "tumor"})
    d = m.merge(meta, on="case_id", how="left", validate="1:1")
    assert len(d) == 1308 and d.tumor.notna().all()
    d["metadata_phase"] = d.metadata_phase.fillna("missing")

    # gland pool regression against S3 (every tumor+ case)
    s3 = pd.read_csv(ATTN).groupby("case_id").ref_global_median.first()
    j = d.set_index("case_id").gland_hu.to_frame().join(s3, how="inner")
    gdiff = (j.gland_hu - j.ref_global_median).abs()
    assert len(j) == 981 and gdiff.max() < 0.05, (len(j), gdiff.max())  # S3 CSV is written at %.4g

    # ---------------------------------------------------------- phase rule (fixed in core.py)
    ok = d.aorta_hu.notna() & d.pv_hu.notna()
    X = d.loc[ok, ["aorta_hu", "pv_hu"]].to_numpy()
    gmm = fit_gmm(X)
    names = core.name_clusters(gmm.means_)
    post = gmm.predict_proba(X)
    label, pmax, best = core.assign(post, names)
    d["est_phase"] = "unmeasurable"
    d["posterior"] = np.nan
    d["argmax_phase"] = "unmeasurable"
    d.loc[ok, "est_phase"] = label
    d.loc[ok, "posterior"] = pmax
    d.loc[ok, "argmax_phase"] = best
    d["pv_enh_frac"] = np.nan
    d.loc[ok, "pv_enh_frac"] = core.pv_enhancement_fraction(d.loc[ok, "pv_hu"], gmm.means_, names)
    d["low_vox_flag"] = (d.aorta_n_vox < core.MIN_VOX) | (d.pv_n_vox < core.MIN_VOX)
    d["early_arterial"] = (d.est_phase == "arterial") & (d.pv_enh_frac < core.EARLY_ART_F)

    S = [f"# S12 analysis ({len(d)} cases; {int(ok.sum())} measurable, fit on those)\n"]
    S.append("## Gland pool regression vs S3\n")
    S.append(f"981 tumor+ cases, max |gland_hu - S3 ref_global_median| = {gdiff.max():.4f} HU (S3 CSV is %.4g)\n")

    S.append("## GMM (k=3, full covariance, n_init=20, seed 0)\n")
    inv = {v: k for k, v in names.items()}
    cen = pd.DataFrame({p: {"aorta_mu": gmm.means_[inv[p], 0], "pv_mu": gmm.means_[inv[p], 1],
                            "aorta_sd": np.sqrt(gmm.covariances_[inv[p]][0, 0]),
                            "pv_sd": np.sqrt(gmm.covariances_[inv[p]][1, 1]),
                            "corr": gmm.covariances_[inv[p]][0, 1] / np.sqrt(
                                gmm.covariances_[inv[p]][0, 0] * gmm.covariances_[inv[p]][1, 1]),
                            "weight": gmm.weights_[inv[p]], "aorta-pv": gmm.means_[inv[p], 0] - gmm.means_[inv[p], 1]}
                        for p in core.PHASES}).T
    cen.index.name = "cluster"
    S.append(md(cen, "{:.2f}") + "\n")

    # stability: other seeds, and refit without low-voxel cases
    agree = []
    for seed in range(1, 11):
        g2 = fit_gmm(X, seed=seed)
        b2 = np.array([core.name_clusters(g2.means_)[i] for i in g2.predict_proba(X).argmax(1)])
        agree.append((b2 == best).mean())
    lv = d.loc[ok, "low_vox_flag"].to_numpy()
    g3 = fit_gmm(X[~lv])
    b3 = np.array([core.name_clusters(g3.means_)[i] for i in g3.predict_proba(X).argmax(1)])
    S.append(f"Argmax agreement with seeds 1-10: min {min(agree):.4f}, max {max(agree):.4f}. "
             f"Refit without {int(lv.sum())} low-voxel cases: argmax agreement {(b3 == best).mean():.4f}.\n")

    # how many components does the scatter support? (description only; the rule stays k=3)
    S.append("## BIC by number of components (description only)\n")
    bic = {}
    fits = {}
    for k in range(1, 7):
        g = fit_gmm(X, k=k)
        fits[k] = g
        bic[k] = g.bic(X)
    bt = pd.DataFrame({"BIC": bic, "dBIC_vs_k3": {k: v - bic[3] for k, v in bic.items()}})
    bt.index.name = "k"
    S.append(md(bt, "{:.1f}") + "\n")
    for k in (4, 5):
        g = fits[k]
        o = np.argsort(g.means_[:, 0])
        t = pd.DataFrame({"aorta_mu": g.means_[o, 0], "pv_mu": g.means_[o, 1], "weight": g.weights_[o],
                          "aorta-pv": g.means_[o, 0] - g.means_[o, 1],
                          "n_argmax": np.bincount(g.predict(X), minlength=k)[o]})
        t.index.name = f"k={k} comp"
        S.append(f"### k={k} components (sorted by aorta mean)\n")
        S.append(md(t, "{:.2f}") + "\n")
        # where do k=4/5 components land in the k=3 labels?
        ct = pd.crosstab(pd.Series(np.argsort(o)[g.predict(X)], name=f"k{k}_comp(sorted)"),
                         pd.Series(best, name="k3_argmax"))
        S.append(md(ct, "{:.0f}") + "\n")

    # ---------------------------------------------------------- report tables
    S.append("## Confusion: metadata phase (rows) x estimated phase (cols)\n")
    conf = pd.crosstab(d.metadata_phase, d.est_phase).reindex(index=META_ORDER, columns=EST_ORDER).fillna(0).astype(int)
    conf = conf.loc[conf.sum(1) > 0]
    conf["total"] = conf.sum(1)
    conf.loc["total"] = conf.sum(0)
    S.append(md(conf, "{:.0f}") + "\n")
    S.append("Same, argmax phase (no uncertain class):\n")
    confb = pd.crosstab(d.metadata_phase, d.argmax_phase).reindex(
        index=META_ORDER, columns=list(core.PHASES) + ["unmeasurable"]).fillna(0).astype(int)
    confb = confb.loc[confb.sum(1) > 0]
    S.append(md(confb, "{:.0f}") + "\n")
    sure = d[d.est_phase.isin(core.PHASES) & d.metadata_phase.isin(["Non-contrast", "Arterial", "Venous"])]
    agree_md = (sure.metadata_phase.str.lower() == sure.est_phase).mean()
    S.append(f"Agreement among confident estimates with a 3-class metadata label: {agree_md:.3f} (n={len(sure)})\n")

    S.append("## Metadata-'Arterial' scans: what do they look like?\n")
    art = d[d.metadata_phase == "Arterial"]
    n_art = len(art)
    rows = {}
    for lab, mask in [("est venous", art.est_phase == "venous"),
                      ("est arterial, early (PV enh. frac < 0.5)", art.early_arterial),
                      ("est arterial, late (PV enh. frac >= 0.5)", (art.est_phase == "arterial") & ~art.early_arterial),
                      ("est non-contrast", art.est_phase == "non-contrast"),
                      ("uncertain", art.est_phase == "uncertain"),
                      ("unmeasurable", art.est_phase == "unmeasurable")]:
        k = int(mask.sum())
        lo, hi = core.wilson(k, n_art)
        rows[lab] = {"n": k, "pct": 100 * k / n_art, "ci_lo": 100 * lo, "ci_hi": 100 * hi}
    t = pd.DataFrame(rows).T
    t.index.name = f"metadata Arterial (n={n_art})"
    S.append(md(t) + "\n")
    S.append("PV enhancement fraction among estimated-arterial scans (all metadata labels): "
             + str(d.loc[d.est_phase == "arterial", "pv_enh_frac"].describe().round(3).to_dict()) + "\n")

    S.append("## Gland HU (S3 v3 pool) by estimated phase -- descriptive, not in the rule\n")
    gt = pd.DataFrame({p: describe(d.loc[d.est_phase == p, "gland_hu"]) for p in EST_ORDER}).T
    gt.index.name = "est_phase"
    S.append(md(gt) + "\n")
    gt2 = pd.DataFrame({p: describe(d.loc[d.metadata_phase == p, "gland_hu"]) for p in META_ORDER}).T
    gt2.index.name = "metadata_phase"
    S.append(md(gt2) + "\n")
    S.append("Gland median HU, metadata (rows) x estimated (cols):\n")
    gx = d.pivot_table(index="metadata_phase", columns="est_phase", values="gland_hu", aggfunc="median") \
        .reindex(index=[p for p in META_ORDER if p in d.metadata_phase.values], columns=EST_ORDER)
    S.append(md(gx) + "\n")
    S.append("Early vs late arterial gland HU: "
             + str({lab: describe(d.loc[msk, "gland_hu"]) for lab, msk in
                    [("early", d.early_arterial), ("late", (d.est_phase == "arterial") & ~d.early_arterial)]}) + "\n")

    S.append("## Tumor+ vs tumor- phase mix\n")
    for col, order in (("est_phase", EST_ORDER), ("metadata_phase", META_ORDER)):
        mix = pd.crosstab(d[col], d.tumor.map({1: "tumor+", 0: "tumor-"})).reindex(order).fillna(0).astype(int)
        mix = mix.loc[mix.sum(1) > 0]
        pct = mix / mix.sum(0) * 100
        tab = pd.concat({"n": mix, "%": pct}, axis=1)
        tab.index.name = col
        S.append(md(tab) + "\n")
        core3 = mix.reindex([p for p in order if p.lower() in core.PHASES]).dropna()
        chi2, p, dof, _ = chi2_contingency(core3.to_numpy())
        S.append(f"chi-square on the 3 phase classes ({col}): chi2={chi2:.2f}, dof={dof}, p={p:.3g}\n")
    S.append("Estimated-phase mix by metadata phase and tumor status:\n")
    tri = pd.crosstab([d.metadata_phase, d.tumor], d.est_phase).reindex(columns=EST_ORDER).fillna(0).astype(int)
    S.append(md(tri.reset_index().set_index(["metadata_phase"]), "{:.0f}") + "\n")

    S.append("## Slice thickness by estimated phase (mm, along slice axis)\n")
    th = pd.DataFrame({p: describe(d.loc[d.est_phase == p, "slice_thickness_mm"]) for p in EST_ORDER}).T
    tb = pd.crosstab(d.est_phase, pd.cut(d.slice_thickness_mm, [0, 2, 4, 99], labels=["<=2", "2-4", ">4"])) \
        .reindex(EST_ORDER)
    th = th.join((tb.div(tb.sum(1), axis=0) * 100).add_prefix("% "))
    th.index.name = "est_phase"
    S.append(md(th, "{:.2f}") + "\n")
    th2 = pd.crosstab([d.est_phase, d.tumor], columns="median_thick", values=d.slice_thickness_mm,
                      aggfunc="median").reindex(EST_ORDER, level=0)
    S.append("Median thickness by estimated phase x tumor:\n" + md(th2.reset_index().set_index("est_phase"), "{:.2f}") + "\n")

    S.append("## QC\n")
    S.append(f"low-voxel flag (<{core.MIN_VOX} eroded voxels in aorta or PV): {int(d.low_vox_flag.sum())}; "
             f"unmeasurable: {int((~ok).sum())}; aorta & veins overlap > 0: {int((d.ov_aorta_pv > 0).sum())}; "
             f"aorta & postcava > 1000 vox: {int((d.ov_aorta_ivc > 1000).sum())}\n")
    S.append("Est phase of low-voxel cases: " + str(d.loc[d.low_vox_flag, "est_phase"].value_counts().to_dict()) + "\n")
    S.append("Posterior quantiles (measurable): " + str(d.posterior.quantile([.05, .1, .25, .5]).round(3).to_dict()) + "\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(S))

    # ---------------------------------------------------------- deliverable CSV
    def qc(r):
        notes = []
        if r.est_phase == "unmeasurable":
            notes.append("aorta or PV mask empty after erosion")
        elif r.low_vox_flag:
            notes.append(f"<{core.MIN_VOX} eroded vessel voxels")
        if r.ov_aorta_ivc > 1000:
            notes.append("aorta mask overlaps postcava >1000 vox")
        if not r.aorta_affine_ok or not r.veins_affine_ok:
            notes.append("mask affine != CT (voxels aligned; CT grid used)")
        return "; ".join(notes)
    d["qc_note"] = d.apply(qc, axis=1)
    keep = ["case_id", "aorta_hu", "pv_hu", "gland_hu", "aorta_n_vox", "pv_n_vox", "low_vox_flag",
            "est_phase", "posterior", "argmax_phase", "pv_enh_frac", "early_arterial", "metadata_phase",
            "slice_thickness_mm", "aorta_iqr", "pv_iqr", "qc_note"]
    out = d[keep].sort_values("case_id").astype({"aorta_n_vox": "Int64", "pv_n_vox": "Int64"})
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.4g")

    # ---------------------------------------------------------- figures
    dd = d[ok]
    scatter(dd, "metadata_phase", META_ORDER, META_COLORS,
            "Aorta vs portal-vein HU, coloured by METADATA phase", FIG_DIR / "SESSION_12_scatter_metadata.png")
    scatter(dd, "est_phase", EST_ORDER, COLORS,
            "Aorta vs portal-vein HU, coloured by ESTIMATED phase (GMM k=3, 2-SD ellipses)",
            FIG_DIR / "SESSION_12_scatter_estimated.png", gmm=gmm, names=names)
    print(OUT_MD.read_text())
    print(f"wrote {OUT_CSV}, {OUT_MD}, figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
