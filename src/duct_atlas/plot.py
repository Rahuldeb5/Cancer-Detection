"""Duct atlas figures, QC image and the numbers for the README.

  PYTHONPATH=src .venv/bin/python -m duct_atlas.plot [--tag full] [--no-qc]
  PYTHONPATH=src .venv/bin/python -m duct_atlas.plot --split tumorpos   (optional tumor+/- figure)

Reads work/duct_atlas/<tag>/{atlas.npz, cases.csv} (written by duct_atlas.run) and writes
  results/duct_atlas/duct_atlas.png      normalized frame (gland lengths)
  results/duct_atlas/duct_atlas_mm.png   physical frame (mm, same rotation, no scaling)
  work/duct_atlas/<tag>/projections.npz  raw (unsmoothed) 2D projections behind every panel
  work/duct_atlas/<tag>/qc_cases.png     12 random included cases in the atlas frame (not a deliverable)
and prints the summary statistics quoted in the README.

Columns: axial (u vs ap, summed along si, anterior up) and coronal (u vs si, summed along ap,
superior up). Rows: A section tint + duct overlay, B duct alone, C duct rate.
Display-only Gaussian smoothing (sigma 1.5 bins) is applied to every projection before drawing;
vmax and the duct alpha are computed on the displayed (smoothed) values.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

from duct_atlas.case import ACC_KEYS, process_case
from duct_atlas.grid import MM_GRID, NORM_GRID
from duct_atlas.run import WORK
from tumorlib import io

OUT_DIR = io.REPO_ROOT / "results" / "duct_atlas"
SECTION_COLOR = {"head": "#0072B2", "body": "#E69F00", "tail": "#CC79A7"}  # Okabe-Ito, as plot_pancreas_map.py
SMOOTH_SIGMA = 1.5
VMAX_PCTILE = 99.5
RATE_TISSUE_FLOOR = 0.01
WHITE_BLACK = LinearSegmentedColormap.from_list("wb", ["#ffffff", "#000000"])
VIEWS = {"axial": (2, "ap", "posterior  <-  -> anterior"), "coronal": (1, "si", "inferior  <-  -> superior")}


# ------------------------------------------------------------------ data
def load(tag: str):
    with np.load(WORK / tag / "atlas.npz") as z:
        atlas = {g: {k: z[f"{g}__{k}"] for k in ACC_KEYS} for g in ("norm", "mm")}
    cases = pd.read_csv(WORK / tag / "cases.csv")
    return atlas, cases


def projections(arrs: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """{view: {key: (n_u, n_v) raw projection}} - axial sums out si (axis 2), coronal sums out ap (axis 1)."""
    return {view: {k: a.sum(axis=ax) for k, a in arrs.items()} for view, (ax, _, _) in VIEWS.items()}


def smooth(a: np.ndarray) -> np.ndarray:
    return gaussian_filter(a, SMOOTH_SIGMA)


def vmax_of(a: np.ndarray) -> float:
    nz = a[a > 1e-9 * a.max()] if a.max() > 0 else a
    return float(np.percentile(nz, VMAX_PCTILE)) if nz.size else 1.0


def section_bounds(ok: pd.DataFrame, mm: bool) -> tuple[float, float]:
    s = 1.0 if not mm else ok["L_mm"]
    return float((ok["b_head_body"] * s).median()), float((ok["b_body_tail"] * s).median())


# ------------------------------------------------------------------ figure
def tint_rgba(p: dict[str, np.ndarray]) -> np.ndarray:
    """(n_v, n_u, 4) image: dominant section colour, opacity from tissue density (as plot_pancreas_map.py)."""
    sec = np.stack([smooth(p[s]) for s in ("head", "body", "tail")])
    tissue = smooth(p["tissue"])
    dom = np.argmax(sec, axis=0)
    rgb = np.zeros(tissue.shape + (3,))
    for i, s in enumerate(("head", "body", "tail")):
        rgb[dom == i] = to_rgb(SECTION_COLOR[s])
    pos = tissue[tissue > 1e-9 * tissue.max()]
    alpha = np.clip((tissue / np.percentile(pos, 99)) ** 0.5, 0, 1) * 0.9
    return np.dstack([rgb, alpha]).transpose(1, 0, 2)


def duct_rgba(d: np.ndarray, vmax: float) -> np.ndarray:
    alpha = np.clip(np.clip(d, 0, None) / vmax, 0, 1) ** 0.5
    img = np.zeros(d.shape + (4,))
    img[..., 3] = alpha
    return img.transpose(1, 0, 2)


def atlas_figure(proj, grid, ok: pd.DataFrame, mm: bool, path, title_extra: str = ""):
    unit = "mm" if mm else "gland lengths"
    ext_u = (grid.lo[0], grid.hi[0])
    extent = {"axial": [*ext_u, grid.lo[1], grid.hi[1]], "coronal": [*ext_u, grid.lo[2], grid.hi[2]]}
    hb, bt = section_bounds(ok, mm)
    n = len(ok)

    w = 15.5 if not mm else 17.0
    fig, axes = plt.subplots(3, 2, figsize=(w, 13.5), constrained_layout=True)
    for col, (view, (_, vname, vlabel)) in enumerate(VIEWS.items()):
        p = proj[view]
        d = smooth(p["duct"])
        t = smooth(p["tissue"])
        vmax = vmax_of(d)
        ex = extent[view]

        ax = axes[0, col]
        ax.imshow(tint_rgba(p), origin="lower", extent=ex, aspect="equal", interpolation="bilinear")
        ax.imshow(duct_rgba(d, vmax), origin="lower", extent=ex, aspect="equal", interpolation="bilinear")
        ax.set_title(f"A  {view}: tissue tinted by dominant section, duct in black   (vmax {vmax:.3g} mm$^3$)",
                     fontsize=10, loc="left")

        ax = axes[1, col]
        im = ax.imshow(d.T, origin="lower", extent=ex, aspect="equal", cmap=WHITE_BLACK, interpolation="bilinear",
                       norm=PowerNorm(gamma=0.5, vmin=0, vmax=vmax, clip=True))
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
        cb.set_label(f"duct mm$^3$ per bin, summed over {n} cases", fontsize=8)
        ax.set_title(f"B  {view}: duct alone   (opacity = (v/vmax)$^{{0.5}}$, vmax {vmax:.3g} mm$^3$)",
                     fontsize=10, loc="left")

        ax = axes[2, col]
        rate = np.where(t >= RATE_TISSUE_FLOOR * t.max(), d / np.maximum(t, 1e-12), np.nan)
        im = ax.imshow(rate.T, origin="lower", extent=ex, aspect="equal", cmap="Greens",
                       vmin=0, vmax=float(np.nanpercentile(rate, 99.5)), interpolation="bilinear")
        ax.set_facecolor("#ffffff")
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
        cb.set_label("duct mm$^3$ / tissue mm$^3$", fontsize=8)
        ax.set_title(f"C  {view}: duct rate (blank where tissue < {RATE_TISSUE_FLOOR:.0%} of its max)",
                     fontsize=10, loc="left")

        for r in range(3):
            ax = axes[r, col]
            for x in (hb, bt):
                ax.axvline(x, color="#333333", ls="--", lw=0.8, alpha=0.6)
            ax.axhline(0, color="#999999", lw=0.5, alpha=0.5)
            ax.set_ylabel(f"{vlabel}\n({vname}, {unit} from gland centre)", fontsize=9)
            ax.set_xlabel(f"head end  ->  tail end   (u, {unit})", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.15)
        top = extent[view][3]
        u0, u1 = (0.0, 1.0) if not mm else (0.0, float(ok["L_mm"].median()))
        for x, lab in (((u0 + hb) / 2, "head"), ((hb + bt) / 2, "body"), ((bt + u1) / 2, "tail")):
            axes[0, col].text(x, top * 0.92, lab, ha="center", va="top", fontsize=9, color="#222222")

    bin_txt = f"{grid.step:g} gland length" if not mm else f"{grid.step:g} mm"
    frame_txt = ("every gland aligned on PC1 (head->tail) and scaled to its own length (u: 0 = head end, 1 = tail end)"
                 if not mm else "every gland aligned on PC1 (head->tail), NOT scaled (u_mm: 0 = 1st-percentile head end)")
    fig.suptitle(f"PanTS pancreatic-duct atlas, {'normalized' if not mm else 'physical'} frame{title_extra}   "
                 f"N = {n} cases, bin {bin_txt}\n{frame_txt}; median section boundaries (dashed) "
                 f"head|body {hb:.3g}, body|tail {bt:.3g} {'' if not mm else 'mm'}", fontsize=11)
    handles = [Patch(facecolor=SECTION_COLOR[s], label=f"{s} tissue") for s in SECTION_COLOR]
    handles.append(Patch(facecolor="black", label="duct (opacity = annotated duct volume)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ QC
def qc_figure(ok: pd.DataFrame, path, n: int = 12, seed: int = 10):
    ids = ok["case_id"].sample(n, random_state=seed).tolist()
    fig, axes = plt.subplots(4, 6, figsize=(22, 11), sharex=True, sharey=True)
    rng = np.random.default_rng(seed)
    for i, cid in enumerate(ids):
        row, pts = process_case(cid, keep_points=True)
        for j, view in enumerate(VIEWS):
            ax = axes[i // 6 * 2 + j, i % 6]   # case i: column i%6, rows (axial, coronal) of block i//6
            yc = 1 if view == "axial" else 2  # points are (u, ap, si)
            for s in ("head", "body", "tail"):
                q = pts[s]
                q = q[rng.choice(len(q), min(3000, len(q)), replace=False)] if len(q) else q
                ax.scatter(q[:, 0], q[:, yc], s=1, c=SECTION_COLOR[s], alpha=0.3, lw=0)
            d = pts["duct"]
            ax.scatter(d[:, 0], d[:, yc], s=2, c="black", lw=0)
            ax.set_title(f"{cid[-5:]} {view}  L={row['L_mm']:.0f} mm", fontsize=8)
            ax.axvline(0, color="#888", lw=0.5)
            ax.axvline(1, color="#888", lw=0.5)
            ax.set_xlim(-0.15, 1.15)
            ax.set_ylim(-0.4, 0.4)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
    fig.suptitle("QC: 12 random included cases in the atlas frame (rows alternate axial u-ap / coronal u-si); "
                 "black = duct", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ numbers
def ridge_profile(arr_norm: dict[str, np.ndarray], step: int = 5) -> pd.DataFrame:
    """Per u slab: duct mass, duct rate, and the duct's ap/si centroid and spread (gland lengths)."""
    g = NORM_GRID
    u = g.centers(0)
    ap, si = g.centers(1), g.centers(2)
    rows = []
    for i in range(0, len(u), step):
        d = arr_norm["duct"][i:i + step]
        t = arr_norm["tissue"][i:i + step]
        m = d.sum()
        r = {"u_lo": round(g.edges(0)[i], 3), "u_hi": round(g.edges(0)[min(i + step, len(u))], 3),
             "duct_mm3": m, "tissue_mm3": t.sum(), "rate": m / t.sum() if t.sum() else np.nan}
        if m > 0:
            pa, ps = d.sum(axis=(0, 2)), d.sum(axis=(0, 1))
            for name, prof, c in (("ap", pa, ap), ("si", ps, si)):
                mu = (prof * c).sum() / m
                r[f"{name}_mean"] = mu
                r[f"{name}_sd"] = np.sqrt((prof * (c - mu) ** 2).sum() / m)
            tp = t.sum(axis=(0, 2)), t.sum(axis=(0, 1))
            for name, prof, c in (("ap", tp[0], ap), ("si", tp[1], si)):
                mu = (prof * c).sum() / prof.sum()
                r[f"tissue_{name}_sd"] = np.sqrt((prof * (c - mu) ** 2).sum() / prof.sum())
        rows.append(r)
    return pd.DataFrame(rows)


def coverage_profile(ok: pd.DataFrame) -> pd.DataFrame:
    """Fraction of included cases whose duct p5-p95 u-interval covers each u."""
    u = np.round(np.arange(-0.1, 1.11, 0.05), 2)
    cov = [((ok["duct_u_p5"] <= x) & (ok["duct_u_p95"] >= x)).mean() for x in u]
    return pd.DataFrame({"u": u, "frac_cases_covering": np.round(cov, 3)})


def report(atlas, cases: pd.DataFrame):
    q = lambda s: " / ".join(f"{v:.3g}" for v in s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    print(f"cases: {len(cases)}")
    print(cases["status"].value_counts().to_string())
    duct = cases[cases["status"] != "no_duct"]
    print(f"\nwith duct >= 8 mm^3: {len(duct)}; duct 0 < v < 8 mm^3: "
          f"{int(((cases.status == 'no_duct') & (cases.duct_mm3 > 0)).sum())}; duct exactly 0: "
          f"{int((cases.duct_mm3 == 0).sum())}")
    for col in ("shape_mismatch", "load_error", "error"):
        sub = cases[cases.status == col]
        if len(sub):
            print(f"{col}: {sub.case_id.tolist()}  {sub.msgs.tolist()}")
    for k in ("body", "tail", "duct"):
        sub = cases[cases[f"{k}_affine_differs"] == True]  # noqa: E712
        print(f"{k} affine differs from head affine: {len(sub)} {sub.case_id.tolist()[:20]}")
    print("head affine vs CT (cohort only):", cases["head_vs_ct_affine"].value_counts().to_dict())
    msgs = cases["msgs"].dropna()
    msgs = msgs[msgs.astype(str).str.len() > 0]
    print(f"cases with loader messages: {len(msgs)}")
    for cid, m in zip(cases.loc[msgs.index, "case_id"].head(15), msgs.head(15)):
        print(f"   {cid}: {str(m)[:200]}")

    ok = cases[cases.status == "ok"]
    print(f"\nincluded N = {len(ok)}")
    print("L_mm  quantiles 1/5/25/50/75/95/99:", q(ok.L_mm))
    print("PC1 var share quantiles:", q(ok.var_share))
    print("env cm3 quantiles:", q(ok.env_cm3))
    print("duct mm3 quantiles:", q(ok.duct_mm3))
    print("duct u-extent (p5-p95) mm quantiles:", q((ok.duct_u_p95 - ok.duct_u_p5) * ok.L_mm))
    print("duct u-extent (p5-p95) / L quantiles:", q(ok.duct_u_p95 - ok.duct_u_p5))
    print("duct inside head|body|tail fraction: max", ok.duct_in_env_frac.max())
    framed = cases[cases.var_share.notna()]
    print(f"\nframes built: {len(framed)}; eigh PC1 negated by head->tail rule: {int(framed.eig_flipped.sum())} "
          f"(eigh sign is arbitrary); head->tail NOT toward patient left: {int((~framed.leftward.astype(bool)).sum())}"
          f" of which included: {int((~ok.leftward.astype(bool)).sum())}")
    print("   not-leftward included:", ok.loc[~ok.leftward.astype(bool), "case_id"].tolist()[:30])

    hb = ok.dropna(subset=["body_u"])
    order = (hb.head_u < hb.body_u) & (hb.body_u < hb.tail_u)
    print(f"\nhead<body<tail centroid on u: {int(order.sum())}/{len(hb)} with body; empty body included: "
          f"{int(ok.body_u.isna().sum())}; head<tail: {int((ok.head_u < ok.tail_u).sum())}/{len(ok)}")
    if (~order).any():
        print("   violators:", hb.loc[~order, ["case_id", "head_u", "body_u", "tail_u"]].round(3).to_string(index=False))
    b1, b2 = section_bounds(ok, False)
    print(f"median boundaries u: head|body {b1:.3f} (lesion map 0.38, diff {b1 - 0.38:+.3f}); body|tail {b2:.3f} "
          f"(0.68, diff {b2 - 0.68:+.3f})")
    print(f"median boundaries mm: {section_bounds(ok, True)}")
    print("median section centroids u:", ok[["head_u", "body_u", "tail_u"]].median().round(3).to_dict())

    cal = pd.read_csv(io.REPO_ROOT / "results/geometry_results/duct_caliber.csv")
    m = cal.merge(cases, on="case_id", how="left")
    have = m.status.notna() & (m.status != "no_duct")
    print(f"\ncohort (duct_caliber.csv) {len(cal)} cases: MPD present {int(cal.mpd_present.sum())}; here duct>=8mm^3 "
          f"{int(have.sum())} (diff {int(have.sum()) - int(cal.mpd_present.sum()):+d}); included {int((m.status == 'ok').sum())}")
    print(pd.crosstab(m.mpd_present, have.rename("duct>=8 here")).to_string())

    for g in ("norm", "mm"):
        tot = ok.duct_mm3.sum()
        ing = atlas[g]["duct"].sum()
        print(f"[{g}] duct in grid {ing:.6g} vs per-case sum {tot:.6g} (ratio {ing / tot:.6f}); "
              f"tissue {atlas[g]['tissue'].sum() / ok.tissue_mm3.sum():.6f}")
    print("\nridge profile (gland lengths; u slabs of 0.05):")
    print(ridge_profile(atlas["norm"]).round(4).to_string(index=False))
    print("\nduct coverage profile:")
    print(coverage_profile(ok).T.to_string(header=False))
    return ok


# ------------------------------------------------------------------ optional tumor+ / tumor- split
def split_figure(full_tag: str, pos_tag: str, path):
    """tumor+ vs tumor- (results/PanTS_metadata_new.csv `tumor?`), normalized frame. The atlas is a sum
    over cases, so tumor- = full - tumor+ (pos_tag = a run over the included tumor+ ids only).
    Duct rows are divided by N (mm^3 per bin per case) so the groups share one scale."""
    full, cases = load(full_tag)
    pos, pos_cases = load(pos_tag)
    ok = cases[cases.status == "ok"]
    pos_ids = set(pos_cases.loc[pos_cases.status == "ok", "case_id"])
    a = ok[ok.case_id.isin(pos_ids)].set_index("case_id").sort_index()
    b = pos_cases.set_index("case_id").loc[a.index]
    assert np.allclose(a[["duct_mm3", "L_mm", "var_share"]], b[["duct_mm3", "L_mm", "var_share"]]), "rerun != full run"
    neg = {k: np.clip(full["norm"][k] - pos["norm"][k], 0, None) for k in ACC_KEYS}
    groups = {"tumor+": (pos["norm"], len(pos_ids)), "tumor-": (neg, len(ok) - len(pos_ids))}
    g = NORM_GRID
    extent = {"axial": [g.lo[0], g.hi[0], g.lo[1], g.hi[1]], "coronal": [g.lo[0], g.hi[0], g.lo[2], g.hi[2]]}
    proj = {name: projections(arrs) for name, (arrs, _) in groups.items()}
    hb, bt = section_bounds(ok, False)

    fig, axes = plt.subplots(4, 2, figsize=(15.5, 17), constrained_layout=True)
    for col, view in enumerate(VIEWS):
        _, vname, vlabel = VIEWS[view]
        per_case = {nm: smooth(proj[nm][view]["duct"]) / n for nm, (_, n) in groups.items()}
        vmax = max(vmax_of(v) for v in per_case.values())
        rates = {}
        for nm in groups:
            t = smooth(proj[nm][view]["tissue"])
            rates[nm] = np.where(t >= RATE_TISSUE_FLOOR * t.max(), smooth(proj[nm][view]["duct"]) / np.maximum(t, 1e-12), np.nan)
        rmax = max(float(np.nanpercentile(r, 99.5)) for r in rates.values())
        for r, nm in enumerate(groups):
            n = groups[nm][1]
            ax = axes[r, col]
            im = ax.imshow(per_case[nm].T, origin="lower", extent=extent[view], aspect="equal", cmap=WHITE_BLACK,
                           norm=PowerNorm(gamma=0.5, vmin=0, vmax=vmax, clip=True), interpolation="bilinear")
            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01).set_label("duct mm$^3$ per bin per case", fontsize=8)
            ax.set_title(f"{nm} (N={n})  {view}: duct per case   (vmax {vmax:.3g})", fontsize=10, loc="left")
            ax = axes[r + 2, col]
            im = ax.imshow(rates[nm].T, origin="lower", extent=extent[view], aspect="equal", cmap="Greens",
                           vmin=0, vmax=rmax, interpolation="bilinear")
            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01).set_label("duct mm$^3$ / tissue mm$^3$", fontsize=8)
            ax.set_title(f"{nm} (N={n})  {view}: duct rate", fontsize=10, loc="left")
        for r in range(4):
            ax = axes[r, col]
            for x in (hb, bt):
                ax.axvline(x, color="#333333", ls="--", lw=0.8, alpha=0.6)
            ax.axhline(0, color="#999999", lw=0.5, alpha=0.5)
            ax.set_ylabel(f"{vlabel}\n({vname}, gland lengths)", fontsize=9)
            ax.set_xlabel("head end  ->  tail end   (u, gland lengths)", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.15)
    fig.suptitle(f"Duct atlas split by tumor? (PanTS metadata), normalized frame, bin {g.step:g}; same colour scale "
                 f"within each row type; dashed = median section boundaries (all {len(ok)} cases) {hb:.3g} / {bt:.3g}",
                 fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    # numbers: share of duct and tissue per section, and duct u-centroid, per group
    u = g.centers(0)
    for nm, (arrs, n) in groups.items():
        d, t = arrs["duct"].sum(axis=(1, 2)), arrs["tissue"].sum(axis=(1, 2))
        sh = lambda v, lo, hi: v[(u >= lo) & (u < hi)].sum() / v.sum()
        print(f"{nm} N={n}: duct mm3/case {d.sum() / n:.0f}; duct share head/body/tail "
              f"{sh(d, -9, hb):.3f}/{sh(d, hb, bt):.3f}/{sh(d, bt, 9):.3f}; tissue share "
              f"{sh(t, -9, hb):.3f}/{sh(t, hb, bt):.3f}/{sh(t, bt, 9):.3f}; duct u-centroid {(d * u).sum() / d.sum():.3f}")
    grp = ok.assign(tumor=ok.case_id.isin(pos_ids))
    print(grp.groupby("tumor")[["duct_mm3", "L_mm"]].median().round(1).to_string())
    print(grp.assign(ext_mm=(grp.duct_u_p95 - grp.duct_u_p5) * grp.L_mm).groupby("tumor")["ext_mm"].median().round(1).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--no-qc", action="store_true")
    ap.add_argument("--split", help="tag of a tumor+ subset run -> duct_atlas_by_tumor.png only")
    a = ap.parse_args()
    if a.split:
        split_figure(a.tag, a.split, OUT_DIR / "duct_atlas_by_tumor.png")
        return
    atlas, cases = load(a.tag)
    ok = report(atlas, cases)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    proj = {g: projections(atlas[g]) for g in ("norm", "mm")}
    np.savez_compressed(WORK / a.tag / "projections.npz",
                        **{f"{g}__{v}__{k}": arr for g, d in proj.items() for v, dd in d.items() for k, arr in dd.items()})
    suffix = "" if a.tag == "full" else f"_{a.tag}"
    out = OUT_DIR if a.tag == "full" else WORK / a.tag
    atlas_figure(proj["norm"], NORM_GRID, ok, False, out / f"duct_atlas{suffix}.png")
    atlas_figure(proj["mm"], MM_GRID, ok, True, out / f"duct_atlas_mm{suffix}.png")
    if not a.no_qc:
        qc_figure(ok, WORK / a.tag / "qc_cases.png")
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
