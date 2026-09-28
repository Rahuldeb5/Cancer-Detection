"""Pancreas-shaped maps of where every non-excluded lesion sits, one panel per size bin.

Background = pancreas tissue (union of pancreas | head | body | tail masks, sampled equally from
every case and overlaid in a common gland-aligned frame -- see pancreas_map.py), tinted by which
section (head/body/tail) dominates each spot and faded by tissue density, so the composite reads
as a pancreas silhouette. Black dots = lesion centroids.

  pancreas_map_axial_by_size.png    u (head->tail) vs anterior offset  (anterior up)
  pancreas_map_coronal_by_size.png  u (head->tail) vs superior offset  (superior up)
  lesion_pancreas_map.csv           per-lesion coordinates (units of gland length; x gland_length_mm for mm)

Also prints how lesions distribute inside the head/body/tail versus where tissue volume lies.

Usage (from repo root): .venv/bin/python3 src/lesion-sectioning/plot_pancreas_map.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from scipy.stats import ks_2samp

WORK_DIR = Path("src/lesion-sectioning/work")
OUT_DIR = Path("results/lesion_pancreas_map")
BINS = ["<5mm", "5-10mm", "10-15mm", "15-20mm", "20-40mm", "40mm+"]
SECTION_COLOR = {1: "#0072B2", 2: "#E69F00", 3: "#CC79A7"}  # head, body, tail (Okabe-Ito)
SECTION_NAME = {1: "head", 2: "body", 3: "tail"}
U_RANGE = (-0.1, 1.1)
V_RANGE = (-0.45, 0.45)
NX, NY = 72, 54


def load():
    les = pd.read_csv(WORK_DIR / "pancreas_map_lesions.csv")
    meta = pd.read_csv("results/lesion_duct_position/lesion_duct_position.csv")[
        ["case_id", "lesion_id", "diam_mm", "diam_bin", "section"]]
    les = les.merge(meta, on=["case_id", "lesion_id"], how="left")
    les["diam_bin"] = pd.Categorical(les["diam_bin"], categories=BINS, ordered=True)
    tis = dict(np.load(WORK_DIR / "pancreas_map_tissue.npz"))
    bounds = pd.read_csv(WORK_DIR / "section_boundaries.csv")
    return les, tis, bounds


def territory_image(tis: dict, ycol: str) -> np.ndarray:
    xe = np.linspace(*U_RANGE, NX + 1)
    ye = np.linspace(*V_RANGE, NY + 1)
    hs = {c: gaussian_filter(np.histogram2d(tis["u"][tis["section"] == c], tis[ycol][tis["section"] == c],
                                            bins=[xe, ye])[0], 1.0) for c in (0, 1, 2, 3)}
    total = sum(hs.values())
    dom = np.argmax(np.stack([hs[1], hs[2], hs[3]]), axis=0) + 1
    rgb = np.zeros((NX, NY, 3))
    for c, col in SECTION_COLOR.items():
        rgb[dom == c] = to_rgb(col)
    alpha = np.clip((total / np.percentile(total[total > 0], 99)) ** 0.5, 0, 1) * 0.9
    return np.dstack([rgb, alpha]).transpose(1, 0, 2)  # (ny, nx, 4) for origin="lower"


def map_figure(les, tis, bounds, ycol, ylabel, title, path):
    img = territory_image(tis, ycol)
    hb, bt = bounds["boundary_head_body"].median(), bounds["boundary_body_tail"].median()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.6), sharex=True, sharey=True)
    for i, (ax, b) in enumerate(zip(axes.flat, BINS)):
        ax.imshow(img, origin="lower", extent=[*U_RANGE, *V_RANGE], aspect="equal", zorder=1, interpolation="bilinear")
        for x in (hb, bt):
            ax.axvline(x, color="black", ls="--", lw=0.8, alpha=0.55, zorder=2)
        sub = les[les["diam_bin"] == b]
        inside = sub[sub["u"].between(*U_RANGE) & sub[ycol].between(*V_RANGE)]
        ax.scatter(inside["u"], inside[ycol], s=24, c="black", edgecolors="white", linewidths=0.6, alpha=0.9, zorder=3)
        ax.set_xlim(*U_RANGE)
        ax.set_ylim(*V_RANGE)
        out = len(sub) - len(inside)
        ax.set_title(f"{b}   (n={len(sub)}" + (f", {out} off-map" if out else "") + ")", fontsize=11)
        ax.grid(alpha=0.2, zorder=0)
        if i == 0:
            for x, lab in ((hb / 2, "head"), ((hb + bt) / 2, "body"), ((bt + 1) / 2, "tail")):
                ax.text(x, V_RANGE[1] - 0.02, lab, ha="center", va="top", fontsize=9, color="#222222")
    for ax in axes[1]:
        ax.set_xlabel("position along gland:  head end (0)  ->  tail end (1)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    handles = [Patch(facecolor=SECTION_COLOR[c], label=f"{SECTION_NAME[c]} tissue") for c in (1, 2, 3)]
    handles.append(Line2D([], [], marker="o", color="none", markerfacecolor="black", markeredgecolor="white",
                          markersize=8, label="lesion centroid"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.text(0.5, 0.075, "every gland aligned on its long axis and scaled to its own length (1 unit ~ 12 cm); tint = dominant "
             "section, opacity = tissue density; dashed = median section boundaries", ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=[0, 0.11, 1, 0.97], h_pad=2.0)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def print_stats(les, tis):
    t_sec = tis["section"]
    print("\n=== share of lesions per section vs share of gland tissue ===")
    tissue_share = {SECTION_NAME[c]: (t_sec == c).mean() for c in (1, 2, 3)}
    rows = []
    for grp, m in (("<20mm", les["diam_mm"] < 20), (">=20mm", les["diam_mm"] >= 20)):
        s = les[m]
        row = {"lesions": grp, "n": len(s)}
        for sec in ("head", "body", "tail"):
            row[f"{sec} lesions %"] = 100 * (s["section"] == sec).mean()
            row[f"{sec} tissue %"] = 100 * tissue_share[sec]
        rows.append(row)
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    print("\n=== inside the HEAD: lesion vs head-tissue position (gland-length units) ===")
    print("(u: 0 head end -> 1 tail end; ap: + anterior; si: + superior; centered on the gland centroid)")
    head_t = t_sec == 1
    rows = []
    for grp, m in (("<10mm", les["diam_mm"] < 10), ("10-20mm", les["diam_mm"].between(10, 20, inclusive="left")),
                   ("<20mm", les["diam_mm"] < 20), (">=20mm", les["diam_mm"] >= 20)):
        s = les[m & (les["section"] == "head")]
        row = {"lesions": grp, "n_in_head": len(s)}
        for col in ("u", "ap", "si"):
            row[f"{col}_lesion"] = s[col].median()
            row[f"{col}_tissue"] = float(np.median(tis[col][head_t]))
            row[f"{col}_KSp"] = ks_2samp(s[col], tis[col][head_t]).pvalue if len(s) >= 5 else np.nan
        rows.append(row)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main() -> None:
    les, tis, bounds = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = les[["case_id", "lesion_id", "diam_mm", "diam_bin", "section", "u", "ap", "si", "gland_length_mm"]].rename(
        columns={"u": "u_head_to_tail", "ap": "ap_offset", "si": "si_offset"})
    out.sort_values(["case_id", "lesion_id"]).round(3).to_csv(OUT_DIR / "lesion_pancreas_map.csv", index=False)

    map_figure(les, tis, bounds, "ap", "posterior <-  -> anterior\n(gland lengths from centre)",
               "Where lesions sit on the pancreas - axial view (anterior up), by lesion diameter",
               OUT_DIR / "pancreas_map_axial_by_size.png")
    map_figure(les, tis, bounds, "si", "inferior <-  -> superior\n(gland lengths from centre)",
               "Where lesions sit on the pancreas - coronal view (superior up), by lesion diameter",
               OUT_DIR / "pancreas_map_coronal_by_size.png")
    print(f"{len(les)} lesions; wrote {OUT_DIR}/lesion_pancreas_map.csv and 2 figures")
    print_stats(les, tis)


if __name__ == "__main__":
    main()
