"""Two 6-panel figures (one panel per size bin) + a per-lesion CSV, from duct_position.py output.

  duct_cross_section_by_size.png   cross-section around the duct: x = anterior(+)/posterior(-)
      offset, y = superior(+)/inferior(-) offset from the duct centerline (mm), duct at the origin.
      Grey = where pancreas tissue lies in that plane (tissue sample, pooled over the whole gland).
  duct_distance_along_gland_by_size.png   x = position along the gland head(0)->tail(1),
      y = distance from lesion centroid to nearest duct voxel (mm). Grey = where tissue lies.

Only lesions with duct_valid are drawn (a duct segment within 10 mm of the lesion along the
gland axis); the grey background is sampled under the same rule. Marker color AND shape encode
the section, so identity never depends on color alone.

Usage (from repo root): .venv/bin/python3 src/lesion-sectioning/plot_duct_position.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from scipy.stats import ks_2samp

WORK_DIR = Path("src/lesion-sectioning/work")
OUT_DIR = Path("results/lesion_duct_position")
BINS = ["<5mm", "5-10mm", "10-15mm", "15-20mm", "20-40mm", "40mm+"]
SECTION_STYLE = {"head": ("#0072B2", "o"), "body": ("#E69F00", "s"), "tail": ("#CC79A7", "^")}  # Okabe-Ito
SECTION_NAME = {1: "head", 2: "body", 3: "tail"}
CROSS_LIM = 50.0
DIST_LIM = 60.0


def load():
    les = pd.read_csv(WORK_DIR / "lesion_duct_position.csv")
    inc = pd.read_csv(WORK_DIR / "lesion_inclusion.csv")[["case_id", "lesion_id", "diam_mm", "diam_bin"]]
    les = les.merge(inc, on=["case_id", "lesion_id"], how="left")
    les["diam_bin"] = pd.Categorical(les["diam_bin"], categories=BINS, ordered=True)
    tis = dict(np.load(WORK_DIR / "tissue_samples.npz"))
    bounds = pd.read_csv(WORK_DIR / "section_boundaries.csv")

    # lesions whose 3-voxel ring touches no head/body/tail mask: assign by gland position instead
    hb, bt = bounds["boundary_head_body"].median(), bounds["boundary_body_tail"].median()
    un = les["section"] == "unassigned"
    les["section_source"] = np.where(un, "pc1_boundary", "mask")
    les.loc[un, "section"] = np.select([les.loc[un, "pc1_norm"] < hb, les.loc[un, "pc1_norm"] < bt],
                                       ["head", "body"], "tail")
    return les, tis, bounds


def draw_points(ax, sub, xcol, ycol):
    for sec, (color, marker) in SECTION_STYLE.items():
        s = sub[sub["section"] == sec]
        ax.scatter(s[xcol], s[ycol], s=34, c=color, marker=marker, edgecolors="white", linewidths=0.7,
                   alpha=0.95, zorder=3)


def legend(fig):
    handles = [Line2D([], [], marker=m, color="none", markerfacecolor=c, markeredgecolor="white",
                      markersize=9, label=sec) for sec, (c, m) in SECTION_STYLE.items()]
    fig.legend(handles=handles, title="lesion section", loc="lower center", ncol=3, frameon=False)


def cross_section_fig(les, tis, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), sharex=True, sharey=True)
    edges = np.arange(-CROSS_LIM, CROSS_LIM + 1, 2)
    H, _, _ = np.histogram2d(tis["offset_ap_mm"], tis["offset_si_mm"], bins=[edges, edges])
    for ax, b in zip(axes.flat, BINS):
        ax.imshow(H.T, origin="lower", extent=[-CROSS_LIM, CROSS_LIM, -CROSS_LIM, CROSS_LIM], cmap="Greys",
                  norm=PowerNorm(0.5, vmax=np.percentile(H[H > 0], 99)), aspect="equal", zorder=1)
        allb = les[les["diam_bin"] == b]
        sub = allb[allb["duct_valid"]]
        inside = sub[(sub["offset_ap_mm"].abs() <= CROSS_LIM) & (sub["offset_si_mm"].abs() <= CROSS_LIM)]
        draw_points(ax, inside, "offset_ap_mm", "offset_si_mm")
        ax.plot(0, 0, marker="+", color="black", ms=13, mew=2, zorder=4)
        ax.set_xlim(-CROSS_LIM, CROSS_LIM)
        ax.set_ylim(-CROSS_LIM, CROSS_LIM)
        ax.set_title(f"{b}   ({len(inside)} shown / {len(allb)} lesions"
                     + (f", {len(sub) - len(inside)} beyond ±{CROSS_LIM:g} mm" if len(sub) > len(inside) else "") + ")",
                     fontsize=11)
        ax.grid(alpha=0.25, zorder=0)
    for ax in axes[1]:
        ax.set_xlabel("offset from duct (mm):  posterior  <-  ->  anterior")
    for ax in axes[:, 0]:
        ax.set_ylabel("offset from duct (mm):  inferior  <-  ->  superior")
    fig.suptitle("Lesion position around the main pancreatic duct (+ = duct), by lesion diameter", fontsize=14)
    fig.text(0.5, 0.075, "grey = pancreas tissue density in the same plane; lesions drawn only where the duct mask runs "
             "within 10 mm of the lesion along the gland", ha="center", fontsize=9, color="#444444")
    legend(fig)
    fig.tight_layout(rect=[0, 0.11, 1, 0.97], h_pad=2.5)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def distance_fig(les, tis, bounds, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), sharex=True, sharey=True)
    xe = np.linspace(-0.05, 1.05, 45)
    ye = np.arange(0, DIST_LIM + 1, 2)
    H, _, _ = np.histogram2d(tis["pc1_norm"], tis["dist_mm"], bins=[xe, ye])
    hb, bt = bounds["boundary_head_body"].median(), bounds["boundary_body_tail"].median()
    for i, (ax, b) in enumerate(zip(axes.flat, BINS)):
        ax.imshow(H.T, origin="lower", extent=[-0.05, 1.05, 0, DIST_LIM], cmap="Greys",
                  norm=PowerNorm(0.5, vmax=np.percentile(H[H > 0], 99)), aspect="auto", zorder=1)
        for x in (hb, bt):
            ax.axvline(x, color="black", ls="--", lw=0.8, alpha=0.6, zorder=2)
        allb = les[les["diam_bin"] == b]
        sub = allb[allb["duct_valid"]]
        inside = sub[sub["dist_centroid_to_duct_mm"] <= DIST_LIM]
        draw_points(ax, inside, "pc1_norm", "dist_centroid_to_duct_mm")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, DIST_LIM)
        ax.set_title(f"{b}   ({len(inside)} shown / {len(allb)} lesions"
                     + (f", {len(sub) - len(inside)} beyond {DIST_LIM:g} mm" if len(sub) > len(inside) else "") + ")",
                     fontsize=11)
        ax.grid(alpha=0.25, zorder=0)
        if i == 0:
            for x, lab in ((hb / 2, "head"), ((hb + bt) / 2, "body"), ((bt + 1.0) / 2, "tail")):
                ax.text(x, DIST_LIM * 0.96, lab, ha="center", va="top", fontsize=9, color="#333333")
    for ax in axes[1]:
        ax.set_xlabel("position along gland:  head (0)  ->  tail (1)")
    for ax in axes[:, 0]:
        ax.set_ylabel("distance, lesion centroid to nearest duct voxel (mm)")
    fig.suptitle("Lesion distance from the main pancreatic duct along the gland, by lesion diameter", fontsize=14)
    fig.text(0.5, 0.075, "grey = pancreas tissue density in the same coordinates; dashed = median head|body and body|tail "
             "boundaries; lesions drawn only where the duct runs within 10 mm of the lesion", ha="center",
             fontsize=9, color="#444444")
    legend(fig)
    fig.tight_layout(rect=[0, 0.11, 1, 0.97], h_pad=2.5)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def print_stats(les, tis):
    tis_sec = tis["section"]
    print("\n=== coverage by size bin ===")
    cov = les.groupby("diam_bin", observed=True).agg(n=("lesion_id", "size"), duct_present=("duct_present", "sum"),
                                                     duct_valid=("duct_valid", "sum"))
    print(cov.to_string())

    v = les[les["duct_valid"]]
    print("\n=== duct-valid lesions vs tissue (centroid distance to duct; position along gland) ===")
    rows = []
    for b in BINS:
        s = v[v["diam_bin"] == b]
        if len(s) < 5:
            rows.append({"bin": b, "n": len(s)})
            continue
        rows.append({"bin": b, "n": len(s),
                     "lesion_med_dist": s["dist_centroid_to_duct_mm"].median(), "tissue_med_dist": np.median(tis["dist_mm"]),
                     "KS_p_dist": ks_2samp(s["dist_centroid_to_duct_mm"], tis["dist_mm"]).pvalue,
                     "lesion_med_pc1": s["pc1_norm"].median(), "tissue_med_pc1": np.median(tis["pc1_norm"]),
                     "KS_p_pc1": ks_2samp(s["pc1_norm"], tis["pc1_norm"]).pvalue,
                     "surface<=5mm": (s["dist_surface_to_duct_mm"] <= 5).mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== within each section: lesion vs tissue, position in section (0 proximal..1 distal) and duct distance ===")
    rows = []
    for grp, mask in (("<20mm", v["diam_mm"] < 20), (">=20mm", v["diam_mm"] >= 20)):
        for code, sec in SECTION_NAME.items():
            s = v[mask & (v["section"] == sec)]
            t = tis_sec == code
            if len(s) < 5:
                rows.append({"size": grp, "section": sec, "n": len(s)})
                continue
            rows.append({"size": grp, "section": sec, "n": len(s),
                         "lesion_med_pos": s["pos_in_section"].median(), "tissue_med_pos": np.median(tis["pos_in_section"][t]),
                         "KS_p_pos": ks_2samp(s["pos_in_section"].dropna(), tis["pos_in_section"][t]).pvalue,
                         "lesion_med_dist": s["dist_centroid_to_duct_mm"].median(), "tissue_med_dist": np.median(tis["dist_mm"][t]),
                         "KS_p_dist": ks_2samp(s["dist_centroid_to_duct_mm"], tis["dist_mm"][t]).pvalue})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main() -> None:
    les, tis, bounds = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cols = ["case_id", "lesion_id", "diam_mm", "diam_bin", "section", "section_source", "pos_in_section", "pc1_norm", "duct_present",
            "duct_valid", "dist_surface_to_duct_mm", "dist_centroid_to_duct_mm", "offset_ap_mm", "offset_si_mm",
            "offset_rl_mm"]
    les[cols].sort_values(["case_id", "lesion_id"]).round(3).to_csv(OUT_DIR / "lesion_duct_position.csv", index=False)

    cross_section_fig(les, tis, OUT_DIR / "duct_cross_section_by_size.png")
    distance_fig(les, tis, bounds, OUT_DIR / "duct_distance_along_gland_by_size.png")
    print(f"wrote {OUT_DIR}/lesion_duct_position.csv and 2 figures")
    print_stats(les, tis)


if __name__ == "__main__":
    main()
