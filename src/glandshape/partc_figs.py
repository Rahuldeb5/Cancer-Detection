"""S15 Part C item 15: 12 visual-QC PNGs (EXPLORATORY run), for Rahul / Dr. Akin.

Picks (fixed rule, by case_id order within each group, no score looked at):
  4 head-heavy glands: highest ratio median width (u 0.05-0.25) / median width (u 0.40-0.80), non-tip stops, any status
  2 target misses (S14 group, evaluable)   2 control >= 40 mm detected   2 reference 20-40 detected   2 tumor- glands
Each PNG: outline (max projection) + centerline + every 5th stop's 2 width axes + footprint stops (red) + top-3 peaks
(yellow); the width map; the v5 primary score map with the footprint outlined and top-3 peaks marked.

Usage: PYTHONPATH=src .venv/bin/python -m glandshape.partc_figs   -> work/gland_shape/partC/qc/*.png
"""
from __future__ import annotations

import contextlib
import io as sio

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from glandshape import centerline as C  # noqa: E402
from glandshape import footprint as FP  # noqa: E402
from glandshape import orient as OR  # noqa: E402
from glandshape import partc as PC  # noqa: E402
from glandshape import partc_run as R  # noqa: E402
from glandshape import profile as P  # noqa: E402
from glandshape import qc_cohort as QCC  # noqa: E402
from glandshape import scores as S  # noqa: E402
from tumorlib import io as tio  # noqa: E402

QCDIR = R.OUTW / "qc"


def head_heavy(cases: pd.DataFrame, n=4):
    rat = []
    for cid in cases.case_id:
        with np.load(R.PROF_C / f"{cid}.npz") as d:
            u, w, tip = d["u"], d["w"], d["tip"]
        med = np.nanmedian(np.where(tip[:, None], np.nan, w), axis=1)
        h = np.nanmedian(med[(u >= 0.05) & (u <= 0.25)])
        b = np.nanmedian(med[(u >= 0.40) & (u <= 0.80)])
        rat.append((h / b if b > 0 else np.nan, cid))
    rat = sorted([r for r in rat if np.isfinite(r[0])], reverse=True)
    return [(c, None, f"head-heavy (head/body width {r:.2f})") for r, c in rat[:n]]


def draw(cid, lid, label, path):
    ref, _ = R.load_reference()
    ol, g = QCC.load_outline(cid)
    with contextlib.redirect_stdout(sio.StringIO()):
        at = OR.aorta_test(ol, tio.load_mask(cid, "aorta"))
    cl = C.build_centerline(ol, OR.anterior_world(at))
    pr = P.compute_profile(ol, cl)
    sc = S.primary_scores(pr, ref)["primary"]
    peaks = PC.top_peaks(sc, pr.s)
    fp = np.zeros_like(sc, bool)
    title = f"{cid} — {label} — {R.LABEL}"
    if lid is not None:
        with np.load(QCC.SPARSE / f"{cid}.npz") as d:
            shape, gi, gl = tuple(int(v) for v in d["shape"]), d["gt_idx"], d["gt_lab"]
        xyz = np.stack(np.unravel_index(gi[gl == lid].astype(np.int64), shape), 1)
        fp = FP.footprint(ol, cl, pr, xyz)
        if fp.any():
            pct = PC.within_scan_percentile(sc, fp)["percentile"]
            title += f"\nlesion {lid}: footprint {int(fp.sum())} cells, within-scan percentile {pct:.0f}, top-3 hit " \
                     f"{any(fp[i, j] for i, j in peaks)}"
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
    zax = int(np.argmax(np.abs(ol.R[2])))
    keep = [a for a in range(3) if a != zax]
    ax[0].imshow(ol.mask.max(axis=zax).T, origin="lower", cmap="gray")
    ax[0].plot(cl.dense[:, keep[0]], cl.dense[:, keep[1]], "r-", lw=1)
    ax[0].plot(cl.P[0, keep[0]], cl.P[0, keep[1]], "go", ms=6)
    dirs = P.ray_dirs(cl)
    for i in range(0, len(cl.s), 5):
        for k in (0, 6):
            if np.isfinite(pr.w[i, k]):
                o, d_ = pr.origin[i], dirs[i, k]
                a, b = o - pr.r[i, k + 12] * d_, o + pr.r[i, k] * d_
                ax[0].plot([a[keep[0]], b[keep[0]]], [a[keep[1]], b[keep[1]]], "c-", lw=0.5)
    st = FP.footprint_stops(fp)
    if len(st):
        ax[0].plot(cl.P[st, keep[0]], cl.P[st, keep[1]], "r.", ms=6, label="footprint stops")
    for i, j in peaks:
        ax[0].plot(cl.P[i, keep[0]], cl.P[i, keep[1]], "y*", ms=12)
    tipi = np.nonzero(pr.tip)[0]
    ax[0].plot(cl.P[tipi, keep[0]], cl.P[tipi, keep[1]], "mx", ms=4)
    ax[0].set_title("outline, centerline (green = head end), width axes; red = footprint, * = top-3, x = tip stops",
                    fontsize=7)
    ext = (0, cl.L, -7.5, 172.5)
    im = ax[1].imshow(pr.w.T, aspect="auto", origin="lower", extent=ext, cmap="viridis")
    plt.colorbar(im, ax=ax[1], label="width (mm)")
    ax[1].set_title("width map (12 axes)", fontsize=8)
    im = ax[2].imshow(sc.T, aspect="auto", origin="lower", extent=ext, cmap="magma")
    plt.colorbar(im, ax=ax[2], label="v5 primary bulge z")
    if fp.any():
        ax[2].contour(np.repeat(np.repeat(fp.T.astype(float), 1, 0), 1, 1), levels=[0.5], colors="cyan",
                      extent=ext, linewidths=1)
    for i, j in peaks:
        ax[2].plot(pr.s[i], j * 15, "y*", ms=12)
    ax[2].set_title("primary score; cyan = lesion footprint; * = top-3 peaks", fontsize=8)
    for a in ax[1:]:
        a.set_xlabel("s (mm, head -> tail)"); a.set_ylabel("axis (deg)")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=85)
    plt.close(fig)


def main():
    QCDIR.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(R.OUTW / "lesions_joined.csv")
    cases = pd.read_csv(R.OUTW / "cases.csv")
    kept = cases[cases.gland_drop.fillna("") == ""]
    ev = d[d.evaluable.astype(bool)].sort_values(["case_id", "lesion_id"])
    meta = pd.read_csv(R.REPO / "results/PanTS_metadata_new.csv", usecols=["PanTS ID", "tumor?"])
    neg = sorted(set(meta.loc[meta["tumor?"] == 0, "PanTS ID"]) & set(kept.case_id))
    picks = head_heavy(kept)
    for col, lab in (("target", "TARGET miss (20-40 mm, missed by 502)"), ("control", "CONTROL >= 40 mm detected"),
                     ("reference", "REFERENCE 20-40 mm detected")):
        for r in ev[ev[col].astype(bool)].head(2).itertuples():
            picks.append((r.case_id, int(r.lesion_id), f"{lab}, {r.diam_mm:.0f} mm, {r.est_phase}, {r.attenuation_v3}"))
    picks += [(c, None, "tumor-negative gland") for c in neg[:2]]
    for n, (cid, lid, lab) in enumerate(picks, 1):
        draw(cid, lid, lab, QCDIR / f"{n:02d}_{cid}{'' if lid is None else f'_L{lid}'}.png")
        print(n, cid, lid, lab)


if __name__ == "__main__":
    main()
