"""S15 Part A real-data checks on the 1308 cohort outlines (no lesion mask, no score is read).

Per case: 502 outline from the S13 sparse cache (panc_idx = channel 0 > 0.5), components (2nd > 10% of 1st reported),
aorta-posterior test on the CT grid, the head->tail sign, and (v2) the gland QC of glandshape.qc: centerline + profile
are run (no HU, no score) only to apply the drop rules. S14's targets are then counted by case_id (counts only). The sign check against S10's GT-signed PC1
(work/duct_atlas/full/cases.csv, e1 from the GT head/tail masks) is EVALUATION ONLY: the tool never reads it.
Optional --smoke N: full profile (with CT HU) on the first N tumor-negative cases + a QC figure, for timing/memory.

Usage: PYTHONPATH=src .venv/bin/python -m glandshape.qc_cohort [--workers 4] [--smoke 6]
Output: work/gland_shape/partA/cohort_qc.csv, cohort_qc.txt, smoke_*.png
"""
from __future__ import annotations

import argparse
from functools import partial
import contextlib
import io as sio
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from glandshape import centerline as C
from glandshape import orient as OR
from glandshape import outline as O
from glandshape import profile as P
from glandshape import qc as QC
from glandshape import footprint as FP
from glandshape import scores as S
from tumorlib import io as tio

REPO = Path(__file__).resolve().parents[2]
SPARSE = Path.home() / "research/nnunet_probs_502"
OUT = REPO / "work/gland_shape/partA"
S10_CASES = REPO / "work/duct_atlas/full/cases.csv"
S10_FLIP = REPO / "work/duct_atlas/aorta_flip_check.csv"


def load_outline(case_id: str):
    with np.load(SPARSE / f"{case_id}.npz") as d:
        shape, sp, panc = tuple(int(v) for v in d["shape"]), np.asarray(d["spacing"], float), d["panc_idx"]
    g = tio.grid(case_id)
    if g is None:
        raise RuntimeError("no CT header")
    if tuple(g[0]) != shape or not np.allclose(g[2], sp, atol=1e-4):
        raise RuntimeError(f"cache grid {shape} {sp} != CT {g[0]} {g[2]}")
    return O.outline_from_flat(panc, shape, sp, g[1]), g


def target_table() -> pd.DataFrame:
    """S14's targets, read (not redefined) with S14's documented filter; asserts n = 48 / 46."""
    ml = pd.read_csv(REPO / "results/rescue_check_502/missed_lesions.csv")
    parts = []
    for b, n_exp, name in (("20-40", 48, "target"), ("10-20", 46, "secondary")):
        tg = ml[(ml.diam_bin == b) & ml.largest_in_case.astype(bool) & (ml.panc_category == "included")
                & (ml.tier != "separated")].copy()
        assert len(tg) == n_exp, (b, len(tg))
        tg["group"] = name
        parts.append(tg[["case_id", "lesion_id", "gt_vox", "group"]])
    return pd.concat(parts, ignore_index=True)


def _target_footprints(case_id, ol, cl, pr, lesions) -> list[dict]:
    """Counts-only footprint facts for this case's S14 target lesions (GT lesion = evaluation geometry only)."""
    with np.load(SPARSE / f"{case_id}.npz") as d:
        shape, gt_idx, gt_lab = tuple(int(v) for v in d["shape"]), d["gt_idx"], d["gt_lab"]
    rows = []
    for lid, gt_vox, group in lesions:
        sel = gt_idx[gt_lab == lid]
        assert len(sel) == gt_vox, (case_id, lid, len(sel), gt_vox)   # join check (gotcha 2)
        xyz = np.stack(np.unravel_index(sel.astype(np.int64), shape), 1)
        fp = FP.footprint(ol, cl, pr, xyz)
        st = FP.footprint_stops(fp)
        rows.append(dict(case_id=case_id, lesion_id=lid, group=group, n_cells=int(fp.sum()), n_stops=len(st),
                         evaluable=bool(fp.any()), fold_frac=float(pr.fold[st].mean()) if len(st) else np.nan,
                         fold_heavy=FP.fold_heavy(fp, pr), censored_fp=FP.censored_footprint(fp, pr),
                         chance_fraction=FP.chance_fraction(fp, pr)))
    return rows


def one(case_id: str, targets: dict | None = None) -> dict:
    out = {"case_id": case_id}
    t0 = time.time()
    try:
        ol, g = load_outline(case_id)
        if ol is None:
            out["status"], out["qc_drop"], out["qc_flags"] = "empty", "empty", ""
            return out
        out.update(n_comp=ol.n_comp, size1=ol.size1, size2=ol.size2, frac2=ol.frac2, fragmented=ol.fragmented,
                   vol_cm3=ol.vol_mm3 / 1000, iso_shape="x".join(map(str, ol.mask.shape)))
        buf = sio.StringIO()
        with contextlib.redirect_stdout(buf):
            aorta = tio.load_mask(case_id, "aorta")
        out["aorta_msg"] = buf.getvalue().strip().replace("\n", " | ")
        out.update(OR.aorta_test(ol, aorta))
        del aorta
        e1 = C.head_tail_axis(ol)
        out.update(e1_x=e1[0], e1_y=e1[1], e1_z=e1[2])
        out["status"] = "ok"
        out["qc_flags"] = ";".join(QC.flags(ol))
        drop = QC.outline_drop(ol)
        if drop is None:
            try:
                cl = C.build_centerline(ol, OR.anterior_world(out))
                out.update(cl_L_mm=cl.L, bbox_diag_mm=QC.bbox_diag_mm(ol), n_stops=len(cl.s))
                drop = QC.centerline_drop(ol, cl.L)
            except Exception as e:
                drop, out["qc_error"] = "centerline_error", f"{type(e).__name__}: {e}"
        if drop is None:
            try:
                pr = P.compute_profile(ol, cl)
                out.update(n_invalid=int(np.logical_not(pr.valid).sum()), n_fold=int(pr.fold.sum()),
                           n_fallback=int(pr.origin_fallback.sum()), n_censored=int(pr.censored.sum()),
                           n_wcens=int(pr.w_censored.sum()),
                           max_shift_mm=float(np.nanmax(pr.origin_shift)) if pr.valid.any() else np.nan)
                if targets and case_id in targets:
                    out["targets"] = _target_footprints(case_id, ol, cl, pr, targets[case_id])
            except Exception as e:
                drop, out["qc_error"] = "profile_error", f"{type(e).__name__}: {e}"
        out["qc_drop"] = drop or ""
    except Exception as e:
        out["status"] = f"error: {type(e).__name__}: {e}"
    out["secs"] = time.time() - t0
    return out


def seam_metric(m: np.ndarray) -> tuple[int, float]:
    """Consecutive-stop jumps: per pair, median over columns of |delta| / the map's median value. Returns
    (#pairs with a median relative jump > 15%, the largest median relative jump)."""
    med = np.nanmedian(m)
    with np.errstate(all="ignore"):
        j = np.nanmedian(np.abs(np.diff(m, axis=0)), axis=1) / med
    j = j[np.isfinite(j)]
    return int((j > 0.15).sum()), float(j.max()) if len(j) else np.nan


def smoke(case_ids, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    fig, axes = plt.subplots(len(case_ids), 3, figsize=(17, 3.2 * len(case_ids)))
    for ax_row, cid in zip(np.atleast_2d(axes), case_ids):
        t0 = time.time()
        ol, g = load_outline(cid)
        with contextlib.redirect_stdout(sio.StringIO()):
            aorta = tio.load_mask(cid, "aorta")
        at = OR.aorta_test(ol, aorta)
        del aorta
        try:
            cl = C.build_centerline(ol, OR.anterior_world(at))
        except ValueError as e:
            rows.append(f"{cid}: centerline FAILED ({e}); outline {ol.vol_mm3 / 1000:.2f} cm^3")
            ax_row[0].set_title(f"{cid}: centerline failed ({e})", fontsize=8)
            continue
        ct = P.make_ct_sampler(ol, g[0], lambda sl: tio.load_ct(cid, sl))
        pr = P.compute_profile(ol, cl, ct_sampler=ct)
        pr2 = P.compute_profile(ol, cl, do_fold=False, origin_mode="centroid")      # v2, comparison only
        bl, mad = S.bulge_local(pr.w)
        secs = time.time() - t0
        hu_in = np.nanmedian(pr.hu[:, :, 2].astype(float))
        s2, s3, s3r = seam_metric(pr2.r), seam_metric(pr.w), seam_metric(pr.r)
        rows.append(f"{cid}: L {cl.L:.0f} mm, {len(cl.s)} stops, invalid {int(np.logical_not(pr.valid).sum())}, "
                    f"origin fallbacks {int(pr.origin_fallback.sum())}, fold {int(pr.fold.sum())}, censored widths "
                    f"{int(pr.w_censored.sum())}, width MAD 15/30/45 {'/'.join(f'{m:.2f}' for m in mad)} mm, "
                    f"median HU 2 mm inside {hu_in:.0f}, {secs:.1f} s | seams (pairs > 15% jump, max jump): "
                    f"v2 centroid radius {s2[0]} ({s2[1]:.0%}), v3 width {s3[0]} ({s3[1]:.0%}), "
                    f"v3 radius {s3r[0]} ({s3r[1]:.0%})")
        R = ol.R
        zax = int(np.argmax(np.abs(R[2])))
        keep = [a for a in range(3) if a != zax]
        a0 = ax_row[0]
        a0.imshow(ol.mask.max(axis=zax).T, origin="lower", cmap="gray")
        a0.plot(cl.dense[:, keep[0]], cl.dense[:, keep[1]], "r-", lw=1)
        a0.plot(cl.P[0, keep[0]], cl.P[0, keep[1]], "go", ms=6)
        for i in range(0, len(cl.s), 5):
            for k in (0, 6):
                if np.isfinite(pr.w[i, k]):
                    o, d = pr.origin[i], P.ray_dirs(cl)[i, k]
                    a, b = o - pr.r[i, k + 12] * d, o + pr.r[i, k] * d
                    a0.plot([a[keep[0]], b[keep[0]]], [a[keep[1]], b[keep[1]]], "c-", lw=0.5)
        if pr.fold.any():
            a0.plot(cl.P[pr.fold, keep[0]], cl.P[pr.fold, keep[1]], "yx", ms=5)
        a0.set_title(f"{cid} (green = head end, yellow x = fold, cyan = 2 of 12 width axes)", fontsize=8)
        for ax, m, ext, lab, ttl in ((ax_row[1], pr2.r, (-7.5, 352.5), "r (mm)", "v2 radius, centroid origin"),
                                     (ax_row[2], pr.w, (-7.5, 172.5), "width (mm)", "v3 WIDTH, centerline origin")):
            im = ax.imshow(m.T, aspect="auto", origin="lower", extent=(0, cl.L, *ext), cmap="viridis")
            ax.set_xlabel("s (mm, head -> tail)"); ax.set_ylabel("theta / axis (deg)")
            ax.set_title(ttl, fontsize=8)
            plt.colorbar(im, ax=ax, label=lab)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=80)
    plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = tio.case_ids()
    if a.limit:
        ids = ids[: a.limit]
    t0 = time.time()
    tt = target_table()
    tdict = {}
    for r in tt.itertuples():
        tdict.setdefault(r.case_id, []).append((int(r.lesion_id), int(r.gt_vox), r.group))
    with mp.Pool(a.workers) as pool:
        rows = pool.map(partial(one, targets=tdict), ids, chunksize=4)
    trows = [x for r in rows for x in r.pop("targets", [])]
    tfp = pd.DataFrame(trows)
    tfp.to_csv(OUT / "target_footprints.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cohort_qc.csv", index=False)

    L = [f"S15 Part A cohort outline / aorta / orientation checks: {len(df)} cases, {time.time() - t0:.0f} s"]
    L.append(f"status: {df.status.value_counts().to_dict()}")
    ok = df[df.status == "ok"]
    L.append(f"components of the raw predicted region: {ok.n_comp.value_counts().sort_index().to_dict()}")
    fr = ok[ok.fragmented.astype(bool)]
    L.append(f"2nd component > 10% of the 1st: {len(fr)}")
    for _, r in fr.sort_values("frac2", ascending=False).iterrows():
        L.append(f"  {r.case_id}: frac2 {r.frac2:.2f} (sizes {r.size1}/{r.size2} vox, {r.n_comp} comps)")
    L.append(f"outline volume cm^3: median {ok.vol_cm3.median():.1f}, P1 {ok.vol_cm3.quantile(.01):.1f}, "
             f"P99 {ok.vol_cm3.quantile(.99):.1f}; median secs/case {df.secs.median():.2f}")
    # aorta
    L.append(f"aorta status: {ok.aorta_status.value_counts().to_dict()}; call (v2, |dy| <= 10 mm inconclusive): "
             f"{ok.aorta_call.value_counts().to_dict()}")
    for _, r in ok[ok.aorta_call == "inconclusive"].sort_values("aorta_dy_mm").iterrows():
        L.append(f"  INCONCLUSIVE {r.case_id}: dy {r.aorta_dy_mm:+.1f} mm")
    dy = ok.aorta_dy_mm.dropna()
    L.append(f"aorta - gland y (CT world, slab): median {dy.median():.1f} mm, P1 {dy.quantile(.01):.1f}, "
             f"P99 {dy.quantile(.99):.1f}; y_flipped (dy > +10 mm): {int(ok.y_flipped.astype(bool).sum())}")
    srt = np.sort(dy.values)
    gaps = np.diff(srt)
    L.append(f"  largest dy values: {np.round(srt[-15:], 1).tolist()}")
    L.append(f"  values in (-15, +15) mm: {int(((dy > -15) & (dy < 15)).sum())}")
    for _, r in ok[ok.y_flipped.astype(bool)].iterrows():
        L.append(f"  FLIPPED {r.case_id}: dy {r.aorta_dy_mm:+.1f} mm (aorta voxels in slab {r.aorta_n_slab})")
    # gland QC (v2)
    meta = pd.read_csv(REPO / "results/PanTS_metadata_new.csv", usecols=["PanTS ID", "tumor?"])
    q = df.merge(meta.rename(columns={"PanTS ID": "case_id", "tumor?": "tumor"}), on="case_id", how="left")
    q["qc_drop"] = q.qc_drop.fillna("")
    q["frag"] = q.qc_flags.fillna("").str.contains("fragmented")
    L.append("gland QC (v2): drop reasons by tumor status (0 = tumor-, 1 = tumor+):")
    L.append("    " + pd.crosstab(q.qc_drop.replace("", "kept"), q.tumor).to_string().replace("\n", "\n    "))
    kept = q[q.qc_drop == ""]
    L.append(f"  kept {len(kept)} (fragmented-flagged {int(kept.frag.sum())}); profiles: stops median "
             f"{kept.n_stops.median():.0f}, cases with fold flags {int((kept.n_fold > 0).sum())}, cases with origin "
             f"fallbacks {int((kept.n_fallback > 0).sum())}, cases with censored rays {int((kept.n_censored > 0).sum())}")
    for _, r in q[q.qc_drop.isin(["centerline_too_long", "centerline_error", "profile_error"])].iterrows():
        L.append(f"  DROP {r.case_id} ({r.qc_drop}): L {r.get('cl_L_mm', np.nan):.0f} mm vs bbox diag "
                 f"{r.get('bbox_diag_mm', np.nan):.0f} mm {r.get('qc_error', '') if isinstance(r.get('qc_error'), str) else ''}")
    kept_rows = q[q.qc_drop == ""]
    L.append(f"censored widths (a half-ray capped at 40 mm): {int((kept_rows.n_wcens > 0).sum())}/{len(kept_rows)} kept "
             f"glands; ray-origin fallbacks (centerline point outside its plane's gland): "
             f"{int((kept_rows.n_fallback > 0).sum())} glands, {int(kept_rows.n_fallback.sum())} stops, "
             f"max shift median {kept_rows.max_shift_mm.median():.1f} mm (P95 {kept_rows.max_shift_mm.quantile(.95):.1f})")
    # S14 targets: counts only (case-level QC applied to each lesion's case)
    for b, n_exp, name in (("20-40", 48, "target"), ("10-20", 46, "secondary")):
        tg = tt[tt.group == name]
        m = tg[["case_id", "lesion_id"]].merge(q[["case_id", "qc_drop", "frag"]], on="case_id", how="left")
        m["qc_drop"] = m.qc_drop.fillna("not_run")
        m["frag"] = m.frag.fillna(False).astype(bool)
        L.append(f"S14 {name} ({b} mm, n={len(m)}): dropped {int((m.qc_drop != '').sum())} "
                 f"{m.loc[m.qc_drop != '', 'qc_drop'].value_counts().to_dict()}; kept and fragmented-flagged "
                 f"{int(((m.qc_drop == '') & m.frag).sum())}; kept clean {int(((m.qc_drop == '') & np.logical_not(m.frag)).sum())}")
        if len(tfp):
            f = tfp[tfp.group == name]
            L.append(f"  footprints of the {len(f)} kept: not evaluable (no footprint cell) {int(np.logical_not(f.evaluable.astype(bool)).sum())}; "
                     f"> half of footprint stops fold-over {int(f.fold_heavy.astype(bool).sum())}; footprint contains a "
                     f"censored width {int(f.censored_fp.astype(bool).sum())}; footprint stops median {f.n_stops.median():.0f}, "
                     f"footprint fraction of the map median {f.chance_fraction.median():.3f}")
    if S10_FLIP.exists():
        f10 = pd.read_csv(S10_FLIP)
        m = ok.merge(f10, on="case_id", how="left")
        m["s10_flip"] = m.aorta_minus_gland_y_mm > 0
        L.append("  vs S10 flip audit (GT head affine, whole aorta; carried-over table, compared now):")
        L.append("    " + pd.crosstab(m.y_flipped.astype(bool), m.s10_flip, dropna=False).to_string().replace("\n", "\n    "))
    # orientation (evaluation only)
    if S10_CASES.exists():
        s10 = pd.read_csv(S10_CASES, usecols=["case_id", "e1_x", "e1_y", "e1_z", "leftward", "status"])
        m = ok.merge(s10.rename(columns={"e1_x": "g_x", "e1_y": "g_y", "e1_z": "g_z", "status": "s10_status"}),
                     on="case_id", how="left")
        has = m.g_x.notna()
        dot = (m.e1_x * m.g_x + m.e1_y * m.g_y + m.e1_z * m.g_z)[has]
        L.append(f"head->tail sign vs S10 GT-signed PC1 (EVALUATION ONLY): {int((dot > 0).sum())}/{int(has.sum())} agree; "
                 f"GT head->tail points to patient right (S10 leftward False): {int((m.leftward[has] == False).sum())}; "
                 f"|cos| median {np.median(np.abs(dot)):.3f}")
        for _, r in m[has & ((m.e1_x * m.g_x + m.e1_y * m.g_y + m.e1_z * m.g_z) <= 0)].iterrows():
            L.append(f"  DISAGREE {r.case_id}: S10 leftward {r.leftward}, dot "
                     f"{r.e1_x * r.g_x + r.e1_y * r.g_y + r.e1_z * r.g_z:+.2f}")
    if a.smoke:
        meta = pd.read_csv(REPO / "results/PanTS_metadata_new.csv")
        neg = set(meta.loc[meta["tumor?"] == 0, "PanTS ID"])
        cands = [c for c in ok.sort_values("vol_cm3").case_id if c in neg]
        pick = [cands[i] for i in np.linspace(0, len(cands) - 1, a.smoke).astype(int)]
        pick += [c for c in ("PanTS_00007401", "PanTS_00000708") if c not in pick]
        L.append(f"smoke profile on {len(pick)} tumor-negative cases (smallest -> largest outline):")
        L += ["  " + s for s in smoke(pick, OUT / "smoke_profiles.png")]
    txt = "\n".join(L)
    (OUT / "cohort_qc.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
