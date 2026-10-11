"""S15 Part B steps 8-9 (+ round-4 item 6), PC: normal reference and insertion tests on the 300 S9 bank glands.

Inputs: ~/research/gland_profiles/bank_pred/{case}.npz (bank_extract.py: 502 5-fold ensemble pancreas-region
probability, float16 crop), bank CTs (ct_normal_bank/, header + HU along rays), bank aorta masks.
No GT pancreas or lesion mask is read.

1. Split ONCE, before QC: rng(SPLIT_SEED).permutation of the sorted 300 bank ids -> first 250 reference, last 50
   held out. Recorded in work/gland_shape/partB/bank_split.csv.
2. Per gland: outline = p0 > 0.5 (same rule as the cohort), gland QC (glandshape.qc: drop empty, < 5 or > 250 cm^3,
   centerline failures; fragmented kept and flagged), aorta test, centerline, profile with HU. Profiles saved to
   ~/research/gland_profiles/bank/{case}.npz.
3. Reference = the kept reference glands: width median / MAD per (u bin 0.05, axis) and the atrophy reference;
   atrophy threshold = P95 of the reference glands' case atrophy, each scored leave-one-out. Saved to
   ~/research/gland_profiles/bank_reference.npz.
4. Held-out (kept) glands, scored with the v4 PRIMARY (residual width) and the secondary (raw width):
   a. top cell (u, axis) distributions; share of top cells at the head-body transition u in [0.2, 0.45]
      (S10 median head/body boundary 0.364, carried over) vs that band's share of scoreable cells;
   b. top-3 peaks (partc.top_peaks) per gland -> heldout_peaks.json (Part C's empirical chance).
5. Insertion test (item 9) on the held-out kept glands. One surface point per gland, fixed before any insertion:
   rng(INSERT_SEED + gland index) picks a valid stop with 0.1 <= u <= 0.9 and one of the 24 ray directions; the
   point = ray origin + r * direction (on the outline surface). Spheres of diameter 15, 20, 30 mm centred there are
   added to the native-grid predicted region (half outside), the whole pipeline is rerun, and the within-scan
   percentile (partc.within_scan_percentile) of the primary score over the sphere's footprint (footprint.py with the
   sphere as the "lesion") is reported. Gate (fixed in the brief): 20 mm median percentile >= 95.
   Shrink: native voxels within 2 mm of the outline surface whose nearest centerline stop has u > 0.5 are removed;
   the pipeline is rerun; case atrophy is compared with the threshold and with the 50 unmodified held-out glands.

Usage: PYTHONPATH=src .venv/bin/python -m glandshape.bank_profile [--workers 4]
"""
from __future__ import annotations

import os
from pathlib import Path

BANK = Path.home() / "research/datasets/pants/ct_normal_bank"
os.environ["TUMORLIB_CT_ROOT"] = str(BANK)  # before tumorlib is imported

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io as sio  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from glandshape import centerline as C  # noqa: E402
from glandshape import core as K  # noqa: E402
from glandshape import footprint as FP  # noqa: E402
from glandshape import orient as OR  # noqa: E402
from glandshape import outline as O  # noqa: E402
from glandshape import partc as PC  # noqa: E402
from glandshape import profile as P  # noqa: E402
from glandshape import qc as QC  # noqa: E402
from glandshape import scores as S  # noqa: E402
from tumorlib import io as tio  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PRED = Path.home() / "research/gland_profiles/bank_pred"
PROF = Path.home() / "research/gland_profiles/bank"
REF_FILE = Path.home() / "research/gland_profiles/bank_reference.npz"
OUT = REPO / "work/gland_shape/partB"
SPLIT_SEED = 15
N_REF = 250
INSERT_SEED = 1500
SIZES_MM = (15.0, 20.0, 30.0)
SHRINK_MM = 2.0
HB_BAND = (0.2, 0.45)


# ------------------------------------------------------------------ pipeline pieces
def load_pred_coords(cid: str):
    with np.load(PRED / f"{cid}.npz") as d:
        p0, off, shape = d["p0"], d["offset"], tuple(int(v) for v in d["shape"])
    g = tio.grid(cid)
    if g is None or tuple(g[0]) != shape:
        raise RuntimeError(f"pred shape {shape} != CT {None if g is None else g[0]}")
    xyz = np.argwhere(p0.astype(np.float32) > K.PANC_THRESHOLD) + off
    return xyz, shape, g


def run_pipeline(xyz, shape, g, aorta, with_hu=False, cid=None):
    """-> dict(outline, cl, pr, drop, flags, aorta) ; drop is a QC reason or None."""
    ol = O.outline_from_coords(xyz, shape, g[2], g[1])
    res = dict(outline=ol, cl=None, pr=None, flags=QC.flags(ol), aorta=None)
    res["drop"] = QC.outline_drop(ol)
    if res["drop"]:
        return res
    at = OR.aorta_test(ol, aorta)
    res["aorta"] = at
    try:
        cl = C.build_centerline(ol, OR.anterior_world(at))
    except Exception as e:
        res["drop"] = f"centerline_error: {e}"
        return res
    res["drop"] = QC.centerline_drop(ol, cl.L)
    if res["drop"]:
        return res
    try:
        ct = P.make_ct_sampler(ol, g[0], lambda sl: tio.load_ct(cid, sl)) if with_hu else None
        res["pr"] = P.compute_profile(ol, cl, ct_sampler=ct)
    except Exception as e:
        res["drop"] = f"profile_error: {e}"
        return res
    res["cl"] = cl
    return res


def _aorta(cid):
    with contextlib.redirect_stdout(sio.StringIO()):
        return tio.load_mask(cid, "aorta")


def save_profile(cid, res):
    pr, cl = res["pr"], res["cl"]
    PROF.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROF / f"{cid}.npz", s=pr.s, u=pr.u, r=pr.r.astype(np.float32), w=pr.w.astype(np.float32),
                        A=pr.A, valid=pr.valid, fold=pr.fold, censored=pr.censored, w_censored=pr.w_censored,
                        origin=pr.origin, origin_fallback=pr.origin_fallback, hu=pr.hu, L=pr.L, P=cl.P, T=cl.T,
                        N=cl.N, B=cl.B, offset=res["outline"].offset, aorta_dy=res["aorta"]["aorta_dy_mm"],
                        aorta_call=res["aorta"]["aorta_call"])


class Prof:
    def __init__(self, f):
        with np.load(f, allow_pickle=False) as d:
            for k in ("s", "u", "r", "w", "A", "valid", "fold", "w_censored"):
                setattr(self, k, d[k])


# ------------------------------------------------------------------ pass 1: every bank gland
def pass1(cid):
    t0 = time.time()
    row = {"case_id": cid}
    try:
        xyz, shape, g = load_pred_coords(cid)
        res = run_pipeline(xyz, shape, g, _aorta(cid), with_hu=True, cid=cid)
        ol = res["outline"]
        row.update(drop=res["drop"] or "", flags=";".join(res["flags"]),
                   vol_cm3=ol.vol_mm3 / 1000 if ol else np.nan, frac2=ol.frac2 if ol else np.nan)
        if res["aorta"]:
            row.update(aorta_dy_mm=res["aorta"]["aorta_dy_mm"], aorta_call=res["aorta"]["aorta_call"])
        if not res["drop"]:
            pr = res["pr"]
            row.update(L_mm=pr.L, n_stops=len(pr.s), n_fold=int(pr.fold.sum()), n_wcens=int(pr.w_censored.sum()),
                       n_fallback=int(pr.origin_fallback.sum()))
            save_profile(cid, res)
    except Exception as e:
        row["drop"] = f"error: {type(e).__name__}: {e}"
    row["secs"] = time.time() - t0
    return row


# ------------------------------------------------------------------ insertion / shrink
def insertion_point(res, gi):
    pr, cl = res["pr"], res["cl"]
    rng = np.random.default_rng(INSERT_SEED + gi)
    cand = np.nonzero(pr.valid & (pr.u >= 0.1) & (pr.u <= 0.9))[0]
    i = int(rng.choice(cand))
    k = int(rng.integers(K.N_DIRS))
    p_iso = pr.origin[i] + pr.r[i, k] * P.ray_dirs(cl)[i, k]
    return res["outline"].iso_to_world(p_iso[None])[0], i, k


def ball_coords(center_w, diam, shape, g):
    A = g[1]
    Ainv = np.linalg.inv(A)
    cn = (Ainv @ np.r_[center_w, 1.0])[:3]
    rad = diam / 2
    sp = np.asarray(g[2])
    lo = np.maximum(np.floor(cn - rad / sp) - 1, 0).astype(int)
    hi = np.minimum(np.ceil(cn + rad / sp) + 2, np.asarray(shape)).astype(int)
    grid = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
    w = grid @ A[:3, :3].T + A[:3, 3]
    return grid[np.linalg.norm(w - center_w, axis=1) <= rad]


def shrink_coords(xyz, res, shape, g):
    """Remove native voxels within SHRINK_MM of the region surface whose nearest centerline stop has u > 0.5."""
    sp = np.asarray(g[2])
    pad = np.ceil(SHRINK_MM / sp).astype(int) + 2
    lo = np.maximum(xyz.min(0) - pad, 0)
    hi = np.minimum(xyz.max(0) + 1 + pad, np.asarray(shape))
    m = np.zeros(tuple(hi - lo), bool)
    m[tuple((xyz - lo).T)] = True
    dist = distance_transform_edt(m, sampling=sp)                       # distance to the nearest outside voxel
    near = m & (dist <= SHRINK_MM)
    cand = np.argwhere(near) + lo
    ol, cl, pr = res["outline"], res["cl"], res["pr"]
    world = cand @ g[1][:3, :3].T + g[1][:3, 3]
    _, j = cKDTree(ol.iso_to_world(cl.P)).query(world)
    rem = cand[pr.u[j] > 0.5]
    keep = np.ones(len(xyz), bool)
    key = lambda a: np.ravel_multi_index(a.T, shape)  # noqa: E731
    keep[np.isin(key(xyz), key(rem))] = False
    return xyz[keep], len(rem)


def pass2(args):
    """Held-out gland gi: insertions and shrink. Scores against the bank reference (loaded from REF_FILE)."""
    cid, gi = args
    ref = load_reference()
    out = {"case_id": cid, "rows": []}
    try:
        xyz, shape, g = load_pred_coords(cid)
        aorta = _aorta(cid)
        base = run_pipeline(xyz, shape, g, aorta)
        pt, i0, k0 = insertion_point(base, gi)
        out.update(ins_stop=i0, ins_dir=k0, ins_u=float(base["pr"].u[i0]))
        for diam in SIZES_MM:
            ball = ball_coords(pt, diam, shape, g)
            xyz2 = np.unique(np.vstack([xyz, ball]), axis=0)
            res = run_pipeline(xyz2, shape, g, aorta)
            row = dict(case_id=cid, kind=f"sphere{diam:.0f}", drop=res["drop"] or "")
            if not res["drop"]:
                pr, cl = res["pr"], res["cl"]
                fp = FP.footprint(res["outline"], cl, pr, ball)
                prim = S.bulge_residual(pr.w, pr.u, ref)["primary"]
                sec = S.bulge_multiscale(pr.w)["primary"]
                a, b = PC.within_scan_percentile(prim, fp), PC.within_scan_percentile(sec, fp)
                row.update(fp_cells=int(fp.sum()), pct_primary=a["percentile"], pct_raw=b["percentile"],
                           top3_primary=PC.top3_hit(prim, pr.s, fp), top3_raw=PC.top3_hit(sec, pr.s, fp),
                           chance_uniform=PC.chance_top3_uniform(fp, np.isfinite(prim), pr.s),
                           n_windows=a["n_windows"])
            out["rows"].append(row)
        xyz3, nrem = shrink_coords(xyz, base, shape, g)
        res = run_pipeline(xyz3, shape, g, aorta)
        row = dict(case_id=cid, kind="shrink_tail_2mm", drop=res["drop"] or "", removed_vox=nrem)
        if not res["drop"]:
            z = S.atrophy_z(res["pr"].A, res["pr"].u, ref)
            mag, sgn, _ = S.case_atrophy(z)
            row.update(atrophy_mag=mag, atrophy_signed=sgn)
        out["rows"].append(row)
        z0 = S.atrophy_z(base["pr"].A, base["pr"].u, ref)
        m0, s0, _ = S.case_atrophy(z0)
        out["rows"].append(dict(case_id=cid, kind="unmodified", drop="", atrophy_mag=m0, atrophy_signed=s0))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# ------------------------------------------------------------------ reference
def save_reference(ref, thr, ids):
    REF_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(REF_FILE, w_med=ref.w_med, w_mad=ref.w_mad, atro_med=ref.atro_med, atro_mad=ref.atro_mad,
             atw_med=ref.atw_med, atw_mad=ref.atw_mad, n_glands=ref.n_glands, atrophy_thr=thr,
             ids=np.array(ids))


def load_reference() -> S.Reference:
    with np.load(REF_FILE) as d:
        return S.Reference(d["w_med"], d["w_mad"], d["atro_med"], d["atro_mad"], int(d["n_glands"]),
                           d["atw_med"], d["atw_mad"])


def loo_threshold(profs):
    mags = []
    for j in range(len(profs)):
        r = S.build_reference([p for i, p in enumerate(profs) if i != j])
        mags.append(S.case_atrophy(S.atrophy_z(profs[j].A, profs[j].u, r))[0])
    return float(np.nanpercentile(mags, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip_pass1", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = sorted(f.stem for f in PRED.glob("PanTS_*.npz"))
    assert len(ids) == 300, len(ids)
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(ids))
    split = pd.DataFrame({"case_id": [ids[i] for i in perm], "role": ["reference"] * N_REF + ["heldout"] * (300 - N_REF)})
    split.to_csv(OUT / "bank_split.csv", index=False)
    L = [f"S15 Part B bank glands: split seed {SPLIT_SEED}, {N_REF} reference / {300 - N_REF} held out (before QC)"]

    t0 = time.time()
    if not a.skip_pass1:
        with mp.Pool(a.workers) as pool:
            rows = pool.map(pass1, list(split.case_id), chunksize=2)
        pd.DataFrame(rows).to_csv(OUT / "bank_pass1.csv", index=False)
    q = pd.read_csv(OUT / "bank_pass1.csv").merge(split, on="case_id")
    q["drop"] = q["drop"].fillna("")
    q["frag"] = q["flags"].fillna("").str.contains("fragmented")
    L.append(f"pass 1: {time.time() - t0:.0f} s")
    L.append("QC drops by role:\n" + pd.crosstab(q["drop"].replace("", "kept"), q.role).to_string())
    L.append(f"fragmented-flagged among kept: {q[(q['drop'] == '') & q.frag].groupby('role').size().to_dict()}")
    L.append(f"aorta calls: {q.aorta_call.value_counts().to_dict()}")
    kept = q[q["drop"] == ""]
    L.append(f"kept glands: length median {kept.L_mm.median():.0f} mm; with fold flags {int((kept.n_fold > 0).sum())}; "
             f"with censored widths {int((kept.n_wcens > 0).sum())}")

    ref_ids = list(kept[kept.role == "reference"].case_id)
    ho_ids = list(kept[kept.role == "heldout"].case_id)
    ref_profs = [Prof(PROF / f"{c}.npz") for c in ref_ids]
    ref = S.build_reference(ref_profs)
    thr = loo_threshold(ref_profs)
    save_reference(ref, thr, ref_ids)
    L.append(f"reference: {ref.n_glands} glands; empty (u bin, axis) cells {int(np.isnan(ref.w_med).sum())}/240; "
             f"atrophy LOO P95 threshold |z| {thr:.2f}")
    L.append("reference median width by u bin (mm, median over axes): " +
             " ".join(f"{v:.0f}" for v in np.nanmedian(ref.w_med, 1)))

    # held-out: top cells (item 6) and top-3 peaks
    tops, peaks = [], {}
    for c in ho_ids:
        p = Prof(PROF / f"{c}.npz")
        prim = S.bulge_residual(p.w, p.u, ref)["primary"]
        sec = S.bulge_multiscale(p.w)["primary"]
        for name, sc in (("primary_residual", prim), ("secondary_raw", sec)):
            i, j = np.unravel_index(np.nanargmax(sc), sc.shape)
            band = (p.u >= HB_BAND[0]) & (p.u <= HB_BAND[1])
            tops.append(dict(case_id=c, score=name, u=float(p.u[i]), axis=int(j), z=float(sc[i, j]),
                             band_share=float(np.isfinite(sc[band]).sum() / np.isfinite(sc).sum())))
        peaks[c] = [(float(p.u[i]), int(j)) for i, j in PC.top_peaks(prim, p.s)]
    tp = pd.DataFrame(tops)
    tp.to_csv(OUT / "heldout_top_cells.csv", index=False)
    (OUT / "heldout_peaks.json").write_text(json.dumps(peaks))
    for name, d in tp.groupby("score"):
        inband = ((d.u >= HB_BAND[0]) & (d.u <= HB_BAND[1])).mean()
        L.append(f"held-out top cell, {name} (n={len(d)}): u deciles " +
                 str(np.histogram(d.u, bins=np.linspace(0, 1, 11))[0].tolist()) +
                 f"; axis counts {np.bincount(d.axis, minlength=12).tolist()}; in head-body band u {HB_BAND}: "
                 f"{inband:.0%} vs band's share of scoreable cells {d.band_share.median():.0%}; top z median "
                 f"{d.z.median():.2f}")

    # insertion / shrink
    t1 = time.time()
    with mp.Pool(a.workers) as pool:
        res = pool.map(pass2, [(c, gi) for gi, c in enumerate(ho_ids)], chunksize=1)
    rows = [r for o in res for r in o["rows"]]
    errs = [(o["case_id"], o["error"]) for o in res if "error" in o]
    ins = pd.DataFrame(rows)
    ins.to_csv(OUT / "insertion.csv", index=False)
    L.append(f"insertion pass: {time.time() - t1:.0f} s; errors {errs}")
    for diam in SIZES_MM:
        d = ins[(ins.kind == f"sphere{diam:.0f}") & (ins["drop"].fillna("") == "")]
        L.append(f"sphere {diam:.0f} mm (n={len(d)}, dropped {int((ins.kind == f'sphere{diam:.0f}').sum() - len(d))}): "
                 f"primary percentile median {d.pct_primary.median():.1f} (IQR {d.pct_primary.quantile(.25):.1f}-"
                 f"{d.pct_primary.quantile(.75):.1f}); raw {d.pct_raw.median():.1f}; top-3 hit primary "
                 f"{d.top3_primary.mean():.0%} / raw {d.top3_raw.mean():.0%} vs uniform chance median "
                 f"{d.chance_uniform.median():.1%}")
    d20 = ins[(ins.kind == "sphere20") & (ins["drop"].fillna("") == "")]
    L.append(f"GATE insertion 20 mm median percentile >= 95: {d20.pct_primary.median():.1f} -> "
             f"{'PASS' if d20.pct_primary.median() >= 95 else 'FAIL'}")
    sh = ins[(ins.kind == "shrink_tail_2mm") & (ins["drop"].fillna("") == "")]
    un = ins[ins.kind == "unmodified"]
    pct = [100 * np.mean(un.atrophy_mag < m) for m in sh.atrophy_mag]
    L.append(f"shrink tail 2 mm (n={len(sh)}): flagged (|z| > {thr:.2f} and z < 0) "
             f"{np.mean((sh.atrophy_mag > thr) & (sh.atrophy_signed < 0)):.0%}; unmodified flagged "
             f"{np.mean((un.atrophy_mag > thr) & (un.atrophy_signed < 0)):.0%}; median percentile of the shrunk case "
             f"atrophy among the unmodified held-out glands {np.median(pct):.0f}; removed voxels median "
             f"{sh.removed_vox.median():.0f}")
    txt = "\n".join(L)
    (OUT / "bank_profile.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
