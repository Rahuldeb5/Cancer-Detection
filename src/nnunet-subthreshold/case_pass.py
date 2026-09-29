"""Per-case pass for session S2: what does nnU-Net's tumor probability look like below its
own 0.5 threshold, inside the lesions it misses?

One worker call per case (out-of-fold validation predictions only). It writes one pickle
per case to work/nnunet_subthreshold/cases/ (resumable) and one float16 probability crop
per case to work/nnunet_subthreshold/probs/. summarize.py turns the pickles into the
deliverables.

Per case it records:
  case     argmax==seg mismatches and max |p0+p1-1| over the whole volume (step-0 checks,
           now cohort-wide); saved-seg localization ("a predicted component touches GT" is
           the same as "some predicted voxel overlaps GT"); peak p1 in the whole volume and
           in the search region, and in the search region minus every lesion dilated 2 mm
  lesion   peak p1 inside the GT lesion and inside the lesion dilated 2 mm (Euclidean, mm),
           as p, clipped logit and unclipped logit (log p1 - log p0); how many of the case's
           own non-lesion search-region voxels are >= that peak
  comps    for each threshold t: 26-connected components of p1 > t over the whole volume,
           with peak voxel/value, size, which lesions they touch (overlap) and which lesions'
           2 mm dilation contains the peak voxel ("peak hit"), and search-region membership
  hist     p1 histograms of search-region voxels (lesion / non-lesion), on a linear-p grid and
           on the clipped-logit grid, plus saturation counts

The lesion mask is never used to build the search region (tumorlib.search_region).

Usage (server, repo root):
  ~/nnunet_setup/.venv/bin/python src/nnunet-subthreshold/case_pass.py --phantom       # oracle only
  ~/nnunet_setup/.venv/bin/python src/nnunet-subthreshold/case_pass.py --cases A B     # a few cases
  ~/nnunet_setup/.venv/bin/python src/nnunet-subthreshold/case_pass.py --workers 4     # phantom, then all
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label as cc_label, maximum_position

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from tumorlib import envelope, io as tio  # noqa: E402
from tumorlib.lesions import CC_STRUCT, bbox_slices, label_lesions  # noqa: E402

CASE_DIR = C.WORK / "cases"
PROB_DIR = C.WORK / "probs"
NUM_WORKERS = 4
Z_CHUNK = 32

# histogram grids (search-region voxels)
P_EDGES = np.linspace(0.0, 1.0, 101)                        # linear p, 0.01 wide
L_EDGES = np.concatenate(([-np.inf], np.arange(-16.0, 16.01, 0.25), [np.inf]))  # clipped logit
SAT_LO, SAT_HI = 0.01, 0.99


# ------------------------------------------------------------------ whole-volume scan
def volume_scan(P: np.ndarray, seg: np.ndarray) -> dict:
    """Chunked over the stored z axis (contiguous in P) so no full-size float temp is made.
    Also returns the bbox of p1 > min(THRESHOLDS): every component at every threshold is in it."""
    t_lo = min(C.THRESHOLDS)
    nz = P.shape[1]
    mism, maxdev, n_one, n_p0zero = 0, 0.0, 0, 0
    pmax, lmax = -np.inf, -np.inf
    any_x = np.zeros(P.shape[3], bool)
    any_y = np.zeros(P.shape[2], bool)
    any_z = np.zeros(nz, bool)
    for s in range(0, nz, Z_CHUNK):
        b0, b1 = P[0, s:s + Z_CHUNK], P[1, s:s + Z_CHUNK]          # (z, y, x)
        mism += int(((b1 > b0) != seg[:, :, s:s + Z_CHUNK].transpose(2, 1, 0).astype(bool)).sum())
        maxdev = max(maxdev, float(np.abs(b0.astype(np.float64) + b1 - 1.0).max()))
        n_one += int((b1 == 1.0).sum())
        n_p0zero += int((b0 == 0.0).sum())
        pmax = max(pmax, float(b1.max()))
        k = int(np.argmax(b1))  # raw logit at the p1 argmax is the chunk max only if p1 < 1; handle ties below
        top = b1 >= b1.flat[k]
        lmax = max(lmax, float(C.raw_logit(b1[top], b0[top]).max()))
        m = b1 > t_lo
        if m.any():
            any_z[s:s + Z_CHUNK] |= m.any(axis=(1, 2))
            any_y |= m.any(axis=(0, 2))
            any_x |= m.any(axis=(0, 1))
    bb = None
    if any_x.any():
        rng = [np.nonzero(a)[0] for a in (any_x, any_y, any_z)]
        bb = tuple(slice(int(r[0]), int(r[-1]) + 1) for r in rng)
    return dict(argmax_mismatch=mism, max_sum_dev=maxdev, n_p1_eq_1=n_one, n_p0_eq_0=n_p0zero,
                vol_peak_p=pmax, vol_peak_raw_logit=lmax, bbox_lo=bb)


# ------------------------------------------------------------------ core analysis
def lesion_geometry(labels: np.ndarray, n: int, sp) -> list[dict]:
    """Per lesion: padded crop slices, lesion crop, 2 mm Euclidean dilation crop."""
    sp = np.asarray(sp, float)
    pad = np.ceil(C.DIL_MM / sp).astype(int) + 1
    out = []
    for j in range(1, n + 1):
        sl = bbox_slices(labels == j, pad)
        les = labels[sl] == j
        dil = distance_transform_edt(~les, sampling=sp) <= C.DIL_MM
        out.append(dict(lesion_id=j, sl=sl, les=les, dil=dil, gt_vox=int(les.sum())))
    return out


def _in_crop(pt, sl, crop) -> bool:
    rel = [p - s.start for p, s in zip(pt, sl)]
    return all(0 <= r < n for r, n in zip(rel, crop.shape)) and bool(crop[tuple(rel)])


def analyze(case_id: str, fold: int, p1: np.ndarray, p0: np.ndarray, seg: np.ndarray, gt: np.ndarray,
            sr: np.ndarray | None, sp, vol: dict) -> dict:
    """Everything except IO. p1/p0 are (x, y, z) arrays or views; seg/gt/sr uint8 {0,1}."""
    labels, n = label_lesions(gt)
    geo = lesion_geometry(labels, n, sp)
    vox_mm3 = float(np.prod(sp))
    seg_b, gt_b = seg.astype(bool), gt.astype(bool)

    case = dict(case_id=case_id, fold=fold, gt_tumor=int(n > 0), n_lesions=n,
                sp_x=sp[0], sp_y=sp[1], sp_z=sp[2],
                seg_any=bool(seg_b.any()), seg_touches_gt=bool((seg_b & gt_b).any()),
                p05_vs_seg_mismatch=None, **{k: v for k, v in vol.items() if k != "bbox_lo"})
    ssl = bbox_slices(seg_b)
    case["n_seg_comps"] = int(cc_label(seg_b[ssl], structure=CC_STRUCT)[1]) if ssl is not None else 0

    # ---- search region
    sr_ok = sr is not None and sr.any()
    case["sr_ok"] = bool(sr_ok)
    hist = dict(p_les=np.zeros(len(P_EDGES) - 1, np.int64), p_bg=np.zeros(len(P_EDGES) - 1, np.int64),
                l_les=np.zeros(len(L_EDGES) - 1, np.int64), l_bg=np.zeros(len(L_EDGES) - 1, np.int64))
    bg_vals = np.zeros(0, np.float32)
    if sr_ok:
        srb = bbox_slices(sr)
        srv = sr[srb].astype(bool)
        p1s = np.asarray(p1[srb], dtype=np.float32)
        les_s = gt_b[srb]
        dil_s = np.zeros(srv.shape, bool)
        for g in geo:  # paste each lesion's 2 mm dilation into search-region-bbox coords
            dst, src = [], []
            for a, b in zip(g["sl"], srb):
                lo, hi = max(a.start, b.start), min(a.stop, b.stop)
                if lo >= hi:
                    break
                dst.append(slice(lo - b.start, hi - b.start))
                src.append(slice(lo - a.start, hi - a.start))
            else:
                dil_s[tuple(dst)] |= g["dil"][tuple(src)]
        all_sr = p1s[srv]
        bg_vals = p1s[srv & ~dil_s]
        for key, vals in (("les", p1s[srv & les_s]), ("bg", p1s[srv & ~les_s])):
            hist[f"p_{key}"] += np.histogram(vals, P_EDGES)[0]
            hist[f"l_{key}"] += np.histogram(C.clip_logit(vals), L_EDGES)[0]
        case.update(sr_vox=int(srv.sum()), sr_peak_p=float(all_sr.max()),
                    sr_nonlesion_vox=int(bg_vals.size),
                    sr_nonlesion_peak_p=float(bg_vals.max()) if bg_vals.size else np.nan,
                    sr_nonlesion_p50=float(np.median(bg_vals)) if bg_vals.size else np.nan,
                    sr_nonlesion_p99=float(np.percentile(bg_vals, 99)) if bg_vals.size else np.nan,
                    sr_n_mid=int(((all_sr >= SAT_LO) & (all_sr <= SAT_HI)).sum()),
                    sr_n_ge_hi_clip=int((all_sr >= 1 - C.EPS).sum()),
                    sr_n_le_lo_clip=int((all_sr <= C.EPS).sum()),
                    sr_n_eq_1=int((all_sr == 1.0).sum()))
        del p1s, les_s, dil_s, all_sr
    bg_sorted = np.sort(bg_vals)

    # ---- lesions
    lrows = []
    for g in geo:
        les_idx = tuple(a + s.start for a, s in zip(np.nonzero(g["les"]), g["sl"]))
        dil_idx = tuple(a + s.start for a, s in zip(np.nonzero(g["dil"]), g["sl"]))
        r = dict(case_id=case_id, lesion_id=g["lesion_id"], gt_vox=g["gt_vox"],
                 seg_touches=bool(seg_b[les_idx].any()),
                 in_sr_frac=float(sr[les_idx].mean()) if sr_ok else np.nan)
        for tag, idx in (("les", les_idx), ("dil2", dil_idx)):
            v1 = np.asarray(p1[idx], dtype=np.float32)
            v0 = np.asarray(p0[idx], dtype=np.float32)
            k = int(np.argmax(v1))
            rl = C.raw_logit(v1, v0)
            r.update({f"peak_p_{tag}": float(v1[k]), f"peak_logit_{tag}": float(C.clip_logit(v1[k])),
                      f"peak_raw_logit_{tag}": float(rl.max()), f"mean_p_{tag}": float(v1.mean()),
                      f"peak_xyz_{tag}": tuple(int(a[k]) for a in idx)})
            if bg_sorted.size:  # own-case non-lesion search-region voxels at or above this peak
                r[f"sr_bg_frac_ge_{tag}"] = float(1.0 - np.searchsorted(bg_sorted, v1[k], "left") / bg_sorted.size)
        lrows.append(r)

    # ---- components at each threshold, whole volume
    crows = []
    bb = vol["bbox_lo"]
    if bb is not None:
        c1 = np.ascontiguousarray(p1[bb], dtype=np.float32)
        off = np.array([s.start for s in bb])
        sr_bb = sr[bb].astype(bool) if sr_ok else None
        les_in_bb = []
        for g in geo:  # lesion voxels in bbox coords, for the touch test
            coords = np.stack(np.nonzero(g["les"]), 1) + [s.start for s in g["sl"]] - off
            keep = np.all((coords >= 0) & (coords < c1.shape), axis=1)
            les_in_bb.append(tuple(coords[keep].T))
        for t in C.THRESHOLDS:
            lab, nc = cc_label(c1 > t, structure=CC_STRUCT)
            if nc == 0:
                continue
            ids = np.arange(1, nc + 1)
            sizes = np.bincount(lab.ravel(), minlength=nc + 1)[1:]
            peaks = maximum_position(c1, lab, ids)
            in_sr = np.bincount(lab[sr_bb], minlength=nc + 1)[1:] if sr_ok else np.zeros(nc, int)
            touch = {c: [] for c in ids}
            for g, lc in zip(geo, les_in_bb):
                if len(lc) and len(lc[0]):
                    for c in np.unique(lab[lc]):
                        if c:
                            touch[int(c)].append(g["lesion_id"])
            for i, c in enumerate(ids):
                gp = tuple(int(a) for a in np.asarray(peaks[i]) + off)
                pk = float(c1[peaks[i]])
                hits = [g["lesion_id"] for g in geo if _in_crop(gp, g["sl"], g["dil"])]
                crows.append(dict(case_id=case_id, fold=fold, t=t, comp_id=int(c), n_vox=int(sizes[i]),
                                  vol_mm3=sizes[i] * vox_mm3, peak_p=pk, peak_logit=float(C.clip_logit(pk)),
                                  peak_raw_logit=float(C.raw_logit(pk, p0[gp])),
                                  peak_x=gp[0], peak_y=gp[1], peak_z=gp[2],
                                  peak_in_sr=bool(sr[gp]) if sr_ok else False,
                                  frac_in_sr=float(in_sr[i] / sizes[i]),
                                  touch_ids=";".join(map(str, touch[int(c)])),
                                  peakhit_ids=";".join(map(str, hits))))
            if t == 0.5:
                full = np.zeros(seg.shape, bool)
                full[bb] = c1 > 0.5
                case["p05_vs_seg_mismatch"] = int((full != seg_b).sum())
                del full
            del lab
        if case["p05_vs_seg_mismatch"] is None:
            case["p05_vs_seg_mismatch"] = int(seg_b.sum())
    else:
        case["p05_vs_seg_mismatch"] = int(seg_b.sum())
    return dict(case=case, lesions=lrows, comps=crows, hist=hist)


# ------------------------------------------------------------------ export
def export_crop(case_id: str, fold: int, p1: np.ndarray, sr: np.ndarray | None, sp, affine) -> dict:
    """float16 p1 (and float16 clipped logit) on the search-region bbox + 20 mm."""
    if sr is None or not sr.any():
        return dict(export="skipped: empty search region")
    pad = np.ceil(C.EXPORT_MARGIN_MM / np.asarray(sp, float)).astype(int)
    box = bbox_slices(sr, pad)
    crop = np.asarray(p1[box], dtype=np.float32)
    PROB_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PROB_DIR / f"{case_id}.npz",
        prob=crop.astype(np.float16), logit=C.clip_logit(crop).astype(np.float16),
        crop_start=np.array([s.start for s in box]), crop_stop=np.array([s.stop for s in box]),
        full_shape=np.array(p1.shape), affine=np.asarray(affine), spacing=np.asarray(sp),
        fold=np.array(fold), case_id=np.array(case_id))
    return dict(export="ok", export_shape=crop.shape,
                export_n_f16_eq_1=int((crop.astype(np.float16) == 1.0).sum()),
                export_n_f16_eq_0=int((crop.astype(np.float16) == 0.0).sum()))


# ------------------------------------------------------------------ worker
def process(job: tuple[str, int]) -> tuple[str, str]:
    case_id, fold = job
    out = CASE_DIR / f"{case_id}.pkl"
    if out.exists():
        return case_id, "cached"
    t0 = time.time()
    try:
        sp = tio.spacing(case_id)
        sr = envelope.search_region(case_id)       # before loading the big array
        seg, aff = C.load_seg(case_id, fold)
        gt, _ = C.load_gt(case_id)
        P = C.load_probs(case_id, fold)
        p1, p0 = C.xyz(P, 1), C.xyz(P, 0)
        if not (p1.shape == seg.shape == gt.shape) or (sr is not None and sr.shape != seg.shape):
            return case_id, f"shape mismatch p1 {p1.shape} seg {seg.shape} gt {gt.shape}"
        vol = volume_scan(P, seg)
        res = analyze(case_id, fold, p1, p0, seg, gt, sr, sp, vol)
        res["case"].update(export_crop(case_id, fold, p1, sr, sp, aff))
        res["case"]["npz_dtype"] = str(P.dtype)
        res["case"]["seconds"] = round(time.time() - t0, 1)
        CASE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.rename(out)
        return case_id, f"ok {res['case']['seconds']}s"
    except Exception as e:
        traceback.print_exc()
        return case_id, f"error {type(e).__name__}: {e}"


# ------------------------------------------------------------------ phantom oracle
def phantom() -> None:
    """Known-answer volume on an anisotropic grid (0.8, 0.8, 2.5 mm):
      lesion 1: sphere r=6 mm, p1 = 0.95 in its core           -> found at every t
      lesion 2: sphere r=3 mm, p1 = 0.2 blob centred 1.6 mm (2 voxels in x) outside its edge, not touching
                                                               -> peak hit (<=2 mm) but no touch, only for t < 0.2
      lesion 3: sphere r=3 mm, p1 = 0.004 inside               -> hit only at t = 0.003, 0.001
      FP blob far away, p1 = 0.6                               -> FP for t < 0.6
      touch-but-far-peak: a 0.08 shell overlapping lesion 1's edge (not connected to its 0.95 core) whose
                          own peak (0.4) sits 5.8 mm outside  -> touch hit, peak-rule FP, for t < 0.08
    Background p1 = 1e-9 (below the clip floor).
    """
    sp = (0.8, 0.8, 2.5)
    shape = (120, 120, 40)
    g = np.stack(np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, sp)], indexing="ij"), -1)

    def ball(c, r):
        return np.linalg.norm(g - np.asarray(c), axis=-1) <= r

    gt = np.zeros(shape, np.uint8)
    c1, c2, c3 = (30, 30, 50), (70, 30, 50), (30, 70, 50)
    gt[ball(c1, 6)] = 1
    gt[ball(c2, 3)] = 1
    gt[ball(c3, 3)] = 1
    p1 = np.full(shape, 1e-9, np.float32)
    p1[ball(c1, 4)] = 0.95
    # lesion 2: blob just outside, 2 voxels (1.6 mm) beyond the lesion edge in x -> no overlap
    edge2 = (c2[0] + 3.0 + 1.6, c2[1], c2[2])
    blob2 = ball(edge2, 0.5)
    assert not (blob2 & ball(c2, 3)).any()
    p1[blob2] = 0.2
    p1[ball(c3, 1.5)] = 0.004
    p1[ball((80, 80, 50), 3)] = 0.6
    # low-level shell hanging off lesion 1's surface, peak ~5.8 mm outside; a 1-voxel gap separates it
    # from lesion 1's core (shell reaches x index 31, core starts at 33)
    far = (c1[0] - 6 - 5, c1[1], c1[2])
    shell = ball(far, 6.0) & ~ball(c1, 4)
    assert (shell & ball(c1, 6)).any()
    p1[shell] = np.maximum(p1[shell], 0.08)
    p1[ball(far, 1)] = 0.4
    p0 = (1.0 - p1).astype(np.float32)
    seg = (p1 > 0.5).astype(np.uint8)
    sr = np.ones(shape, np.uint8)
    P = np.stack([p0, p1]).transpose(0, 3, 2, 1)  # stored (C, z, y, x)
    vol = volume_scan(np.ascontiguousarray(P), seg)
    res = analyze("phantom", 0, C.xyz(P, 1), C.xyz(P, 0), seg, gt, sr, sp, vol)

    comps = res["comps"]
    # ids follow label_lesions' raster order, so look them up rather than assume 1/2/3
    labs = label_lesions(gt)[0]
    ids = [int(labs[tuple(int(round(v / s)) for v, s in zip(c, sp))]) for c in (c1, c2, c3)]
    assert sorted(ids) == [1, 2, 3], ids
    L = {k + 1: i for k, i in enumerate(ids)}          # phantom lesion k -> lesion_id
    lr = {k: r for k in L for r in res["lesions"] if r["lesion_id"] == L[k]}

    def hit(t, k, key):
        return any(str(L[k]) in c[key].split(";") for c in comps if c["t"] == t)

    def n_fp(t, key):
        return sum(1 for c in comps if c["t"] == t and not c[key])

    expect = {  # t -> (L1, L2, L3) peak hit
        0.9: (1, 0, 0), 0.5: (1, 0, 0), 0.1: (1, 1, 0), 0.03: (1, 1, 0), 0.003: (1, 1, 1), 0.001: (1, 1, 1)}
    for t, exp in expect.items():
        got = tuple(int(hit(t, j, "peakhit_ids")) for j in (1, 2, 3))
        assert got == exp, (t, got, exp)
    for t in (0.9, 0.5, 0.1, 0.003):
        assert not hit(t, 2, "touch_ids"), t      # lesion 2's blob never overlaps it
    assert hit(0.003, 3, "touch_ids") and not hit(0.01, 3, "touch_ids")
    # FP counts (peak rule): t=0.9 -> 0; 0.5 -> FP blob; 0.3 -> FP blob + far peak 0.4
    assert n_fp(0.9, "peakhit_ids") == 0 and n_fp(0.5, "peakhit_ids") == 1 and n_fp(0.3, "peakhit_ids") == 2
    # t=0.03: L1 core, shell (touches L1, peak outside), L2 blob (peak hit, no touch), FP blob
    c03 = [c for c in comps if c["t"] == 0.03]
    assert len(c03) == 4, len(c03)
    assert n_fp(0.03, "peakhit_ids") == 2 and n_fp(0.03, "touch_ids") == 2
    shell_c = [c for c in c03 if c["touch_ids"] == str(L[1]) and c["peakhit_ids"] == ""]
    assert len(shell_c) == 1 and abs(shell_c[0]["peak_p"] - 0.4) < 1e-6
    assert n_fp(0.1, "peakhit_ids") == 2 and n_fp(0.1, "touch_ids") == 3   # shell gone; far 0.4 blob + FP; L2 blob no touch
    assert lr[1]["seg_touches"] and not lr[2]["seg_touches"] and not lr[3]["seg_touches"]
    assert abs(lr[2]["peak_p_dil2"] - 0.2) < 1e-6 and lr[2]["peak_p_les"] < 1e-6
    assert abs(lr[3]["peak_p_les"] - 0.004) < 1e-6
    assert abs(lr[1]["peak_logit_les"] - np.log(0.95 / 0.05)) < 1e-4
    assert abs(lr[2]["peak_logit_les"] - np.log(1e-7 / (1 - 1e-7))) < 1e-6   # clipped at the floor
    assert lr[2]["peak_raw_logit_les"] < -20                                    # unclipped sees below it
    assert res["case"]["argmax_mismatch"] == 0 and res["case"]["p05_vs_seg_mismatch"] == 0
    assert res["case"]["max_sum_dev"] < 1e-6
    print("phantom: all checks passed "
          f"({len(comps)} component rows; lesion peaks p {[round(lr[j]['peak_p_dil2'], 4) for j in (1, 2, 3)]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phantom", action="store_true", help="run the phantom oracle only")
    ap.add_argument("--cases", nargs="*", help="only these case ids")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = ap.parse_args()
    phantom()
    if args.phantom:
        return
    fold = C.fold_of_case()
    ids = args.cases or sorted(fold)
    jobs = [(c, fold[c]) for c in ids]
    print(f"{len(jobs)} cases, {args.workers} workers -> {CASE_DIR}", flush=True)
    t0 = time.time()
    bad = []
    with mp.Pool(min(args.workers, NUM_WORKERS), maxtasksperchild=10) as pool:
        for i, (cid, status) in enumerate(pool.imap_unordered(process, jobs, chunksize=1), 1):
            if not (status.startswith("ok") or status == "cached"):
                bad.append((cid, status))
                print(f"  {cid}: {status}", flush=True)
            if i % 50 == 0 or i == len(jobs) or len(jobs) <= 10:
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}s) last {cid} {status}", flush=True)
    print(f"done; {len(bad)} failures: {bad}")


if __name__ == "__main__":
    main()
