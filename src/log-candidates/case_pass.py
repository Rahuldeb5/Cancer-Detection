"""S4 per-case pass: run the scale-space LoG detector on one case and score it against GT.

One worker call per case, one case in memory at a time, resumable (one pickle per case in
work/log_candidates/cases/). No 1 mm array is ever cached to disk.

Per case:
  region   tumorlib.search_region (hole-filled pancreas envelope + 3 mm; never reads the lesion
           mask), bbox-cropped with a PAD_MM margin so the widest kernel (4*6 mm) never reaches
           past the crop, then CT and region resampled to 1 mm isotropic and HU clipped.
  detect   logblob.detect -> per-scale peaks, the pooled scale-space candidate list (top K per
           polarity), the max-over-scales z map, and the per-scale noise MAD.
  lesions  EVALUATION ONLY, after detection: coverage of each lesion by the search region (on the
           native grid), the rank of the best peak landing in the lesion's 2 mm dilation (per
           scale and pooled), the continuous max z in that dilation per scale, and the same max
           compared with N_PATCH random lesion-sized patches elsewhere in this case's region.
  false    every candidate peak's distance to the nearest GT lesion voxel, the region volume, and
           the region volume more than FP_EXCL_MM from every lesion (the denominator for false
           peaks measured on positive scans).
  context  nearest anatomical structure to each candidate peak (vessels / ducts / duodenum ...).

Usage (from repo root, PC):
  .venv/bin/python src/log-candidates/case_pass.py --cases PanTS_00000003   # one case
  .venv/bin/python src/log-candidates/case_pass.py --workers 4              # all 1308
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
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
import logblob as LB  # noqa: E402
from tumorlib import envelope, io as tio  # noqa: E402
from tumorlib.lesions import bbox_slices, label_lesions  # noqa: E402
from tumorlib.resample import resample_ct_to_isotropic  # noqa: E402

WORK = REPO / "work" / "log_candidates"
CASE_DIR = WORK / "cases"
NUM_WORKERS = 4

PAD_MM = 30.0          # > 4 * max(sigma) = 24 mm, so no peak inside the region sees the crop edge
DIL_MM = 2.0           # "a peak is in the lesion" = inside the lesion dilated 2 mm (S2's rule)
FP_EXCL_MM = 10.0      # a peak further than this from every GT lesion in the case is a false peak
N_PATCH = 200          # random lesion-sized patches per lesion
PATCH_SEED = 20260929
Z_GRID = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0)
STRUCTURES = ("pancreatic_duct", "common_bile_duct", "duodenum", "stomach", "aorta", "postcava",
              "veins", "superior_mesenteric_artery", "celiac_artery", "gall_bladder")
STRUCT_MAX_MM = 30.0   # beyond this a peak is reported as "none"


# ------------------------------------------------------------------ grids
def native_to_iso_index(n_iso: int, spacing_axis: float, n_crop: int) -> np.ndarray:
    """Crop voxel index for each isotropic index: p_crop = p_iso / spacing (resample.py's map),
    clipped to the crop. Used to carry integer labels onto the 1 mm grid by nearest neighbour --
    linear interpolation of a label array would invent labels."""
    return np.clip(np.round(np.arange(n_iso) / spacing_axis).astype(int), 0, n_crop - 1)


def nn_resample_labels(lab_crop: np.ndarray, iso_shape, spacing) -> np.ndarray:
    gi = [native_to_iso_index(n, s, m) for n, s, m in zip(iso_shape, spacing, lab_crop.shape)]
    return lab_crop[np.ix_(*gi)]


# ------------------------------------------------------------------ per-lesion regions
def lesion_regions(lab_iso: np.ndarray, n: int, dil_mm: float = DIL_MM):
    """(dil_id, overlap_vox): uint16 map of which lesion's `dil_mm` dilation covers each voxel.

    Built per lesion on its own padded sub-crop. Where two dilations meet, the lower lesion_id
    keeps the voxel and the collision is counted (it only affects which lesion a shared peak is
    credited to, and both are reported).
    """
    dil_id = np.zeros(lab_iso.shape, np.uint16)
    overlap = 0
    for j in range(1, n + 1):
        comp = lab_iso == j
        sl = bbox_slices(comp, int(np.ceil(dil_mm)) + 1)
        if sl is None:
            continue
        sub = comp[sl]
        dil = distance_transform_edt(~sub, sampling=(1.0, 1.0, 1.0)) <= dil_mm
        tgt = dil_id[sl]
        overlap += int((dil & (tgt > 0)).sum())
        tgt[dil & (tgt == 0)] = j
    return dil_id, overlap


# ------------------------------------------------------------------ random patches
def patch_scores(smap: np.ndarray, centres: np.ndarray, radius_mm: float) -> np.ndarray:
    """max of `smap` inside a ball of radius_mm around each centre (1 mm isotropic grid)."""
    offs = LB.ball_offsets(radius_mm, (1.0, 1.0, 1.0))
    shape = np.asarray(smap.shape)
    out = np.empty(len(centres))
    for i, c in enumerate(centres):
        q = offs + c
        q = q[np.all((q >= 0) & (q < shape), axis=1)]
        out[i] = smap[q[:, 0], q[:, 1], q[:, 2]].max() if len(q) else -np.inf
    return out


# ------------------------------------------------------------------ structures
def nearest_structures(case_id: str, crop, spacing, pts_mm: np.ndarray, names=STRUCTURES):
    """(labels, distances_mm) of the nearest structure to each point, plus the masks that failed.

    Points are in mm from the crop origin, which is the same frame as crop voxel index * spacing,
    so nothing has to be resampled. A KD-tree over the structure's voxel centres inside the crop
    is cheaper than a distance transform of the whole crop for a handful of query points.
    """
    if len(pts_mm) == 0:
        return np.array([], dtype=object), np.array([]), []
    sp = np.asarray(spacing, float)
    best = np.full(len(pts_mm), np.inf)
    lab = np.full(len(pts_mm), "none", dtype=object)
    missing = []
    for name in names:
        m = tio.load_mask(case_id, name)
        if m is None:
            missing.append(name)
            continue
        sub = m[crop]
        del m
        idx = np.argwhere(sub)
        if len(idx) == 0:
            continue
        d, _ = cKDTree(idx * sp).query(pts_mm, k=1)
        upd = d < best
        best[upd] = d[upd]
        lab[np.asarray(upd)] = name
    lab[best > STRUCT_MAX_MM] = "none"
    return lab, best, missing


# ------------------------------------------------------------------ the pass
def analyze(case_id: str, fold: int, ct_iso: np.ndarray, sr_iso: np.ndarray, lab_iso: np.ndarray,
            n_les: int, vol_mm3: dict[int, float], in_sr_frac: dict[int, float], spacing,
            crop_start, rng) -> dict:
    """Everything after the arrays are on the 1 mm grid. No file I/O, so it is phantom-testable."""
    iso = (1.0, 1.0, 1.0)
    dil_id, dil_overlap = lesion_regions(lab_iso, n_les)
    probe = {f"les{j}": np.flatnonzero((dil_id == j) & sr_iso) for j in range(1, n_les + 1)}
    res = LB.detect(ct_iso, sr_iso, iso, probe_idx=probe)
    sigmas = res["sigmas"]

    les_any = lab_iso > 0
    d_les = (distance_transform_edt(~les_any, sampling=iso) if les_any.any()
             else np.full(lab_iso.shape, np.inf, np.float32))
    far = sr_iso & (d_les > FP_EXCL_MM)

    case = dict(case_id=case_id, fold=fold, gt_tumor=int(n_les > 0), n_lesions=n_les,
                sp_x=spacing[0], sp_y=spacing[1], sp_z=spacing[2],
                crop_start=tuple(int(v) for v in crop_start), iso_shape=tuple(ct_iso.shape),
                sr_vox_iso=int(sr_iso.sum()), sr_vol_cm3=float(sr_iso.sum()) / 1000.0,
                sr_vol_excl_tumor_cm3=float(far.sum()) / 1000.0,
                les_vox_iso=int(les_any.sum()), dil_overlap_vox=dil_overlap,
                noise_mad={float(s): float(res["noise"][s]) for s in sigmas})

    # ---- candidate peaks (pooled over scales)
    peaks = []
    for p in res["peaks"]:
        q = (p["ix"], p["iy"], p["iz"])
        peaks.append(dict(case_id=case_id, fold=fold, gt_tumor=case["gt_tumor"], **p,
                          nx=float(crop_start[0] + p["ix"] / spacing[0]),
                          ny=float(crop_start[1] + p["iy"] / spacing[1]),
                          nz=float(crop_start[2] + p["iz"] / spacing[2]),
                          dist_lesion_mm=float(d_les[q]), in_lesion=int(lab_iso[q]),
                          in_dil2=int(dil_id[q]), is_false=bool(d_les[q] > FP_EXCL_MM)))

    # ---- per-scale peak counts, split into in-lesion / false, for the FP-vs-scale table
    scale_rows = []
    for pol in LB.POLARITIES:
        for s in sigmas:
            sub = [p for p in res["scale_peaks"] if p["polarity"] == pol and p["sigma_mm"] == s]
            zz = np.array([p["z"] for p in sub])
            dd = np.array([d_les[(p["ix"], p["iy"], p["iz"])] for p in sub]) if sub else np.zeros(0)
            scale_rows.append(dict(case_id=case_id, polarity=pol, sigma_mm=s, n_peaks=len(sub),
                                   z_min_kept=float(zz.min()) if len(zz) else np.nan,
                                   **{f"n_ge_{t:g}": int((zz >= t).sum()) for t in Z_GRID},
                                   **{f"nfp_ge_{t:g}": int(((zz >= t) & (dd > FP_EXCL_MM)).sum())
                                      for t in Z_GRID}))

    # ---- per lesion
    smask = {pol: np.where(sr_iso, res["smax"][pol], -np.inf) for pol in LB.POLARITIES}
    cand_far = np.argwhere(far)
    lesions = []
    for j in range(1, n_les + 1):
        r = dict(case_id=case_id, lesion_id=j, gt_vox_iso=int((lab_iso == j).sum()),
                 in_sr_frac=in_sr_frac[j], vol_mm3=vol_mm3[j],
                 in_sr_frac_iso=float(sr_iso[lab_iso == j].mean()) if (lab_iso == j).any() else np.nan)
        dil = dil_id == j
        r_eq = (3.0 * vol_mm3[j] / (4.0 * np.pi)) ** (1.0 / 3.0)
        for pol in LB.POLARITIES:
            zs = res["probe"][f"les{j}"][pol]
            for s, v in zip(sigmas, zs):
                r[f"z_{pol}_s{s:g}"] = v
            # rank of the best peak of each scale's own list that lands in the 2 mm dilation
            for s in sigmas:
                hit = [p["scale_rank"] for p in res["scale_peaks"]
                       if p["polarity"] == pol and p["sigma_mm"] == s and dil_id[(p["ix"], p["iy"], p["iz"])] == j]
                r[f"rank_{pol}_s{s:g}"] = min(hit) if hit else np.nan
            hit = [p for p in res["peaks"] if p["polarity"] == pol and dil_id[(p["ix"], p["iy"], p["iz"])] == j]
            best = min(hit, key=lambda p: p["rank"]) if hit else None
            r[f"rank_{pol}"] = best["rank"] if best else np.nan
            r[f"z_{pol}"] = best["z"] if best else np.nan
            r[f"sigma_{pol}"] = best["sigma_mm"] if best else np.nan
            r[f"blobness_{pol}"] = best["blobness"] if best else np.nan
            r[f"n_peaks_{pol}"] = sum(1 for p in res["peaks"] if p["polarity"] == pol)
            # continuous max over the 2 mm dilation, region-restricted and not
            sm = smask[pol][dil]
            r[f"smax_dil2_{pol}"] = float(sm.max()) if sm.size and np.isfinite(sm).any() else np.nan
            r[f"smax_dil2_free_{pol}"] = float(res["smax"][pol][dil].max()) if dil.any() else np.nan
            # ... versus random lesion-sized patches elsewhere in this case's search region
            if len(cand_far) >= 10 and np.isfinite(r[f"smax_dil2_{pol}"]):
                take = rng.choice(len(cand_far), min(N_PATCH, len(cand_far)), replace=False)
                ps = patch_scores(smask[pol], cand_far[take], r_eq + DIL_MM)
                ps = ps[np.isfinite(ps)]
                r[f"patch_n_{pol}"] = int(ps.size)
                r[f"patch_frac_ge_{pol}"] = float((ps >= r[f"smax_dil2_{pol}"]).mean()) if ps.size else np.nan
                r[f"patch_med_{pol}"] = float(np.median(ps)) if ps.size else np.nan
                r[f"patch_p95_{pol}"] = float(np.percentile(ps, 95)) if ps.size else np.nan
            else:
                for k in ("patch_n", "patch_frac_ge", "patch_med", "patch_p95"):
                    r[f"{k}_{pol}"] = np.nan
        r["patch_radius_mm"] = r_eq + DIL_MM
        lesions.append(r)
    return dict(case=case, peaks=peaks, scales=scale_rows, lesions=lesions)


def process(job: tuple[str, int, bool]) -> tuple[str, str]:
    case_id, fold, do_struct = job
    out = CASE_DIR / f"{case_id}.pkl"
    if out.exists():
        return case_id, "cached"
    t0 = time.time()
    try:
        sp = tio.spacing(case_id)
        sr_full = envelope.search_region(case_id)
        if sp is None or sr_full is None:
            return case_id, "no spacing / search region"
        if not sr_full.any():
            rec = dict(case=dict(case_id=case_id, fold=fold, sr_ok=False), peaks=[], scales=[], lesions=[])
            CASE_DIR.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as f:
                pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)
            return case_id, "empty search region (recorded)"

        les_full = tio.load_mask(case_id, "pancreatic_lesion")
        if les_full is None:
            return case_id, "lesion mask failed"
        labels_full, n_les = label_lesions(les_full)
        del les_full
        vol_vox = float(np.prod(sp))
        vol_mm3, in_sr_frac = {}, {}
        for j in range(1, n_les + 1):
            sl = bbox_slices(labels_full == j)
            comp = labels_full[sl] == j
            vol_mm3[j] = float(comp.sum()) * vol_vox
            in_sr_frac[j] = float(sr_full[sl][comp].mean())

        pad = np.ceil(PAD_MM / np.asarray(sp, float)).astype(int)
        tight = bbox_slices(sr_full)                 # the region itself
        crop = bbox_slices(sr_full, pad)             # ... padded, clipped to the volume
        sr_crop, lab_crop = sr_full[crop].copy(), labels_full[crop].copy()
        # achieved margin between the region and the crop edge, per axis and side, in mm: this is
        # what says whether the requested PAD_MM (> 4*sigma_max) was available or clipped away
        margin_mm = [float(v) * sp[a] for a, (t_, c_) in enumerate(zip(tight, crop))
                     for v in (t_.start - c_.start, c_.stop - t_.stop)]
        del sr_full, labels_full

        ct_crop = tio.load_ct(case_id, crop)
        if ct_crop is None:
            return case_id, "CT failed"
        ct_iso, iso, _ = resample_ct_to_isotropic(ct_crop, sp, None)
        del ct_crop
        sr_iso = resample_ct_to_isotropic(sr_crop.astype(np.float32), sp, None)[0] > 0.5
        if not sr_iso.any():
            return case_id, "search region vanished on the 1 mm grid"
        lab_iso = nn_resample_labels(lab_crop, ct_iso.shape, sp)
        del sr_crop, lab_crop

        rng = np.random.default_rng(PATCH_SEED + int(case_id[-8:]))
        rec = analyze(case_id, fold, ct_iso, sr_iso, lab_iso, n_les, vol_mm3, in_sr_frac,
                      sp, [s.start for s in crop], rng)
        rec["case"]["sr_ok"] = True
        rec["case"]["crop_shape_native"] = tuple(int(s.stop - s.start) for s in crop)
        rec["case"]["crop_margin_mm"] = float(min(margin_mm))
        rec["case"]["crop_margin_mm_per_side"] = tuple(round(v, 2) for v in margin_mm)
        del ct_iso, lab_iso
        if do_struct and rec["peaks"]:
            pts = np.array([[p["ix"], p["iy"], p["iz"]] for p in rec["peaks"]], float)
            lab, dist, missing = nearest_structures(case_id, crop, sp, pts)
            for p, l, d in zip(rec["peaks"], lab, dist):
                p["struct"] = str(l)
                p["struct_mm"] = float(d)
            rec["case"]["missing_structures"] = ";".join(missing)
        rec["case"]["seconds"] = round(time.time() - t0, 1)
        CASE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.rename(out)
        return case_id, f"ok {rec['case']['seconds']}s"
    except Exception as e:
        traceback.print_exc()
        return case_id, f"error {type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", help="only these case ids")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--no-structures", action="store_true", help="skip the nearest-structure labels")
    a = ap.parse_args()
    fold = {c: k - 1 for k in range(1, 6) for c in tio.fold_ids(k)}
    ids = a.cases or sorted(fold)
    jobs = [(c, fold[c], not a.no_structures) for c in ids]
    print(f"{len(jobs)} cases, {min(a.workers, NUM_WORKERS)} workers -> {CASE_DIR}", flush=True)
    t0 = time.time()
    bad = []
    with mp.Pool(min(a.workers, NUM_WORKERS), maxtasksperchild=8) as pool:
        for i, (cid, status) in enumerate(pool.imap_unordered(process, jobs, chunksize=1), 1):
            if not (status.startswith("ok") or status == "cached"):
                bad.append((cid, status))
                print(f"  {cid}: {status}", flush=True)
            if i % 25 == 0 or i == len(jobs) or len(jobs) <= 5:
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}s) last {cid} {status}", flush=True)
    print(f"done in {time.time() - t0:.0f}s; {len(bad)} failures: {bad}")


if __name__ == "__main__":
    main()
