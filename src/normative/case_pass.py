"""PHASE 1 per-case pass: CT + search region -> cached patch feature tables (work/normative/).

One case per worker call, resumable (one gzip pickle per case), at most 4 workers, CPU only.
Volumes are never cached; only the per-patch tables are.

Per case:
  region    tumorlib envelope (pancreas | head | body | tail) -> fill_search_region (the same
            hole-filled, dilated region as tumorlib.search_region; never reads the lesion mask),
            bbox crop padded by PAD_MM, CT + region resampled to 1 mm iso (S4's recipe)
  features  features.patch_features for r = 6 and 10 mm at 4 mm lattice centres inside the region
  context   u (gland frame from pancreas.nii.gz PCA, head/tail sign fix; S1 method), distance to
            veins / duodenum / SMA / aorta (GT masks -> upper bound), phase, slice thickness
  eval      COHORT ONLY and only after the features exist: lesion labels (tumorlib.label_lesions)
            -> which lesion's 2 mm dilation holds each centre, distance from each centre to the
            nearest lesion, per-lesion volume / equivalent radius / search-region coverage / u.

Modes:
  bank     the 300 normal-bank cases (CTs in ct_normal_bank/)
  cohort   the 1308 evaluation cases
  insert   PHASE 3b: 50 cohort negatives x (sphere diameter x dHU, plus dHU = 0 controls), with
           S4's LoG pass (log-candidates/case_pass.analyze) run on the same planted CT

Usage (repo root):
  PYTHONPATH=src .venv/bin/python -m normative.case_pass bank   [--cases ...] [--workers 4]
  PYTHONPATH=src .venv/bin/python -m normative.case_pass cohort
  PYTHONPATH=src .venv/bin/python -m normative.case_pass insert
"""
from __future__ import annotations

import argparse
import gzip
import multiprocessing as mp
import pickle
import sys
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

from normative import common as C
from normative import features as FT
from tumorlib import envelope as tenv
from tumorlib import io as tio
from tumorlib.lesions import bbox_slices, label_lesions
from tumorlib.resample import resample_ct_to_isotropic

sys.path.insert(0, str(C.REPO / "src" / "log-candidates"))
import case_pass as S4  # noqa: E402  (S4's per-case analysis; imported, never modified)

DEFAULT_CT_ROOT = tio.CT_ROOT
OUT = {"bank": C.WORK / "features" / "bank", "cohort": C.WORK / "features" / "cohort",
       "insert": C.WORK / "features" / "insert"}
BANK_IDS = C.WORK / "pool" / "bank_ids.txt"
INSERT_IDS = C.WORK / "insert_case_ids.txt"
INSERT_DEPTH_MM = 2.0     # sphere centre at least this deep inside the envelope (fallback: anywhere in it)


def bank_ids() -> list[str]:
    return [l.strip() for l in BANK_IDS.read_text().splitlines() if l.strip()]


def use_ct_root(case_id: str, bank: set[str]) -> None:
    tio.CT_ROOT = C.BANK_CT_ROOT if case_id in bank else DEFAULT_CT_ROOT


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=3) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(path)


def load(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------ loading
def world_mm(native_idx: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return native_idx @ affine[:3, :3].T + affine[:3, 3]


def mask_points(m: np.ndarray) -> np.ndarray:
    sl = bbox_slices(m)
    if sl is None:
        return np.zeros((0, 3))
    return np.argwhere(m[sl]) + [s.start for s in sl]


def prepare(case_id: str) -> dict:
    """Everything up to (not including) the 1 mm resample. Never touches the lesion mask."""
    g = tio.grid(case_id)
    if g is None:
        raise RuntimeError("no CT grid")
    shape, affine, sp = g
    thick, slice_ax = C.slice_thickness(affine)
    masks = {}
    for name in tenv.ENVELOPE_MASKS:
        m = tio.load_mask(case_id, name)
        if m is None:
            raise RuntimeError(f"mask {name} failed")
        masks[name] = m
    env = masks["pancreas"].copy()
    for name in tenv.ENVELOPE_MASKS[1:]:
        np.bitwise_or(env, masks[name], out=env)
    rec = dict(case_id=case_id, sp=sp, affine=affine, shape=shape, thick=thick, slice_axis=slice_ax,
               env_vox=int(env.sum()))
    # gland frame (S1): pancreas.nii.gz point cloud, head -> tail sign; envelope if pancreas is empty
    pts = mask_points(masks["pancreas"])
    rec["frame_source"] = "pancreas"
    if len(pts) == 0:
        pts, rec["frame_source"] = mask_points(env), "envelope"
    hp, tp = mask_points(masks["pancreas_head"]), mask_points(masks["pancreas_tail"])
    if len(pts) >= 10 and len(hp) and len(tp):
        rec["frame"] = FT.gland_frame(world_mm(pts.astype(float), affine),
                                      world_mm(hp.astype(float), affine).mean(0),
                                      world_mm(tp.astype(float), affine).mean(0))
    else:
        rec["frame"], rec["frame_source"] = None, "none"
    del masks, pts, hp, tp
    if not env.any():
        rec["sr_ok"] = False
        return rec
    sr = tenv.fill_search_region(env, sp)
    pad = np.ceil(C.PAD_MM / np.asarray(sp, float)).astype(int)
    crop = bbox_slices(sr, pad)
    rec.update(sr_ok=True, crop=crop, crop_start=np.array([s.start for s in crop]),
               sr_crop=sr[crop].copy(), env_crop=env[crop].copy())
    del sr, env
    ct = tio.load_ct(case_id, crop)
    if ct is None:
        raise RuntimeError("CT failed")
    rec["ct_crop"] = ct
    return rec


def to_iso(rec: dict, ct_crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ct_iso, _, _ = resample_ct_to_isotropic(ct_crop, rec["sp"], None)
    sr_iso = resample_ct_to_isotropic(rec["sr_crop"].astype(np.float32), rec["sp"], None)[0] > 0.5
    return ct_iso, sr_iso


def iso_to_world(rec: dict, centres_iso: np.ndarray) -> np.ndarray:
    native = rec["crop_start"] + centres_iso * (C.ISO_MM / np.asarray(rec["sp"], float))
    return world_mm(native, rec["affine"])


def structure_distances(rec: dict, iso_shape, centres: np.ndarray) -> tuple[dict, list[str]]:
    """Distance (mm, capped at CONTEXT_MAX_MM) from each centre to each context structure, from the GT
    mask carried to the 1 mm grid by nearest neighbour (S4's nn_resample_labels)."""
    out, missing = {}, []
    for name in C.CONTEXT_STRUCTURES:
        m = tio.load_mask(rec["case_id"], name)
        if m is None:
            out[name] = np.full(len(centres), np.nan, np.float32)
            missing.append(name)
            continue
        sub = m[rec["crop"]]
        del m
        iso = S4.nn_resample_labels(sub, iso_shape, rec["sp"]).astype(bool)
        if not iso.any():
            out[name] = np.full(len(centres), C.CONTEXT_MAX_MM, np.float32)
            continue
        d = distance_transform_edt(~iso)
        out[name] = np.minimum(d[tuple(centres.T)], C.CONTEXT_MAX_MM).astype(np.float32)
    return out, missing


def feature_block(ct_iso: np.ndarray, centres: np.ndarray) -> dict:
    maps = FT.case_maps(ct_iso)
    return {r: FT.patch_features(maps, centres, r) for r in C.RADII_MM}


# ------------------------------------------------------------------ bank / cohort
def process(job: tuple[str, str]) -> tuple[str, str]:
    mode, case_id = job
    out = OUT[mode] / f"{case_id}.pkl.gz"
    if out.exists():
        return case_id, "cached"
    t0 = time.time()
    try:
        bank = set(bank_ids())
        if (mode == "bank") != (case_id in bank):
            raise RuntimeError("bank/cohort membership mismatch")
        use_ct_root(case_id, bank)
        with C.quiet() as log:
            rec = prepare(case_id)
        meta = {k: rec[k] for k in ("case_id", "sp", "thick", "slice_axis", "env_vox", "frame_source", "sr_ok")}
        meta["frame"] = rec["frame"]
        if not rec["sr_ok"]:
            save(out, dict(case=meta, log=log.getvalue()))
            return case_id, "empty envelope (recorded)"
        ct_iso, sr_iso = to_iso(rec, rec["ct_crop"])
        rec.pop("ct_crop")
        centres = FT.grid_centres(sr_iso)
        F = feature_block(ct_iso, centres)
        del ct_iso
        u = FT.u_of(iso_to_world(rec, centres.astype(float)), rec["frame"])
        dist, missing = structure_distances(rec, sr_iso.shape, centres)
        meta.update(crop_start=rec["crop_start"], iso_shape=sr_iso.shape, n_centres=len(centres),
                    sr_vol_cm3=float(sr_iso.sum()) / 1000.0, missing_structures=";".join(missing))
        res = dict(case=meta, centres=centres.astype(np.int16), F={r: v for r, v in F.items()},
                   u=u.astype(np.float32), dist=dist)
        if mode == "cohort":                         # evaluation bookkeeping, AFTER the features
            res.update(lesion_bookkeeping(rec, sr_iso, centres))
        res["log"] = log.getvalue()
        res["case"]["seconds"] = round(time.time() - t0, 1)
        save(out, res)
        return case_id, f"ok {res['case']['seconds']}s n={len(centres)}"
    except Exception as e:
        traceback.print_exc()
        return case_id, f"error {type(e).__name__}: {e}"


def lesion_bookkeeping(rec: dict, sr_iso: np.ndarray, centres: np.ndarray) -> dict:
    case_id, sp = rec["case_id"], rec["sp"]
    with C.quiet():
        les = tio.load_mask(case_id, "pancreatic_lesion")
    if les is None:
        raise RuntimeError("lesion mask failed")
    lab_full, n = label_lesions(les)
    del les
    vox_vol = float(np.prod(sp))
    lesions = []
    for j in range(1, n + 1):
        sl = bbox_slices(lab_full == j)
        comp = lab_full[sl] == j
        idx = np.argwhere(comp) + [s.start for s in sl]
        vol = float(comp.sum()) * vox_vol
        lesions.append(dict(case_id=case_id, lesion_id=j, gt_vox=int(comp.sum()), vol_mm3=vol,
                            r_eq_mm=(3.0 * vol / (4.0 * np.pi)) ** (1 / 3),
                            in_sr_frac=float(rec_sr_full_frac(rec, idx)),
                            u_centroid=float(FT.u_of(world_mm(idx.mean(0, keepdims=True), rec["affine"]),
                                                     rec["frame"])[0])))
    lab_iso = S4.nn_resample_labels(lab_full[rec["crop"]], sr_iso.shape, sp)
    del lab_full
    dil_id, overlap = S4.lesion_regions(lab_iso, n, C.DIL_MM)
    anyl = lab_iso > 0
    d = (distance_transform_edt(~anyl) if anyl.any() else np.full(lab_iso.shape, np.inf, np.float32))
    c = tuple(centres.T)
    for L in lesions:
        j = L["lesion_id"]
        L["gt_vox_iso"] = int((lab_iso == j).sum())
        L["n_centres_dil2"] = int((dil_id[c] == j).sum())
    far_vox = sr_iso & (d > C.FP_EXCL_MM)
    return dict(dil_id=dil_id[c].astype(np.uint16), dist_les=np.minimum(d[c], 999).astype(np.float32),
                lesions=lesions, sr_vol_far_cm3=float(far_vox.sum()) / 1000.0, dil_overlap=overlap,
                n_lesions=n)


def rec_sr_full_frac(rec: dict, idx_native: np.ndarray) -> float:
    """Fraction of a lesion's native voxels inside the search region (outside the crop -> outside)."""
    st = rec["crop_start"]
    rel = idx_native - st
    ok = np.all((rel >= 0) & (rel < np.asarray(rec["sr_crop"].shape)), axis=1)
    inside = np.zeros(len(idx_native), bool)
    inside[ok] = rec["sr_crop"][tuple(rel[ok].T)] > 0
    return inside.mean() if len(inside) else np.nan


# ------------------------------------------------------------------ insertion (PHASE 3b)
def insertion_conditions() -> list[tuple[float, float]]:
    conds = [(d, h) for d in C.INSERT_DIAMS_MM for h in C.INSERT_DHU]
    return conds + [(d, 0.0) for d in C.INSERT_DIAMS_MM]            # dHU = 0: null controls


def pick_location(rec: dict, rng) -> np.ndarray:
    """Random envelope voxel at least INSERT_DEPTH_MM inside the gland (crop-frame mm, sub-voxel jitter)."""
    env = rec["env_crop"].astype(bool)
    depth = distance_transform_edt(env, sampling=rec["sp"])
    cand = np.argwhere(depth >= INSERT_DEPTH_MM)
    if len(cand) == 0:
        cand = np.argwhere(env)
    v = cand[rng.integers(len(cand))]
    return (v + rng.uniform(-0.5, 0.5, 3)) * np.asarray(rec["sp"], float)


def process_insert(case_id: str) -> tuple[str, str]:
    out = OUT["insert"] / f"{case_id}.pkl.gz"
    if out.exists():
        return case_id, "cached"
    t0 = time.time()
    try:
        use_ct_root(case_id, set())
        with C.quiet() as log:
            rec = prepare(case_id)
        if not rec["sr_ok"]:
            return case_id, "no search region"
        rng = np.random.default_rng(C.SEED + int(case_id[-8:]))
        c_mm = pick_location(rec, rng)                      # one location, all conditions (paired)
        ct0 = rec.pop("ct_crop")
        runs = []
        for diam, dhu in insertion_conditions():
            R = diam / 2.0
            ct = ct0.copy()
            if dhu != 0.0:
                sl, blk = FT.sphere_profile(ct.shape, rec["sp"], c_mm, R, dhu)
                ct[sl] += blk
            ct_iso, sr_iso = to_iso(rec, ct)
            del ct
            centres = FT.grid_centres(sr_iso)
            F = feature_block(ct_iso, centres)
            dc = np.linalg.norm(centres - c_mm / C.ISO_MM, axis=1)
            # S4's LoG on the same planted CT, with the sphere as a one-lesion label map
            g = np.indices(sr_iso.shape, dtype=np.float32)
            dd = np.sqrt(sum((g[i] - c_mm[i]) ** 2 for i in range(3)))
            del g
            lab_iso = (dd <= R).astype(np.uint16)
            del dd
            in_sr = float(sr_iso[lab_iso > 0].mean()) if lab_iso.any() else np.nan
            s4rng = np.random.default_rng(C.SEED + int(case_id[-8:]) + int(diam * 10) + int(dhu + 100))
            s4 = S4.analyze(case_id, -1, ct_iso, sr_iso, lab_iso, 1, {1: 4 / 3 * np.pi * R ** 3},
                            {1: in_sr}, rec["sp"], rec["crop_start"], s4rng)
            L = s4["lesions"][0]
            runs.append(dict(diam=diam, dhu=dhu, centres=centres.astype(np.int16), F=F,
                             dist_centre=dc.astype(np.float32), in_sr_frac=in_sr,
                             s4={k: L.get(k) for k in ("patch_frac_ge_dark", "patch_frac_ge_bright",
                                                        "smax_dil2_dark", "smax_dil2_bright",
                                                        "rank_dark", "rank_bright", "patch_n_dark")}))
            del ct_iso, sr_iso, lab_iso
        u = FT.u_of(iso_to_world(rec, runs[0]["centres"].astype(float)), rec["frame"])
        meta = {k: rec[k] for k in ("case_id", "sp", "thick", "slice_axis", "frame_source")}
        meta.update(centre_mm=c_mm, crop_start=rec["crop_start"], seconds=round(time.time() - t0, 1))
        save(out, dict(case=meta, runs=runs, u=u.astype(np.float32), log=log.getvalue()))
        return case_id, f"ok {meta['seconds']}s"
    except Exception as e:
        traceback.print_exc()
        return case_id, f"error {type(e).__name__}: {e}"


def pick_insert_cases() -> list[str]:
    """N_INSERT_CASES cohort negatives with a search region (from the cohort pass), seed 42."""
    neg = []
    for c in tio.case_ids():
        rec = load(OUT["cohort"] / f"{c}.pkl.gz")
        if rec["case"].get("sr_ok") and rec.get("n_lesions", 1) == 0:
            neg.append(c)
    pick = sorted(np.random.default_rng(C.SEED).choice(sorted(neg), C.N_INSERT_CASES, replace=False))
    INSERT_IDS.write_text("\n".join(pick) + "\n")
    print(f"insertion cases: {len(pick)} of {len(neg)} cohort negatives with a search region")
    return pick


def run_insert_job(job):
    return process_insert(job[1])


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bank", "cohort", "insert"])
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    a = ap.parse_args()
    if a.cases:
        ids = a.cases
    elif a.mode == "bank":
        ids = bank_ids()
    elif a.mode == "cohort":
        ids = tio.case_ids()
    else:
        if not INSERT_IDS.exists():
            pick_insert_cases()
        ids = [l.strip() for l in INSERT_IDS.read_text().splitlines() if l.strip()]
    assert a.mode == "bank" or not set(ids) & set(bank_ids()), "a bank case in an evaluation run"
    assert a.mode != "bank" or not set(ids) & set(tio.case_ids()), "a cohort case in the bank"
    fn = run_insert_job if a.mode == "insert" else process
    jobs = [(a.mode, c) for c in ids]
    w = min(a.workers, C.NUM_WORKERS)
    print(f"{a.mode}: {len(jobs)} cases, {w} workers -> {OUT[a.mode]}", flush=True)
    t0, bad = time.time(), []
    with mp.Pool(w, maxtasksperchild=4) as pool:
        for i, (cid, status) in enumerate(pool.imap_unordered(fn, jobs, chunksize=1), 1):
            if not (status.startswith("ok") or status == "cached"):
                bad.append((cid, status))
                print(f"  {cid}: {status}", flush=True)
            if i % 20 == 0 or i == len(jobs) or len(jobs) <= 8:
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}s) last {cid} {status}", flush=True)
    print(f"done in {time.time() - t0:.0f}s; {len(bad)} not ok: {bad}")


if __name__ == "__main__":
    main()
