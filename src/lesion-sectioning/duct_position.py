"""Where each (non-excluded) lesion sits relative to the main pancreatic duct and
inside its head/body/tail section, plus a matching sample of pancreas tissue so
lesion positions can be judged against where gland tissue actually is.

Per lesion (excluded_lesions.csv rows are skipped):
  section               head/body/tail = which mask holds most of (lesion + 3-voxel ring)
                        -- the ring makes lesions carved out of the organ masks assignable
  pc1_norm              0 = head end .. 1 = tail end of that gland (same frame as pca_hitrate.py)
  pos_in_section        0 = proximal end .. 1 = distal end of the lesion's own section
                        (section extent = 5th-95th percentile of its mask along PC1)
  dist_centroid_to_duct_mm / dist_surface_to_duct_mm   nearest duct voxel, 3-D, mm
  offset_{rl,ap,si}_mm  lesion centroid minus the duct centerline at the lesion's PC1
                        position, in WORLD axes (RAS+: ap>0 anterior, si>0 superior)
  duct_present          case has any duct voxels
  duct_valid            a duct segment lies within 10 mm (along the gland axis) of the lesion's PC1 position

Why "valid": the duct mask is often fragmentary (median coverage ~half the gland), so a
distance/offset to the duct is only meaningful where the duct actually runs alongside the
lesion (within 10 mm along the gland axis). Everything duct-related is computed only under that rule, for lesions AND for the
tissue sample.

World axes are used for up/down because PCA axes 2/3 are not consistently anatomical
(PC2 is superior-inferior in ~65% of cases, anterior-posterior in ~35%).

Tissue sample: N_TISSUE random voxels per case from the mask envelope (pancreas | head |
body | tail -- pancreas.nii.gz alone has lesions carved out), equal weight per case.

Usage (from repo root): .venv/bin/python3 src/lesion-sectioning/duct_position.py
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, label as cc_label
from scipy.spatial import cKDTree

from pca_hitrate import CC_STRUCT, MASK_ROOT, PCTILE_CLIP, build_frame, case_ids, load_bool, voxel_to_mm

WORK_DIR = Path("src/lesion-sectioning/work")
EXCLUDED_CSV = Path("results/lesion_location_results/excluded_lesions.csv")
NUM_WORKERS = 4
N_TISSUE = 1500
CENTERLINE_BIN_MM = 3.0
MAX_GAP_MM = 10.0
SECTIONS = ("head", "body", "tail")
SECTION_CODE = {"none": 0, "head": 1, "body": 2, "tail": 3}

_ex = pd.read_csv(EXCLUDED_CSV)
EXCLUDED = set(zip(_ex["case_id"], _ex["lesion_id"]))


def process_case(case_id: str):
    seg = MASK_ROOT / case_id / "segmentations"
    paths = {"pancreas": "pancreas", "head": "pancreas_head", "body": "pancreas_body",
             "tail": "pancreas_tail", "lesion": "pancreatic_lesion"}
    for f in paths.values():
        if not (seg / f"{f}.nii.gz").exists():
            return [], None, None
    lesion = load_bool(seg / "pancreatic_lesion.nii.gz")
    if not lesion.any():
        return [], None, None

    M = {k: load_bool(seg / f"{f}.nii.gz") for k, f in paths.items() if k != "lesion"}
    M["lesion"] = lesion
    duct_path = seg / "pancreatic_duct.nii.gz"
    duct = load_bool(duct_path) if duct_path.exists() else np.zeros_like(lesion)
    if any(m.shape != lesion.shape for m in (*M.values(), duct)):
        print(f"{case_id}: shape mismatch, skipping")
        return [], None, None
    if not (M["pancreas"].any() and M["head"].any() and M["tail"].any()):
        return [], None, None  # PC1 needs head+tail centroids (same limitation as pca_hitrate.py)

    aff = nib.load(seg / "pancreas.nii.gz").affine
    panc_mm = voxel_to_mm(np.argwhere(M["pancreas"]), aff)
    head_c = voxel_to_mm(np.argwhere(M["head"]), aff).mean(axis=0)
    tail_c = voxel_to_mm(np.argwhere(M["tail"]), aff).mean(axis=0)
    cen, R = build_frame(panc_mm, head_c, tail_c)
    axis1 = R[:, 0]
    p1 = (panc_mm - cen) @ axis1
    p1_lo, p1_hi = np.percentile(p1, PCTILE_CLIP)

    def to_norm(pc1_mm):
        return (pc1_mm - p1_lo) / (p1_hi - p1_lo)

    ext = {}
    for s in SECTIONS:
        v = to_norm((voxel_to_mm(np.argwhere(M[s]), aff) - cen) @ axis1)
        if len(v) >= 20:
            ext[s] = tuple(np.percentile(v, [5, 95]))

    duct_present = bool(duct.any())
    usable = False
    if duct_present:
        dmm = voxel_to_mm(np.argwhere(duct), aff)
        tree = cKDTree(dmm)
        d_pc1 = (dmm - cen) @ axis1
        bins = np.floor((d_pc1 - d_pc1.min()) / CENTERLINE_BIN_MM).astype(int)
        ub = np.unique(bins)
        c_pc1 = np.array([d_pc1[bins == b].mean() for b in ub])
        c_xyz = np.array([dmm[bins == b].mean(axis=0) for b in ub])
        usable = len(ub) >= 2

    def centerline(pc1_mm: np.ndarray):
        xyz = np.stack([np.interp(pc1_mm, c_pc1, c_xyz[:, k]) for k in range(3)], axis=1)
        i = np.searchsorted(c_pc1, pc1_mm).clip(1, len(c_pc1) - 1)
        gap = np.minimum(np.abs(pc1_mm - c_pc1[i - 1]), np.abs(pc1_mm - c_pc1[i]))
        return xyz, gap <= MAX_GAP_MM  # np.interp holds the end value flat, so <=10 mm past an end is tolerated

    labeled, n = cc_label(lesion, structure=CC_STRUCT)
    rows = []
    for k in range(1, n + 1):
        if (case_id, k) in EXCLUDED:
            continue
        comp = labeled == k
        idx = np.argwhere(comp)
        mm = voxel_to_mm(idx, aff)
        c_l = mm.mean(axis=0)
        pc1_mm = float((c_l - cen) @ axis1)
        pc1n = float(to_norm(pc1_mm))

        lo = np.maximum(idx.min(axis=0) - 4, 0)
        hi = np.minimum(idx.max(axis=0) + 5, lesion.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        grown = binary_dilation(comp[sl], structure=CC_STRUCT, iterations=3)
        votes = {s: int((grown & M[s][sl]).sum()) for s in SECTIONS}
        section = max(votes, key=votes.get) if max(votes.values()) > 0 else "unassigned"
        pos = np.nan
        if section in ext:
            a, b = ext[section]
            pos = (pc1n - a) / (b - a)

        row = {"case_id": case_id, "lesion_id": k, "n_vox": len(idx), "section": section,
               "pc1_norm": pc1n, "pos_in_section": pos, "duct_present": duct_present,
               "dist_centroid_to_duct_mm": np.nan, "dist_surface_to_duct_mm": np.nan,
               "duct_valid": False, "offset_rl_mm": np.nan, "offset_ap_mm": np.nan, "offset_si_mm": np.nan}
        if duct_present:
            row["dist_centroid_to_duct_mm"] = float(tree.query(c_l)[0])
            row["dist_surface_to_duct_mm"] = float(tree.query(mm)[0].min())
            if usable:
                cxyz, ok = centerline(np.array([pc1_mm]))
                if ok[0]:
                    off = c_l - cxyz[0]
                    row.update(duct_valid=True, offset_rl_mm=off[0], offset_ap_mm=off[1], offset_si_mm=off[2])
        rows.append(row)

    tissue = bounds = None
    if usable and all(s in ext for s in SECTIONS):
        env = M["pancreas"] | M["head"] | M["body"] | M["tail"]
        ev = np.argwhere(env)
        rng = np.random.default_rng(int(case_id.split("_")[-1]))
        tv = ev[rng.choice(len(ev), min(N_TISSUE, len(ev)), replace=False)]
        tmm = voxel_to_mm(tv, aff)
        t_p1 = (tmm - cen) @ axis1
        cxyz, ok = centerline(t_p1)
        sec = np.zeros(len(tv), dtype=np.uint8)
        for s in SECTIONS[::-1]:
            sec[M[s][tuple(tv.T)]] = SECTION_CODE[s]
        t_n = to_norm(t_p1)
        t_pos = np.full(len(tv), np.nan)
        for s in SECTIONS:
            m = sec == SECTION_CODE[s]
            a, b = ext[s]
            t_pos[m] = (t_n[m] - a) / (b - a)
        off = tmm - cxyz
        tissue = {"pc1_norm": t_n[ok].astype(np.float32), "pos_in_section": t_pos[ok].astype(np.float32),
                  "dist_mm": tree.query(tmm)[0][ok].astype(np.float32),
                  "offset_ap_mm": off[ok, 1].astype(np.float32), "offset_si_mm": off[ok, 2].astype(np.float32),
                  "section": sec[ok]}
        bounds = {"case_id": case_id,
                  "boundary_head_body": 0.5 * (ext["head"][1] + ext["body"][0]),
                  "boundary_body_tail": 0.5 * (ext["body"][1] + ext["tail"][0])}
    return rows, tissue, bounds


def main() -> None:
    ids = case_ids()
    print(f"{len(ids)} case ids, {NUM_WORKERS} workers, {len(EXCLUDED)} excluded lesions skipped")
    with mp.Pool(NUM_WORKERS) as pool:
        results = pool.map(process_case, ids)

    rows = [r for rs, _, _ in results for r in rs]
    df = pd.DataFrame(rows)
    tissue = [t for _, t, _ in results if t is not None]
    bounds = pd.DataFrame([b for _, _, b in results if b is not None])

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(WORK_DIR / "lesion_duct_position.csv", index=False)
    bounds.to_csv(WORK_DIR / "section_boundaries.csv", index=False)
    np.savez_compressed(WORK_DIR / "tissue_samples.npz",
                        **{k: np.concatenate([t[k] for t in tissue]) for k in tissue[0]})
    print(f"{len(df)} lesions; duct present {int(df.duct_present.sum())}; duct valid {int(df.duct_valid.sum())}; "
          f"sections {df.section.value_counts().to_dict()}")
    print(f"tissue sample from {len(tissue)} cases, {sum(len(t['pc1_norm']) for t in tissue)} voxels")


if __name__ == "__main__":
    main()
