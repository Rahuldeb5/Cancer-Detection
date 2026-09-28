"""Gland-aligned coordinates for every non-excluded lesion and for a tissue sample of the
pancreas envelope (pancreas | head | body | tail), so lesions can be drawn on a
pancreas-shaped map pooled across patients.

Frame per case (all in world/RAS+ mm, then divided by the gland length so glands of
different sizes overlay):
  u      position along the gland, 0 = head end .. 1 = tail end (identical to pc1_norm in
         pca_hitrate.py; PC1 sign-fixed head->tail from the head/tail mask centroids)
  ap     offset from the gland centroid along the anterior direction perpendicular to PC1
         (world +y made orthogonal to PC1)          -> "axial" map: u vs ap
  si     offset along the superior direction perpendicular to PC1 and to that ap axis
         (world +z made orthogonal)                 -> "coronal" map: u vs si
ap/si are in units of gland length (1 unit = the 1st-99th percentile PC1 extent, ~12 cm). Anatomical axes are used instead of PCA axes 2/3 because those
are not consistently anatomical across patients (PC2 is superior-inferior in ~65% of cases).

Tissue sample: N_TISSUE random envelope voxels per case, equal weight per case, tagged with
the section (head/body/tail/none) of the voxel. Lesions in excluded_lesions.csv are skipped.

Usage (from repo root): .venv/bin/python3 src/lesion-sectioning/pancreas_map.py
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label

from pca_hitrate import CC_STRUCT, MASK_ROOT, PCTILE_CLIP, build_frame, case_ids, load_bool, voxel_to_mm

WORK_DIR = Path("src/lesion-sectioning/work")
EXCLUDED_CSV = Path("results/lesion_location_results/excluded_lesions.csv")
NUM_WORKERS = 4
N_TISSUE = 1500
SECTION_CODE = {"head": 1, "body": 2, "tail": 3}

_ex = pd.read_csv(EXCLUDED_CSV)
EXCLUDED = set(zip(_ex["case_id"], _ex["lesion_id"]))
WORLD_Y = np.array([0.0, 1.0, 0.0])
WORLD_Z = np.array([0.0, 0.0, 1.0])


def anatomical_axes(a1: np.ndarray):
    ap = WORLD_Y - (WORLD_Y @ a1) * a1
    ap /= np.linalg.norm(ap)
    si = WORLD_Z - (WORLD_Z @ a1) * a1 - (WORLD_Z @ ap) * ap
    si /= np.linalg.norm(si)
    return ap, si


def process_case(case_id: str):
    seg = MASK_ROOT / case_id / "segmentations"
    names = {"pancreas": "pancreas", "head": "pancreas_head", "body": "pancreas_body", "tail": "pancreas_tail"}
    if not (seg / "pancreatic_lesion.nii.gz").exists() or any(not (seg / f"{f}.nii.gz").exists() for f in names.values()):
        return [], None
    lesion = load_bool(seg / "pancreatic_lesion.nii.gz")
    if not lesion.any():
        return [], None
    M = {k: load_bool(seg / f"{f}.nii.gz") for k, f in names.items()}
    if any(m.shape != lesion.shape for m in M.values()):
        print(f"{case_id}: shape mismatch, skipping")
        return [], None
    if not (M["pancreas"].any() and M["head"].any() and M["tail"].any()):
        return [], None  # PC1 needs head+tail centroids (same limitation as pca_hitrate.py)

    aff = nib.load(seg / "pancreas.nii.gz").affine
    panc_mm = voxel_to_mm(np.argwhere(M["pancreas"]), aff)
    head_c = voxel_to_mm(np.argwhere(M["head"]), aff).mean(axis=0)
    tail_c = voxel_to_mm(np.argwhere(M["tail"]), aff).mean(axis=0)
    cen, R = build_frame(panc_mm, head_c, tail_c)
    a1 = R[:, 0]
    ap, si = anatomical_axes(a1)
    p1_lo, p1_hi = np.percentile((panc_mm - cen) @ a1, PCTILE_CLIP)
    L = p1_hi - p1_lo

    def coords(mm: np.ndarray) -> np.ndarray:
        rel = mm - cen
        return np.stack([((rel @ a1) - p1_lo) / L, (rel @ ap) / L, (rel @ si) / L], axis=1)

    labeled, n = cc_label(lesion, structure=CC_STRUCT)
    rows = []
    for k in range(1, n + 1):
        if (case_id, k) in EXCLUDED:
            continue
        c = coords(voxel_to_mm(np.argwhere(labeled == k), aff).mean(axis=0, keepdims=True))[0]
        rows.append({"case_id": case_id, "lesion_id": k, "u": c[0], "ap": c[1], "si": c[2], "gland_length_mm": L})

    env = M["pancreas"] | M["head"] | M["body"] | M["tail"]
    ev = np.argwhere(env)
    rng = np.random.default_rng(int(case_id.split("_")[-1]))
    tv = ev[rng.choice(len(ev), min(N_TISSUE, len(ev)), replace=False)]
    tc = coords(voxel_to_mm(tv, aff)).astype(np.float32)
    sec = np.zeros(len(tv), dtype=np.uint8)
    for s in ("tail", "body", "head"):  # masks are disjoint; order is irrelevant but explicit
        sec[M[s][tuple(tv.T)]] = SECTION_CODE[s]
    return rows, {"u": tc[:, 0], "ap": tc[:, 1], "si": tc[:, 2], "section": sec}


def main() -> None:
    ids = case_ids()
    print(f"{len(ids)} case ids, {NUM_WORKERS} workers, {len(EXCLUDED)} excluded lesions skipped")
    with mp.Pool(NUM_WORKERS) as pool:
        results = pool.map(process_case, ids)
    rows = [r for rs, _ in results for r in rs]
    tissue = [t for _, t in results if t is not None]

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(WORK_DIR / "pancreas_map_lesions.csv", index=False)
    np.savez_compressed(WORK_DIR / "pancreas_map_tissue.npz", **{k: np.concatenate([t[k] for t in tissue]) for k in tissue[0]})
    print(f"{len(rows)} lesions; tissue sample {sum(len(t['u']) for t in tissue)} voxels from {len(tissue)} cases")


if __name__ == "__main__":
    main()
