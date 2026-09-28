"""Reorient each patient's pancreas onto a common PCA frame and project every
lesion's centroid into it, so lesion position can be pooled across patients
despite hugely varying gland size/shape/scanner orientation.

Frame construction, per case:
  - PCA of the pancreas point cloud in mm (world/RAS+) space, not voxel-index
    space -- voxel spacing is anisotropic (e.g. 0.8mm in-plane, 7.5mm through
    plane in some PanTS cases), so index-space PCA would just measure that
    anisotropy, not gland shape. All voxel->mm conversion goes through
    pancreas.nii.gz's own affine (never lesion's -- see loader.py/main.py's
    notes elsewhere in this repo about occasional corrupted lesion affines);
    body/head/tail/lesion share the pancreas mask's voxel grid, so applying
    its affine to their voxel indices is valid.
  - PC1 (largest-variance axis) is sign-fixed to point from the pancreas_head
    centroid to the pancreas_tail centroid -- i.e. PC1=0 end is the head side.
  - PC2 is sign-fixed against nibabel's world +y (Anterior in RAS+), the one
    direction that is guaranteed consistent across cases regardless of each
    file's own voxel axis order, since apply_affine always lands in RAS+ world
    space. PC3 = PC1 x PC2 to keep a right-handed frame (its own sign is
    otherwise unconstrained/uninterpreted here).
  - Each axis is min-max normalized using the 1st/99th percentile of the
    PANCREAS's own projected extent (not the lesion's), so "0.0..1.0 along
    PC1" means "head end..tail end of THIS gland" for every patient.

Usage (from repo root): .venv/bin/python3 src/lesion-sectioning/pca_hitrate.py
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure, label as cc_label

MASK_ROOT = Path("/home/rahuldeb5/research/datasets/pants/masks/mask_only")
FOLD_DIR = Path("src/data")
RESULTS_DIR = Path("src/lesion-sectioning/work")
NUM_WORKERS = 4

CC_STRUCT = generate_binary_structure(3, 3)
PCTILE_CLIP = (1.0, 99.0)  # avoids a stray voxel/segmentation speckle stretching the axis
WORLD_ANTERIOR = np.array([0.0, 1.0, 0.0])  # nibabel world space is always RAS+: +y = Anterior


def case_ids() -> list[str]:
    ids: set[str] = set()
    for i in range(1, 6):
        for line in (FOLD_DIR / f"fold_{i}_ids.txt").read_text().splitlines():
            line = line.strip()
            if line:
                ids.add(line)
    return sorted(ids)


def load_bool(path: Path) -> np.ndarray:
    img = nib.load(path)
    raw = np.nan_to_num(np.asanyarray(img.dataobj), nan=0.0)
    return raw > 0.5


def voxel_to_mm(coords_vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """(n,3) voxel indices -> (n,3) mm world coordinates (RAS+)."""
    homog = np.hstack([coords_vox, np.ones((coords_vox.shape[0], 1))])
    return homog @ affine.T[:, :3]


def build_frame(panc_mm: np.ndarray, head_centroid_mm: np.ndarray, tail_centroid_mm: np.ndarray):
    centroid = panc_mm.mean(axis=0)
    centered = panc_mm - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
    order = np.argsort(eigvals)[::-1]
    pcs = eigvecs[:, order]

    pc1, pc2 = pcs[:, 0].copy(), pcs[:, 1].copy()
    if np.dot(pc1, tail_centroid_mm - head_centroid_mm) < 0:
        pc1 = -pc1
    if np.dot(pc2, WORLD_ANTERIOR) < 0:
        pc2 = -pc2
    pc3 = np.cross(pc1, pc2)

    R = np.stack([pc1, pc2, pc3], axis=1)
    return centroid, R


def process_case(case_id: str) -> list[dict]:
    seg_dir = MASK_ROOT / case_id / "segmentations"
    paths = {
        "pancreas": seg_dir / "pancreas.nii.gz",
        "body": seg_dir / "pancreas_body.nii.gz",
        "head": seg_dir / "pancreas_head.nii.gz",
        "tail": seg_dir / "pancreas_tail.nii.gz",
        "lesion": seg_dir / "pancreatic_lesion.nii.gz",
    }
    for p in paths.values():
        if not p.exists():
            return []

    pancreas_img = nib.load(paths["pancreas"])
    affine = pancreas_img.affine
    masks = {name: load_bool(p) for name, p in paths.items()}
    ref_shape = masks["pancreas"].shape
    if any(m.shape != ref_shape for m in masks.values()):
        print(f"{case_id}: shape mismatch, skipping")
        return []

    if not (masks["pancreas"].any() and masks["head"].any() and masks["tail"].any() and masks["lesion"].any()):
        return []  # need head+tail centroids to fix PC1, and at least one lesion to report

    panc_mm = voxel_to_mm(np.argwhere(masks["pancreas"]), affine)
    head_mm = voxel_to_mm(np.argwhere(masks["head"]), affine).mean(axis=0)
    tail_mm = voxel_to_mm(np.argwhere(masks["tail"]), affine).mean(axis=0)

    centroid, R = build_frame(panc_mm, head_mm, tail_mm)
    proj_panc = (panc_mm - centroid) @ R
    p1_lo, p1_hi = np.percentile(proj_panc[:, 0], PCTILE_CLIP)
    p2_lo, p2_hi = np.percentile(proj_panc[:, 1], PCTILE_CLIP)
    p3_lo, p3_hi = np.percentile(proj_panc[:, 2], PCTILE_CLIP)

    labeled, n = cc_label(masks["lesion"], structure=CC_STRUCT)
    rows = []
    for k in range(1, n + 1):
        comp = labeled == k
        comp_mm = voxel_to_mm(np.argwhere(comp), affine)
        centroid_l_mm = comp_mm.mean(axis=0)
        proj_l = (centroid_l_mm - centroid) @ R
        rows.append({
            "case_id": case_id,
            "lesion_id": k,
            "n_components": n,
            "pc1_norm": (proj_l[0] - p1_lo) / (p1_hi - p1_lo),  # 0=head end, 1=tail end
            "pc2_norm": (proj_l[1] - p2_lo) / (p2_hi - p2_lo),  # 0=posterior-most, 1=anterior-most
            "pc3_norm": (proj_l[2] - p3_lo) / (p3_hi - p3_lo),  # gland "thickness" axis
            "pancreas_length_mm": float(p1_hi - p1_lo),
        })
    return rows


def main() -> None:
    ids = case_ids()
    print(f"{len(ids)} case ids to load, {NUM_WORKERS} workers")

    with mp.Pool(NUM_WORKERS) as pool:
        results = pool.map(process_case, ids)

    rows = [row for rows in results for row in rows]
    print(f"{len(rows)} lesion rows total")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "lesion_pca_position.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
