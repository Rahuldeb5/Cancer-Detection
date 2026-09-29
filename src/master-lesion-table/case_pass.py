"""Per-case mask pass for the master lesion table (session S1).

One pass over all 1308 cohort cases, straight from the PanTS masks, to get the
things no existing CSV stores consistently:
  - voxel spacing from nib.affines.voxel_sizes(CT affine) -- never header zooms or
    np.diag -- plus which voxel axis is the slice axis (the one most aligned with
    world superior-inferior), since the slice axis is not always the last one
  - an independent recount of every lesion component (26-connectivity, nibabel
    x,y,z order) so gt_vox can be checked against every upstream CSV
  - Feret diameter with that spacing, and head/body/tail region for ALL lesions
    (the existing region column only covers the 1142 non-excluded ones)

Region rule = duct_position.py's: which of head/body/tail holds most voxels of
(lesion + 3-voxel ring). The ring is what makes lesions carved out of the organ
masks assignable. "unassigned" if the ring touches none of the three.

Outputs (git-ignored work dir):
  work/master_lesion_table/case_spacing.csv   one row per case (1308)
  work/master_lesion_table/lesion_recount.csv one row per lesion component

Usage (from repo root):
  .venv/bin/python3 src/master-lesion-table/case_pass.py --phantom   # oracle only
  .venv/bin/python3 src/master-lesion-table/case_pass.py             # phantom, then real data
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, generate_binary_structure, label as cc_label
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

CT_ROOT = Path("/home/rahuldeb5/research/datasets/pants/ct_staging")
MASK_ROOT = Path("/home/rahuldeb5/research/datasets/pants/masks/mask_only")
FOLD_DIR = Path("src/data")
WORK_DIR = Path("work/master_lesion_table")
NUM_WORKERS = 4

CC_STRUCT = generate_binary_structure(3, 3)
SECTIONS = ("head", "body", "tail")
RING_ITERS = 3


def fold_of_case() -> dict[str, int]:
    """fold_k_ids.txt <-> nnU-Net fold k-1."""
    out: dict[str, int] = {}
    for k in range(1, 6):
        for line in (FOLD_DIR / f"fold_{k}_ids.txt").read_text().splitlines():
            if line.strip():
                out[line.strip()] = k - 1
    return out


def load_bool(path: Path) -> np.ndarray:
    raw = nib.load(path).dataobj.get_unscaled()  # int8 on disk; scl_slope is NaN in PanTS
    if np.issubdtype(raw.dtype, np.floating):
        raw = np.nan_to_num(raw, nan=0.0)
    return raw > 0.5


def spacing_info(affine: np.ndarray) -> tuple[np.ndarray, int]:
    """Voxel sizes (mm, voxel-axis order) and the index of the slice axis.

    Slice axis = voxel axis whose direction cosine has the largest world-z
    (superior-inferior) component, so it is found even when that axis is first.
    """
    vs = nib.affines.voxel_sizes(affine)
    cos_z = np.abs(affine[2, :3]) / vs
    return vs, int(np.argmax(cos_z))


def feret_mm(coords_vox: np.ndarray, spacing) -> float:
    """Max pairwise voxel-center distance in mm (same as main.py / join_lesion_attenuation.py)."""
    pts = coords_vox * np.asarray(spacing)
    if len(pts) < 2:
        return 0.0
    if len(pts) > 200:
        try:
            pts = pts[ConvexHull(pts).vertices]
        except Exception:  # noqa: BLE001 - coplanar / degenerate
            idx = np.random.default_rng(0).choice(len(pts), 200, replace=False)
            pts = pts[idx]
    return float(pdist(pts).max())


def lesion_rows(lesion: np.ndarray, sections: dict[str, np.ndarray], spacing) -> list[dict]:
    """One row per 26-connected lesion component. Pure array math (phantom-testable)."""
    labeled, n = cc_label(lesion, structure=CC_STRUCT)
    rows = []
    for k in range(1, n + 1):
        comp = labeled == k
        idx = np.argwhere(comp)
        lo = np.maximum(idx.min(axis=0) - (RING_ITERS + 1), 0)
        hi = np.minimum(idx.max(axis=0) + RING_ITERS + 2, lesion.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        grown = binary_dilation(comp[sl], structure=CC_STRUCT, iterations=RING_ITERS)
        votes = {s: int((grown & sections[s][sl]).sum()) for s in SECTIONS}
        region = max(votes, key=votes.get) if max(votes.values()) > 0 else "unassigned"
        rows.append({"lesion_id": k, "n_components": n, "gt_vox": len(idx),
                     "diam_mm": feret_mm(idx, spacing), "region": region,
                     **{f"ring_vox_{s}": votes[s] for s in SECTIONS}})
    return rows


def process_case(args: tuple[str, int]) -> tuple[dict, list[dict]]:
    case_id, fold = args
    ct_img = nib.load(CT_ROOT / case_id / "ct.nii.gz")  # header only; no voxel data read
    vs, slice_ax = spacing_info(ct_img.affine)
    hz = np.asarray(ct_img.header.get_zooms()[:3], dtype=float)
    seg = MASK_ROOT / case_id / "segmentations"
    les_img = nib.load(seg / "pancreatic_lesion.nii.gz")
    case = {"case_id": case_id, "fold": fold, "shape": str(tuple(ct_img.shape)),
            "spacing_x_mm": vs[0], "spacing_y_mm": vs[1], "spacing_z_mm": vs[2],
            "slice_axis": slice_ax, "slice_thickness_mm": vs[slice_ax],
            "header_zooms_match": bool(np.allclose(hz, vs, atol=1e-4)),
            "lesion_affine_matches_ct": bool(np.allclose(les_img.affine, ct_img.affine, atol=1e-3)),
            "lesion_shape_matches_ct": les_img.shape == ct_img.shape}
    lesion = load_bool(seg / "pancreatic_lesion.nii.gz")
    case["mask_nonempty"] = bool(lesion.any())
    if not case["mask_nonempty"]:
        return case, []
    sections = {s: load_bool(seg / f"pancreas_{s}.nii.gz") if (seg / f"pancreas_{s}.nii.gz").exists()
                else np.zeros_like(lesion) for s in SECTIONS}
    rows = lesion_rows(lesion, sections, vs)
    for r in rows:
        r["case_id"] = case_id
    return case, rows


def phantom() -> None:
    """Known-answer checks: component count/order, voxel counts, Feret, region ring vote,
    and slice-axis detection on an axis-permuted affine."""
    sp = (0.8, 0.8, 2.5)
    les = np.zeros((40, 40, 20), bool)
    les[5:10, 5:10, 2:4] = True           # 5x5x2 = 50 vox; first in raster order -> id 1
    les[30, 30, 15] = True                # single voxel -> id 2
    head = np.zeros_like(les); head[:20] = True
    tail = np.zeros_like(les); tail[20:] = True
    body = np.zeros_like(les)
    rows = lesion_rows(les, {"head": head, "body": body, "tail": tail}, sp)
    assert [r["gt_vox"] for r in rows] == [50, 1], rows
    exp = np.sqrt((4 * 0.8) ** 2 + (4 * 0.8) ** 2 + (1 * 2.5) ** 2)
    assert abs(rows[0]["diam_mm"] - exp) < 1e-9 and rows[1]["diam_mm"] == 0.0
    assert rows[0]["region"] == "head" and rows[1]["region"] == "tail"
    # lesion carved out of the body mask (0 voxels of overlap) is still assigned by its ring
    carved = np.zeros_like(les); carved[3:12, 3:12, 0:6] = True; carved &= ~les
    rc = lesion_rows(les, {"head": np.zeros_like(les), "body": carved, "tail": np.zeros_like(les)}, sp)
    assert rc[0]["region"] == "body" and rc[1]["region"] == "unassigned"
    # slice axis FIRST in voxel order (like some PanTS files): spacing (3.0, 0.7, 0.9)
    aff = np.array([[0, 0.7, 0, 0], [0, 0, -0.9, 0], [3.0, 0, 0, 0], [0, 0, 0, 1]], float)
    vs, ax = spacing_info(aff)
    assert np.allclose(vs, [3.0, 0.7, 0.9]) and ax == 0, (vs, ax)
    # oblique: slice axis tilted 20 deg still found
    t = np.deg2rad(20)
    aff2 = np.diag([0.7, 0.7, 5.0, 1.0]); aff2[:3, 2] = [0, 5 * np.sin(t), 5 * np.cos(t)]
    vs2, ax2 = spacing_info(aff2)
    assert np.allclose(vs2, [0.7, 0.7, 5.0]) and ax2 == 2
    print("phantom OK: gt_vox, Feret, ring-vote region, slice-axis detection")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phantom", action="store_true", help="run the oracle only")
    args = ap.parse_args()
    phantom()
    if args.phantom:
        return
    folds = fold_of_case()
    items = sorted(folds.items())
    print(f"{len(items)} cases, {NUM_WORKERS} workers", flush=True)
    with mp.Pool(NUM_WORKERS) as pool:
        results = pool.map(process_case, items, chunksize=8)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cases = pd.DataFrame([c for c, _ in results])
    lesions = pd.DataFrame([r for _, rs in results for r in rs])
    cases.to_csv(WORK_DIR / "case_spacing.csv", index=False)
    lesions.to_csv(WORK_DIR / "lesion_recount.csv", index=False)
    print(f"wrote {len(cases)} cases, {len(lesions)} lesion components to {WORK_DIR}")


if __name__ == "__main__":
    main()
