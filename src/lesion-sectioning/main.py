"""Per-lesion-component anatomical location within the pancreas.

For every lesion connected component, computes how much of its volume falls
inside pancreas_body.nii.gz / pancreas_head.nii.gz / pancreas_tail.nii.gz.
If none of the three has meaningful overlap, falls back to pancreas.nii.gz
(the subregion masks are usually an exact voxel-for-voxel partition of
pancreas.nii.gz, but not always -- see PanTS_00000099, where pancreas.nii.gz
is ~87k voxels bigger than body+head+tail combined -- so a lesion can land
in "pancreas but no subregion" or, if pancreas.nii.gz itself misses it too,
"extrapancreatic").

Connected-component numbering mirrors src/nnunet/join_lesion_attenuation.py's
GT labeling exactly (np.asanyarray bool load, no crop, 26-connectivity, raster
order) so lesion_id here joins directly onto lesion_dice_attenuation.csv.

Usage (from repo root): .venv/bin/python3 src/lesion-sectioning/main.py
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label as cc_label
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

MASK_ROOT = Path("/home/rahuldeb5/research/datasets/pants/masks/mask_only")
FOLD_DIR = Path("src/data")
RESULTS_DIR = Path("src/lesion-sectioning/work")
NUM_WORKERS = 4

CC_STRUCT = generate_binary_structure(3, 3)  # full 26-connectivity, axis-order agnostic

DIAM_EDGES_MM = (0.0, 5.0, 10.0, 15.0, 20.0, 40.0, np.inf)
DIAM_LABELS = ["<5mm", "5-10mm", "10-15mm", "15-20mm", "20-40mm", "40mm+"]

MAJORITY_FRAC = 0.5   # primary region must hold at least this much of the lesion
DOMINANT_FRAC = 0.9   # at/above this, call it "complete" rather than "mostly"
MEANINGFUL_FRAC = 0.1  # below this a region's overlap is noise, not a real span


def case_ids() -> list[str]:
    ids: set[str] = set()
    for i in range(1, 6):
        for line in (FOLD_DIR / f"fold_{i}_ids.txt").read_text().splitlines():
            line = line.strip()
            if line:
                ids.add(line)
    return sorted(ids)


def load_bool(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(path)
    raw = np.nan_to_num(np.asanyarray(img.dataobj), nan=0.0)
    return raw > 0.5, img.header.get_zooms()[:3]


def feret_mm(coords_vox: np.ndarray, spacing) -> float:
    """Max pairwise voxel-center distance in mm -- the lesion's long-axis diameter."""
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


def diam_bin(diam_mm: float) -> str:
    idx = int(np.digitize([diam_mm], DIAM_EDGES_MM[1:-1])[0])
    return DIAM_LABELS[idx]


def classify_location(frac_body: float, frac_head: float, frac_tail: float, frac_panc_general: float):
    """Two-tier check, mirroring what was asked: is the lesion mostly/completely in one
    of body/head/tail? If not, is it at least mostly/completely in pancreas.nii.gz overall
    (a real subregion-boundary lesion, or a case where body+head+tail under-covers
    pancreas.nii.gz -- see PanTS_00000099)? If not even that, it's extrapancreatic.

    Returns (location, certainty).
    """
    fracs = {"body": frac_body, "head": frac_head, "tail": frac_tail}
    primary = max(fracs, key=fracs.get)
    primary_frac = fracs[primary]

    if primary_frac >= DOMINANT_FRAC:
        return primary, "complete"
    if primary_frac >= MAJORITY_FRAC:
        return primary, "mostly"

    if frac_panc_general >= DOMINANT_FRAC:
        return "pancreas_junction", "complete_no_subregion_majority"
    if frac_panc_general >= MAJORITY_FRAC:
        return "pancreas_junction", "mostly_no_subregion_majority"
    return "extrapancreatic", "not_mostly_in_pancreas"


def regions_touched(frac_body: float, frac_head: float, frac_tail: float, frac_panc_general: float) -> str:
    """Human-readable breakdown of every region holding >= MEANINGFUL_FRAC of the lesion,
    informational only -- does not drive `location`/`certainty`."""
    parts = [(r, f) for r, f in (("head", frac_head), ("body", frac_body), ("tail", frac_tail))
             if f >= MEANINGFUL_FRAC]
    frac_extrapancreatic = 1.0 - frac_panc_general
    if frac_extrapancreatic >= MEANINGFUL_FRAC:
        parts.append(("extrapancreatic", frac_extrapancreatic))
    parts.sort(key=lambda t: -t[1])
    return "+".join(f"{r}:{f:.2f}" for r, f in parts) if parts else "none"


def load_case(case_id: str) -> dict | None:
    seg_dir = MASK_ROOT / case_id / "segmentations"
    paths = {
        "lesion": seg_dir / "pancreatic_lesion.nii.gz",
        "pancreas": seg_dir / "pancreas.nii.gz",
        "body": seg_dir / "pancreas_body.nii.gz",
        "head": seg_dir / "pancreas_head.nii.gz",
        "tail": seg_dir / "pancreas_tail.nii.gz",
    }
    for p in paths.values():
        if not p.exists():
            print(f"{case_id}: missing {p}")
            return None

    masks: dict[str, np.ndarray] = {}
    ref_shape = None
    zooms = None
    for name, p in paths.items():
        m, z = load_bool(p)
        if ref_shape is None:
            ref_shape, zooms = m.shape, z
        elif m.shape != ref_shape:
            print(f"{case_id}: shape mismatch on {name}: {m.shape} vs {ref_shape}")
            return None
        masks[name] = m

    return {"case_id": case_id, "zooms": zooms, **masks}


def process_case(case_id: str) -> list[dict]:
    case = load_case(case_id)
    if case is None:
        return []
    lesion = case["lesion"]
    if not lesion.any():
        return []

    zooms = case["zooms"]
    voxvol = float(np.prod(zooms))
    labeled, n = cc_label(lesion, structure=CC_STRUCT)

    # Crop to lesion|pancreas bbox before the EDT -- scipy's exact distance
    # transform allocates scratch scaling with the FULL input volume, and this
    # box always fully contains every lesion voxel, so cropping only changes
    # cost, not the nearest-pancreas-voxel distance for any lesion voxel in it.
    roi = lesion | case["pancreas"]
    coords_roi = np.argwhere(roi)
    lo, hi = coords_roi.min(axis=0), coords_roi.max(axis=0) + 1
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    panc_crop = case["pancreas"][sl]
    dist_from_panc = distance_transform_edt(~panc_crop, sampling=zooms) if panc_crop.any() else None

    rows = []
    for k in range(1, n + 1):
        comp = labeled == k
        n_vox = int(comp.sum())
        coords = np.argwhere(comp)
        diam = feret_mm(coords, zooms)

        n_body = int((comp & case["body"]).sum())
        n_head = int((comp & case["head"]).sum())
        n_tail = int((comp & case["tail"]).sum())
        n_panc = int((comp & case["pancreas"]).sum())
        dist_to_pancreas_mm = float(dist_from_panc[comp[sl]].min()) if dist_from_panc is not None else np.nan

        frac_body = n_body / n_vox
        frac_head = n_head / n_vox
        frac_tail = n_tail / n_vox
        frac_panc = n_panc / n_vox

        location, certainty = classify_location(frac_body, frac_head, frac_tail, frac_panc)

        rows.append({
            "case_id": case_id,
            "lesion_id": k,
            "n_components": n,
            "n_vox": n_vox,
            "vol_mm3": n_vox * voxvol,
            "diam_mm": diam,
            "diam_bin": diam_bin(diam),
            "frac_body": frac_body,
            "frac_head": frac_head,
            "frac_tail": frac_tail,
            "frac_pancreas_general": frac_panc,
            "location": location,
            "certainty": certainty,
            "regions_touched": regions_touched(frac_body, frac_head, frac_tail, frac_panc),
            "dist_to_pancreas_mm": dist_to_pancreas_mm,
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
    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "lesion_location.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")

    if len(df):
        summary = (
            df.groupby(["diam_bin", "location"], observed=True)
            .size()
            .rename("n_lesions")
            .reset_index()
        )
        summary["diam_bin"] = pd.Categorical(summary["diam_bin"], categories=DIAM_LABELS, ordered=True)
        summary = summary.sort_values(["diam_bin", "n_lesions"], ascending=[True, False])
        summary_path = RESULTS_DIR / "location_by_diam_bin_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"wrote {summary_path}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
