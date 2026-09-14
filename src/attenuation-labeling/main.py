import multiprocessing as mp
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from scipy.ndimage import label, generate_binary_structure, distance_transform_edt

CT_ROOT = Path("/home/rahuldeb5/research/datasets/pants/ct_staging")
MASK_ROOT = Path("/home/rahuldeb5/research/datasets/pants/masks/mask_only")
FOLD_DIR = Path("src/data")
RESULTS_DIR = Path("results")
NUM_WORKERS = 2


def case_ids() -> list[str]:
    ids: set[str] = set()
    for i in range(1, 6):
        for line in (FOLD_DIR / f"fold_{i}_ids.txt").read_text().splitlines():
            line = line.strip()
            if line:
                ids.add(line)
    return sorted(ids)


def load_case(case_id: str) -> dict | None:
    """Load CT + lesion/pancreas/MPD masks for one case, all on the same grid.

    Returns None (with a printed reason) instead of raising, so one bad case
    doesn't kill a loop over ~1000. Masks are bool; CT is float32 HU.
    """
    seg_dir = MASK_ROOT / case_id / "segmentations"
    ct_path = CT_ROOT / case_id / "ct.nii.gz"
    lesion_path = seg_dir / "pancreatic_lesion.nii.gz"
    pancreas_path = seg_dir / "pancreas.nii.gz"
    mpd_path = seg_dir / "pancreatic_duct.nii.gz"

    for p in (ct_path, lesion_path, pancreas_path, mpd_path):
        if not p.exists():
            print(f"{case_id}: missing {p}")
            return None

    ct_img = nib.load(ct_path)
    lesion_img = nib.load(lesion_path)
    pancreas_img = nib.load(pancreas_path)
    mpd_img = nib.load(mpd_path)

    # Shape mismatch means the files can't be the same grid at all -- reject.
    # Affine mismatch is only a warning: a handful of cases carry a corrupted
    # affine header on lesion/mpd (traced to a stale/wrong header, not a real
    # resampling difference -- confirmed by checking that lesion voxels sit
    # in/near the pancreas mask in raw index space, which they wouldn't if the
    # arrays were actually on different grids). The code never applies any
    # mask's affine to a transform -- only shape-aligned indexing and the CT's
    # own header.get_zooms() -- so a bad affine here doesn't affect results.
    imgs = {"ct": ct_img, "lesion": lesion_img, "pancreas": pancreas_img, "mpd": mpd_img}
    ref_shape = ct_img.shape
    ref_affine = ct_img.affine
    for name, img in imgs.items():
        if img.shape != ref_shape:
            print(f"{case_id}: shape mismatch on {name}: {img.shape} vs ct {ref_shape}")
            return None
        if not np.allclose(img.affine, ref_affine, atol=1e-3):
            print(f"{case_id}: WARNING affine mismatch on {name} -- proceeding, trusting raw voxel grid")

    ct = ct_img.get_fdata(dtype=np.float32)  # applies scl_slope/inter -> real HU

    def load_bool_mask(img) -> np.ndarray:
        # get_unscaled() reads the on-disk dtype (int8 label masks) directly --
        # np.asanyarray(img.dataobj) applies nibabel's scl_slope/inter scaling
        # and silently upcasts to float64, ballooning memory 8x on large volumes
        raw = img.dataobj.get_unscaled()
        if np.issubdtype(raw.dtype, np.floating):
            raw = np.nan_to_num(raw, nan=0.0)
        return raw > 0.5

    return {
        "case_id": case_id,
        "ct": ct,
        "lesion": load_bool_mask(lesion_img),
        "pancreas": load_bool_mask(pancreas_img),
        "mpd": load_bool_mask(mpd_img),
        "zooms": ct_img.header.get_zooms()[:3],
    }

ATTENUATION_THRESHOLD_HU = 10.0


def classify_attenuation(delta_hu: float) -> str:
    if np.isnan(delta_hu):
        return "unknown"
    if delta_hu < -ATTENUATION_THRESHOLD_HU:
        return "hypoattenuating"
    if delta_hu > ATTENUATION_THRESHOLD_HU:
        return "hyperattenuating"
    return "isoattenuating"


def process_scan(case: dict, dilate_margin_mm: float = 2.5) -> list[dict]:
    """One row per connected lesion component. Pure array math -- no I/O here,
    so this function alone is what the phantom tests exercise."""
    zooms = case["zooms"]
    if not case["lesion"].any():
        return []

    # Crop to the bounding box of every mask used below before running the
    # distance transforms. scipy's exact EDT allocates scratch memory that
    # scales with the FULL input volume (not just its output), which blows
    # up past 10GB on full-body CTs with mostly-empty padding around a small
    # pancreas. This box always fully contains lesion/pancreas/mpd, so the
    # true nearest-foreground voxel for any query point inside it is also
    # inside it -- cropping changes nothing about the result, only the cost.
    roi = case["lesion"] | case["pancreas"] | case["mpd"]
    coords = np.argwhere(roi)
    lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))

    ct = case["ct"][sl]
    lesion = case["lesion"][sl]
    pancreas = case["pancreas"][sl]
    mpd = case["mpd"][sl]

    labeled, n = label(lesion, structure=generate_binary_structure(3, 3))

    # exclude every lesion component + a dilate_margin_mm shell around it from
    # the parenchyma reference pool -- computed once over the WHOLE lesion mask
    # so a second tumor elsewhere in the pancreas can't contaminate component i's
    # reference. Reused unchanged for every component below.
    dist_from_any_lesion = distance_transform_edt(~lesion, sampling=zooms)
    panc_pool = pancreas & ~(dist_from_any_lesion <= dilate_margin_mm)
    panc_hu = ct[panc_pool]
    panc_hu_median = float(np.median(panc_hu)) if panc_pool.any() else np.nan
    del dist_from_any_lesion  # full-volume float64; drop before allocating the MPD one below

    # computed once for the whole mask -- reused per component instead of
    # rebuilding this full-volume float64 array on every loop iteration
    dist_from_mpd = distance_transform_edt(~mpd, sampling=zooms) if mpd.any() else None

    rows = []
    for i in range(n):
        comp = labeled == i + 1
        n_vox = int(comp.sum())
        vol_mm3 = n_vox * float(np.prod(zooms))

        tumor_hu = ct[comp]

        tumor_hu_median = float(np.median(tumor_hu))
        delta_hu = tumor_hu_median - panc_hu_median if panc_pool.any() else np.nan

        if dist_from_mpd is not None:
            dist_to_mpd = float(dist_from_mpd[comp].min())
        else:
            dist_to_mpd = np.nan

        rows.append(
            {
                "case_id": case["case_id"],
                "lesion_id": i + 1,
                "n_components": n,
                "n_vox": n_vox,
                "vol_mm3": vol_mm3,
                "tumor_hu_mean": float(tumor_hu.mean()),
                "tumor_hu_median": tumor_hu_median,
                "panc_hu_median": panc_hu_median,
                "n_panc_vox": int(panc_pool.sum()),
                "delta_hu": delta_hu,
                "attenuation": classify_attenuation(delta_hu),
                "mpd_present": bool(mpd.any()),
                "dist_to_mpd_mm": dist_to_mpd,
            }
        )
    return rows


def process_case(case_id: str) -> list[dict]:
    case = load_case(case_id)
    if case is None:
        return []
    rows = process_scan(case)
    for r in rows:
        print(
            f"{case_id} lesion {r['lesion_id']}/{r['n_components']}: "
            f"vol={r['vol_mm3']:.0f}mm3 delta_hu={r['delta_hu']:.1f} "
            f"dist_to_mpd={r['dist_to_mpd_mm']}"
        )
    return rows


def main() -> None:
    ids = case_ids()
    print(f"{len(ids)} case ids to load, {NUM_WORKERS} workers")

    with mp.Pool(NUM_WORKERS) as pool:
        results = pool.map(process_case, ids)

    all_rows = [row for rows in results for row in rows]
    print(f"{len(all_rows)} lesion rows total")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "attenuation_labels.csv"
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
