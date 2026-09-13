import nibabel as nib
import numpy as np
from pathlib import Path

from scipy.ndimage import label, generate_binary_structure, distance_transform_edt

CT_ROOT = Path("/home/rahuldeb5/research/datasets/pants/ct_staging")
MASK_ROOT = Path("/home/rahuldeb5/research/datasets/pants/masks/mask_only")
FOLD_DIR = Path("src/data")


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

    # grid check before trusting any voxel-to-voxel comparison across files
    imgs = {"ct": ct_img, "lesion": lesion_img, "pancreas": pancreas_img, "mpd": mpd_img}
    ref_shape = ct_img.shape
    ref_affine = ct_img.affine
    for name, img in imgs.items():
        if img.shape != ref_shape:
            print(f"{case_id}: shape mismatch on {name}: {img.shape} vs ct {ref_shape}")
            return None
        if not np.allclose(img.affine, ref_affine, atol=1e-3):
            print(f"{case_id}: affine mismatch on {name}")
            return None

    ct = ct_img.get_fdata(dtype=np.float32)  # applies scl_slope/inter -> real HU

    def load_bool_mask(img) -> np.ndarray:
        raw = np.nan_to_num(np.asanyarray(img.dataobj), nan=0.0)
        return raw > 0.5

    return {
        "case_id": case_id,
        "ct": ct,
        "lesion": load_bool_mask(lesion_img),
        "pancreas": load_bool_mask(pancreas_img),
        "mpd": load_bool_mask(mpd_img),
        "zooms": ct_img.header.get_zooms()[:3],
    }

def process_scan(case: dict, dilate_margin_mm: float = 2.5) -> list[dict]:
    """One row per connected lesion component. Pure array math -- no I/O here,
    so this function alone is what the phantom tests exercise."""
    ct = case["ct"]
    zooms = case["zooms"]
    labeled, n = label(case["lesion"], structure=generate_binary_structure(3, 3))

    # exclude every lesion component + a dilate_margin_mm shell around it from
    # the parenchyma reference pool -- computed once over the WHOLE lesion mask
    # so a second tumor elsewhere in the pancreas can't contaminate component i's
    # reference. Reused unchanged for every component below.
    dist_from_any_lesion = distance_transform_edt(~case["lesion"], sampling=zooms)
    panc_pool = case["pancreas"] & ~(dist_from_any_lesion <= dilate_margin_mm)
    panc_hu = ct[panc_pool]
    panc_hu_median = float(np.median(panc_hu)) if panc_pool.any() else np.nan

    rows = []
    for i in range(n):
        comp = labeled == i + 1
        n_vox = int(comp.sum())
        vol_mm3 = n_vox * float(np.prod(zooms))

        tumor_hu = ct[comp]

        tumor_hu_median = float(np.median(tumor_hu))
        delta_hu = tumor_hu_median - panc_hu_median if panc_pool.any() else np.nan

        mpd = case["mpd"]
        if mpd.any():
            dist_to_mpd = float(
                distance_transform_edt(~mpd, sampling=zooms)[comp].min()
            )
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
                "mpd_present": bool(mpd.any()),
                "dist_to_mpd_mm": dist_to_mpd,
            }
        )
    return rows


def main() -> None:
    ids = case_ids()
    print(f"{len(ids)} case ids to load")
    all_rows: list[dict] = []
    for case_id in ids:
        case = load_case(case_id)
        if case is None:
            continue
        rows = process_scan(case)
        all_rows.extend(rows)
        for r in rows:
            print(
                f"{case_id} lesion {r['lesion_id']}/{r['n_components']}: "
                f"vol={r['vol_mm3']:.0f}mm3 delta_hu={r['delta_hu']:.1f} "
                f"dist_to_mpd={r['dist_to_mpd_mm']}"
            )
    print(f"{len(all_rows)} lesion rows total")


if __name__ == "__main__":
    main()
