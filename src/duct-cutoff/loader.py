"""Load GT CBD + MPD duct masks for one case and resample both to isotropic
spacing, so downstream skeletonization/EDT geometry isn't biased by PanTS's
anisotropic (~1x1x1.5mm) native spacing (skeletonize/EDT operate on the voxel
lattice and don't know about physical units).

Duplicates rather than imports geometry/loader.py's load_ducts(): "duct-cutoff"
and "geometry" aren't valid Python package names (hyphens), so a normal
`import` can't cross that folder boundary -- same reason attenuation-labeling/
carries its own copy of case_ids() instead of importing it.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

MASK_ROOT = Path("/home/rahuldeb5/research/datasets/pants/masks/mask_only")
FOLD_DIR = Path("src/data")

CROP_PAD_VOX = 5  # native-grid voxels of padding kept around each mask's bbox


def case_ids() -> list[str]:
    ids: set[str] = set()
    for i in range(1, 6):
        for line in (FOLD_DIR / f"fold_{i}_ids.txt").read_text().splitlines():
            line = line.strip()
            if line:
                ids.add(line)
    return sorted(ids)


def load_ducts(case_id: str) -> dict | None:
    """Load CBD + MPD masks for one case, on the same grid, at native spacing.

    Returns None (with a printed reason) instead of raising, so one bad
    case doesn't kill a loop over ~1000.
    """
    seg_dir = MASK_ROOT / case_id / "segmentations"
    cbd_path = seg_dir / "common_bile_duct.nii.gz"
    mpd_path = seg_dir / "pancreatic_duct.nii.gz"

    for p in (cbd_path, mpd_path):
        if not p.exists():
            print(f"{case_id}: missing {p}")
            return None

    try:
        cbd_img = nib.load(cbd_path)
        mpd_img = nib.load(mpd_path)
    except Exception as e:  # e.g. un-pulled git-LFS pointer stub ("not a gzip file")
        print(f"{case_id}: unreadable mask ({type(e).__name__}: {e})")
        return None

    ref_shape = cbd_img.shape
    ref_affine = cbd_img.affine
    if mpd_img.shape != ref_shape:
        print(f"{case_id}: shape mismatch cbd {ref_shape} vs mpd {mpd_img.shape}")
        return None
    if not np.allclose(mpd_img.affine, ref_affine, atol=1e-3):
        print(f"{case_id}: affine mismatch cbd vs mpd")
        return None

    def load_bool_mask(img) -> np.ndarray:
        # get_unscaled() reads the on-disk dtype (int8 label masks) directly --
        # np.asanyarray(img.dataobj) applies nibabel's scl_slope/inter scaling
        # and silently upcasts to float64, doubling memory right before we
        # need a float32 buffer for the resample below.
        raw = img.dataobj.get_unscaled()
        if np.issubdtype(raw.dtype, np.floating):
            raw = np.nan_to_num(raw, nan=0.0)
        return raw > 0.5

    spacing = nib.affines.voxel_sizes(ref_affine)  # array-axis order, robust to permuted affines

    return {
        "case_id": case_id,
        "cbd": load_bool_mask(cbd_img),
        "mpd": load_bool_mask(mpd_img),
        "spacing": tuple(float(s) for s in spacing),
        "affine": ref_affine,
    }


def _bbox_crop(mask: np.ndarray, pad: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Crop to the mask's bounding box + pad. Returns (cropped, offset), where
    offset is this crop's origin in the ORIGINAL (native, uncropped) array's
    voxel indices -- needed later to map isotropic-space points back to
    world mm via the native affine.

    Unlike largest_component()/clean_mask() in geometry.py, this drops no
    voxels and merges no components -- it only avoids handing the resampler (and
    later, skeletonize/EDT) a full CT-grid-sized array when the duct itself
    occupies a tiny fraction of it, which is the same full-volume memory trap
    attenuation-labeling/main.py documents for distance_transform_edt.
    """
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + 1 + pad, mask.shape)
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    return mask[sl], tuple(int(x) for x in lo)


def resample_mask_to_isotropic(
    mask: np.ndarray, spacing: tuple[float, float, float], target_mm: float = 1.0
) -> tuple[np.ndarray, tuple[float, float, float], tuple[int, int, int]]:
    """Bbox-crop, then resample a *binary* mask to exactly target_mm isotropic.

    Linear interpolation on the float-cast mask + 0.5 threshold, not
    nearest-neighbor: the duct is only 2-4 voxels across at native spacing, so
    nearest-neighbor upsampling would just duplicate the existing blocky
    boundary, and caliber/derivative estimates are sensitive to it. Safe for a
    single binary label -- nothing to blend it with.

    Uses affine_transform with an explicit scale, NOT scipy's zoom(): zoom()
    aligns first/last voxel *centers* and rounds the output shape per array, so
    the effective spacing drifts up to ~1% and differs between two crops of
    the same grid (observed on real PanTS ducts). Here output index o maps to
    input index o / factor, so spacing is exactly target_mm and voxel 0 of the
    output is voxel 0 of the crop -- which makes the crop_offset world-mm
    formula in load_ducts_isotropic exact.

    Returns (resampled_mask, spacing_iso, crop_offset); an empty input mask
    comes back as a zero-size array.
    """
    iso = (target_mm,) * 3
    if not mask.any():
        return np.zeros((0, 0, 0), dtype=bool), iso, (0, 0, 0)

    cropped, offset = _bbox_crop(mask, CROP_PAD_VOX)

    factors = np.asarray(spacing) / target_mm
    out_shape = tuple(int(np.floor((n - 1) * f)) + 1 for n, f in zip(cropped.shape, factors))
    resampled = affine_transform(
        cropped.astype(np.float32), np.diag(1.0 / factors), output_shape=out_shape, order=1, mode="constant", cval=0.0
    ) > 0.5
    return resampled, iso, offset


def load_ducts_isotropic(case_id: str, target_mm: float = 1.0) -> dict | None:
    """load_ducts() + bbox-crop + resample both duct masks to isotropic
    spacing. cbd and mpd are cropped independently (their bboxes differ), so
    each carries its own crop_offset -- world-mm conversion for a point
    p_iso in one duct's isotropic-space array is:

        p_native = crop_offset + p_iso * (spacing_iso / native_spacing)   # per axis
        world_mm = affine @ [*p_native, 1]

    That composition (and the actual orientation/world-mm logic) belongs in
    centerline.py, not here -- this just returns what's needed to do it.
    """
    case = load_ducts(case_id)
    if case is None:
        return None

    cbd_iso, spacing_iso, cbd_offset = resample_mask_to_isotropic(case["cbd"], case["spacing"], target_mm)
    mpd_iso, _, mpd_offset = resample_mask_to_isotropic(case["mpd"], case["spacing"], target_mm)

    return {
        "case_id": case_id,
        "cbd": cbd_iso,
        "mpd": mpd_iso,
        "cbd_offset": cbd_offset,
        "mpd_offset": mpd_offset,
        "spacing": spacing_iso,
        "native_spacing": case["spacing"],
        "affine": case["affine"],
    }
