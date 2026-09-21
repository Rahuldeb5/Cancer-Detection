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
from scipy.ndimage import zoom

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

    cbd_img = nib.load(cbd_path)
    mpd_img = nib.load(mpd_path)

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
        # need a float32 buffer for zoom() below.
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
    voxels and merges no components -- it only avoids handing zoom() (and
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
    """Bbox-crop, then resample a *binary* mask to isotropic spacing.

    order=1 (linear) on the float-cast mask + 0.5 threshold, not order=0
    (nearest-neighbor): the duct is only 2-4 voxels across at native spacing,
    so nearest-neighbor upsampling would just duplicate the existing blocky
    boundary instead of reconstructing a smoother one, and downstream
    caliber/derivative estimates are sensitive to that boundary. Linear is
    safe for a single binary label -- no risk of blending two different
    label IDs together, since there's only one here.

    Returns (resampled_mask, new_spacing, crop_offset); new_spacing is
    recomputed from the actual output shape rather than assumed to be exactly
    target_mm, since zoom() rounds output shape to the nearest integer voxel
    count and the zoom factor it actually applied can differ slightly.
    """
    if not mask.any():
        return mask, spacing, (0, 0, 0)

    cropped, offset = _bbox_crop(mask, CROP_PAD_VOX)

    zoom_factors = tuple(s / target_mm for s in spacing)
    resampled = zoom(cropped.astype(np.float32), zoom_factors, order=1) > 0.5

    new_spacing = tuple(
        s * (n_in / n_out) for s, n_in, n_out in zip(spacing, cropped.shape, resampled.shape)
    )
    return resampled, new_spacing, offset


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

    cbd_iso, cbd_spacing, cbd_offset = resample_mask_to_isotropic(case["cbd"], case["spacing"], target_mm)
    mpd_iso, mpd_spacing, mpd_offset = resample_mask_to_isotropic(case["mpd"], case["spacing"], target_mm)

    # both start on the same grid/spacing (load_ducts checked this), so a
    # fixed target_mm must resample them to the same spacing even though
    # their crops/offsets differ -- if this ever fires, resample_mask_to_isotropic
    # itself has a bug, not the input data.
    assert cbd_spacing == mpd_spacing, (
        f"{case_id}: cbd/mpd resampled to different spacing ({cbd_spacing} vs "
        f"{mpd_spacing}) despite sharing native spacing -- should be impossible"
    )

    return {
        "case_id": case_id,
        "cbd": cbd_iso,
        "mpd": mpd_iso,
        "cbd_offset": cbd_offset,
        "mpd_offset": mpd_offset,
        "spacing": cbd_spacing,
        "native_spacing": case["spacing"],
        "affine": case["affine"],
    }
