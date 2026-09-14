import numpy as np
from scipy.ndimage import label, generate_binary_structure, distance_transform_edt
from skimage.morphology import skeletonize

def largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = label(mask, structure=generate_binary_structure(3,3))
    
    if n == 0:
        return mask

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    biggest_label = sizes.argmax()
    return labeled == biggest_label

def crop_and_pad(mask: np.ndarray, pad: int = 10) -> np.ndarray:
    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1               # +1: argwhere gives inclusive indices, slices are exclusive
    tight = mask[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    return np.pad(tight, pad_width=pad, mode="constant", constant_values=False)

def clean_mask(mask: np.ndarray, pad: int = 10) -> np.ndarray:
    mask = largest_component(mask)
    return crop_and_pad(mask, pad=pad)


def edt_field(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    return distance_transform_edt(mask, sampling=spacing)


def skeleton(mask: np.ndarray) -> np.ndarray:
    return skeletonize(mask)


def radius_profile(edt: np.ndarray, skel: np.ndarray) -> np.ndarray:
    return edt[skel]


def caliber_profile(radii: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    corrected_radii = radii - 0.5 * min(spacing)
    return 2 * corrected_radii


def caliber_stats(calibers: np.ndarray) -> dict:
    """Order statistics of the caliber profile -- no filtering, nothing dropped."""
    if calibers.size == 0:
        return {"p50_mm": np.nan, "p95_mm": np.nan, "p99_mm": np.nan, "max_mm": np.nan, "n_skel": 0}
    return {
        "p50_mm": float(np.percentile(calibers, 50)),
        "p95_mm": float(np.percentile(calibers, 95)),
        "p99_mm": float(np.percentile(calibers, 99)),
        "max_mm": float(calibers.max()),
        "n_skel": int(calibers.size),
    }


def max_caliber_mm(mask: np.ndarray, spacing: tuple[float, float, float], pad: int = 10) -> dict:
    """Full pipeline, steps 2-7: raw boolean mask + spacing in, caliber stats dict out."""
    cleaned = clean_mask(mask, pad=pad)
    if not cleaned.any():
        return {"present": False, **caliber_stats(np.array([]))}

    edt = edt_field(cleaned, spacing)
    skel = skeleton(cleaned)
    radii = radius_profile(edt, skel)
    calibers = caliber_profile(radii, spacing)
    return {"present": True, **caliber_stats(calibers)}


