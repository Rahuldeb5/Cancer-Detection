import nibabel as nib
import numpy as np
from pathlib import Path

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


def load_ducts(case_id: str) -> dict | None:
    """Load CBD + MPD masks for one case, on the same grid.

    Returns None (with a printed reason) instead of raising, so one bad
    case doesn't kill a loop over ~1000. Spacing comes from the affine's
    column norms (nib.affines.voxel_sizes), not header pixdim or
    np.diag(affine) -- some PanTS affines carry the scale factors off
    the main diagonal (axis-permuted orientation), so diag() can silently
    read a zero or a rotation cross-term instead of the true spacing.
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

    # grid check before trusting any voxel-to-voxel comparison across files
    ref_shape = cbd_img.shape
    ref_affine = cbd_img.affine
    if mpd_img.shape != ref_shape:
        print(f"{case_id}: shape mismatch cbd {ref_shape} vs mpd {mpd_img.shape}")
        return None
    if not np.allclose(mpd_img.affine, ref_affine, atol=1e-3):
        print(f"{case_id}: affine mismatch cbd vs mpd")
        return None

    def load_bool_mask(img) -> np.ndarray:
        raw = np.nan_to_num(np.asanyarray(img.dataobj), nan=0.0)
        return raw > 0.5

    spacing = nib.affines.voxel_sizes(ref_affine)  # array-axis order, robust to permuted affines

    return {
        "case_id": case_id,
        "cbd": load_bool_mask(cbd_img),
        "mpd": load_bool_mask(mpd_img),
        "spacing": tuple(float(s) for s in spacing),
        "affine": ref_affine,
        "axcodes": nib.aff2axcodes(ref_affine),
    }
