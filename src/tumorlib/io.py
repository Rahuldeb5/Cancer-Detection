"""Case lists and NIfTI loading for PanTS, in nibabel's native (x, y, z) array order.

Conventions (CONTEXT.md "Data gotchas"):
  * Masks are read with dataobj.get_unscaled(): the on-disk int8 stays int8, no
    float64 upcast. The 0.5 threshold is applied in *scaled* units but evaluated
    in raw units, because the int8 files come in two encodings (on-disk headers,
    1308-case cohort, 5 pancreas/lesion masks each): 3130 files are raw {0,1} with
    slope 1, and 3410 are raw {-128,127} with slope 1/255, inter 0.502. `raw != 0`
    would light up the whole volume for the second kind. (img.header shows
    scl_slope=NaN for every file only because nibabel moves the scaling onto
    img.dataobj.slope/inter when it loads an image.)
  * Spacing comes from nib.affines.voxel_sizes(CT affine): never header zooms or
    np.diag. The CT grid is authoritative because 9 cases carry corrupted *mask*
    affines on an otherwise aligned voxel grid.
  * Any problem (missing file, un-pulled git-LFS stub, unreadable gzip, grid
    mismatch) returns None with a printed reason, so a loop over ~1300 cases
    survives one bad case.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FOLD_DIR = REPO_ROOT / "src" / "data"
MASK_ROOT = Path(os.environ.get("TUMORLIB_MASK_ROOT", "/home/rahuldeb5/research/datasets/pants/masks/mask_only"))
CT_ROOT = Path(os.environ.get("TUMORLIB_CT_ROOT", "/home/rahuldeb5/research/datasets/pants/ct_staging"))

N_FOLDS = 5
LFS_STUB_MAX_BYTES = 1024  # real .nii.gz masks are KBs+; LFS pointer stubs are ~131 bytes
MAX_FG_FRACTION = 0.30     # a mask covering >30% of the scan was misread, not segmented


# ------------------------------------------------------------------ case lists
def fold_ids(k: int) -> list[str]:
    """Case ids in src/data/fold_{k}_ids.txt (k = 1..5 <-> nnU-Net fold k-1)."""
    lines = (FOLD_DIR / f"fold_{k}_ids.txt").read_text().splitlines()
    return [s.strip() for s in lines if s.strip()]


def case_ids() -> list[str]:
    """The 1308-case cohort: sorted union of the five fold files."""
    return sorted({c for k in range(1, N_FOLDS + 1) for c in fold_ids(k)})


# ------------------------------------------------------------------ paths
def mask_path(case_id: str, name: str) -> Path:
    return MASK_ROOT / case_id / "segmentations" / f"{name}.nii.gz"


def ct_path(case_id: str) -> Path:
    return CT_ROOT / case_id / "ct.nii.gz"


def _open(path: Path, case_id: str):
    """nib.load() with every failure mode turned into None + printed reason."""
    if not path.exists():
        print(f"{case_id}: missing {path}")
        return None
    size = path.stat().st_size
    if size < LFS_STUB_MAX_BYTES:
        head = path.read_bytes()[:64]
        why = "git-LFS pointer stub" if head.startswith(b"version https://git-lfs") else f"only {size} bytes"
        print(f"{case_id}: {path.name} is not real data ({why}); run a scoped `git lfs pull -I`")
        return None
    try:
        return nib.load(path)
    except Exception as e:  # e.g. "not a gzip file"
        print(f"{case_id}: unreadable {path} ({type(e).__name__}: {e})")
        return None


# ------------------------------------------------------------------ grid
@lru_cache(maxsize=64)
def grid(case_id: str) -> tuple[tuple[int, int, int], np.ndarray, tuple[float, float, float]] | None:
    """(shape, affine, spacing_mm) of the case's CT, read from the header only."""
    img = _open(ct_path(case_id), case_id)
    if img is None:
        return None
    spacing = tuple(float(s) for s in nib.affines.voxel_sizes(img.affine))
    return tuple(img.shape[:3]), img.affine.copy(), spacing


def spacing(case_id: str) -> tuple[float, float, float] | None:
    """Voxel size in mm per array axis (x, y, z), from the CT affine."""
    g = grid(case_id)
    return None if g is None else g[2]


# ------------------------------------------------------------------ loaders
def _raw_threshold(dataobj) -> float:
    """Raw-unit value equivalent to 0.5 in scaled units: raw*slope + inter > 0.5."""
    slope, inter = float(dataobj.slope), float(dataobj.inter)
    if not np.isfinite(slope) or slope <= 0 or not np.isfinite(inter):
        return 0.5
    return (0.5 - inter) / slope


def load_mask(case_id: str, name: str) -> np.ndarray | None:
    """Binary mask `name` (e.g. "pancreatic_lesion", "pancreas_head") as a uint8
    {0,1} array on the CT grid, or None with a printed reason.

    Peak memory ~2x the voxel count in bytes: the int8 raw array plus the uint8 output.
    """
    img = _open(mask_path(case_id, name), case_id)
    if img is None:
        return None

    g = grid(case_id)
    if g is not None:
        shape, affine, _ = g
        if tuple(img.shape[:3]) != shape:
            print(f"{case_id}: {name} shape {img.shape} != CT {shape}; rejecting")
            return None
        if not np.allclose(img.affine, affine, atol=1e-3):
            print(f"{case_id}: WARNING {name} affine differs from CT; using the CT grid (voxels are aligned)")

    try:
        raw = img.dataobj.get_unscaled()
    except Exception as e:  # truncated gzip etc. only surfaces when the data is read
        print(f"{case_id}: unreadable data in {name} ({type(e).__name__}: {e})")
        return None
    if np.issubdtype(raw.dtype, np.floating):
        raw = np.nan_to_num(raw, nan=0.0, copy=False)

    out = np.empty(raw.shape, dtype=np.uint8)
    np.greater(raw, _raw_threshold(img.dataobj), out=out, casting="unsafe")
    del raw

    frac = out.mean(dtype=np.float64)
    if frac > MAX_FG_FRACTION:
        print(f"{case_id}: {name} covers {frac:.1%} of the scan -- misread mask, rejecting")
        return None
    return out


def load_ct(case_id: str, crop: tuple[slice, ...] | None = None) -> np.ndarray | None:
    """CT in HU as float32, optionally only the `crop` sub-block (array-index slices).

    Cropping here avoids ever holding the full volume as float32 (~800 MB for the
    largest case): nibabel only allocates the requested block.
    """
    img = _open(ct_path(case_id), case_id)
    if img is None:
        return None
    try:
        block = img.dataobj[crop] if crop is not None else img.dataobj
        return np.asarray(block, dtype=np.float32)
    except Exception as e:
        print(f"{case_id}: unreadable CT data ({type(e).__name__}: {e})")
        return None
