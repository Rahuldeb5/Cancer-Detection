"""Assemble ``Dataset501_PanTSTumor`` (binary tumor) in ``$nnUNet_raw``.

Inputs:
  - CT volumes  : ``<CT_STAGING>/PanTS_XXXXXXXX/ct.nii.gz``           (from ``extract_cts.sh``)
  - lesion masks: ``masks/mask_only/PanTS_XXXXXXXX/segmentations/pancreatic_lesion.nii.gz``
                  (git-LFS working tree, materialised by ``git lfs pull --include='**/pancreatic_lesion.nii.gz'``)

Outputs:
  - ``imagesTr/PanTS_XXXXXXXX_0000.nii.gz``  (hard-link to the staged CT, or a
                                              rewritten copy if its direction cosines
                                              are not orthonormal -- a few PanTS CTs
                                              carry ~1e-4 numerical drift that ITK,
                                              and therefore nnU-Net's default reader,
                                              rejects outright)
  - ``labelsTr/PanTS_XXXXXXXX.nii.gz``       (lesion mask, binarised to {0,1} uint8,
                                              written on the CT's exact affine/header)
  - ``dataset.json``

Label scheme: background=0, tumor=1 (pancreas is NOT a class).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

FOLD_DIR = Path("src/data")
CT_STAGING = Path("/home/rahuldeb5/research/datasets/pants/ct_staging")
MASK_STAGING = Path("/home/rahuldeb5/research/datasets/pants/masks/mask_only")
RAW_DIR = Path(os.environ["nnUNet_raw"]) / "Dataset501_PanTSTumor"

IMAGES_TR = RAW_DIR / "imagesTr"
LABELS_TR = RAW_DIR / "labelsTr"

# ITK refuses direction cosines whose off-orthonormality exceeds ~1e-4.
ORTHO_TOL = 1e-5


def fold_ids() -> list[str]:
    ids: set[str] = set()
    for i in range(1, 6):
        for line in (FOLD_DIR / f"fold_{i}_ids.txt").read_text().splitlines():
            line = line.strip()
            if line:
                ids.add(line)
    return sorted(ids)


def is_orthonormal(affine: np.ndarray) -> bool:
    r = affine[:3, :3]
    d = r / np.linalg.norm(r, axis=0)
    return bool(np.max(np.abs(d.T @ d - np.eye(3))) < ORTHO_TOL)


def orthonormalized(affine: np.ndarray) -> np.ndarray:
    """Nearest affine with orthonormal direction cosines; spacing + origin preserved."""
    r = affine[:3, :3]
    spacing = np.linalg.norm(r, axis=0)
    d = r / spacing
    u, _, vt = np.linalg.svd(d)
    d_ortho = u @ vt
    if np.linalg.det(d_ortho) < 0:  # keep handedness
        u[:, -1] *= -1
        d_ortho = u @ vt
    out = np.eye(4)
    out[:3, :3] = d_ortho * spacing
    out[:3, 3] = affine[:3, 3]
    return out


def prepare_ct(cid: str) -> tuple[Path | None, str | None, bool]:
    """Return (final_ct_path, error, was_rewritten)."""
    src = CT_STAGING / cid / "ct.nii.gz"
    if not src.exists():
        return None, "missing staged CT", False
    dst = IMAGES_TR / f"{cid}_0000.nii.gz"

    src_img = nib.load(src)
    if is_orthonormal(src_img.affine):
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
        return dst, None, False

    # sanitise: rewrite with orthonormal direction cosines
    fixed = orthonormalized(src_img.affine)
    data = np.asanyarray(src_img.dataobj)
    hdr = src_img.header.copy()
    out = nib.Nifti1Image(data, fixed, hdr)
    out.set_qform(fixed, code=1)
    out.set_sform(fixed, code=1)
    if dst.exists():
        dst.unlink()
    nib.save(out, dst)
    return dst, None, True


# A pancreatic tumor filling more than this fraction of the scan is not physical --
# it means the mask was misread (e.g. the PanTS lesion files carry scl_slope=NaN,
# which makes SimpleITK's BinaryThreshold light up the whole volume).
MAX_FG_FRACTION = 0.30


def read_binary_mask(path: Path) -> np.ndarray:
    """Lesion mask -> {0,1} uint8. nibabel already ignores the NaN scl_slope in the
    PanTS headers (SimpleITK does NOT -- its BinaryThreshold then lights up the whole
    volume, which is the bug this rewrite fixes). nan_to_num guards the edge case."""
    raw = np.nan_to_num(np.asanyarray(nib.load(path).dataobj), nan=0.0)
    return (raw > 0.5).astype(np.uint8)


def build_label(cid: str, ct_path: Path, force: bool) -> str | None:
    src = MASK_STAGING / cid / "segmentations" / "pancreatic_lesion.nii.gz"
    if not src.exists():
        return "missing lesion mask"
    dst = LABELS_TR / f"{cid}.nii.gz"
    if dst.exists() and not force:
        return None

    ct = nib.load(ct_path)
    les = nib.load(src)
    if les.shape != ct.shape:
        return f"grid mismatch CT{ct.shape} vs lesion{les.shape}"

    seg = read_binary_mask(src)
    frac = float(seg.mean())
    if frac > MAX_FG_FRACTION:
        return f"implausible tumor mask: {frac:.1%} of scan is foreground"

    affine = ct.affine
    out = nib.Nifti1Image(seg, affine)  # fresh header -> scl_slope=1, scl_inter=0
    out.header.set_zooms(nib.load(ct_path).header.get_zooms()[:3])
    out.set_data_dtype(np.uint8)
    out.set_qform(affine, code=1)
    out.set_sform(affine, code=1)
    if dst.exists():
        dst.unlink()
    nib.save(out, dst)
    return None


def main() -> None:
    ids = fold_ids()
    IMAGES_TR.mkdir(parents=True, exist_ok=True)
    LABELS_TR.mkdir(parents=True, exist_ok=True)
    print(f"building Dataset501_PanTSTumor from {len(ids)} cases -> {RAW_DIR}", flush=True)

    errs: list[tuple[str, str]] = []
    rewritten: list[str] = []
    for n, cid in enumerate(ids, 1):
        ct_path, e, was_rewritten = prepare_ct(cid)
        if was_rewritten:
            rewritten.append(cid)
        if e is None:
            e = build_label(cid, ct_path, force=was_rewritten)
        if e:
            errs.append((cid, e))
        if n % 200 == 0 or n == len(ids):
            print(f"  {n}/{len(ids)}  (rewritten CTs: {len(rewritten)}, errors: {len(errs)})", flush=True)

    n_img = len(list(IMAGES_TR.glob("*_0000.nii.gz")))
    n_lab = len(list(LABELS_TR.glob("*.nii.gz")))
    print(f"imagesTr: {n_img}   labelsTr: {n_lab}   sanitised CTs: {rewritten}")

    if errs:
        print(f"ERRORS ({len(errs)}):")
        for cid, e in errs[:30]:
            print(f"  {cid}: {e}")
        sys.exit(1)

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(ids),
        "file_ending": ".nii.gz",
        "dataset_name": "Dataset501_PanTSTumor",
        "description": "PanTS-Mini binary pancreatic-tumor segmentation; 1308-case 5-fold CV subset.",
    }
    (RAW_DIR / "dataset.json").write_text(json.dumps(dataset_json, indent=4))
    print(f"wrote {RAW_DIR / 'dataset.json'}")


if __name__ == "__main__":
    main()
