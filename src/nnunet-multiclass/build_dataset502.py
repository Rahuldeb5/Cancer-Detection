"""Assemble ``Dataset502_PanTSPancLesion`` (pancreas + lesion) in ``$nnUNet_raw``.

Same 1308 cases, same CT grids and the same direction-cosine repair as
``testing/nnunet/build_dataset.py`` (whose ``is_orthonormal`` / ``orthonormalized`` are
imported here rather than reimplemented, so the 6 drifted CTs get byte-identical
treatment). Dataset501 is only ever READ.

imagesTr: hard-linked from ``Dataset501_PanTSTumor/imagesTr`` when that is still on this
machine (zero extra bytes, and it inherits the already-repaired affines). If a case is
missing there, it is rebuilt from ``ct_staging`` exactly as build_dataset.py does.

labelsTr: ``labels502.compose_label(envelope, lesion)`` written on the image's exact
affine/header, uint8, scl_slope=1/inter=0. Masks are read through
``tumorlib.io.load_mask`` -- the two on-disk int8 encodings make ``raw != 0`` light up
whole volumes, and SimpleITK thresholding has the same failure mode (CONTEXT.md gotcha 1).

    source ~/research/nnunet_env/env.sh
    PYTHONPATH=src .venv/bin/python src/nnunet-multiclass/build_dataset502.py [--force] [--nproc 3]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "nnunet-multiclass"))
sys.path.insert(0, str(REPO / "testing" / "nnunet"))

from tumorlib import io as tio                                   # noqa: E402
from labels502 import (                                          # noqa: E402
    DATASET_NAME,
    ENVELOPE_MASKS,
    LESION_MASK,
    MAX_LESION_FRACTION,
    MAX_PANCREAS_FRACTION,
    compose_label,
    dataset_json,
)
# build_dataset.py reads $nnUNet_raw at import time; it writes nothing on import.
from build_dataset import is_orthonormal, orthonormalized      # noqa: E402

RAW_ROOT = Path(os.environ["nnUNet_raw"])
DS501 = RAW_ROOT / "Dataset501_PanTSTumor"
RAW_DIR = RAW_ROOT / DATASET_NAME
IMAGES_TR = RAW_DIR / "imagesTr"
LABELS_TR = RAW_DIR / "labelsTr"
CT_STAGING = tio.CT_ROOT


# --------------------------------------------------------------------- images
def prepare_image(cid: str) -> tuple[Path | None, str | None, str]:
    """(image_path, error, how) with how in {"exists", "link501", "link_ct", "rewrite"}."""
    dst = IMAGES_TR / f"{cid}_0000.nii.gz"
    if dst.exists():
        return dst, None, "exists"

    src501 = DS501 / "imagesTr" / f"{cid}_0000.nii.gz"
    if src501.exists():
        # Dataset501's copy is already orthonormal (build_dataset.py repaired it there).
        if not is_orthonormal(nib.load(src501).affine):
            return None, "Dataset501 image has non-orthonormal cosines", "link501"
        try:
            os.link(src501, dst)
        except OSError:
            shutil.copyfile(src501, dst)
        return dst, None, "link501"

    src = CT_STAGING / cid / "ct.nii.gz"
    if not src.exists():
        return None, "missing staged CT", "link_ct"
    src_img = nib.load(src)
    if is_orthonormal(src_img.affine):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
        return dst, None, "link_ct"

    fixed = orthonormalized(src_img.affine)
    out = nib.Nifti1Image(np.asanyarray(src_img.dataobj), fixed, src_img.header.copy())
    out.set_qform(fixed, code=1)
    out.set_sform(fixed, code=1)
    nib.save(out, dst)
    return dst, None, "rewrite"


# --------------------------------------------------------------------- labels
def load_envelope(cid: str) -> tuple[np.ndarray | None, str | None]:
    """Union of the four gland masks, uint8, CT grid. No morphology."""
    out = None
    for name in ENVELOPE_MASKS:
        m = tio.load_mask(cid, name)
        if m is None:
            return None, f"unreadable mask {name}"
        out = m if out is None else np.bitwise_or(out, m, out=out)
    return out, None


def build_one(args) -> dict:
    cid, force = args
    r: dict = {"id": cid, "error": None}
    img_p, err, how = prepare_image(cid)
    r["image_how"] = how
    if err:
        r["error"] = err
        return r

    dst = LABELS_TR / f"{cid}.nii.gz"
    img = nib.load(img_p)

    les = tio.load_mask(cid, LESION_MASK)
    if les is None:
        r["error"] = f"unreadable mask {LESION_MASK}"
        return r
    env, err = load_envelope(cid)
    if err:
        r["error"] = err
        return r

    if les.shape != img.shape[:3] or env.shape != img.shape[:3]:
        r["error"] = f"grid mismatch image{img.shape[:3]} lesion{les.shape} envelope{env.shape}"
        return r

    lab = compose_label(env, les)
    n_les = int((lab == 2).sum())
    n_panc = int((lab == 1).sum())
    r["env_vox"] = int(env.sum(dtype=np.int64))
    r["lesion_vox"] = n_les
    r["panc_only_vox"] = n_panc
    r["fg_vox"] = n_panc + n_les
    r["panc_frac"] = r["fg_vox"] / lab.size
    r["lesion_frac"] = n_les / lab.size
    if r["panc_frac"] > MAX_PANCREAS_FRACTION:
        r["error"] = f"pancreas region {r['panc_frac']:.1%} > {MAX_PANCREAS_FRACTION:.0%} of scan"
        return r
    if r["lesion_frac"] > MAX_LESION_FRACTION:
        r["error"] = f"lesion {r['lesion_frac']:.1%} > {MAX_LESION_FRACTION:.0%} of scan"
        return r

    if dst.exists() and not force:
        r["label_how"] = "exists"
        return r

    affine = img.affine
    out = nib.Nifti1Image(lab, affine)        # fresh header -> scl_slope=1, scl_inter=0
    out.header.set_zooms(img.header.get_zooms()[:3])
    out.set_data_dtype(np.uint8)
    out.set_qform(affine, code=1)
    out.set_sform(affine, code=1)
    tmp = dst.with_suffix(".tmp.nii.gz")
    nib.save(out, tmp)
    os.replace(tmp, dst)
    r["label_how"] = "written"

    # geometry must match the image exactly, re-read from disk
    chk = nib.load(dst)
    if chk.shape != img.shape[:3] or not np.allclose(chk.affine, affine, atol=1e-6):
        r["error"] = "written label geometry does not match the image"
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rewrite labels that already exist")
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("BUILD502_NPROC", "3")))
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (smoke test)")
    a = ap.parse_args()

    ids = tio.case_ids()
    if a.limit:
        ids = ids[: a.limit]
    IMAGES_TR.mkdir(parents=True, exist_ok=True)
    LABELS_TR.mkdir(parents=True, exist_ok=True)
    print(f"building {DATASET_NAME} from {len(ids)} cases -> {RAW_DIR}  (nproc {a.nproc})", flush=True)

    rows: list[dict] = []
    with Pool(a.nproc) as pool:
        for n, r in enumerate(pool.imap_unordered(build_one, [(c, a.force) for c in ids], chunksize=1), 1):
            rows.append(r)
            if n % 100 == 0 or n == len(ids):
                ne = sum(1 for x in rows if x["error"])
                print(f"  {n}/{len(ids)}  errors: {ne}", flush=True)

    errs = [r for r in rows if r["error"]]
    how = {}
    for r in rows:
        how[r["image_how"]] = how.get(r["image_how"], 0) + 1
    print(f"imagesTr: {len(list(IMAGES_TR.glob('*_0000.nii.gz')))}   "
          f"labelsTr: {len(list(LABELS_TR.glob('*.nii.gz')))}   image sources: {how}")
    empty_env = sorted(r["id"] for r in rows if not r["error"] and r.get("env_vox", 1) == 0)
    print(f"cases with an empty gland envelope: {len(empty_env)} {empty_env}")

    if errs:
        print(f"ERRORS ({len(errs)}):")
        for r in errs[:30]:
            print(f"  {r['id']}: {r['error']}")
        sys.exit(1)

    if not a.limit:
        (RAW_DIR / "dataset.json").write_text(json.dumps(dataset_json(len(ids)), indent=4))
        print(f"wrote {RAW_DIR / 'dataset.json'}")


if __name__ == "__main__":
    main()
