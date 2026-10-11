"""S8 per-case pass (server, where the OOF exports live). One pickle per (model, case) in
work/nnunet_multiclass/cases/{model}/, resumable. Both models run through core.evaluate_case().

Inputs per case (fold = the case's own out-of-fold model, fold_k_ids.txt <-> fold k-1):
  seg   fold_f/validation/{case}.nii.gz      (501 {0,1}; 502 {0,1,2})
  P     fold_f/validation/{case}.npz         'probabilities' (2, z, y, x) float32
  gt    Dataset501 gt_segmentations          lesion GT for BOTH models (S7: 502 label 2 == 501 label, 0 voxels differ;
                                             re-asserted here per case)
  gland Dataset502 gt_segmentations > 0      502 only, pancreas-region Dice
  sr    tumorlib.search_region (server duct_masks), spacing from the S2 ctgrid symlink farm (gt headers)
  excluded lesion_ids: master_lesions.csv rows with excluded == True (S1 `separated`)

Usage (server, repo root):
  PYTHONPATH=src ~/nnunet_setup/.venv/bin/python -m multiclass_eval.case_pass --cases PanTS_00000771 PanTS_00003513 PanTS_00000020
  PYTHONPATH=src ~/nnunet_setup/.venv/bin/python -m multiclass_eval.case_pass --workers 5
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from multiclass_eval import core as K
from multiclass_eval.core import C
from tumorlib import envelope, io as tio

WORK = K.REPO / "work" / "nnunet_multiclass"
CASE_DIR = WORK / "cases"
RESULTS = Path.home() / "research/nnunet_env/nnUNet_results"
PRED_ROOT = {
    "501": RESULTS / "Dataset501_PanTSTumor/nnUNetTrainer__nnUNetPlans__3d_fullres",
    "502": RESULTS / "Dataset502_PanTSPancLesion/nnUNetTrainer__nnUNetPlans__3d_fullres",
}
GT502_DIR = Path.home() / "research/nnunet_env/nnUNet_preprocessed/Dataset502_PanTSPancLesion/gt_segmentations"
MASTER = K.REPO / "results/master_lesion_table/master_lesions.csv"


def load_label_map(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """nnU-Net-written uint8 label map (seg or gt_segmentations), nibabel order, values untouched."""
    img = nib.load(path)
    arr = np.asarray(img.dataobj)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{path}: non-integer label map {arr.dtype}")
    return arr.astype(np.uint8), img.affine


def process(job: tuple[str, str, int, tuple[int, ...]]) -> tuple[str, str, str]:
    model, case_id, fold, excluded = job
    out = CASE_DIR / model / f"{case_id}.pkl"
    if out.exists():
        return model, case_id, "cached"
    t0 = time.time()
    try:
        v = PRED_ROOT[model] / f"fold_{fold}" / "validation"
        sp = tio.spacing(case_id)
        sr = envelope.search_region(case_id)
        seg, seg_aff = load_label_map(v / f"{case_id}.nii.gz")
        gt, gt_aff = C.load_gt(case_id)
        gt_gland = None
        if model == "502":
            g502, _ = load_label_map(GT502_DIR / f"{case_id}.nii.gz")
            if not np.array_equal(g502 == 2, gt.astype(bool)):
                return model, case_id, "error: 502 GT label 2 != 501 GT"
            if not set(np.unique(g502)) <= {0, 1, 2}:
                return model, case_id, f"error: 502 GT labels {np.unique(g502)}"
            gt_gland = (g502 > 0).astype(np.uint8)
            del g502
        with np.load(v / f"{case_id}.npz") as d:
            P = d["probabilities"]
        if P.ndim != 4 or P.shape[0] != 2:
            return model, case_id, f"error: npz shape {P.shape}"
        res = K.evaluate_case(model, case_id, fold, seg, gt, P, sr, sp, list(excluded), gt_gland)
        res["case"].update(npz_dtype=str(P.dtype), seg_gt_affine_close=bool(np.allclose(seg_aff, gt_aff, atol=1e-4)),
                           seconds=round(time.time() - t0, 1))
        del P
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.rename(out)
        return model, case_id, f"ok {res['case']['seconds']}s"
    except Exception as e:
        traceback.print_exc()
        return model, case_id, f"error {type(e).__name__}: {e}"


def jobs(models, cases=None) -> list[tuple[str, str, int, tuple[int, ...]]]:
    fold = C.fold_of_case()
    m = pd.read_csv(MASTER)
    excl = m[m.excluded].groupby("case_id").lesion_id.apply(lambda s: tuple(sorted(int(x) for x in s))).to_dict()
    ids = cases or sorted(fold)
    # interleave models so a partial run is still paired
    return [(mo, c, fold[c], excl.get(c, ())) for c in ids for mo in models]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--models", nargs="*", default=list(K.MODELS))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    js = jobs(args.models, args.cases)
    print(f"{len(js)} jobs, {args.workers} workers -> {CASE_DIR}", flush=True)
    t0 = time.time()
    bad = []
    with mp.Pool(args.workers, maxtasksperchild=10) as pool:
        for i, (mo, cid, status) in enumerate(pool.imap_unordered(process, js, chunksize=1), 1):
            if not (status.startswith("ok") or status == "cached"):
                bad.append((mo, cid, status))
                print(f"  {mo} {cid}: {status}", flush=True)
            if i % 50 == 0 or i == len(js) or len(js) <= 10:
                print(f"  {i}/{len(js)} ({time.time() - t0:.0f}s) last {mo} {cid} {status}", flush=True)
    print(f"done; {len(bad)} failures: {bad}")


if __name__ == "__main__":
    main()
