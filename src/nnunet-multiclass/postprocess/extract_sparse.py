"""S13 step 0 (server, where the Dataset502 OOF exports live): one small sparse cache per case, so the PC never
needs the ~100 GB of float32 npz files. Pure data movement; every decision is made on the PC (core.py).

Per case -> OUT/{case}.npz (all indices are flat C-order indices into the nibabel (x, y, z) volume):
  shape, spacing (tumorlib affine spacing, the same values S8 used), fold
  p_idx (uint32), p_val (float32)   lesion sigmoid (npz channel 1) at every voxel with p > FLOOR (1e-4)
  panc_idx (uint32)                 predicted pancreas region: npz channel 0 > 0.5
  gt_idx (uint32), gt_lab (uint16)  lesion GT (Dataset501 gt_segmentations, as S8) labelled by tumorlib label_lesions
  n_gt05, vol_peak_p                checks: #(p > 0.5) and the whole-volume peak, compared with S8's pickles on the PC

FLOOR: the lowest t_low on the S13 grid is 1e-4. 502's probabilities come from fp16 logits (relative p step 0.78% near
1e-4, measured on 4 cases), so float32 values resolve p > 1e-4 without ties at the threshold.

Usage (server, repo root):
  PYTHONPATH=src ~/nnunet_setup/.venv/bin/python src/nnunet-multiclass/postprocess/extract_sparse.py --cases PanTS_00000020
  PYTHONPATH=src ~/nnunet_setup/.venv/bin/python src/nnunet-multiclass/postprocess/extract_sparse.py --workers 4
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "nnunet-subthreshold"))
import common as C  # noqa: E402  (sets the server tumorlib env: duct_masks + ctgrid symlink farm)

from tumorlib import io as tio  # noqa: E402
from tumorlib.lesions import label_lesions  # noqa: E402

FLOOR = 1e-4
PRED_ROOT = Path.home() / "research/nnunet_env/nnUNet_results/Dataset502_PanTSPancLesion/nnUNetTrainer__nnUNetPlans__3d_fullres"
OUT = Path.home() / "research/nnunet_probs_502"


def flat_xyz(mask_zyx: np.ndarray, shape_xyz) -> np.ndarray:
    """Flat C-order indices into the (x, y, z) volume of the True voxels of a (z, y, x) array."""
    z, y, x = np.nonzero(mask_zyx)
    return np.ravel_multi_index((x, y, z), shape_xyz).astype(np.uint32)


def process(job: tuple[str, int]) -> tuple[str, str]:
    case_id, fold = job
    out = OUT / f"{case_id}.npz"
    if out.exists():
        return case_id, "cached"
    t0 = time.time()
    try:
        sp = tio.spacing(case_id)
        gt, _ = C.load_gt(case_id)
        labels, n = label_lesions(gt)
        with np.load(PRED_ROOT / f"fold_{fold}" / "validation" / f"{case_id}.npz") as d:
            P = d["probabilities"]
        if P.ndim != 4 or P.shape[0] != 2 or P.dtype != np.float32:
            return case_id, f"error: npz {P.shape} {P.dtype}"
        shape = tuple(int(s) for s in P.shape[:0:-1])  # (x, y, z)
        if shape != gt.shape or int(np.prod(shape)) >= 2 ** 32:
            return case_id, f"error: shape {shape} vs gt {gt.shape}"
        p = P[1]
        m = p > FLOOR
        z, y, x = np.nonzero(m)
        p_idx = np.ravel_multi_index((x, y, z), shape).astype(np.uint32)
        p_val = p[z, y, x].astype(np.float32)
        panc_idx = flat_xyz(P[0] > 0.5, shape)
        gx = np.nonzero(labels)
        gt_idx = np.ravel_multi_index(gx, shape).astype(np.uint32)
        gt_lab = labels[gx].astype(np.uint16)
        n05, peak = int((p > 0.5).sum()), float(p.max())
        del P, p, m
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, case_id=np.array(case_id), fold=np.array(fold), shape=np.array(shape),
                            spacing=np.asarray(sp, float), p_idx=p_idx, p_val=p_val, panc_idx=panc_idx,
                            gt_idx=gt_idx, gt_lab=gt_lab, n_lesions=np.array(n), n_gt05=np.array(n05),
                            vol_peak_p=np.array(peak), floor=np.array(FLOOR))
        tmp.rename(out)
        return case_id, f"ok {time.time() - t0:.1f}s p>{FLOOR:g} {p_idx.size} panc {panc_idx.size}"
    except Exception as e:
        traceback.print_exc()
        return case_id, f"error {type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    fold = C.fold_of_case()
    ids = args.cases or sorted(fold)
    jobs = [(c, fold[c]) for c in ids]
    print(f"{len(jobs)} cases, {args.workers} workers -> {OUT}", flush=True)
    t0, bad = time.time(), []
    with mp.Pool(min(args.workers, 4), maxtasksperchild=20) as pool:
        for i, (cid, status) in enumerate(pool.imap_unordered(process, jobs, chunksize=1), 1):
            if not (status.startswith("ok") or status == "cached"):
                bad.append((cid, status))
                print(f"  {cid}: {status}", flush=True)
            if i % 50 == 0 or i == len(jobs) or len(jobs) <= 10:
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}s) last {cid} {status}", flush=True)
    print(f"done; {len(bad)} failures: {bad}")


if __name__ == "__main__":
    main()
