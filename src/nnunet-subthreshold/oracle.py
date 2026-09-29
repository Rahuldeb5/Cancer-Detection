"""Step 0 oracle (session S2): check the nnU-Net probability files on 3 known cases
before trusting them on 1308.

Cases (chosen from master_lesions.csv):
  PanTS_00000771  fold 0, one 40.7 mm lesion, OOF lesion Dice 0.921 (large, found)
  PanTS_00003513  fold 0, one 9.4 mm lesion, missed (detected_any False), 1.0 mm slices
  PanTS_00000020  fold 0, tumor-negative; carried-over max tumor prob 0.9994

Checks per case:
  a. npz 'probabilities' is (2, z, y, x); channel 1 .transpose(2,1,0) has the .nii shape,
     and seg / gt / pancreas-mask affines agree
  b. argmax over channels (transposed) equals the saved segmentation, voxel for voxel
  c. p0 + p1 == 1 (max abs deviation)
  d. alignment known answer: lesion Dice recomputed from the saved seg equals the master
     table; mean p1 inside GT under the transpose vs under each single-axis flip
     (a wrong orientation would not put the probability mass on the tumor)
  e. float ceiling: dtype, how many voxels sit at p1 == 1.0 exactly, smallest p0

Usage (server, repo root): ~/nnunet_setup/.venv/bin/python src/nnunet-subthreshold/oracle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from tumorlib import envelope, io as tio  # noqa: E402
from tumorlib.lesions import label_lesions  # noqa: E402

CASES = ("PanTS_00000771", "PanTS_00003513", "PanTS_00000020")


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int(np.logical_and(a, b).sum()) / s if s else float("nan")


def check(case_id: str, fold: int, master: pd.DataFrame) -> None:
    print(f"\n=== {case_id} (fold {fold})")
    seg, seg_aff = C.load_seg(case_id, fold)
    gt, gt_aff = C.load_gt(case_id)
    P = C.load_probs(case_id, fold)
    p1, p0 = C.xyz(P, 1), C.xyz(P, 0)
    panc_aff = C.nib.load(Path(C.os.environ["TUMORLIB_MASK_ROOT"]) / case_id / "segmentations/pancreas.nii.gz").affine

    # a. layout
    print(f"a. npz shape {P.shape} dtype {P.dtype}; channel-1 transposed {p1.shape}; seg {seg.shape}; gt {gt.shape}")
    assert P.ndim == 4 and P.shape[0] == 2, "expected 2 channels"
    assert p1.shape == seg.shape == gt.shape, "transposed channel 1 must match the .nii shape"
    print(f"   affines equal: seg~gt {np.allclose(seg_aff, gt_aff, atol=1e-4)}, "
          f"seg~pancreas mask {np.allclose(seg_aff, panc_aff, atol=1e-3)}")
    sp = tio.spacing(case_id)
    row = master[master.case_id == case_id]
    if len(row):
        ms = row[["spacing_x_mm", "spacing_y_mm", "spacing_z_mm"]].iloc[0].to_numpy(float)
        print(f"   spacing (gt header via tumorlib) {np.round(sp, 6)} vs master {ms}: max diff {np.abs(np.array(sp) - ms).max():.2e} mm")

    # b + c + e, in z-chunks of the stored array (contiguous blocks)
    mism = 0
    maxdev = 0.0
    n1 = n0zero = 0
    min_p0 = np.inf
    for s in range(0, P.shape[1], 32):
        blk = P[:, s:s + 32]
        am = (np.argmax(blk, axis=0) == 1).transpose(2, 1, 0)
        mism += int((am != seg[:, :, s:s + 32].astype(bool)).sum())
        maxdev = max(maxdev, float(np.abs(blk[0].astype(np.float64) + blk[1] - 1.0).max()))
        n1 += int((blk[1] == 1.0).sum())
        n0zero += int((blk[0] == 0.0).sum())
        pos = blk[0][blk[0] > 0]
        if pos.size:
            min_p0 = min(min_p0, float(pos.min()))
    print(f"b. argmax != saved seg: {mism} of {seg.size} voxels (seg has {int(seg.sum())} fg voxels)")
    print(f"c. max |p0 + p1 - 1| = {maxdev:.3e}")
    print(f"e. max p1 = {float(p1.max()):.9f}; voxels with p1 == 1.0 exactly: {n1}; p0 == 0: {n0zero}; "
          f"smallest positive p0 = {min_p0:.3e}  (float32 1 - eps/2 = {1 - np.finfo(np.float32).epsneg:.9f})")
    thr05 = int(((p1 > 0.5) != seg.astype(bool)).sum())
    print(f"   (p1 > 0.5) != saved seg: {thr05} voxels")

    # d. alignment
    labels, n = label_lesions(gt)
    print(f"d. GT lesions: {n}; gt_vox {[int((labels == j).sum()) for j in range(1, n + 1)]}; "
          f"master gt_vox {row.sort_values('lesion_id').gt_vox.tolist()}")
    if n:
        g = gt.astype(bool)
        print(f"   Dice(saved seg, GT) = {dice(seg.astype(bool), g):.4f}; seg touches GT: {bool((seg.astype(bool) & g).any())}; "
              f"master lesion_dice {row.lesion_dice.round(4).tolist()}")
        from scipy.ndimage import label as cc_label
        from tumorlib.lesions import CC_STRUCT
        slab, _ = cc_label(seg.astype(bool), structure=CC_STRUCT)
        for j in range(1, n + 1):
            les = labels == j
            touching = np.isin(slab, np.unique(slab[les & (slab > 0)]))
            print(f"   lesion {j}: Dice vs union of touching predicted components = {dice(touching, les):.4f} (master definition)")
        mean_in = float(p1[g].mean())
        flips = {ax: float(np.flip(p1, axis=ax)[g].mean()) for ax in range(3)}
        print(f"   mean p1 inside GT: transposed {mean_in:.4g}; flipped x {flips[0]:.4g}, y {flips[1]:.4g}, z {flips[2]:.4g}")
        peak = float(p1[g].max())
        print(f"   peak p1 inside GT {peak:.4g} (clipped logit {C.clip_logit(peak):.2f}); "
              f"raw logit max inside GT {C.raw_logit(p1[g], p0[g]).max():.2f}")
    else:
        print(f"   negative case: max p1 = {float(p1.max()):.4f} (carried over: 0.9994); seg fg voxels {int(seg.sum())}")

    sr = envelope.search_region(case_id)
    if sr is not None:
        srb = sr.astype(bool)
        print(f"   search region: {int(srb.sum())} voxels; GT voxels inside it {int((srb & gt.astype(bool)).sum())}/{int(gt.sum())}; "
              f"max p1 in it {float(p1[srb].max()) if srb.any() else float('nan'):.4g}")


def main() -> None:
    fold = C.fold_of_case()
    master = pd.read_csv(C.REPO / "results/master_lesion_table/master_lesions.csv")
    for cid in CASES:
        check(cid, fold[cid], master)


if __name__ == "__main__":
    main()
