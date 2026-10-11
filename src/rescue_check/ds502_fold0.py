"""S11 optional step: which Dataset501-missed lesions does Dataset502 (multi-class, fold 0 only) find?

Reads ONLY fold_0/validation/{case}.nii.gz from the server (rsync pull: a read on the server, nothing is
written there, no GPU). Lesion prediction = label == 2. A lesion is DETECTED by 502 when any label-2
voxel overlaps it (the same touch rule as 501's detected_any). GT lesions come from the PC masks via
tumorlib (lesion_id = tumorlib.label_lesions, nibabel order) and are checked against master gt_vox.

Fold 0 only (264 cases, fold_1_ids.txt): n is small, and 502 has only one fold trained.

Output: work/rescue_check/ds502_fold0_lesions.csv (fold-0 lesions: case_id, lesion_id, gt_vox, detected_502) and
ds502_fold0_cases.csv (any_label2, n_comp_502, n_fp_502 per case). analyze.py picks them up if present.

Usage (repo root, PC): PYTHONPATH=src .venv/bin/python -m rescue_check.ds502_fold0 [--fetch] [--workers 3]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from tumorlib import io as tio
from scipy.ndimage import label as cc_label

from tumorlib.lesions import CC_STRUCT, bbox_slices, label_lesions

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work" / "rescue_check"
SEG_DIR = WORK / "ds502_fold0_seg"
SERVER = "rahul@deep-server.tail8e65db.ts.net"
REMOTE = ("research/nnunet_env/nnUNet_results/Dataset502_PanTSPancLesion/"
          "nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/")
LESION_LABEL = 2


def fetch() -> None:
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a", "--include=*.nii.gz", "--exclude=*", f"{SERVER}:{REMOTE}", f"{SEG_DIR}/"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def process(case_id: str) -> list[dict] | str:
    img = nib.load(SEG_DIR / f"{case_id}.nii.gz")
    pred = np.asarray(img.dataobj).astype(np.uint8)
    g = tio.grid(case_id)
    if g is None or tuple(pred.shape) != tuple(g[0]):
        return f"{case_id}: pred shape {pred.shape} != CT grid {None if g is None else g[0]}"
    if not np.allclose(img.affine, g[1], atol=1e-3):
        print(f"{case_id}: WARNING 502 seg affine differs from CT affine (grid shapes agree)")
    les = tio.load_mask(case_id, "pancreatic_lesion")
    if les is None:
        return f"{case_id}: lesion mask unreadable"
    labels, n = label_lesions(les)
    p = pred == LESION_LABEL
    rows = []
    for j in range(1, n + 1):
        sl = bbox_slices(labels == j)
        m = labels[sl] == j
        rows.append(dict(case_id=case_id, lesion_id=j, gt_vox=int(m.sum()), detected_502=bool(p[sl][m].any())))
    # label-2 components (26-conn) and how many touch no GT lesion (same FP rule as S2's touch rule)
    n_comp = n_fp = 0
    psl = bbox_slices(p)
    if psl is not None:
        cl, n_comp = cc_label(p[psl], structure=CC_STRUCT)
        touched = set(np.unique(cl[labels[psl] > 0])) - {0}
        n_fp = n_comp - len(touched)
    labels_seen = sorted(int(v) for v in np.unique(pred))
    rows.append(dict(case_id=case_id, lesion_id=0, gt_vox=0, detected_502=bool(p.any()),
                     labels=";".join(map(str, labels_seen)), n_comp_502=int(n_comp), n_fp_502=int(n_fp)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="rsync fold_0/validation/*.nii.gz from the server first")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    if args.fetch:
        fetch()
    cases = tio.fold_ids(1)   # fold_1_ids.txt <-> nnU-Net fold 0
    missing = [c for c in cases if not (SEG_DIR / f"{c}.nii.gz").exists()]
    assert not missing, f"{len(missing)} fold-0 predictions missing, e.g. {missing[:3]}"
    with mp.Pool(args.workers, maxtasksperchild=20) as pool:
        res = pool.map(process, cases, chunksize=4)
    bad = [r for r in res if isinstance(r, str)]
    assert not bad, bad
    allrows = pd.DataFrame([r for rs in res for r in rs])
    case_rows, out = allrows[allrows.lesion_id == 0], allrows[allrows.lesion_id > 0].drop(columns=["labels", "n_comp_502", "n_fp_502"])
    print(f"{len(cases)} fold-0 cases; label values seen: {sorted(set(';'.join(case_rows['labels']).split(';')))}; "
          f"cases with any label-2 voxel: {int(case_rows.detected_502.sum())}")
    master = pd.read_csv(REPO / "results/master_lesion_table/master_lesions.csv")
    m0 = master[master.fold == 0]
    j = m0[["case_id", "lesion_id", "gt_vox"]].merge(out, on=["case_id", "lesion_id"], how="outer",
                                                     suffixes=("", "_502"), indicator=True)
    assert (j._merge == "both").all() and (j.gt_vox == j.gt_vox_502).all(), "fold-0 lesion join / gt_vox mismatch"
    print(f"{len(out)} fold-0 lesions join master with equal gt_vox")
    WORK.mkdir(parents=True, exist_ok=True)
    out.to_csv(WORK / "ds502_fold0_lesions.csv", index=False)
    case_rows[["case_id", "detected_502", "labels", "n_comp_502", "n_fp_502"]].rename(columns={"detected_502": "any_label2"}).to_csv(
        WORK / "ds502_fold0_cases.csv", index=False)
    print(f"wrote {WORK / 'ds502_fold0_lesions.csv'}")


if __name__ == "__main__":
    main()
