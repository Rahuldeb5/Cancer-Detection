"""One row per nnU-Net predicted tumor component, with the features needed to test
whether duct geometry helps rank/suppress components beyond nnU-Net's own scores.

Runs on the deep-server (predictions live there). Inputs:
  PRED_ROOT/fold_k/validation/<case>.nii.gz + .npz   nnU-Net out-of-fold predictions
  GT_DIR/<case>.nii.gz                                 GT tumor labels
  MASK_ROOT/<case>/segmentations/*.nii.gz              ducts + pancreas (+head/body/tail), copied from the PC
  PC_RESULTS/duct_cutoff/{cutoff_per_duct.csv,profiles/}   oriented duct profiles from src/duct-cutoff/main.py
  PC_RESULTS/geometry_results/duct_{mask_quality,dilation_flags}.csv   Session 2 duct flags

Output: results/component_features/components.csv (+ skipped.csv)

Feature groups (the eval compares models built on each):
  nnunet   -- vol, max/mean prob, rank within case, #components in case
  pancreas -- overlap with / distance to pancreas GT, head/body/tail fractions
  duct     -- distance to MPD/CBD masks, to the oriented MPD head end, to the dilated
              MPD segment, to a detected cutoff; case-level dilation flags
Label: is_tp = component overlaps any GT tumor voxel.

Usage (repo root on the server):
  ~/nnunet_setup/.venv/bin/python src/duct-cutoff/component_features.py --workers 8
  ... --limit 20      # smoke test
"""
import argparse
import multiprocessing as mp
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, generate_binary_structure
from scipy.ndimage import label as cc_label

HOME = Path.home()
PRED_ROOT = HOME / "research/nnunet_env/nnUNet_results/Dataset501_PanTSTumor/nnUNetTrainer__nnUNetPlans__3d_fullres"
GT_DIR = HOME / "research/nnunet_env/nnUNet_preprocessed/Dataset501_PanTSTumor/gt_segmentations"
MASK_ROOT = HOME / "research/datasets/pants/duct_masks"
PC_RESULTS = HOME / "research/duct_cutoff_pc"
OUT_DIR = Path("results/component_features")

CC = generate_binary_structure(3, 3)
MASKS = ("pancreatic_duct", "common_bile_duct", "pancreas", "pancreas_head", "pancreas_body", "pancreas_tail")
DILATE_MM = {"mpd": 3.21, "cbd": 8.05}
CROP_MARGIN_VOX = 10


def load_bool(path: Path) -> np.ndarray:
    raw = nib.load(path).dataobj.get_unscaled()
    if np.issubdtype(raw.dtype, np.floating):
        raw = np.nan_to_num(raw, nan=0.0)
    return raw > 0.5


def dist_field(target: np.ndarray, sp) -> np.ndarray | None:
    """mm distance to the nearest target voxel. Exact inside the crop because every
    target voxel is inside it (the crop bbox includes all masks)."""
    return distance_transform_edt(~target, sampling=sp) if target.any() else None


def min_dist_to_points(comp_world: np.ndarray, pts_world: np.ndarray) -> float:
    if pts_world is None or len(pts_world) == 0:
        return np.nan
    # component voxels can be many; subsample for the point-set distance
    if len(comp_world) > 4000:
        comp_world = comp_world[np.linspace(0, len(comp_world) - 1, 4000).astype(int)]
    best = np.inf
    for chunk in np.array_split(pts_world, max(1, len(pts_world) // 256)):
        d = np.linalg.norm(comp_world[:, None, :] - chunk[None, :, :], axis=2).min()
        best = min(best, d)
    return float(best)


def load_profile(case_id: str, duct: str):
    f = PC_RESULTS / "duct_cutoff/profiles" / f"{case_id}_{duct}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z["path_world_mm"], z["caliber"]


def process(job) -> tuple[list[dict], dict | None]:
    fold, case_id, case_meta = job
    try:
        vdir = PRED_ROOT / f"fold_{fold}" / "validation"
        pimg = nib.load(vdir / f"{case_id}.nii.gz")
        pred = np.asarray(pimg.dataobj) > 0
        sp = tuple(float(s) for s in nib.affines.voxel_sizes(pimg.affine))
        gt = load_bool(GT_DIR / f"{case_id}.nii.gz")
        if gt.shape != pred.shape:
            return [], {"case_id": case_id, "reason": f"gt shape {gt.shape} != pred {pred.shape}"}
        if not pred.any():
            return [], {"case_id": case_id, "reason": "no predicted voxels", "gt_tumor": int(gt.any())}

        prob = np.load(vdir / f"{case_id}.npz")["probabilities"][1].transpose(2, 1, 0)
        if prob.shape != pred.shape:
            return [], {"case_id": case_id, "reason": f"prob shape {prob.shape} != pred {pred.shape}"}

        seg = MASK_ROOT / case_id / "segmentations"
        mask_affine = nib.load(seg / "common_bile_duct.nii.gz").affine  # same affine loader.py used for profiles/cutoffs
        masks = {n: load_bool(seg / f"{n}.nii.gz") for n in MASKS}
        masks_ok = all(m.shape == pred.shape for m in masks.values())

        # crop everything to one bbox containing prediction + all masks (EDT memory)
        roi = pred.copy()
        if masks_ok:
            for m in masks.values():
                roi |= m
        idx = np.argwhere(roi)
        lo = np.maximum(idx.min(0) - CROP_MARGIN_VOX, 0)
        hi = np.minimum(idx.max(0) + 1 + CROP_MARGIN_VOX, pred.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        pred, gt, prob = pred[sl], gt[sl], prob[sl]
        if masks_ok:
            masks = {n: m[sl] for n, m in masks.items()}

        comp_lab, n_comp = cc_label(pred, structure=CC)
        gt_lab, n_gt = cc_label(gt, structure=CC)
        gt_sizes = np.bincount(gt_lab.ravel())
        vox_mm3 = float(np.prod(sp))

        d_panc = d_mpd = d_cbd = None
        if masks_ok:
            d_panc = dist_field(masks["pancreas"], sp)
            d_mpd = dist_field(masks["pancreatic_duct"], sp)
            d_cbd = dist_field(masks["common_bile_duct"], sp)

        prof = {d: load_profile(case_id, d) for d in ("mpd", "cbd")}
        to_world = lambda ijk: (ijk + lo) @ mask_affine[:3, :3].T + mask_affine[:3, 3]  # noqa: E731

        rows = []
        for k in range(1, n_comp + 1):
            comp = comp_lab == k
            ijk = np.argwhere(comp)
            p = prob[comp]
            ov = gt_lab[comp]
            hit_labels = np.unique(ov[ov > 0])
            r = dict(
                case_id=case_id, fold=fold, comp_id=k, n_comp_in_case=n_comp,
                gt_tumor=int(n_gt > 0),
                n_vox=int(comp.sum()), vol_mm3=comp.sum() * vox_mm3,
                max_prob=float(p.max()), mean_prob=float(p.mean()),
                p90_prob=float(np.percentile(p, 90)),
                is_tp=int(hit_labels.size > 0),
                overlap_frac=float((ov > 0).mean()),
                matched_gt_diam_mm=float(max(
                    (6 * gt_sizes[h] * vox_mm3 / np.pi) ** (1 / 3) for h in hit_labels
                )) if hit_labels.size else np.nan,
                masks_ok=masks_ok,
            )
            if masks_ok:
                inside = masks["pancreas"][comp]
                r.update(
                    frac_in_pancreas=float(inside.mean()),
                    dist_to_pancreas_mm=float(d_panc[comp].min()) if d_panc is not None else np.nan,
                    frac_head=float(masks["pancreas_head"][comp].mean()),
                    frac_body=float(masks["pancreas_body"][comp].mean()),
                    frac_tail=float(masks["pancreas_tail"][comp].mean()),
                    dist_to_mpd_mm=float(d_mpd[comp].min()) if d_mpd is not None else np.nan,
                    dist_to_cbd_mm=float(d_cbd[comp].min()) if d_cbd is not None else np.nan,
                )
                cw = to_world(ijk)
                for duct in ("mpd", "cbd"):
                    pr = prof[duct]
                    if pr is None:
                        continue
                    path_w, cal = pr
                    r[f"dist_to_{duct}_head_end_mm"] = min_dist_to_points(cw, path_w[:1])
                    dil = np.isfinite(cal) & (cal >= DILATE_MM[duct])
                    r[f"dist_to_{duct}_dilated_seg_mm"] = min_dist_to_points(cw, path_w[dil]) if dil.any() else np.nan
                for duct in ("mpd", "cbd"):
                    cx = case_meta.get(f"{duct}_cutoff_xyz")
                    if cx is not None:
                        r[f"dist_to_{duct}_cutoff_mm"] = min_dist_to_points(cw, np.asarray([cx]))
            r.update({k2: v for k2, v in case_meta.items() if not k2.endswith("_xyz")})
            rows.append(r)

        # rank within case by nnU-Net's own confidence
        order = np.argsort([-r["max_prob"] for r in rows])
        for rank, i in enumerate(order, 1):
            rows[i]["rank_in_case"] = rank
        return rows, None
    except Exception as e:
        traceback.print_exc()
        return [], {"case_id": case_id, "reason": f"error:{type(e).__name__}: {e}"}


def case_metadata() -> dict[str, dict]:
    q = pd.read_csv(PC_RESULTS / "geometry_results/duct_mask_quality.csv").set_index("case_id")
    fl = pd.read_csv(PC_RESULTS / "geometry_results/duct_dilation_flags.csv").set_index("case_id")
    cd = pd.read_csv(PC_RESULTS / "duct_cutoff/cutoff_per_duct.csv")
    meta = {}
    for cid in q.index:
        m = dict(
            mpd_status_s2=q.at[cid, "mpd_status"], cbd_status_s2=q.at[cid, "cbd_status"],
            mpd_dilated=bool(fl.at[cid, "mpd_dilated"]), cbd_dilated=bool(fl.at[cid, "cbd_dilated"]),
            double_duct_dilated=bool(fl.at[cid, "double_duct_dilated"]),
            mpd_p95_mm=fl.at[cid, "mpd_p95_mm"], cbd_p99_mm=fl.at[cid, "cbd_p99_mm"],
        )
        for duct in ("mpd", "cbd"):
            r = cd[(cd.case_id == cid) & (cd.duct == duct)]
            if len(r):
                r = r.iloc[0]
                m[f"{duct}_pipeline_status"] = r.status
                m[f"{duct}_len_mm"] = r.total_len_mm
                m[f"{duct}_head_caliber_mm"] = r.get("caliber_head_mm", np.nan)
                m[f"{duct}_head_end_dilated"] = bool(
                    pd.notna(r.get("caliber_head_mm")) and r.caliber_head_mm >= DILATE_MM[duct])
                m[f"{duct}_cutoff_detected"] = bool(r.detected)
                if r.detected:
                    m[f"{duct}_cutoff_xyz"] = (r.cutoff_x_mm, r.cutoff_y_mm, r.cutoff_z_mm)
        meta[cid] = m
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    meta = case_metadata()
    jobs = []
    for fold in range(5):
        for f in sorted((PRED_ROOT / f"fold_{fold}" / "validation").glob("*.nii.gz")):
            cid = f.name[: -len(".nii.gz")]
            jobs.append((fold, cid, meta.get(cid, {})))
    if args.limit:
        jobs = jobs[:: max(1, len(jobs) // args.limit)][: args.limit]
    print(f"{len(jobs)} cases, {args.workers} workers", flush=True)

    rows, skipped = [], []
    with mp.Pool(args.workers, maxtasksperchild=20) as pool:
        for i, (r, s) in enumerate(pool.imap_unordered(process, jobs), 1):
            rows.extend(r)
            if s:
                skipped.append(s)
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} cases, {len(rows)} components", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["case_id", "comp_id"])
    df.to_csv(args.out / "components.csv", index=False)
    pd.DataFrame(skipped).to_csv(args.out / "skipped.csv", index=False)
    print(f"wrote {args.out / 'components.csv'} ({len(df)} components from {df.case_id.nunique()} cases); "
          f"skipped {len(skipped)} cases")
    if len(df):
        print(df.groupby("gt_tumor").agg(cases=("case_id", "nunique"), comps=("comp_id", "size"),
                                         tp_frac=("is_tp", "mean"), masks_ok=("masks_ok", "mean")).to_string())


if __name__ == "__main__":
    main()
