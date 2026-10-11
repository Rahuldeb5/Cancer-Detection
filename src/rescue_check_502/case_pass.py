"""S14 step 1 (PC, CPU): one pickle per case in work/rescue_check_502/cases/ (resumable).

Inputs (read-only):
  ~/research/nnunet_probs_502/{case}.npz        S13 sparse cache: lesion p > 1e-4, predicted pancreas (ch0 > 0.5),
                                                tumorlib-labelled GT lesions
  work/nnunet_postprocess/atoms/{case}.pkl      S13 atoms -> raw 502 and 502 + D per-lesion scores (S13 core.evaluate)
  PC masks via tumorlib                         search region (placement domain), pancreatic_duct (MPD)

Per case: raw / D case rows (FP components, any prediction). Per lesion (tumor+ only):
  raw / D lesion Dice, touch, predicted-volume ratio
  predicted-pancreas metrics (core.lesion_panc_metrics) and category
  nnU-Net OP: the case's t = 0.01 components of rank <= 3, each one's placement fraction f_i, chance, observed
    hit at the identity placement (must equal the real touch), Dice of the touching OP components (descriptive)
  duct: MPD present, placement fraction g of "within 10 mm of the MPD", observed at the identity placement

Usage (PC, repo root):
  PYTHONPATH=src .venv/bin/python -m rescue_check_502.case_pass --cases PanTS_00000026
  PYTHONPATH=src .venv/bin/python -m rescue_check_502.case_pass --workers 4
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label as cc_label

from rescue_check import core as K11
from rescue_check_502 import core as K
from tumorlib import envelope, io as tio
from tumorlib.lesions import CC_STRUCT

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "nnunet-multiclass" / "postprocess"))
import core as PP  # noqa: E402  (S13 post-processing core, imported unchanged)

SPARSE = Path.home() / "research/nnunet_probs_502"
ATOMS = REPO / "work/nnunet_postprocess/atoms"
WORK = REPO / "work/rescue_check_502"
CASES = WORK / "cases"
D_SETTING = PP.Setting(t_low=K.D_T_LOW, t_seg=K.D_T_SEG, require_core=True)


def _crop_box(xyz: np.ndarray, pad_vox: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    lo = np.maximum(xyz.min(0) - pad_vox, 0)
    hi = np.minimum(xyz.max(0) + 1 + pad_vox, np.asarray(shape))
    return lo, hi


def _to_crop(xyz: np.ndarray, lo, hi) -> np.ndarray:
    a = np.zeros(tuple(int(v) for v in hi - lo), bool)
    a[tuple((xyz - lo).T)] = True
    return a


def nn_components(shape, p_idx, p_val, gl_sparse, t: float) -> list[dict]:
    """26-connected components of p > t from the sparse cache (complete: the cache holds every p > 1e-4).
    Peak voxel = highest p, ties to the lowest flat index (S13 rule)."""
    keep = p_val > t
    if not keep.any():
        return []
    idx, val, gl = p_idx[keep], p_val[keep], gl_sparse[keep]
    xyz = np.stack(np.unravel_index(idx, shape), 1)
    lo = xyz.min(0)
    a = np.zeros(tuple(int(v) for v in xyz.max(0) - lo + 1), bool)
    a[tuple((xyz - lo).T)] = True
    lab, n = cc_label(a, structure=CC_STRUCT)
    lv = lab[tuple((xyz - lo).T)]
    out = []
    order = np.lexsort((idx, -val, lv))
    first = np.r_[True, lv[order][1:] != lv[order][:-1]]
    for pos in order[first]:
        c = int(lv[pos])
        m = lv == c
        inter = {}
        if (gl[m] > 0).any():
            ls, cnt = np.unique(gl[m][gl[m] > 0], return_counts=True)
            inter = {int(a): int(b) for a, b in zip(ls, cnt)}
        out.append(dict(comp=c, n_vox=int(m.sum()), peak_p=float(val[pos]), peak_xyz=xyz[pos], xyz=xyz[m], inter=inter))
    return out


def process(case_id: str) -> tuple[str, str]:
    out = CASES / f"{case_id}.pkl"
    if out.exists():
        return case_id, "cached"
    t0 = time.time()
    try:
        with np.load(SPARSE / f"{case_id}.npz") as z:
            d = {k: z[k] for k in z.files}
        with open(ATOMS / f"{case_id}.pkl", "rb") as f:
            at = pickle.load(f)
        shape = tuple(int(s) for s in d["shape"])
        sp = np.asarray(d["spacing"], float)
        row_raw, les_raw = PP.evaluate(at, PP.BASELINE)
        row_d, les_d = PP.evaluate(at, D_SETTING)
        case = dict(case_id=case_id, fold=int(d["fold"]), gt_tumor=int(at["gt_vox"] > 0), n_lesions=at["n_lesions"],
                    n_fp_raw=row_raw["n_fp_cc"], seg_any_raw=row_raw["seg_any"], dice_raw=row_raw["dice"],
                    n_fp_d=row_d["n_fp_cc"], seg_any_d=row_d["seg_any"], dice_d=row_d["dice"],
                    n_pred_panc=int(d["panc_idx"].size))
        lesions = []
        if at["n_lesions"]:
            lesions = lesion_pass(case_id, d, at, shape, sp, les_raw, les_d, case)
        CASES.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(dict(case=case, lesions=lesions), f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.rename(out)
        return case_id, f"ok {time.time() - t0:.1f}s"
    except Exception as e:
        traceback.print_exc()
        return case_id, f"error {type(e).__name__}: {e}"


def lesion_pass(case_id, d, at, shape, sp, les_raw, les_d, case) -> list[dict]:
    gt_idx = np.asarray(d["gt_idx"], np.int64)
    gt_lab = np.asarray(d["gt_lab"], np.int64)
    gxyz = np.stack(np.unravel_index(gt_idx, shape), 1)
    n_les = int(at["n_lesions"])
    assert n_les == int(gt_lab.max()) and [int((gt_lab == j).sum()) for j in range(1, n_les + 1)] == \
        [int(v) for v in at["les_vox"]], "sparse GT != atoms"
    panc_xyz = np.stack(np.unravel_index(np.asarray(d["panc_idx"], np.int64), shape), 1)

    # ---- predicted pancreas crop: all lesions + predicted pancreas, padded for the closing and the shell
    pad = np.ceil((K.FILL_CLOSE_MM + K.SHELL_MM) / sp).astype(int) + 3
    lo, hi = _crop_box(np.vstack([gxyz, panc_xyz]) if len(panc_xyz) else gxyz, pad, shape)
    panc = _to_crop(panc_xyz, lo, hi) if len(panc_xyz) else np.zeros(tuple(hi - lo), bool)
    labc = np.zeros(tuple(int(v) for v in hi - lo), np.int32)
    labc[tuple((gxyz - lo).T)] = gt_lab
    filled = envelope.fill_search_region(panc.astype(np.uint8), sp, K.FILL_CLOSE_MM, 0.0).astype(bool) \
        if panc.any() else panc
    dist = distance_transform_edt(~panc, sampling=sp) if panc.any() else None

    # ---- placement domain, nnU-Net OP components, MPD
    sr = envelope.search_region(case_id)
    sr_xyz = np.argwhere(sr) if sr is not None else np.zeros((0, 3), int)
    if sr is not None:
        assert sr.shape == shape, f"search region shape {sr.shape} != {shape}"
    del sr
    case["sr_vox"] = len(sr_xyz)
    gl_sparse = np.zeros(d["p_idx"].size, np.int64)
    o = np.argsort(gt_idx)
    pos = np.searchsorted(gt_idx[o], d["p_idx"].astype(np.int64))
    posc = np.minimum(pos, max(gt_idx.size - 1, 0))
    hit = gt_idx[o][posc] == d["p_idx"]
    gl_sparse[hit] = gt_lab[o][posc[hit]]
    comps = nn_components(shape, np.asarray(d["p_idx"], np.int64), np.asarray(d["p_val"], np.float32), gl_sparse,
                          K11.NN_OP_T)
    if comps:
        rk = K11.rank_components(pd.DataFrame({"case_id": case_id, "peak_p": [c["peak_p"] for c in comps]}))
        top = [c for c, r in zip(comps, rk["rank"]) if r <= K11.NN_OP_RANK]
    else:
        top = []
    case["n_comps_t0.01"] = len(comps)
    case["n_op_comps"] = len(top)
    mpd = tio.load_mask(case_id, "pancreatic_duct")
    mpd_present = mpd is not None and bool(mpd.any())
    if mpd is not None:
        assert mpd.shape == shape, f"MPD shape {mpd.shape} != {shape}"
    near, near_lo = None, None
    if mpd_present:
        mxyz = np.argwhere(mpd)
        mlo, mhi = _crop_box(mxyz, np.ceil(K.DUCT_MM / sp).astype(int) + 1, shape)
        near = K.near_region(mpd[tuple(slice(a, b) for a, b in zip(mlo, mhi))], sp, K.DUCT_MM)
        near_lo = mlo
    del mpd
    case["mpd_present"] = mpd_present

    rows = []
    spad = np.ceil(K.SHELL_MM / sp).astype(int) + 1
    for j in range(1, n_les + 1):
        lxyz = gxyz[gt_lab == j]
        llo, lhi = lxyz.min(0), lxyz.max(0) + 1
        # predicted-pancreas metrics on the lesion's padded sub-crop
        slo, shi = np.maximum(llo - spad, lo), np.minimum(lhi + spad, hi)
        sl = tuple(slice(int(a - b), int(c - b)) for a, c, b in zip(slo, shi, lo))
        m = K.lesion_panc_metrics(labc[sl] == j, labc[sl] > 0, panc[sl], filled[sl],
                                  dist[sl] if dist is not None else None, sp)
        r = dict(case_id=case_id, lesion_id=j, gt_vox=len(lxyz), **{f"panc_{k}": v for k, v in m.items()},
                 panc_category=K.panc_category(m["frac_in"], m["frac_filled"], m["dist_mm"]),
                 les_in_panc_atoms=int(at["les_in_panc"][j - 1]))
        for tag, lr in (("raw", les_raw[j - 1]), ("d", les_d[j - 1])):
            assert lr["lesion_id"] == j and lr["gt_vox"] == len(lxyz)
            r.update({f"dice_{tag}": lr["lesion_dice"], f"touch_{tag}": lr["detected_any"],
                      f"vol_ratio_{tag}": lr["pred_vol_ratio"]})

        # nnU-Net OP: placement of each rank <= 3 component (anchor = its peak voxel)
        lcrop = _to_crop(lxyz, llo, lhi)
        fs, obs, inter_u, size_u = [], False, 0, 0
        for c in top:
            clo = c["xyz"].min(0)
            ccrop = _to_crop(c["xyz"], clo, c["xyz"].max(0) + 1)
            R, origin = K.anchor_hits(lcrop, llo, ccrop, c["peak_xyz"] - clo)
            touches = j in c["inter"]
            assert K.hit_at(R, origin, c["peak_xyz"]) == touches, "identity placement != real touch (nnU-Net)"
            fs.append(K.placement_fraction(sr_xyz, R, origin))
            if touches:
                obs = True
                inter_u += c["inter"][j]
                size_u += c["n_vox"]
        r.update(nn_op_hit_obs=obs, nn_op_n_placed=len(top), nn_op_f=";".join(f"{f:.6g}" for f in fs),
                 nn_op_chance=K.chance_any(fs) if len(sr_xyz) else np.nan,
                 nn_op_dice=(2.0 * inter_u / (len(lxyz) + size_u)) if obs else 0.0)

        # duct: lesion placement (anchor = its centroid voxel) against "within 10 mm of the MPD"
        if mpd_present:
            anchor = np.round(lxyz.mean(0)).astype(int) - llo
            R, origin = K.anchor_hits(near, near_lo, lcrop, anchor)
            r.update(duct_near_obs=K.hit_at(R, origin, llo + anchor),
                     duct_near_g=K.placement_fraction(sr_xyz, R, origin))
        else:
            r.update(duct_near_obs=np.nan, duct_near_g=np.nan)
        rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    ids = args.cases or sorted(c for k in range(1, 6) for c in tio.fold_ids(k))
    print(f"{len(ids)} cases, {args.workers} workers -> {CASES}", flush=True)
    t0, bad = time.time(), []
    with mp.Pool(min(args.workers, 4), maxtasksperchild=20) as pool:
        for i, (cid, status) in enumerate(pool.imap_unordered(process, ids, chunksize=1), 1):
            if not (status.startswith("ok") or status == "cached"):
                bad.append((cid, status))
                print(f"  {cid}: {status}", flush=True)
            if i % 100 == 0 or i == len(ids) or len(ids) <= 10:
                print(f"  {i}/{len(ids)} ({time.time() - t0:.0f}s) last {cid} {status}", flush=True)
    print(f"done; {len(bad)} failures: {bad}")


if __name__ == "__main__":
    main()
