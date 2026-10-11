"""S13 step 1 (PC): sparse cache -> per-case atoms + the objective for every grid setting, checked against S8.

Reads ~/research/nnunet_probs_502/{case}.npz (extract_sparse.py) and S8's 502 pickle
work/nnunet_multiclass/cases/502/{case}.pkl; writes work/nnunet_postprocess/atoms/{case}.pkl (resumable).

Per-case regression against S8 (status != "ok" lists what differs; lofo.py refuses to run on any failure):
  - #(p > 0.5) == S8's nn_n_pred (= #(seg == 2)), whole-volume peak == S8's vol_peak_p
  - GT lesion sizes, in tumorlib order, == S8's per-lesion gt_vox
  - components of p > t for t in {0.5, 0.1, 0.01, 0.001}: the multiset of (size, peak, touched lesions) == S8's comps
  - the no-op setting: case Dice, predicted volume, #CCs and every lesion's Dice / IoU / detected == S8's ev_case /
    ev_lesions (evaluate.py through S8's code path)

Usage (PC, repo root):  .venv/bin/python src/nnunet-multiclass/postprocess/atoms_pass.py --workers 4
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as K  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SPARSE = Path.home() / "research/nnunet_probs_502"
S8 = REPO / "work/nnunet_multiclass/cases/502"
WORK = REPO / "work/nnunet_postprocess"
ATOMS = WORK / "atoms"
S8_T = (0.5, 0.1, 0.01, 0.001)


def compare_s8(case: dict, d: dict, s8: dict) -> list[str]:
    bad = []
    p_val = d["p_val"]
    if int((p_val > 0.5).sum()) != int(d["n_gt05"]) or int(d["n_gt05"]) != s8["ev_case"]["nn_n_pred"]:
        bad.append(f"n(p>0.5) {(p_val > 0.5).sum()} / {int(d['n_gt05'])} vs S8 {s8['ev_case']['nn_n_pred']}")
    if not np.isclose(float(d["vol_peak_p"]), s8["case"]["vol_peak_p"], rtol=0, atol=0):
        bad.append("vol_peak_p")
    s8_les = sorted(s8["lesions"], key=lambda r: r["lesion_id"])
    if [r["gt_vox"] for r in s8_les] != [int(v) for v in case["les_vox"]]:
        bad.append("lesion gt_vox / numbering")
    for t in S8_T:
        mine = sorted((int(n), round(float(p), 7), tuple(sorted(tc)))
                      for n, p, tc in zip(case["comps"][t]["n_vox"], case["comps"][t]["peak"], case["comps"][t]["touch"]))
        ref = sorted((int(c["n_vox"]), round(float(c["peak_p"]), 7),
                      tuple(sorted(int(x) for x in c["touch_ids"].split(";") if x)))
                     for c in s8["comps"] if c["t"] == t)
        if mine != ref:
            bad.append(f"comps t={t}: {len(mine)} vs S8 {len(ref)}")
    row, les = K.evaluate(case, K.BASELINE)
    ev = s8["ev_case"]
    if row["pred_vox"] != ev["nn_n_pred"] or row["inter_vox"] != ev["nn_tp"] or row["n_cc"] != s8["case"]["n_seg_comps"]:
        bad.append("baseline pred/tp/n_cc")
    if case["gt_vox"] and not np.isclose(row["dice"], ev["dice"], rtol=1e-12, atol=1e-12):
        bad.append(f"baseline dice {row['dice']} vs {ev['dice']}")
    if not np.isclose(row["pred_vol_mm3"], ev["pred_vol_mm3"], rtol=1e-9):
        bad.append("baseline pred_vol_mm3")
    for a, b in zip(les, s8["ev_lesions"]):
        if a["gt_vox"] != b["gt_vox"] or a["detected_any"] != b["detected_any"] or \
                not np.isclose(a["lesion_dice"], b["lesion_dice"], rtol=1e-12, atol=1e-12) or \
                not np.isclose(a["lesion_iou"], b["lesion_iou"], rtol=1e-12, atol=1e-12):
            bad.append(f"baseline lesion {a['lesion_id']}")
    if len(les) != len(s8["ev_lesions"]):
        bad.append("baseline lesion count")
    return bad


def process(case_id: str) -> tuple[str, str]:
    out = ATOMS / f"{case_id}.pkl"
    if out.exists():
        return case_id, "cached"
    t0 = time.time()
    try:
        with np.load(SPARSE / f"{case_id}.npz") as z:
            d = {k: z[k] for k in z.files}
        if float(d["floor"]) > K.FLOOR:
            return case_id, f"error: cache floor {float(d['floor'])} > {K.FLOOR}"
        case = K.build_case(d["shape"], d["spacing"], d["p_idx"], d["p_val"], d["panc_idx"], d["gt_idx"], d["gt_lab"])
        case.pop("dist")
        with open(S8 / f"{case_id}.pkl", "rb") as f:
            s8 = pickle.load(f)
        bad = compare_s8(case, d, s8)
        settings = K.all_settings()
        # A diagnostic: GT lesion voxels inside the model's predicted pancreas (channel 0 > 0.5), per lesion
        inp = np.isin(d["gt_idx"], d["panc_idx"])
        les_in_panc = np.bincount(d["gt_lab"][inp].astype(np.int64), minlength=case["n_lesions"] + 1)[1:]
        case.update(case_id=case_id, fold=int(d["fold"]), n_panc=int(d["panc_idx"].size), les_in_panc=les_in_panc,
                    grid=K.score_grid(case, settings).astype(np.float64), n_settings=len(settings),
                    s8_ok=not bad, s8_bad=bad, seconds=round(time.time() - t0, 1))
        if int(d["fold"]) != s8["case"]["fold"]:
            case["s8_bad"].append("fold")
            case["s8_ok"] = False
        ATOMS.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(case, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.rename(out)
        return case_id, ("ok" if not bad else f"S8 MISMATCH {bad}") + f" {case['seconds']}s"
    except Exception as e:
        traceback.print_exc()
        return case_id, f"error {type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    ids = args.cases or sorted(p.stem for p in SPARSE.glob("PanTS_*.npz") if not p.stem.endswith(".tmp"))
    print(f"{len(ids)} cases, {args.workers} workers -> {ATOMS}", flush=True)
    t0, bad = time.time(), []
    with mp.Pool(min(args.workers, 4), maxtasksperchild=50) as pool:
        for i, (cid, status) in enumerate(pool.imap_unordered(process, ids, chunksize=2), 1):
            if not (status.startswith("ok") or status == "cached"):
                bad.append((cid, status))
                print(f"  {cid}: {status}", flush=True)
            if i % 100 == 0 or i == len(ids) or len(ids) <= 10:
                print(f"  {i}/{len(ids)} ({time.time() - t0:.0f}s) last {cid} {status}", flush=True)
    print(f"done; {len(bad)} not ok: {bad[:20]}")


if __name__ == "__main__":
    main()
