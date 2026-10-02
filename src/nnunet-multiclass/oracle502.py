"""Pre-preprocessing oracles over ALL 1308 Dataset502 cases.

Hard checks (any failure exits non-zero -- do not preprocess):
  1. (label == 2) equals the Dataset501 binary label voxel for voxel  -> 0 mismatches
  2. label values are a subset of {0, 1, 2}; dtype uint8; scl_slope/inter clean
  3. (label > 0) equals union(4 gland masks, lesion), recomputed from the source masks
  4. label sits on the imagesTr grid (shape + affine)
  5. pancreas region (label > 0) < 10% of the scan; lesion (label == 2) < 30%

Reported (not gating):
  - pancreas volume distribution in cm^3 and the outliers (< 20 or > 250 cm^3)
  - cases whose 4-mask envelope is empty (expect the 11 known negatives; any positive
    among them is called out)
  - lesion voxels outside the original pancreas.nii.gz and outside the 4-mask envelope,
    per lesion, joined to the S1 master table by (case_id, lesion_id) with a gt_vox
    equality assert -- broken out by S1 tier
  - how many of the 70 `separated` lesions are now inside the pancreas REGION purely by
    nesting (all of them, by construction: that is the confound to record, not fix)

Writes work/nnunet_ds502/oracle_per_case.csv, oracle_per_lesion.csv, oracle_summary.json.

    source ~/research/nnunet_env/env.sh
    .venv/bin/python src/nnunet-multiclass/oracle502.py [--nproc 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "nnunet-multiclass"))

from tumorlib import io as tio                                            # noqa: E402
from tumorlib.lesions import label_lesions                                # noqa: E402
from labels502 import (                                                   # noqa: E402
    DATASET_NAME,
    ENVELOPE_MASKS,
    LESION_MASK,
    MAX_LESION_FRACTION,
    MAX_PANCREAS_FRACTION,
    compose_label,
)

RAW_ROOT = Path(os.environ["nnUNet_raw"])
DS502 = RAW_ROOT / DATASET_NAME
DS501 = RAW_ROOT / "Dataset501_PanTSTumor"
OUT = REPO / "work" / "nnunet_ds502"
MASTER = REPO / "results" / "master_lesion_table" / "master_lesions.csv"

PANC_CM3_LO, PANC_CM3_HI = 20.0, 250.0


def _read_label(path: Path):
    img = nib.load(path)
    raw = img.dataobj.get_unscaled()
    return img, np.asarray(raw)


def check_one(cid: str) -> dict:
    r: dict = {"id": cid, "problems": []}
    lab_p = DS502 / "labelsTr" / f"{cid}.nii.gz"
    img_p = DS502 / "imagesTr" / f"{cid}_0000.nii.gz"
    p501 = DS501 / "labelsTr" / f"{cid}.nii.gz"
    for p in (lab_p, img_p, p501):
        if not p.exists():
            r["problems"].append(f"missing {p.name}")
    if r["problems"]:
        return r

    lab_img, lab = _read_label(lab_p)
    ct = nib.load(img_p)

    # -- 2. encoding / dtype / header
    uniq = np.unique(lab).tolist()
    if not set(int(u) for u in uniq) <= {0, 1, 2}:
        r["problems"].append(f"label values {uniq} not subset of {{0,1,2}}")
    if lab_img.get_data_dtype() != np.uint8:
        r["problems"].append(f"dtype {lab_img.get_data_dtype()} != uint8")
    slope, inter = lab_img.header.get_slope_inter()
    if (slope not in (None, 1.0)) or (inter not in (None, 0.0)):
        r["problems"].append(f"scl_slope/inter = {(slope, inter)}")

    # -- 4. geometry
    if lab_img.shape != ct.shape[:3]:
        r["problems"].append(f"label shape {lab_img.shape} != image {ct.shape[:3]}")
        return r
    if not np.allclose(lab_img.affine, ct.affine, atol=1e-6):
        r["problems"].append(
            f"label affine != image affine (maxdiff {np.abs(lab_img.affine - ct.affine).max():.1e})")

    les502 = lab == 2
    fg502 = lab > 0

    # -- 1. lesion channel vs Dataset501
    i501, l501 = _read_label(p501)
    thr501 = tio._raw_threshold(i501.dataobj)
    les501 = l501 > thr501
    if les501.shape != les502.shape:
        r["problems"].append(f"Dataset501 label shape {les501.shape} != {les502.shape}")
    else:
        d = int((les501 != les502).sum())
        r["ds501_lesion_mismatch_vox"] = d
        if d:
            r["problems"].append(f"(label==2) differs from Dataset501 by {d} vox")
    del l501, les501

    # -- 3. foreground vs a fresh recomputation from the source masks
    les_src = tio.load_mask(cid, LESION_MASK)
    env = None
    for name in ENVELOPE_MASKS:
        m = tio.load_mask(cid, name)
        if m is None:
            r["problems"].append(f"unreadable source mask {name}")
            env = None
            break
        env = m if env is None else np.bitwise_or(env, m, out=env)
    if les_src is None:
        r["problems"].append(f"unreadable source mask {LESION_MASK}")
    if env is not None and les_src is not None:
        recomposed = compose_label(env, les_src)
        d = int((recomposed != lab).sum())
        r["recompose_mismatch_vox"] = d
        if d:
            r["problems"].append(f"label differs from a fresh recomposition by {d} vox")
        r["env_vox"] = int(env.sum(dtype=np.int64))
        r["env_empty"] = bool(r["env_vox"] == 0)
        # lesion voxels outside pancreas.nii.gz vs outside the 4-mask envelope
        panc_only = tio.load_mask(cid, "pancreas")
        n_les = int(les_src.sum(dtype=np.int64))
        r["lesion_vox"] = n_les
        if n_les:
            r["lesion_out_env_vox"] = int((les_src.astype(bool) & ~env.astype(bool)).sum())
            if panc_only is not None:
                r["lesion_out_pancreasnii_vox"] = int(
                    (les_src.astype(bool) & ~panc_only.astype(bool)).sum())
            # per-lesion breakdown, lesion_id from the canonical labeller
            lab_les, n = label_lesions(les_src)
            per = []
            for lid in range(1, n + 1):
                sel = lab_les == lid
                v = int(sel.sum())
                per.append({
                    "case_id": cid, "lesion_id": lid, "gt_vox": v,
                    "out_env_vox": int((sel & ~env.astype(bool)).sum()),
                    "out_pancreasnii_vox": (None if panc_only is None
                                            else int((sel & ~panc_only.astype(bool)).sum())),
                })
            r["per_lesion"] = per
            r["n_lesions"] = n
        del panc_only
    del env, les_src

    # -- 5. plausibility + volumes
    spacing = tio.spacing(cid)
    vox_mm3 = float(np.prod(spacing)) if spacing else float("nan")
    r["spacing"] = [round(s, 4) for s in spacing] if spacing else None
    n_fg, n_les = int(fg502.sum(dtype=np.int64)), int(les502.sum(dtype=np.int64))
    r["fg_vox"] = n_fg
    r["label2_vox"] = n_les
    r["panc_frac"] = round(n_fg / lab.size, 7)
    r["lesion_frac"] = round(n_les / lab.size, 7)
    r["panc_cm3"] = round(n_fg * vox_mm3 / 1000.0, 3)
    r["lesion_cm3"] = round(n_les * vox_mm3 / 1000.0, 3)
    if r["panc_frac"] > MAX_PANCREAS_FRACTION:
        r["problems"].append(f"pancreas region {r['panc_frac']:.1%} >= {MAX_PANCREAS_FRACTION:.0%}")
    if r["lesion_frac"] > MAX_LESION_FRACTION:
        r["problems"].append(f"lesion {r['lesion_frac']:.1%} >= {MAX_LESION_FRACTION:.0%}")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("ORACLE502_NPROC", "3")))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    ids = tio.case_ids()
    if a.limit:
        ids = ids[: a.limit]
    print(f"oracle over {len(ids)} cases, nproc {a.nproc}", flush=True)

    rows: list[dict] = []
    with Pool(a.nproc) as pool:
        for n, r in enumerate(pool.imap_unordered(check_one, ids, chunksize=1), 1):
            rows.append(r)
            if n % 100 == 0 or n == len(ids):
                print(f"  {n}/{len(ids)}  cases with problems: "
                      f"{sum(1 for x in rows if x['problems'])}", flush=True)

    per_les = [p for r in rows for p in r.pop("per_lesion", [])]
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.assign(problems=df["problems"].apply("; ".join)).to_csv(OUT / "oracle_per_case.csv", index=False)

    les = pd.DataFrame(per_les)
    tiers: dict = {}
    join_note = "master table not found"
    if len(les) and MASTER.exists():
        m = pd.read_csv(MASTER)[["case_id", "lesion_id", "gt_vox", "tier", "diam_bin"]]
        j = les.merge(m, on=["case_id", "lesion_id"], how="outer",
                      suffixes=("", "_master"), indicator=True)
        both = j[j["_merge"] == "both"]
        bad = int((both["gt_vox"] != both["gt_vox_master"]).sum())
        join_note = (f"rows: ours {len(les)}, master {len(m)}, matched {len(both)}, "
                     f"ours-only {int((j['_merge'] == 'left_only').sum())}, "
                     f"master-only {int((j['_merge'] == 'right_only').sum())}, "
                     f"gt_vox disagreements {bad}")
        assert bad == 0, f"gt_vox mismatch on {bad} joined lesions -- join key is wrong"
        both = both.copy()
        both["any_out_env"] = both["out_env_vox"] > 0
        both["any_out_pancreasnii"] = both["out_pancreasnii_vox"].fillna(0) > 0
        g = both.groupby("tier")
        tiers = {
            str(t): {
                "n_lesions": int(len(s)),
                "n_with_vox_outside_envelope": int(s["any_out_env"].sum()),
                "frac_vox_outside_envelope": round(
                    float(s["out_env_vox"].sum() / s["gt_vox"].sum()), 4),
                "n_with_vox_outside_pancreas_nii": int(s["any_out_pancreasnii"].sum()),
                "frac_vox_outside_pancreas_nii": round(
                    float(s["out_pancreasnii_vox"].sum() / s["gt_vox"].sum()), 4),
            }
            for t, s in g
        }
        both.to_csv(OUT / "oracle_per_lesion.csv", index=False)
    elif len(les):
        les.to_csv(OUT / "oracle_per_lesion.csv", index=False)

    probs = df[df["problems"].apply(bool)]
    panc = df["panc_cm3"].dropna()
    env_empty = df[df["env_empty"].fillna(False)] if "env_empty" in df else df.iloc[:0]
    pos = df["label2_vox"].fillna(0) > 0
    summary = {
        "dataset": DATASET_NAME,
        "n_cases": len(df),
        "n_cases_with_problems": len(probs),
        "hard_checks": {
            "ds501_lesion_mismatch_cases": int((df.get("ds501_lesion_mismatch_vox", pd.Series(dtype=float)).fillna(0) > 0).sum()),
            "recompose_mismatch_cases": int((df.get("recompose_mismatch_vox", pd.Series(dtype=float)).fillna(0) > 0).sum()),
            "max_panc_frac": float(df["panc_frac"].max()),
            "max_lesion_frac": float(df["lesion_frac"].max()),
        },
        "n_label2_nonempty": int(pos.sum()),
        "n_label2_empty": int((~pos).sum()),
        "pancreas_cm3": {
            "min": round(float(panc.min()), 2), "p5": round(float(panc.quantile(.05)), 2),
            "median": round(float(panc.median()), 2), "mean": round(float(panc.mean()), 2),
            "p95": round(float(panc.quantile(.95)), 2), "max": round(float(panc.max()), 2),
        },
        "pancreas_cm3_outliers": {
            f"below_{PANC_CM3_LO:g}": sorted(df.loc[panc.index][df["panc_cm3"] < PANC_CM3_LO]["id"].tolist()),
            f"above_{PANC_CM3_HI:g}": sorted(df.loc[panc.index][df["panc_cm3"] > PANC_CM3_HI]["id"].tolist()),
        },
        "empty_envelope_cases": {
            "n": len(env_empty),
            "ids": sorted(env_empty["id"].tolist()),
            "positives_among_them": sorted(env_empty[env_empty["label2_vox"] > 0]["id"].tolist()),
        },
        "lesion_outside": {
            "total_lesion_vox": int(df["lesion_vox"].fillna(0).sum()),
            "vox_outside_envelope": int(df.get("lesion_out_env_vox", pd.Series(dtype=float)).fillna(0).sum()),
            "vox_outside_pancreas_nii": int(df.get("lesion_out_pancreasnii_vox", pd.Series(dtype=float)).fillna(0).sum()),
            "by_s1_tier": tiers,
            "join": join_note,
        },
        "failures": [{"id": r.id, "why": "; ".join(r.problems)} for r in probs.itertuples()],
    }
    (OUT / "oracle_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== ORACLE SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    if summary["failures"]:
        print(f"\n{len(summary['failures'])} cases with problems:")
        for f in summary["failures"][:60]:
            print(f"  {f['id']}: {f['why']}")
    print(f"\nwrote {OUT}/oracle_summary.json, oracle_per_case.csv, oracle_per_lesion.csv")
    sys.exit(1 if len(probs) else 0)


if __name__ == "__main__":
    main()
