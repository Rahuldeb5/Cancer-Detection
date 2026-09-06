"""Full integrity audit of Dataset501_PanTSTumor (CT images + binary tumor labels).

For every one of the 1308 CV cases:
  - imagesTr / labelsTr present
  - CT geometry (shape + voxel spacing) matches PanTS_metadata.csv
  - label sits on the CT's exact grid (shape + affine), dtype uint8, clean header,
    values a subset of {0, 1}
  - label foreground fraction is physically plausible; connected-component count
  - empty / non-empty label agrees with the metadata `tumor?` flag
  - CT voxels are genuine Hounsfield units: air population present (p1 well below 0),
    a soft-tissue population present, no unsigned-scaling blow-up

On a deterministic random sample (SEED, SAMPLE_N) also runs the expensive checks:
  - labelsTr identical, voxel for voxel, to a fresh read of the source
    pancreatic_lesion.nii.gz  (would catch any silent corruption in build_dataset.py)
  - lesion voxels actually sit on / next to the pancreas (source pancreas.nii.gz,
    only for sample cases whose pancreas mask has been pulled from git-LFS)

Writes audit_summary.json + audit_per_case.csv to $AUDIT_OUT (default logs dir).
Exits non-zero iff any hard check fails.

    python src/nnunet/audit_dataset.py                 # full audit
    python src/nnunet/audit_dataset.py --print-sample  # list the sample ids
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

RAW = Path(os.environ["nnUNet_raw"]) / "Dataset501_PanTSTumor"
IMAGES_TR = RAW / "imagesTr"
LABELS_TR = RAW / "labelsTr"
MASK_STAGING = Path("/home/rahuldeb5/research/datasets/pants/masks/mask_only")
META_CSV = Path("src/data/PanTS_metadata.csv")
OUT_DIR = Path(os.environ.get("AUDIT_OUT", "/home/rahuldeb5/research/nnunet_env/logs"))

SAMPLE_N = 150
SEED = 501
NPROC = int(os.environ.get("AUDIT_NPROC", "6"))
MAX_FG_FRACTION = 0.30
CC_STRUCT = np.ones((3, 3, 3), dtype=int)


def fold_ids() -> list[str]:
    ids: set[str] = set()
    for i in range(1, 6):
        for line in Path(f"src/data/fold_{i}_ids.txt").read_text().splitlines():
            s = line.strip()
            if s:
                ids.add(s)
    return sorted(ids)


def _nums(s: str) -> list[float]:
    return [float(x) for x in str(s).strip("()[] ").replace(",", " ").split()]


def audit_one(args) -> dict:
    cid, meta_shape, meta_spacing, meta_tumor, sampled = args
    r: dict = {"id": cid, "problems": [], "sampled": sampled}
    img_p = IMAGES_TR / f"{cid}_0000.nii.gz"
    lab_p = LABELS_TR / f"{cid}.nii.gz"
    if not img_p.exists():
        r["problems"].append("missing imagesTr")
    if not lab_p.exists():
        r["problems"].append("missing labelsTr")
    if r["problems"]:
        return r

    ct = nib.load(img_p)
    lab = nib.load(lab_p)
    zooms = [float(z) for z in ct.header.get_zooms()[:3]]
    r["ct_shape"] = list(ct.shape)
    r["ct_zooms"] = [round(z, 4) for z in zooms]

    # ---- geometry vs metadata ----
    try:
        if tuple(ct.shape) != tuple(int(round(x)) for x in _nums(meta_shape)):
            r["problems"].append(f"CT shape {tuple(ct.shape)} != metadata {meta_shape}")
    except Exception as e:  # noqa: BLE001
        r["problems"].append(f"meta shape parse {meta_shape!r}: {e}")
    try:
        if not np.allclose(sorted(zooms), sorted(_nums(meta_spacing)), atol=1e-2):
            r["problems"].append(f"CT spacing {r['ct_zooms']} != metadata {meta_spacing}")
    except Exception as e:  # noqa: BLE001
        r["problems"].append(f"meta spacing parse {meta_spacing!r}: {e}")

    # ---- label on the CT grid ----
    if lab.shape != ct.shape:
        r["problems"].append(f"label shape {lab.shape} != CT {ct.shape}")
    if not np.allclose(lab.affine, ct.affine, atol=1e-3):
        r["problems"].append(f"label affine != CT affine (maxdiff {np.abs(lab.affine - ct.affine).max():.1e})")
    if lab.get_data_dtype() != np.uint8:
        r["problems"].append(f"label dtype {lab.get_data_dtype()} != uint8")
    slope, inter = lab.header.get_slope_inter()
    if (slope not in (None, 1.0)) or (inter not in (None, 0.0)):
        r["problems"].append(f"label scl_slope/inter = {(slope, inter)}")

    seg = np.asanyarray(lab.dataobj)
    uniq = np.unique(seg).tolist()
    if not set(uniq) <= {0, 1}:
        r["problems"].append(f"label values {uniq} not subset of {{0,1}}")
    seg = (seg > 0).astype(np.uint8)
    fg = int(seg.sum())
    frac = fg / seg.size
    r["fg_vox"] = fg
    r["fg_frac"] = round(frac, 7)
    if frac > MAX_FG_FRACTION:
        r["problems"].append(f"fg fraction {frac:.1%} implausible")

    r["meta_tumor"] = int(meta_tumor) if meta_tumor is not None else None
    if meta_tumor is not None and (fg > 0) != bool(meta_tumor == 1):
        r["problems"].append(
            f"label {'non-empty' if fg else 'empty'} but metadata tumor?={int(meta_tumor)}"
        )

    if fg:
        cc, ncc = ndimage.label(seg, structure=CC_STRUCT)
        r["n_cc"] = int(ncc)
        r["largest_cc_vox"] = int(np.bincount(cc.ravel())[1:].max())
        r["tumor_ml"] = round(fg * float(np.prod(zooms)) / 1000.0, 3)

    # ---- CT Hounsfield sanity (strided read) ----
    ctd = np.asanyarray(ct.dataobj)[::3, ::3, ::3].astype(np.float32)
    ctd = ctd[np.isfinite(ctd)]
    p1, p50, p99, mx = (float(x) for x in np.percentile(ctd, [1, 50, 99, 100]))
    soft_frac = float(((ctd > -100) & (ctd < 300)).mean())
    r["ct_p1"], r["ct_p50"], r["ct_p99"], r["ct_max"] = round(p1, 1), round(p50, 1), round(p99, 1), round(mx, 1)
    r["ct_soft_frac"] = round(soft_frac, 3)
    if p1 > -300:
        r["problems"].append(f"CT p1={p1:.0f} > -300: no air population -> not HU / windowed")
    if soft_frac < 0.03:
        r["problems"].append(f"CT soft-tissue fraction {soft_frac:.1%} < 3%: no body in view?")
    if mx > 30000:
        r["problems"].append(f"CT max {mx:.0f}: unsigned-scaling blow-up")

    # ---- expensive sample-only checks ----
    if sampled:
        les_p = MASK_STAGING / cid / "segmentations" / "pancreatic_lesion.nii.gz"
        if les_p.exists():
            raw = np.nan_to_num(np.asanyarray(nib.load(les_p).dataobj), nan=0.0)
            src = (raw > 0.5).astype(np.uint8)
            if src.shape != seg.shape:
                r["problems"].append(f"source lesion shape {src.shape} != label {seg.shape} (sample)")
            else:
                d = int(np.abs(src.astype(int) - seg.astype(int)).sum())
                r["sample_label_vs_source_diff"] = d
                if d:
                    r["problems"].append(f"label differs from fresh source read by {d} vox (sample)")
        else:
            r["problems"].append("source lesion file missing (sample)")

        if fg:
            try:
                panc = np.asanyarray(nib.load(MASK_STAGING / cid / "segmentations" / "pancreas.nii.gz").dataobj) > 0.5
            except Exception:  # noqa: BLE001  (git-LFS pointer not pulled)
                panc = None
            if panc is not None and panc.shape == seg.shape:
                inside = float((seg.astype(bool) & ndimage.binary_dilation(panc, iterations=6)).sum()) / fg
                r["lesion_near_pancreas"] = round(inside, 3)
                if inside < 0.5:
                    r["problems"].append(f"only {inside:.0%} of lesion within ~6 vox of pancreas")

    return r


def main() -> None:
    ids = fold_ids()
    meta = pd.read_csv(META_CSV).set_index("PanTS ID")
    rng = np.random.default_rng(SEED)
    sample_idx = set(rng.choice(len(ids), size=min(SAMPLE_N, len(ids)), replace=False).tolist())

    if "--print-sample" in sys.argv:
        print("\n".join(ids[i] for i in sorted(sample_idx)))
        return

    tasks = []
    for i, cid in enumerate(ids):
        m = meta.loc[cid] if cid in meta.index else None
        tasks.append((
            cid,
            None if m is None else m["shape"],
            None if m is None else m["spacing"],
            None if m is None else int(m["tumor?"]),
            i in sample_idx,
        ))

    print(f"auditing {len(tasks)} cases with {NPROC} workers ...", flush=True)
    rows: list[dict] = []
    with Pool(NPROC) as pool:
        for n, r in enumerate(pool.imap_unordered(audit_one, tasks, chunksize=4), 1):
            rows.append(r)
            if n % 200 == 0 or n == len(tasks):
                nf = sum(1 for x in rows if x["problems"])
                print(f"  {n}/{len(tasks)}  cases with problems: {nf}", flush=True)

    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "audit_per_case.csv").write_text(
        df.assign(problems=df["problems"].apply("; ".join)).to_csv(index=False)
    )

    fg = df["fg_vox"].fillna(0)
    pos = df[fg > 0]
    probs = df[df["problems"].apply(bool)]
    by_kind: dict[str, int] = {}
    for pl in df["problems"]:
        for p in pl:
            by_kind[p.split(":")[0].split(" (")[0]] = by_kind.get(p.split(":")[0].split(" (")[0], 0) + 1

    summary = {
        "n_cases": len(df),
        "n_label_nonempty": int((fg > 0).sum()),
        "n_label_empty": int((fg == 0).sum()),
        "n_meta_tumor_pos": int((df["meta_tumor"] == 1).sum()),
        "n_sampled_deep": int(df["sampled"].sum()),
        "n_sample_label_mismatch": int((df["sample_label_vs_source_diff"].fillna(0) > 0).sum())
        if "sample_label_vs_source_diff" in df else 0,
        "lesion_near_pancreas_min": float(df["lesion_near_pancreas"].min())
        if "lesion_near_pancreas" in df and df["lesion_near_pancreas"].notna().any() else None,
        "fg_frac_max": float(df["fg_frac"].max()),
        "tumor_ml_median": float(pos["tumor_ml"].median()) if len(pos) else None,
        "tumor_ml_p95": float(pos["tumor_ml"].quantile(0.95)) if len(pos) else None,
        "tumor_ml_max": float(pos["tumor_ml"].max()) if len(pos) else None,
        "n_cc_max": int(pos["n_cc"].max()) if len(pos) and "n_cc" in pos else None,
        "ct_p1_worst": float(df["ct_p1"].max()),
        "ct_soft_frac_min": float(df["ct_soft_frac"].min()),
        "ct_max_worst": float(df["ct_max"].max()),
        "n_cases_with_problems": len(probs),
        "problem_kinds": by_kind,
        "failures": [{"id": row.id, "why": "; ".join(row.problems)} for row in probs.itertuples()],
    }
    (OUT_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    if summary["failures"]:
        print(f"\n{len(summary['failures'])} cases with problems:")
        for f in summary["failures"][:60]:
            print(f"  {f['id']}: {f['why']}")
    print(f"\nwrote {OUT_DIR/'audit_summary.json'} and {OUT_DIR/'audit_per_case.csv'}")
    hard = len(probs)
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
