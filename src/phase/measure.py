"""S12 step 2: per-case aorta, portal-vein and gland HU for the 1308 cohort -> work/phase/measure.csv.

Per case (one CT read, cropped to the bbox of aorta | veins | envelope | lesion, padded 2 voxels):
  aorta / pv   core.vessel_stats: 6-connected one-voxel erosion, middle 80% of vessel slices along
               the CT slice axis, median + IQR + voxel counts
  gland        core.gland_pool (S3 v3 global pool) on the exact S3 crop bbox(lesion | envelope, pad 2);
               for tumor+ cases it is asserted equal to ref_global_median in
               results/attenuation_v3/attenuation_labels_v3.csv (analyze step)
  QC           voxel overlaps aorta & veins, aorta & postcava, veins & postcava; mask encodings;
               whether each vessel mask affine matches the CT (masks are used on the CT grid regardless:
               S12 checked that the voxel arrays align even where the affine is flipped)
Every mask goes through tumorlib.io.load_mask (gotcha 1).

    PYTHONPATH=src .venv/bin/python -m phase.measure [--limit N] [--workers 4]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from phase import core
from tumorlib import envelope as envmod
from tumorlib import io
from tumorlib.lesions import bbox_slices

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "work" / "phase" / "measure.csv"
MAX_WORKERS = 4


def _sub(outer: tuple[slice, ...], inner: tuple[slice, ...]) -> tuple[slice, ...]:
    """`inner` (full-volume slices) expressed inside the crop `outer`."""
    return tuple(slice(i.start - o.start, i.stop - o.start) for o, i in zip(outer, inner))


def _mask_meta(case_id: str, name: str, ct_affine: np.ndarray) -> dict:
    img = nib.load(io.mask_path(case_id, name))
    slope = float(img.dataobj.slope)
    return {f"{name}_enc": "raw01" if np.isclose(slope, 1.0) else f"scaled{slope:.4g}",
            f"{name}_affine_ok": bool(np.allclose(img.affine, ct_affine, atol=1e-3))}


def measure_case(case_id: str) -> dict:
    t0 = time.time()
    row = {"case_id": case_id}
    g = io.grid(case_id)
    if g is None:
        row["error"] = "no CT"
        return row
    shape, affine, sp = g
    sax = core.slice_axis(affine)
    row.update(slice_axis=sax, slice_thickness_mm=sp[sax])
    for name in (core.AORTA_MASK, core.PV_MASK):
        try:
            row.update(_mask_meta(case_id, name, affine))
        except Exception as e:  # stub / missing: load_mask below reports it
            row[f"{name}_enc"] = f"unreadable:{type(e).__name__}"

    aorta = io.load_mask(case_id, core.AORTA_MASK)
    pv = io.load_mask(case_id, core.PV_MASK)
    env = envmod.envelope(case_id)
    les = io.load_mask(case_id, "pancreatic_lesion")
    if aorta is None or pv is None:
        row["error"] = "vessel mask failed"
        return row
    union = aorta | pv
    if env is not None:
        union |= env
    if les is not None:
        union |= les
    sl = bbox_slices(union, pad=2)
    del union
    if sl is None:
        row["error"] = "all masks empty"
        return row
    aorta, pv = aorta[sl].astype(bool), pv[sl].astype(bool)
    ivc = io.load_mask(case_id, core.IVC_MASK)
    ivc = ivc[sl].astype(bool) if ivc is not None else np.zeros_like(aorta)
    row.update(ov_aorta_pv=int((aorta & pv).sum()), ov_aorta_ivc=int((aorta & ivc).sum()),
               ov_pv_ivc=int((pv & ivc).sum()))
    del ivc

    ct = io.load_ct(case_id, sl)
    if ct is None:
        row["error"] = "CT read failed"
        return row
    for key, m in (("aorta", aorta), ("pv", pv)):
        st = core.vessel_stats(ct, m, sax)
        row.update({f"{key}_hu": st["median"], f"{key}_iqr": st["iqr"], f"{key}_n_vox": st["n_vox"],
                    f"{key}_n_vox_raw": st["n_vox_raw"], f"{key}_n_slices": st["n_slices"]})
    del aorta, pv

    # ---- gland: S3 v3 global pool on S3's own crop
    row.update(gland_hu=np.nan, gland_n_vox=0)
    if env is not None and les is not None:
        les_full_any = bool(les.any())
        sl3 = bbox_slices(les | env, pad=2)
        if sl3 is not None:
            les3, env3 = les[sl3].astype(bool), env[sl3].astype(bool)
            ducts = np.zeros(les3.shape, bool)
            missing = []
            for name in core.DUCTS:
                m = io.load_mask(case_id, name)
                if m is None:
                    missing.append(name)
                else:
                    ducts |= m[sl3].astype(bool)
                del m
            pool = core.gland_pool(env3, les3, ducts, sp)
            med, n = core.gland_median(ct[_sub(sl, sl3)], pool)
            row.update(gland_hu=med, gland_n_vox=n, missing_ducts=";".join(missing))
        row["tumor_mask_nonempty"] = les_full_any
    print(f"{case_id}: aorta {row['aorta_hu']:.0f} ({row['aorta_n_vox']}) pv {row['pv_hu']:.0f} "
          f"({row['pv_n_vox']}) gland {row['gland_hu']:.0f}  {time.time() - t0:.1f}s", flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    workers = min(a.workers, MAX_WORKERS)
    cases = a.cases or io.case_ids()[: a.limit]
    print(f"{len(cases)} cases, {workers} workers", flush=True)
    t0 = time.time()
    with mp.Pool(workers, maxtasksperchild=20) as pool:
        rows = list(pool.imap_unordered(measure_case, cases, chunksize=1))
    df = pd.DataFrame(rows).sort_values("case_id")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(df)} rows, {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
