"""S3 step 2: v3 attenuation measurements for every GT lesion, keyed (case_id, lesion_id).

Per case (everything cropped to the lesion+envelope bbox before any EDT):
  envelope   = pancreas | head | body | tail, 3D hole-filled (no closing: closing would pull
               peripancreatic fat from between lobules into the reference)
  ducts      = pancreatic_duct | common_bile_duct masks
  pool       = envelope - (all lesions + SHELL_MM) - ducts                      -> global reference
  ring_i     = pool within RING_MM[1] of lesion i (pool already excludes <= 5 mm of any lesion)
  core_i     = attn_v3.core_mask (PV-aware depth); tumor median and 20% trimmed mean over it
  sigma      = in-plane Laplacian-residual MAD over the pool; CNR = dHU / sigma
The v2 labels are not touched; this writes work/attenuation_v3/measure_v3.csv.

    .venv/bin/python src/attenuation-v3/measure_v3.py [--limit N]
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, distance_transform_edt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
import attn_v3 as av  # noqa: E402
from tumorlib import envelope as envmod  # noqa: E402
from tumorlib import io  # noqa: E402
from tumorlib.lesions import bbox_slices, label_lesions  # noqa: E402

MASTER = REPO / "results" / "master_lesion_table" / "master_lesions.csv"
OUT = REPO / "work" / "attenuation_v3" / "measure_v3.csv"
NUM_WORKERS = 2
DUCTS = ("pancreatic_duct", "common_bile_duct")


def measure_case(args) -> list[dict]:
    case_id, slice_axis = args
    t0 = time.time()
    sp = io.spacing(case_id)
    les = io.load_mask(case_id, "pancreatic_lesion")
    env = envmod.envelope(case_id)
    if sp is None or les is None or env is None:
        print(f"{case_id}: load failed")
        return []
    sp = np.asarray(sp)
    sl = bbox_slices(les | env, pad=2)
    les, env = les[sl].astype(bool), env[sl].astype(bool)

    ducts = np.zeros(les.shape, bool)
    missing_ducts = []
    for name in DUCTS:
        m = io.load_mask(case_id, name)
        if m is None:
            missing_ducts.append(name)
        else:
            ducts |= m[sl].astype(bool)
        del m
    ct = io.load_ct(case_id, sl)
    if ct is None:
        return []

    labels, n = label_lesions(les)
    env_filled = binary_fill_holes(env)
    d_any = distance_transform_edt(~les, sampling=sp)
    pool = env_filled & (d_any > av.SHELL_MM) & ~ducts
    del d_any
    ref_med, ref_trim = av.robust_stats(ct[pool])
    n_pool = int(pool.sum())
    if n_pool < av.MIN_REF_VOX:
        ref_med = ref_trim = np.nan
    sigma, n_sigma = av.noise_sigma(ct, pool, slice_axis)

    pad = np.ceil(av.RING_MM[1] / sp).astype(int) + 2
    rows = []
    for i in range(1, n + 1):
        sub = bbox_slices(labels == i, pad)
        comp = labels[sub] == i
        core, info = av.core_mask(comp, sp)
        ring = pool[sub] & (distance_transform_edt(~comp, sampling=sp) <= av.RING_MM[1])
        ct_sub = ct[sub]
        whole_med, whole_trim = av.robust_stats(ct_sub[comp])
        core_med, core_trim = av.robust_stats(ct_sub[core])
        n_ring = int(ring.sum())
        ring_med, ring_trim = av.robust_stats(ct_sub[ring]) if n_ring >= av.MIN_REF_VOX else (np.nan, np.nan)
        d_main = core_med - ref_med
        rows.append({
            "case_id": case_id, "lesion_id": i, "gt_vox": int(comp.sum()),
            "n_core": info["n_core"], "core_fallback": info["core_fallback"],
            "core_min_depth_mm": info["core_min_depth_mm"], "max_depth_mm": info["max_depth_mm"],
            "tumor_whole_median": whole_med, "tumor_core_median": core_med, "tumor_core_trim": core_trim,
            "ref_global_median": ref_med, "ref_global_trim": ref_trim, "n_ref_global": n_pool,
            "ref_local_median": ring_med, "ref_local_trim": ring_trim, "n_ref_local": n_ring,
            "dHU_whole_global": whole_med - ref_med,
            "dHU_core_global": d_main,
            "dHU_coretrim_global": core_trim - ref_trim,
            "dHU_core_local": core_med - ring_med,
            "dHU_coretrim_local": core_trim - ring_trim,
            "sigma_hu": sigma, "n_sigma": n_sigma,
            "cnr": d_main / sigma if sigma and np.isfinite(sigma) else np.nan,
            "missing_ducts": ";".join(missing_ducts),
        })
    print(f"{case_id}: {n} lesions, crop {tuple(les.shape)}, pool {n_pool}, sigma {sigma:.1f} "
          f"({time.time() - t0:.1f}s)", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    master = pd.read_csv(MASTER)
    cases = master.groupby("case_id").slice_axis.first().reset_index()
    jobs = list(cases.itertuples(index=False, name=None))[: a.limit]
    print(f"{len(jobs)} tumor+ cases, {NUM_WORKERS} workers", flush=True)
    with mp.Pool(NUM_WORKERS) as pool:
        rows = [r for rs in pool.imap_unordered(measure_case, jobs) for r in rs]
    df = pd.DataFrame(rows).sort_values(["case_id", "lesion_id"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
