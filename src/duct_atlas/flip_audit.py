"""Audit (S10, not a deliverable): is a case's gland affine mirrored in y (anterior <-> posterior)?

The aorta lies posterior to the pancreas, so with a correct affine (aorta centroid - gland centroid).y < 0
(median -35.8 mm over 1594 cases). Both masks are mapped with the pancreas_head affine, so a y-flipped head
affine makes the aorta appear anterior. Runs on the 1608 cases whose aorta mask is real (1308 cohort + 300 S9
bank); also compares the head affine with the CT (ct_staging or ct_normal_bank).

  PYTHONPATH=src .venv/bin/python -m duct_atlas.flip_audit   -> work/duct_atlas/aorta_flip_check.csv
"""
import contextlib, io as sio, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib

from tumorlib import io
from duct_atlas.frame import voxel_to_world
BANK = Path.home() / "research/datasets/pants/ct_normal_bank"

def one(cid):
    out = {"case_id": cid}
    p = io.mask_path(cid, "aorta")
    if p.stat().st_size < 1024:
        return out
    h = nib.load(io.mask_path(cid, "pancreas_head")).affine
    with contextlib.redirect_stdout(sio.StringIO()):
        a = io.load_mask(cid, "aorta")
        g = [io.load_mask(cid, s) for s in ("pancreas_head", "pancreas_body", "pancreas_tail")]
    if a is None or any(m is None for m in g) or not a.any():
        return out
    env = np.argwhere(g[0] | g[1] | g[2])
    if not len(env):
        return out
    ac = voxel_to_world(np.argwhere(a), h).mean(0)
    gc = voxel_to_world(env, h).mean(0)
    out["aorta_minus_gland_y_mm"] = float(ac[1] - gc[1])
    for root, tag in ((io.CT_ROOT, "ct"), (BANK, "bank")):
        cp = root / cid / "ct.nii.gz"
        if cp.exists():
            ca = nib.load(cp).affine
            out["ct_src"] = tag
            out["head_eq_ct"] = bool(np.allclose(ca, h, atol=1e-3))
    return out

if __name__ == "__main__":
    ids = sorted(p.name for p in io.MASK_ROOT.iterdir() if p.name.startswith("PanTS_"))
    ids = [c for c in ids if io.mask_path(c, "aorta").stat().st_size >= 1024]
    with Pool(4) as pool:
        rows = pool.map(one, ids, chunksize=8)
    df = pd.DataFrame(rows)
    df.to_csv("work/duct_atlas/aorta_flip_check.csv", index=False)
    print(len(df), df.ct_src.value_counts(dropna=False).to_dict())
    print(pd.crosstab(df.head_eq_ct, df.aorta_minus_gland_y_mm > 0, dropna=False))
    print(df[(df.aorta_minus_gland_y_mm > 0) | (df.head_eq_ct == False)].to_string())
