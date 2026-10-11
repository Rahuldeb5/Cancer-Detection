"""S12 step 1 check: do the aorta / veins voxel arrays line up with the CT although their affines differ?

Tallies, over the 1308 cohort, how each vessel-mask affine relates to its CT affine (same, translation
only, or a different orientation code). Then, for up to N_PER_KIND cases per kind, compares two
readings of the aorta mask:
  asis      the mask array on the CT voxel grid (what tumorlib.io.load_mask does)
  worldmap  the mask array flipped so that the mask affine's world coordinates match the CT's
and reports, for each, the eroded-aorta median / IQR HU and its overlap with the veins and postcava
masks. A correctly placed aorta is homogeneous (small IQR) and does not overlap either vein.

    PYTHONPATH=src .venv/bin/python -m phase.affine_check   -> work/phase/affine_check{,_tally}.csv
"""
from __future__ import annotations

from collections import Counter

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion

from phase import core
from tumorlib import io

OUT = core.REPO / "work" / "phase" / "affine_check.csv"
N_PER_KIND = 6


def kind(mask_aff, ct_aff) -> str:
    if np.allclose(mask_aff, ct_aff, atol=1e-3):
        return "same"
    a, b = "".join(nib.aff2axcodes(mask_aff)), "".join(nib.aff2axcodes(ct_aff))
    if a == b:
        return "translation only" if np.allclose(mask_aff[:3, :3], ct_aff[:3, :3], atol=1e-3) else "rot/scale"
    return f"mask {a} vs CT {b}"


def main():
    tally = Counter()
    picks: dict[str, list[str]] = {}
    for c in io.case_ids():
        ct_aff = io.grid(c)[1]
        for name in (core.AORTA_MASK, core.PV_MASK):
            k = kind(nib.load(io.mask_path(c, name)).affine, ct_aff)
            tally[(name, k)] += 1
            if name == core.AORTA_MASK and len(picks.setdefault(k, [])) < N_PER_KIND:
                picks[k].append(c)
    t = pd.Series(tally).rename("n").rename_axis(["mask", "kind"]).reset_index()
    print(t.to_string(index=False))
    t.to_csv(OUT.with_name("affine_check_tally.csv"), index=False)

    rows = []
    for k, cases in picks.items():
        for c in cases:
            T = np.linalg.inv(io.grid(c)[1]) @ nib.load(io.mask_path(c, core.AORTA_MASK)).affine
            flip_axes = tuple(i for i in range(3) if T[i, i] < 0)
            a = io.load_mask(c, core.AORTA_MASK).astype(bool)
            v = io.load_mask(c, core.PV_MASK).astype(bool)
            ivc = io.load_mask(c, core.IVC_MASK).astype(bool)
            ct = io.load_ct(c)
            for reading, m in (("asis", a), ("worldmap", np.flip(a, axis=flip_axes) if flip_axes else a)):
                hu = ct[binary_erosion(m)]
                q1, q2, q3 = np.percentile(hu, [25, 50, 75])
                rows.append({"case_id": c, "kind": k, "reading": reading, "flip_axes": str(flip_axes),
                             "median_hu": q2, "iqr_hu": q3 - q1,
                             "ov_veins": int((m & v).sum()), "ov_postcava": int((m & ivc).sum())})
            del ct
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, float_format="%.1f")
    print(df.to_string(index=False, float_format="%.0f"))


if __name__ == "__main__":
    main()
