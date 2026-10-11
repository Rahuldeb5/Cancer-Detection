"""S12 step 3: oracle for the vessel-HU measurement, before any real data.

Synthetic volumes with known HU (vessel VESSEL_HU, surrounding fat FAT_HU) are written to disk as a
fake PanTS case (ct.nii.gz + segmentations/{aorta,veins}.nii.gz) under work/phase/oracle/, in BOTH
on-disk mask encodings:
  enc1  raw {0,1}, slope 1                      (int8 data written as-is)
  enc2  raw {-128,127}, slope 1/255, inter 0.502 (float {0,1} written to an int8 header; nibabel
        picks exactly the scaling PanTS uses)
They are read back with tumorlib.io.load_mask (gotcha 1) and measured with core.vessel_stats, i.e. the
code the cohort run uses.

Volume model: the vessel is a cylinder; each voxel's value is its vessel fraction (4^3 sub-samples
over the voxel box) mixed between fat and vessel, then blurred by a Gaussian PSF (FWHM 1.2 mm
in-plane, 1.0 mm along z) and given 15 HU white noise. The mask is every voxel whose centre lies
inside the cylinder (how an annotator draws it), then SHIFTED by one voxel.

Geometries x slice spacing x shift direction:
  aorta-like: radius 11 mm, axis along z (craniocaudal)
  vein-like : radius 5 mm, axis along x (in-plane, like the splenic vein)
  spacing (0.8, 0.8, 0.8), (0.75, 0.75, 2.5), (0.75, 0.75, 5.0) mm
  shift +1 voxel along x, along y, along z, and along (1,1,1)
Pass: |median - VESSEL_HU| <= TOL_HU for every condition where any eroded voxel survives; a
condition with zero surviving voxels returns NaN (unmeasurable, reported, never a wrong value).
A 2-voxel shift is run for information only (not part of the pass rule).

    PYTHONPATH=src .venv/bin/python -m phase.oracle
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "work" / "phase" / "oracle"
# tumorlib.io reads these at import time: point it at the synthetic case before importing it.
os.environ["TUMORLIB_MASK_ROOT"] = str(ROOT / "masks")
os.environ["TUMORLIB_CT_ROOT"] = str(ROOT / "ct")

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from phase import core  # noqa: E402
from tumorlib import io  # noqa: E402

VESSEL_HU, FAT_HU = 300.0, -100.0
TOL_HU = 5.0
NOISE_HU = 15.0
PSF_FWHM_MM = (1.2, 1.2, 1.0)
FOV_MM = (200.0, 200.0, 80.0)  # realistic in-plane size: a tiny volume gzips below tumorlib's 1 KB LFS-stub guard
SUB = 4
SPACINGS = ((0.8, 0.8, 0.8), (0.75, 0.75, 2.5), (0.75, 0.75, 5.0))
GEOMS = {"aorta_r11_z": (11.0, 2), "vein_r5_x": (5.0, 0)}   # (radius mm, cylinder axis)
SHIFTS = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1), "xyz": (1, 1, 1)}


def _centres(n, sp):
    return (np.arange(n) - (n - 1) / 2.0) * sp


def phantom(spacing, radius, axis, rng):
    """(ct float32, centre-inside mask uint8) for a cylinder through the volume centre."""
    sp = np.asarray(spacing)
    shape = tuple(int(round(f / s)) for f, s in zip(FOV_MM, sp))
    # a small off-centre offset so the cylinder edge does not sit on a voxel-grid symmetry
    off = np.array([0.31, -0.17, 0.23]) * sp
    perp = [a for a in range(3) if a != axis]
    frac = np.zeros(shape, dtype=np.float32)
    sub = (np.arange(SUB) + 0.5) / SUB - 0.5
    c = [_centres(n, s) - o for n, s, o in zip(shape, sp, off)]
    for du in sub:
        for dv in sub:
            u = (c[perp[0]] + du * sp[perp[0]])
            v = (c[perp[1]] + dv * sp[perp[1]])
            inside = (u[:, None] ** 2 + v[None, :] ** 2) <= radius ** 2
            # broadcast the 2D cross-section along the cylinder axis
            frac += np.expand_dims(inside, axis).astype(np.float32) / SUB ** 2 * np.ones(shape, np.float32)
    ct = FAT_HU + (VESSEL_HU - FAT_HU) * frac
    sigma_vox = np.asarray(PSF_FWHM_MM) / 2.3548 / sp
    ct = gaussian_filter(ct, sigma_vox) + rng.normal(0, NOISE_HU, shape).astype(np.float32)
    u, v = np.meshgrid(c[perp[0]], c[perp[1]], indexing="ij")
    centre_in = (u ** 2 + v ** 2) <= radius ** 2
    mask = (np.expand_dims(centre_in, axis) & np.ones(shape, bool)).astype(np.uint8)
    return ct.astype(np.float32), mask


def shift(mask, d):
    out = np.zeros_like(mask)
    src = tuple(slice(0, n - s) if s > 0 else slice(None) for n, s in zip(mask.shape, d))
    dst = tuple(slice(s, None) if s > 0 else slice(None) for s in d)
    out[dst] = mask[src]
    return out


def write_case(case_id, ct, masks: dict, spacing, encoding):
    aff = np.diag(list(spacing) + [1.0])
    (ROOT / "ct" / case_id).mkdir(parents=True, exist_ok=True)
    seg = ROOT / "masks" / case_id / "segmentations"
    seg.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(ct, aff), ROOT / "ct" / case_id / "ct.nii.gz")
    for name, m in masks.items():
        if encoding == "enc1":
            img = nib.Nifti1Image(m.astype(np.int8), aff)
        else:
            img = nib.Nifti1Image(m.astype(np.float32), aff)
            img.set_data_dtype(np.int8)
        nib.save(img, seg / f"{name}.nii.gz")
        # verify the encoding actually on disk
        r = nib.load(seg / f"{name}.nii.gz")
        raw = np.unique(np.asarray(r.dataobj.get_unscaled()))
        want = {"enc1": {0, 1}, "enc2": {-128, 127}}[encoding]
        assert set(raw.tolist()) == want, (encoding, raw)


def main():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    rng = np.random.default_rng(12)
    rows = []
    for gname, (radius, axis) in GEOMS.items():
        for sp in SPACINGS:
            ct, mask = phantom(sp, radius, axis, rng)
            sax = 2  # diagonal affine: slice axis = array z
            for enc in ("enc1", "enc2"):
                for sname, d in SHIFTS.items():
                    for mult in (1, 2):
                        case_id = f"ORACLE_{gname}_{sp[2]}_{enc}_{sname}{mult}"
                        dd = tuple(mult * x for x in d)
                        write_case(case_id, ct, {"aorta": shift(mask, dd)}, sp, enc)
                        io.grid.cache_clear()
                        m = io.load_mask(case_id, "aorta")
                        assert m is not None and np.array_equal(m, shift(mask, dd)), case_id
                        st = core.vessel_stats(io.load_ct(case_id), m, sax)
                        err = st["median"] - VESSEL_HU
                        rows.append({"geometry": gname, "slice_mm": sp[2], "encoding": enc,
                                     "shift": sname, "shift_vox": mult, **st, "err_hu": err,
                                     "gated": mult == 1})
                        shutil.rmtree(ROOT / "ct" / case_id)
                        shutil.rmtree(ROOT / "masks" / case_id)
            # unshifted reference (enc1) for scale
            st = core.vessel_stats(ct, mask, sax)
            rows.append({"geometry": gname, "slice_mm": sp[2], "encoding": "array", "shift": "none",
                         "shift_vox": 0, **st, "err_hu": st["median"] - VESSEL_HU, "gated": True})
    df = pd.DataFrame(rows)
    out = REPO / "work" / "phase" / "oracle_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.2f")
    pd.set_option("display.width", 200)
    print(df[["geometry", "slice_mm", "encoding", "shift", "shift_vox", "n_slices", "n_vox",
              "median", "iqr", "err_hu"]].to_string(index=False, float_format="%.1f"))

    g = df[df.gated]
    measured = g[g.n_vox > 0]
    fails = measured[measured.err_hu.abs() > TOL_HU]
    # the two encodings must give identical measurements (same mask after load_mask)
    e1 = df[df.encoding == "enc1"].set_index(["geometry", "slice_mm", "shift", "shift_vox"])
    e2 = df[df.encoding == "enc2"].set_index(["geometry", "slice_mm", "shift", "shift_vox"])
    enc_same = np.allclose(e1["median"].to_numpy(), e2.loc[e1.index, "median"].to_numpy(), equal_nan=True)
    print(f"\ngated conditions: {len(g)}, measured: {len(measured)}, unmeasurable (0 voxels): "
          f"{(g.n_vox == 0).sum()}, fails (|err|>{TOL_HU} HU): {len(fails)}, "
          f"max |err| {measured.err_hu.abs().max():.2f} HU, encodings identical: {enc_same}")
    print("ORACLE", "PASS" if len(fails) == 0 and enc_same else "FAIL")


if __name__ == "__main__":
    main()
