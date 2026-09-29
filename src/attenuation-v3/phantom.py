"""S3 step 1: phantom for the partial-volume bias of small-lesion dHU.

A sphere (r = 2, 4, 8 mm) of known contrast dHU sits in uniform parenchyma (100 HU).
The scanner is simulated on a 0.25 mm fine grid: in-plane Gaussian PSF (FWHM 1.2 mm) +
z Gaussian (FWHM 1.0 mm) + the voxel box (0.75 mm in-plane, slice thickness in z), then
sampled at coarse voxel centres with a random sub-voxel offset. Noise is Gaussian,
correlated in-plane (sigma 0.7 voxel), scaled to NOISE_HU per voxel. The "annotator" mask
is every coarse voxel whose centre lies inside the true sphere.

Estimators compared (tumor side; reference = median of the uniform background pool):
  v2        whole-mask median, pool excludes lesion + 2.5 mm
  core      attn_v3.core_mask (PV-aware depth >= D mm, else the N deepest voxels), median / trimmed mean
The core parameters (D, N) are chosen here, on the phantom only, then frozen in attn_v3.py.
Also checks noise_sigma() against the known voxel SD (white and correlated noise).

    .venv/bin/python src/attenuation-v3/phantom.py   # ~5 min, writes work/attenuation_v3/phantom_*.csv
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter, uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attn_v3 as av  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "work" / "attenuation_v3"
FINE = 0.25
INPLANE = 0.75
PSF_XY_FWHM, PSF_Z_FWHM = 1.2, 1.0
BG_HU, NOISE_HU, NOISE_CORR_VOX = 100.0, 18.0, 0.7
RADII = (2.0, 4.0, 8.0)
THICK = (0.75, 2.5, 5.0)
DHU = (-30.0, -15.0, 20.0)
N_TRIALS = 100
CORE_GRID = list(product((0.5, 1.0, 2.0), (10, 20, 40, 80)))  # (depth_mm, min_vox) candidates
F = 2.0 * np.sqrt(2.0 * np.log(2.0))


def sphere_template(r, thick, rng):
    """Blurred unit-contrast sphere on the coarse grid, the annotator mask, spacing."""
    step = np.array([INPLANE, INPLANE, thick]) / FINE
    assert np.allclose(step, np.round(step))
    step = step.astype(int)
    half = r + 4.0 + thick                               # template half-width, mm
    n = int(np.ceil(2 * half / FINE)) + 1
    ax = np.arange(n) * FINE
    c = half + rng.uniform(-0.5, 0.5, 3) * np.array([INPLANE, INPLANE, thick])
    x, y, z = np.meshgrid(ax - c[0], ax - c[1], ax - c[2], indexing="ij", sparse=True)
    ind = (x * x + y * y + z * z <= r * r).astype(np.float32)
    sig = np.array([PSF_XY_FWHM, PSF_XY_FWHM, PSF_Z_FWHM]) / F / FINE
    blur = gaussian_filter(ind, sig)
    blur = uniform_filter(blur, size=tuple(step))        # voxel box (even z size shifts -FINE/2)
    shift = np.where(step % 2 == 0, -FINE / 2, 0.0)      # uniform_filter window centre offset
    # coarse centres: every step-th fine sample, phase chosen so the grid spans the template
    coarse = blur[::step[0], ::step[1], ::step[2]]
    cx, cy, cz = (np.arange(s) * st * FINE + sh - cc
                  for s, st, sh, cc in zip(coarse.shape, step, shift, c))
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij", sparse=True)
    mask = X * X + Y * Y + Z * Z <= r * r
    return coarse, mask, (INPLANE, INPLANE, thick)


def embed(coarse, mask, spacing, margin_mm=17.0):
    pad = np.ceil(margin_mm / np.asarray(spacing)).astype(int)
    shape = tuple(s + 2 * p for s, p in zip(coarse.shape, pad))
    t = np.zeros(shape, np.float32)
    m = np.zeros(shape, bool)
    sl = tuple(slice(p, p + s) for p, s in zip(pad, coarse.shape))
    t[sl], m[sl] = coarse, mask
    return t, m


def noise(shape, rng, corr=True):
    w = rng.standard_normal(shape).astype(np.float32)
    if corr:
        w = gaussian_filter(w, (NOISE_CORR_VOX, NOISE_CORR_VOX, 0))
    return w / w.std() * NOISE_HU


def run():
    rng = np.random.default_rng(20260928)
    rows, noise_rows = [], []
    for r, thick in product(RADII, THICK):
        n_empty = 0
        for trial in range(N_TRIALS):
            coarse, mask, sp = sphere_template(r, thick, rng)
            if not mask.any():
                n_empty += 1
                continue
            t, m = embed(coarse, mask, sp)
            d_les = distance_transform_edt(~m, sampling=sp)
            pool25, pool5 = d_les > 2.5, d_les > av.SHELL_MM
            ring = pool5 & (d_les <= av.RING_MM[1])
            cores = {cfg: av.core_mask(m, sp, *cfg) for cfg in CORE_GRID}
            for k, corr in enumerate((True, False)):
                nz = noise(t.shape, rng, corr)
                s, _ = av.noise_sigma(BG_HU + nz, pool5, slice_axis=2)
                noise_rows.append(dict(r=r, thick=thick, correlated=corr, sigma_true=NOISE_HU, sigma_est=s))
                if not corr:
                    continue
                for dhu in DHU:
                    ct = BG_HU + dhu * t + nz
                    ref25 = np.median(ct[pool25])
                    ref5 = np.median(ct[pool5])
                    ref_ring = np.median(ct[ring])
                    base = dict(r=r, thick=thick, trial=trial, dhu_true=dhu, n_mask=int(m.sum()),
                                pv_truth_mean=float(t[m].mean()))
                    rows.append(base | dict(est="v2_whole_median", n_used=int(m.sum()),
                                            dhu=np.median(ct[m]) - ref25))
                    rows.append(base | dict(est="whole_trim_ring", n_used=int(m.sum()),
                                            dhu=av.robust_stats(ct[m])[1] - ref_ring))
                    for (D, N), (core, info) in cores.items():
                        med, tm = av.robust_stats(ct[core])
                        tag = f"core_D{D}_N{N}"
                        rows.append(base | dict(est=f"{tag}_median", n_used=info["n_core"],
                                                fallback=info["core_fallback"], dhu=med - ref5))
                        rows.append(base | dict(est=f"{tag}_trim", n_used=info["n_core"],
                                                fallback=info["core_fallback"],
                                                dhu=tm - np.asarray(av.robust_stats(ct[pool5]))[1]))
        if n_empty:
            print(f"r={r} thick={thick}: {n_empty}/{N_TRIALS} trials had an empty mask (skipped)")
    return pd.DataFrame(rows), pd.DataFrame(noise_rows)


def summarize(df):
    df = df.assign(err=df.dhu - df.dhu_true,
                   called_iso=av.classify(df.dhu.to_numpy()) == "iso",
                   true_cls=av.classify(df.dhu_true.to_numpy()))
    g = df.groupby(["est", "r", "thick", "dhu_true"])
    s = g.agg(n=("err", "size"), bias=("err", "mean"), sd=("err", "std"),
              rmse=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
              frac_iso=("called_iso", "mean"), n_used=("n_used", "median"),
              pv_truth=("pv_truth_mean", "mean")).reset_index()
    s["recovered"] = 1 + s.bias / s.dhu_true            # fraction of true contrast recovered (mean)
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, nz = run()
    df.to_csv(OUT / "phantom_trials.csv", index=False)
    s = summarize(df)
    s.to_csv(OUT / "phantom_summary.csv", index=False)
    pd.set_option("display.width", 220, "display.max_rows", 400)

    # a-priori selection rule: lowest RMSE averaged over every (r, thickness, dHU) condition
    sel = (s[s.est.str.startswith("core_")].groupby("est").rmse.mean().sort_values())
    print("\nCore candidates, mean RMSE over all conditions (HU):")
    print(sel.round(2).to_string())

    show = ["v2_whole_median", "whole_trim_ring", sel.index[0],
            sel.index[0].replace("_median", "_trim") if sel.index[0].endswith("_median")
            else sel.index[0].replace("_trim", "_median")]
    t = s[s.est.isin(show)].pivot_table(index=["r", "thick", "dhu_true"], columns="est",
                                        values=["bias", "frac_iso"]).round(2)
    print("\nBias (HU, mean est - true) and fraction called iso (+-10 HU):")
    print(t.to_string())
    print("\nMedian voxels used by the best core:")
    print(s[s.est == sel.index[0]].pivot_table(index="r", columns="thick", values="n_used").to_string())
    print("\nNoise estimator (true voxel SD = %.0f HU):" % NOISE_HU)
    print(nz.groupby("correlated").sigma_est.describe()[["mean", "std", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
