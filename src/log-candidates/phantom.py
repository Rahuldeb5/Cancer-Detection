"""S4 step 0: phantoms for the scale-space LoG detector, run before any real data.

Four known-answer checks, all on logblob.detect() -- the same function the cohort run calls:

  A. scale law     a uniform ball of radius r in noise peaks at sigma = r/sqrt(3). Asserted to
                   land within one step of the frozen sigma ladder.
  B. anisotropy    the same ball built on a native (0.8, 0.8, 5) mm grid, PSF-blurred and
                   partial-volumed, then resampled to 1 mm by tumorlib, must be localized in
                   plane to within 1 mm. The z error is reported, not asserted: a 5 mm slice
                   cannot localize a centre in z to better than ~half a slice.
  C. null          noise only, no object: false peaks per volume and per 100 cm^3 as a function
                   of the z threshold. This is the floor any real FP rate must be read against.
  D. blob vs tube  a dark ball of radius 2 mm and a long dark tube of radius 2 mm at the same
                   contrast: the eigenvalue term blobness = |l1|/|l3| must separate them.

    .venv/bin/python src/log-candidates/phantom.py            # ~4 min
    .venv/bin/python src/log-candidates/phantom.py --quick    # fewer trials
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
import logblob as LB  # noqa: E402
from tumorlib.resample import resample_ct_to_isotropic  # noqa: E402

OUT = REPO / "work" / "log_candidates"

BG_HU = 60.0            # pancreatic parenchyma, non-contrast-ish
CONTRAST_HU = -50.0     # a clearly hypodense lesion
NOISE_HU = 15.0
NOISE_CORR_VOX = 0.7    # in-plane correlation, as in the S3 phantom
PSF_FWHM_XY, PSF_FWHM_Z = 1.2, 1.0
SUB_MM = 0.2            # sub-voxel sampling step for partial volume
RADII_MM = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
Z_GRID = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0)
FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


# ------------------------------------------------------------------ builders
def _axes(shape, spacing):
    return [np.arange(n) * s for n, s in zip(shape, spacing)]


def ball_fraction(shape, spacing, centre_mm, radius_mm, sub_mm: float = SUB_MM) -> np.ndarray:
    """Per-voxel fill fraction of a ball, by sub-voxel sampling (the partial-volume effect).
    A plain centre-inside test would quantize a 3 mm ball to a handful of voxels."""
    sp = np.asarray(spacing, dtype=np.float64)
    S = np.maximum(np.ceil(sp / sub_mm).astype(int), 1)
    base = _axes(shape, sp)
    acc = np.zeros(shape, np.float32)
    offs = [((np.arange(s) + 0.5) / s - 0.5) * sp[a] for a, s in enumerate(S)]
    for ox in offs[0]:
        x = (base[0] + ox - centre_mm[0])[:, None, None] ** 2
        for oy in offs[1]:
            y = (base[1] + oy - centre_mm[1])[None, :, None] ** 2
            for oz in offs[2]:
                z = (base[2] + oz - centre_mm[2])[None, None, :] ** 2
                acc += (x + y + z <= radius_mm ** 2)
    return acc / float(S.prod())


def tube_fraction(shape, spacing, centre_mm, radius_mm, length_mm, axis: int = 0,
                  sub_mm: float = SUB_MM) -> np.ndarray:
    """Per-voxel fill fraction of a circular cylinder along `axis`, same sub-voxel sampling."""
    sp = np.asarray(spacing, dtype=np.float64)
    S = np.maximum(np.ceil(sp / sub_mm).astype(int), 1)
    base = _axes(shape, sp)
    acc = np.zeros(shape, np.float32)
    offs = [((np.arange(s) + 0.5) / s - 0.5) * sp[a] for a, s in enumerate(S)]
    radial = [a for a in range(3) if a != axis]
    for ox in offs[0]:
        for oy in offs[1]:
            for oz in offs[2]:
                d = [(base[a] + o - centre_mm[a]) for a, o in zip(range(3), (ox, oy, oz))]
                shaped = [d[a].reshape([-1 if a == k else 1 for k in range(3)]) for a in range(3)]
                rho2 = shaped[radial[0]] ** 2 + shaped[radial[1]] ** 2
                acc += (rho2 <= radius_mm ** 2) & (np.abs(shaped[axis]) <= length_mm / 2.0)
    return acc / float(S.prod())


def add_noise(vol, rng, spacing, corr: bool = True, noise_hu: float = NOISE_HU) -> np.ndarray:
    """Gaussian noise scaled to noise_hu per voxel, optionally correlated in plane (axes 0,1),
    which is what real CT noise looks like and what makes the LoG response correlated too."""
    w = rng.standard_normal(vol.shape).astype(np.float32)
    if corr:
        w = gaussian_filter(w, (NOISE_CORR_VOX, NOISE_CORR_VOX, 0.0))
    return vol + w / w.std() * noise_hu


def psf(vol, spacing) -> np.ndarray:
    sig = np.array([PSF_FWHM_XY, PSF_FWHM_XY, PSF_FWHM_Z]) / FWHM / np.asarray(spacing, float)
    return gaussian_filter(vol, sig, mode="nearest")


def flat_ball_idx(shape, spacing, centre_mm, radius_mm) -> np.ndarray:
    """Flat indices of voxels within radius_mm of centre_mm (the probe region)."""
    sp = np.asarray(spacing, float)
    base = _axes(shape, sp)
    d2 = ((base[0] - centre_mm[0])[:, None, None] ** 2 + (base[1] - centre_mm[1])[None, :, None] ** 2
          + (base[2] - centre_mm[2])[None, None, :] ** 2)
    return np.flatnonzero((d2 <= radius_mm ** 2).ravel())


def nearest_ladder(sigma_star: float, sigmas=LB.SIGMAS_MM) -> int:
    return int(np.argmin(np.abs(np.asarray(sigmas) - sigma_star)))


# ------------------------------------------------------------------ A. scale law
def check_scale_law(rng, n_trials: int) -> pd.DataFrame:
    """Ball of radius r, dark on uniform background: which sigma gives the largest response?

    The law is about the *raw* scale-normalized response s^2*LoG, which is what s^2
    normalization is designed to equalize across scales. It is asserted on that, at the ball's
    centre voxel, noiseless and in correlated noise.

    The detector ranks peaks by z = response / MAD(response), and the MAD shrinks with s faster
    than the ball's signal does, so z prefers a coarser scale than r/sqrt(3). That shift is
    reported here (it is a property of the ranking, not an error) and is why sigma is stored per
    peak instead of being read off a single map.
    """
    rows = []
    for r in RADII_MM:
        star = r / np.sqrt(3.0)
        want = nearest_ladder(star)
        half = r + 4.0 * max(LB.SIGMAS_MM) + 4.0     # room for the widest kernel
        n = int(np.ceil(2 * half))
        shape, sp = (n, n, n), (1.0, 1.0, 1.0)
        c = np.array([half + 0.37, half - 0.21, half + 0.11])   # off-grid centre
        frac = ball_fraction(shape, sp, c, r)
        clean = BG_HU + CONTRAST_HU * frac
        mask = np.ones(shape, bool)
        centre = np.ravel_multi_index([int(round(v)) for v in c], shape)
        probe = {"centre": np.array([centre]), "les2": flat_ball_idx(shape, sp, c, 2.0)}
        for noisy in (False, True):
            for trial in range(n_trials if noisy else 1):
                ct = add_noise(clean, rng, sp) if noisy else clean
                res = LB.detect(ct, mask, sp, probe_idx=probe, polarities=("dark",))
                nmad = np.array([res["noise"][s] for s in LB.SIGMAS_MM])
                zc = np.asarray(res["probe"]["centre"]["dark"])
                raw = zc * nmad                      # undo the noise normalization
                z2 = np.asarray(res["probe"]["les2"]["dark"])
                g_raw, g_z = int(np.argmax(raw)), int(np.argmax(z2))
                rows.append(dict(radius_mm=r, sigma_star_mm=star, want_sigma_mm=LB.SIGMAS_MM[want],
                                 noisy=noisy, trial=trial,
                                 raw_sigma_mm=LB.SIGMAS_MM[g_raw], raw_idx_err=g_raw - want,
                                 z_sigma_mm=LB.SIGMAS_MM[g_z], z_idx_err=g_z - want,
                                 z_at_best=float(z2[g_z]),
                                 **{f"raw_s{s:g}": float(v) for s, v in zip(LB.SIGMAS_MM, raw)},
                                 **{f"z_s{s:g}": float(v) for s, v in zip(LB.SIGMAS_MM, z2)}))
    df = pd.DataFrame(rows)
    print("\n== A. scale law: argmax sigma of the raw s^2*LoG at the ball centre vs r/sqrt(3)")
    print(f"{'r':>5} {'r/sqrt3':>8} {'ladder':>7} {'raw noiseless':>14} {'raw noisy mode':>15} "
          f"{'raw |err|<=1':>13} {'z-ranked mode':>14} {'median z':>9}")
    for r, g in df.groupby("radius_mm"):
        cl, ns = g[~g.noisy].iloc[0], g[g.noisy]
        print(f"{r:>5.1f} {cl.sigma_star_mm:>8.2f} {cl.want_sigma_mm:>7.1f} {cl.raw_sigma_mm:>14.1f} "
              f"{ns.raw_sigma_mm.mode().iat[0]:>15.1f} {(ns.raw_idx_err.abs() <= 1).mean():>12.0%} "
              f"{ns.z_sigma_mm.mode().iat[0]:>14.1f} {ns.z_at_best.median():>9.1f}")
    bad = df[df.raw_idx_err.abs() > 1]
    assert len(bad) == 0, f"{len(bad)} trials off the ladder by more than one step:\n{bad.head()}"
    print(f"  OK: the raw-response argmax is within one ladder step of r/sqrt(3) in all "
          f"{len(df)}/{len(df)} runs (noiseless max error {df[~df.noisy].raw_idx_err.abs().max()} steps).")
    shift = df[df.noisy].z_idx_err.median() - df[df.noisy].raw_idx_err.median()
    print(f"  NOTE: ranking by z instead of raw response shifts the preferred scale by "
          f"{shift:+.0f} ladder step(s) (median), because MAD(response) falls with sigma.")
    return df


# ------------------------------------------------------------------ B. anisotropic -> resample
def check_anisotropic(rng, n_trials: int) -> pd.DataFrame:
    """Build at (0.8, 0.8, 5) mm with PSF + partial volume, resample to 1 mm, detect.

    Every trial puts the true centre at a fresh random sub-voxel offset (uniform over one
    native voxel, so up to +-2.5 mm in z), which is the only way to see what the 5 mm slice
    grid actually costs. In-plane error is asserted <= 1 mm; the z error is measured and
    reported as the limitation.
    """
    native = (0.8, 0.8, 5.0)
    rows = []
    for r in (5.0, 8.0):
        half = r + 30.0
        shape = tuple(int(np.ceil(2 * half / s)) for s in native)
        pad = int(np.ceil(4 * max(LB.SIGMAS_MM)))
        for trial in range(n_trials):
            off = rng.uniform(-0.5, 0.5, 3) * np.asarray(native)
            c = np.array([half, half, half]) + off
            frac = ball_fraction(shape, native, c, r)
            clean = psf(BG_HU + CONTRAST_HU * frac, native)
            ct = add_noise(clean, rng, native)
            iso, sp_iso, _ = resample_ct_to_isotropic(ct, native, None)
            mask = np.zeros(iso.shape, bool)                     # search only away from the border
            mask[pad:-pad, pad:-pad, pad:-pad] = True
            res = LB.detect(iso, mask, sp_iso, polarities=("dark",))
            top = res["peaks"][0]
            p = np.array([top["ix"], top["iy"], top["iz"]], float) * np.asarray(sp_iso)
            rows.append(dict(radius_mm=r, trial=trial, off_z_mm=off[2], sigma_mm=top["sigma_mm"],
                             z=top["z"], err_x=p[0] - c[0], err_y=p[1] - c[1], err_z=p[2] - c[2],
                             err_inplane=float(np.hypot(p[0] - c[0], p[1] - c[1])),
                             blobness=top["blobness"]))
    df = pd.DataFrame(rows)
    print("\n== B. native (0.8, 0.8, 5) mm -> 1 mm iso: localization of the top dark peak")
    print("     (centre offset drawn uniformly inside one native voxel each trial)")
    for r, g in df.groupby("radius_mm"):
        print(f"  r={r:>4.1f} mm  sigma mode {g.sigma_mm.mode().iat[0]:>3.1f} "
              f"(r/sqrt3 = {r / np.sqrt(3):.2f})  in-plane err median {g.err_inplane.median():.2f} mm, "
              f"max {g.err_inplane.max():.2f}  |z err| median {g.err_z.abs().median():.2f} mm, "
              f"max {g.err_z.abs().max():.2f}  median z {g.z.median():.0f}")
    assert df.err_inplane.max() <= 1.0, f"in-plane error > 1 mm:\n{df.sort_values('err_inplane').tail()}"
    print(f"  OK: in-plane error <= 1 mm in {len(df)}/{len(df)} trials "
          f"(max {df.err_inplane.max():.2f} mm), so the in-plane peak survives the resampling.")
    print(f"  LIMITATION (measured, not assumed): |z error| median {df.err_z.abs().median():.2f} mm, "
          f"max {df.err_z.abs().max():.2f} mm -- up to ~{0.5 * native[2]:.1f} mm, i.e. half a slice. "
          f"Resampling to 1 mm interpolates, it does not add z information.")
    return df


# ------------------------------------------------------------------ C. noise-only null
def check_null(rng, n_trials: int) -> pd.DataFrame:
    """False peaks per volume with no object present, as a function of the z threshold.

    The mask is a ball of ~200 cm^3, the order of a hole-filled, 3 mm-dilated pancreas
    envelope, so "per volume" here is comparable to "per case" on real data. Both correlated
    and white noise are run: correlation changes the number of independent maxima.
    """
    sp = (1.0, 1.0, 1.0)
    rad = (200_000.0 * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)       # 200 cm^3 = 200000 mm^3
    half = rad + 4.0 * max(LB.SIGMAS_MM) + 4.0
    n = int(np.ceil(2 * half))
    shape = (n, n, n)
    c = np.array([half, half, half])
    mask = np.zeros(shape, bool)
    mask.ravel()[flat_ball_idx(shape, sp, c, rad)] = True
    vol_cm3 = mask.sum() / 1000.0
    rows = []
    for corr in (True, False):
        for trial in range(n_trials):
            ct = add_noise(np.full(shape, BG_HU, np.float32), rng, sp, corr=corr)
            res = LB.detect(ct, mask, sp)
            for pol in LB.POLARITIES:
                zz = np.array([p["z"] for p in res["peaks"] if p["polarity"] == pol])
                rows.append(dict(corr=corr, trial=trial, polarity=pol, vol_cm3=vol_cm3,
                                 n_peaks=len(zz), top_z=float(zz.max()) if len(zz) else np.nan,
                                 z_min_kept=float(zz.min()) if len(zz) else np.nan,
                                 censored=len(zz) >= LB.TOP_K,
                                 **{f"n_ge_{t:g}": int((zz >= t).sum()) for t in Z_GRID}))
    df = pd.DataFrame(rows)
    print(f"\n== C. noise-only null ({vol_cm3:.0f} cm^3 search mask, top {LB.TOP_K} peaks per polarity)")
    print(f"{'noise':>11} {'pol':>7} {'top z':>7} {'50th z':>7} "
          + " ".join(f"z>={t:g}".rjust(7) for t in Z_GRID))
    for (corr, pol), g in df.groupby(["corr", "polarity"]):
        print(f"{'correlated' if corr else 'white':>11} {pol:>7} {g.top_z.mean():>7.1f} "
              f"{g.z_min_kept.mean():>7.1f} "
              + " ".join(f"{g[f'n_ge_{t:g}'].mean():>7.1f}" for t in Z_GRID))
    print(f"  CENSORING: every run keeps all {LB.TOP_K} peaks, so any count that equals "
          f"{LB.TOP_K} is a floor, not a rate. The 50th peak sits at z ~ "
          f"{df.z_min_kept.mean():.1f}, so only thresholds above that are uncensored here.")
    d = df[df["corr"] & (df.polarity == "dark")]
    print("  per 100 cm^3, correlated noise, dark: "
          + ", ".join(f"z>={t:g}: {d[f'n_ge_{t:g}'].mean() * 100 / vol_cm3:.2f}" for t in Z_GRID))
    assert d["n_ge_20"].mean() < 1.0, "pure noise produces z >= 20 peaks: the z scale is not calibrated"
    print("  OK: pure correlated noise gives < 1 peak per volume at z >= 20; the counts above are the"
          " false-peak floor the real FROC has to be read against.")
    return df


# ------------------------------------------------------------------ D. blob vs tube
def check_blob_vs_tube(rng, n_trials: int) -> pd.DataFrame:
    """A dark r=2 mm ball and a long dark r=2 mm tube at the same contrast: does blobness split them?"""
    sp = (1.0, 1.0, 1.0)
    r, length = 2.0, 60.0
    half = 40.0
    n = int(np.ceil(2 * half))
    shape = (n, n, n)
    c = np.array([half + 0.3, half - 0.2, half + 0.1])
    fr_ball = ball_fraction(shape, sp, c, r)
    fr_tube = tube_fraction(shape, sp, c, r, length, axis=0)
    mask = np.zeros(shape, bool)
    pad = int(np.ceil(4 * max(LB.SIGMAS_MM)))
    mask[pad:-pad, pad:-pad, pad:-pad] = True
    probe = {"obj": flat_ball_idx(shape, sp, c, 2.0)}
    rows = []
    for name, frac in (("ball", fr_ball), ("tube", fr_tube)):
        clean = BG_HU + CONTRAST_HU * frac
        for trial in range(n_trials):
            ct = add_noise(clean, rng, sp)
            res = LB.detect(ct, mask, sp, polarities=("dark",))
            cm = np.array([[p["ix"], p["iy"], p["iz"]] for p in res["peaks"]], float)
            d = np.linalg.norm(cm - c, axis=1)
            k = int(np.argmin(d))                                # the peak on the object
            p = res["peaks"][k]
            rows.append(dict(object=name, trial=trial, dist_mm=float(d[k]), rank=p["rank"],
                             sigma_mm=p["sigma_mm"], z=p["z"], blobness=p["blobness"],
                             Ra=p["Ra"], Rb=p["Rb"], n_pos=p["n_pos"]))
    df = pd.DataFrame(rows)
    print("\n== D. r=2 mm dark ball vs r=2 mm dark tube (same contrast), peak on the object")
    for name, g in df.groupby("object"):
        print(f"  {name:>4}: rank {g['rank'].median():>3.0f}  sigma mode {g.sigma_mm.mode().iat[0]:>3.1f}  "
              f"z {g.z.median():>5.1f}  blobness {g.blobness.median():.3f} "
              f"[{g.blobness.min():.3f}-{g.blobness.max():.3f}]  Ra {g.Ra.median():.3f}  "
              f"Rb {g.Rb.median():.3f}  n_pos {g.n_pos.median():.0f}")
    b, t = df[df.object == "ball"].blobness, df[df.object == "tube"].blobness
    assert b.min() > t.max(), f"blobness does not separate ball from tube: ball min {b.min():.3f} <= tube max {t.max():.3f}"
    print(f"  OK: blobness separates them completely -- ball min {b.min():.3f} > tube max {t.max():.3f} "
          f"(a cut at {(b.min() + t.max()) / 2:.2f} is clean on this phantom).")
    return df


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer trials (smoke test)")
    a = ap.parse_args()
    nt = 3 if a.quick else 10
    rng = np.random.default_rng(20260929)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"sigma ladder {LB.SIGMAS_MM} mm, HU clip {LB.HU_CLIP}, min separation {LB.MIN_SEP_MM} mm, "
          f"top K {LB.TOP_K}; {nt} noisy trials per condition")
    out = {"A_scale_law": check_scale_law(rng, nt),
           "B_anisotropic": check_anisotropic(rng, nt),
           "C_null": check_null(rng, max(3, nt // 2)),
           "D_blob_vs_tube": check_blob_vs_tube(rng, nt)}
    for name, df in out.items():
        df.to_csv(OUT / f"phantom_{name}.csv", index=False)
    print(f"\nall phantom checks passed in {time.time() - t0:.0f}s; tables in {OUT}")


if __name__ == "__main__":
    main()
