"""Known-answer (phantom) tests for the S9 normative model (src/normative/).

Written and passing before any real case was processed. The last test is an end-to-end phantom:
a bank of textured noise "glands", a planted smooth sphere in a held-out one, and the full
features -> PCA -> strata -> scores -> same-footprint percentile chain must rank the sphere near the
top while an unplanted control location ranks like a random patch.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

from normative import common as C
from normative import features as FT
from normative import model as M


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize("r", C.RADII_MM)
def test_ball_and_shells_partition(r):
    offs, d = FT.ball(r)
    assert abs(len(offs) - 4 / 3 * np.pi * r ** 3) / (4 / 3 * np.pi * r ** 3) < 0.05
    assert (offs == 0).all(1).sum() == 1
    edges = FT.shell_edges(r)
    assert edges[0][0] == 0 and abs(edges[-1][1] - r) < 1e-9
    cover = np.zeros(len(d), int)
    for a, b in edges:
        cover += (((d > a) | ((a == 0) & (d == 0))) & (d <= b + 1e-9))
    assert (cover == 1).all()                                       # every voxel in exactly one shell
    ring, rd = FT.ball(r + C.RING_MM, inner_mm=r)
    assert rd.min() > r and rd.max() <= r + C.RING_MM + 1e-9


def test_feature_names_lengths():
    assert len(FT.feature_names(6.0)) == 15 and len(FT.feature_names(10.0)) == 17


def test_grid_centres_lattice_and_region():
    reg = np.zeros((30, 30, 30), bool)
    reg[5:22, 3:17, 9:30] = True
    c = FT.grid_centres(reg, 4)
    assert (c % 4 == 0).all() and reg[tuple(c.T)].all()
    brute = np.argwhere(reg)
    assert len(c) == ((brute % 4 == 0).all(1)).sum()


# ------------------------------------------------------------------ features
def _maps(vol):
    return FT.case_maps(vol)


def test_constant_image_features():
    vol = np.full((50, 50, 50), 80.0, np.float32)
    mp = _maps(vol)
    c = np.array([[24, 24, 24]])
    for r in C.RADII_MM:
        f = dict(zip(FT.feature_names(r), FT.patch_features(mp, c, r)[0]))
        assert f["hu_mean"] == pytest.approx(80) and f["hu_std"] == pytest.approx(0, abs=1e-5)
        assert all(f[f"hu_q{q:02d}"] == pytest.approx(80) for q in FT.QUANTILES)
        assert all(abs(v) < 1e-4 for k, v in f.items() if k.startswith("rad_"))
        assert abs(f["ring_contrast"]) < 1e-4 and abs(f["grad_mean"]) < 1e-3
        assert f["entropy"] == pytest.approx(0, abs=1e-9)
        # scipy's truncated (4 sigma) 2nd-derivative kernel does not sum exactly to 0, so a flat
        # image gives sigma^2*LoG ~ -0.0015 * level (-0.12 at 80 HU, sigma 3.5 mm). S4's LoG shares
        # this; it is ~1% of a 30 HU lesion's response, so it is documented rather than corrected.
        assert abs(f["log_raw"]) < 0.003 * 80


def test_hu_clipping_applies():
    vol = np.full((40, 40, 40), 1000.0, np.float32)
    f = FT.patch_features(_maps(vol), np.array([[20, 20, 20]]), 6.0)[0]
    assert f[0] == pytest.approx(C.HU_CLIP[1])


def _ball_vol(shape, centre, radius, bg=80.0, dhu=-40.0):
    g = np.indices(shape).astype(float)
    d = np.sqrt(sum((g[i] - centre[i]) ** 2 for i in range(3)))
    return (bg + dhu * (d <= radius)).astype(np.float32), d


def test_dark_ball_features_have_the_expected_signs():
    vol, _ = _ball_vol((60, 60, 60), (30, 30, 30), 6.0)
    mp = _maps(vol)
    f = dict(zip(FT.feature_names(6.0), FT.patch_features(mp, np.array([[30, 30, 30]]), 6.0)[0]))
    assert f["hu_q50"] == pytest.approx(40) and f["ring_contrast"] == pytest.approx(-40, abs=1)
    assert f["log_raw"] > 0                                         # dark blob -> positive (S4 sign)
    fb = dict(zip(FT.feature_names(6.0),
                  FT.patch_features(_maps(_ball_vol((60, 60, 60), (30, 30, 30), 6.0, dhu=+40)[0]),
                                    np.array([[30, 30, 30]]), 6.0)[0]))
    assert fb["log_raw"] < 0 and fb["ring_contrast"] == pytest.approx(40, abs=1)


def test_log_scale_law_raw_response_peaks_near_matching_radius():
    """sigma = r/sqrt(3) is the scale whose raw sigma^2*LoG peaks for a ball of radius r."""
    resp = {}
    vol, _ = _ball_vol((80, 80, 80), (40, 40, 40), 8.0)
    f = FT.clip_hu(vol)
    for r in (4.0, 6.0, 8.0, 10.0, 12.0):
        resp[r] = FT.log_raw_map(f, r)[40, 40, 40]
    assert max(resp, key=resp.get) == 8.0


def test_entropy_of_two_level_patch_is_one_bit():
    v = np.array([[0.0] * 50 + [200.0] * 50])
    assert FT.entropy_rows(v)[0] == pytest.approx(1.0)
    assert FT.entropy_rows(np.full((1, 10), 300.0))[0] == pytest.approx(0.0)  # top edge stays in range


def test_features_are_rotation_invariant_for_lattice_rotations():
    # odd-sized cube: the centre voxel maps to itself under every lattice rotation / flip
    rng = np.random.default_rng(0)
    vol = gaussian_filter(rng.normal(60, 30, (49, 49, 49)), 1.5).astype(np.float32)
    c = np.array([[24, 24, 24]])
    base = {r: FT.patch_features(_maps(vol), c, r)[0] for r in C.RADII_MM}
    for rot in (np.rot90(vol, 1, (0, 1)), np.rot90(vol, 2, (1, 2)), vol.transpose(2, 0, 1)[::-1]):
        rot = np.ascontiguousarray(rot)
        for r in C.RADII_MM:
            np.testing.assert_allclose(FT.patch_features(_maps(rot), c, r)[0], base[r], rtol=1e-4, atol=1e-3)


# ------------------------------------------------------------------ gland frame
def test_gland_frame_matches_lesion_sectioning_build_frame():
    sys.path.insert(0, str(C.REPO / "src" / "lesion-sectioning"))
    from pca_hitrate import build_frame
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(5000, 3)) * [40, 8, 5]
    rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    pts = raw @ rot.T + [10, -20, 30]
    head, tail = pts[raw[:, 0] < -30].mean(0), pts[raw[:, 0] > 30].mean(0)
    fr = FT.gland_frame(pts, head, tail)
    cen, R = build_frame(pts, head, tail)
    assert np.allclose(fr["axis"], R[:, 0]) and np.allclose(fr["centroid"], cen)
    u = FT.u_of(np.stack([head, tail]), fr)
    assert u[0] < 0.25 and u[1] > 0.75
    uu = FT.u_of(pts, fr)
    assert np.mean((uu >= 0) & (uu <= 1)) == pytest.approx(0.98, abs=0.005)


# ------------------------------------------------------------------ insertion
@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (0.8, 0.8, 2.5), (0.7, 0.7, 5.0)])
def test_sphere_profile_partial_volume_preserves_the_integral(spacing):
    R, dhu = 7.5, -30.0
    shape = tuple(int(60 / s) for s in spacing)
    centre = np.array([30.3, 29.6, 30.9])
    sl, block = FT.sphere_profile(shape, spacing, centre, R, dhu)
    total = block.sum() * np.prod(spacing)
    _, ref = FT.sphere_profile(tuple(int(60 / 0.5) for _ in range(3)), (0.5, 0.5, 0.5), centre, R, dhu, sub_mm=0.5)
    ref_total = ref.sum() * 0.125
    assert total == pytest.approx(ref_total, rel=0.01)
    if spacing == (1.0, 1.0, 1.0):
        full = np.zeros(shape, np.float32)
        full[sl] = block
        cv = full[30, 30, 31]
        assert cv == pytest.approx(dhu, rel=0.02)                 # centre = dHU
        # edge voxels follow the analytic soft-edge profile (d = 6.71 and 7.71 mm, R = 7.5)
        from scipy.special import erfc
        for x in (37, 38):
            d = np.linalg.norm(np.array([x, 30, 31]) - centre)
            assert full[x, 30, 31] == pytest.approx(0.5 * dhu * erfc((d - R) / np.sqrt(2)), abs=1.0)


# ------------------------------------------------------------------ candidates + random sets
def test_grid_local_maxima_returns_bumps_first():
    reg = np.ones((60, 60, 60), bool)
    c = FT.grid_centres(reg, 4)
    s = (np.exp(-((c - [20, 20, 20]) ** 2).sum(1) / 50) * 5 + np.exp(-((c - [44, 44, 44]) ** 2).sum(1) / 50) * 3)
    k = FT.grid_local_maxima(c, s, top_k=5)
    assert tuple(c[k[0]]) == (20, 20, 20) and tuple(c[k[1]]) == (44, 44, 44)
    assert (np.diff(s[k]) <= 0).all()
    d = np.linalg.norm(c[k][:, None] - c[k][None], axis=2) + np.eye(len(k)) * 99
    assert d.min() >= C.CAND_SEP_MM


def test_set_percentile_definition():
    p, f = FT.set_percentile(5.0, np.array([1, 2, 5, 7.0]))
    assert f == 0.5 and p == 50.0


def test_footprint_protocol_is_calibrated_on_a_null_map():
    """A random compact 'lesion' set on an iid null map must have a ~uniform percentile."""
    rng = np.random.default_rng(3)
    reg = np.zeros((80, 80, 80), bool)
    reg[8:72, 20:60, 20:60] = True
    c = FT.grid_centres(reg, 4).astype(float)
    pct = []
    for _ in range(300):
        s = rng.normal(size=len(c))
        n_foot = int(rng.integers(1, 20))
        les = FT.footprint_sets(c, n_foot, rng, n_sets=1)[0]
        d = np.min(np.linalg.norm(c[:, None] - c[les][None], axis=2), axis=1)
        far = np.flatnonzero(d > C.FP_EXCL_MM)
        sets = FT.footprint_sets(c[far], n_foot, rng)
        pct.append(FT.set_percentile(s[les].max(), s[far][sets].max(1))[0])
    pct = np.asarray(pct)
    assert abs(pct.mean() - 50) < 5 and abs(np.mean(pct >= 90) - 0.1) < 0.05


# ------------------------------------------------------------------ model
def test_merge_strata_rule():
    full = {(p, t, u): 5000 for p in M.PHASES for t in M.THICKS for u in M.UTERS}
    k = M.merge_strata(full)
    assert len(set(k.values())) == 18
    c = dict(full)
    c[("arterial", "thick", "u2")] = 100                             # a: merge thickness in (art, u2)
    k = M.merge_strata(c)
    assert k[("arterial", "thin", "u2")] == k[("arterial", "thick", "u2")] == "arterial|*|u2"
    assert k[("arterial", "thin", "u1")] == "arterial|thin|u1"
    c = {kk: (600 if kk[0] == "venous" else 5000) for kk in full}    # b: venous merges u then all
    k = M.merge_strata(c)
    assert {k[kk] for kk in full if kk[0] == "venous"} == {"venous|*|*"}
    c = {kk: (1500 if kk[0] == "venous" else 5000) for kk in full}   # 1500*3 = 4500 per thickness
    k = M.merge_strata(c)
    assert {k[kk] for kk in full if kk[0] == "venous"} == {"venous|*|u1", "venous|*|u2", "venous|*|u3"}


def test_within_case_z_is_leave_one_out():
    s = np.random.default_rng(4).normal(size=40)
    z = M.within_case_z(s)
    for i in (0, 7, 39):
        o = np.delete(s, i)
        assert z[i] == pytest.approx((s[i] - o.mean()) / o.std(ddof=1))


def _bank_frame(rng, n_cases, n_per, d, phase="venous"):
    rows = []
    for ci in range(n_cases):
        X = rng.normal(size=(n_per, d)) * np.linspace(3, 0.5, d)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
        df["case_id"] = f"c{ci}"
        df["phase_group"], df["thick_group"] = phase, "thin"
        df["u"] = rng.uniform(0, 1, n_per)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def test_scores_rank_an_outlier_above_inliers_and_unknown_keys_fall_back():
    rng = np.random.default_rng(5)
    d = 6
    bank = pd.concat([_bank_frame(rng, 10, 700, d, "venous"), _bank_frame(rng, 10, 700, d, "arterial")])
    names = [f"f{i}" for i in range(d)]
    mdl = M.fit(bank, names, 6.0)
    # 7000 patches per phase, all thin: step a merges thickness (thick cells are empty), and each
    # (phase, *, u tercile) keeps ~2330 >= 2000 patches, so the u split survives
    assert mdl.report["n_components"] >= 1
    assert set(mdl.strata) == {f"{p}|*|{u}" for p in ("venous", "arterial") for u in M.UTERS}
    inl = rng.normal(size=(200, d)) * np.linspace(3, 0.5, d)
    out = inl[:5] + 12.0
    a, b, lab = mdl.score(np.vstack([inl, out]), "venous", "thin", np.full(205, 0.5))
    assert a[200:].min() > np.percentile(a[:200], 99) and b[200:].min() > np.percentile(b[:200], 99)
    _, _, lab_u = mdl.score(inl[:3], "unknown", "thin", np.array([0.5, np.nan, 0.1]))
    assert set(lab_u[1].split("+")) == set(mdl.strata)               # phase AND u unknown -> all strata
    assert set(lab_u[0].split("+")) == {"venous|*|u2", "arterial|*|u2"}


# ------------------------------------------------------------------ end-to-end phantom
def _phantom_gland(rng, shape=(70, 60, 50)):
    """Textured 'parenchyma' (smoothed noise, ~80 HU) in an ellipsoidal gland, plus a bright tube."""
    tex = gaussian_filter(rng.normal(0, 25, shape), 1.0) * 2.0 + 80.0
    g = np.indices(shape).astype(float)
    gland = (((g[0] - 35) / 30) ** 2 + ((g[1] - 30) / 14) ** 2 + ((g[2] - 25) / 12) ** 2) <= 1
    tube = ((g[1] - 22) ** 2 + (g[2] - 25) ** 2) <= 4
    vol = np.where(gland, tex, -50.0 + rng.normal(0, 10, shape))
    vol[tube] = 200 + rng.normal(0, 10, tube.sum())
    return vol.astype(np.float32), gland


def test_end_to_end_phantom_detects_planted_sphere():
    rng = np.random.default_rng(7)
    r = 6.0
    names = FT.feature_names(r)
    rows = []
    for ci in range(12):
        vol, gland = _phantom_gland(rng)
        cen = FT.grid_centres(gland)
        F = FT.patch_features(FT.case_maps(vol), cen, r)
        df = pd.DataFrame(F, columns=names)
        df["case_id"], df["phase_group"], df["thick_group"] = f"b{ci}", "venous", "thin"
        df["u"] = cen[:, 0] / 70.0
        rows.append(df)
    mdl = M.fit(pd.concat(rows, ignore_index=True), names, r)

    vol, gland = _phantom_gland(rng)
    sl, blk = FT.sphere_profile(vol.shape, (1, 1, 1), np.array([45.0, 32.0, 26.0]), 7.5, -30.0)
    vol[sl] += blk
    cen = FT.grid_centres(gland)
    F = FT.patch_features(FT.case_maps(vol), cen, r)
    a, b, _ = mdl.score(F, "venous", "thin", cen[:, 0] / 70.0)
    for centre, expect_high in (([45.0, 32.0, 26.0], True), ([20.0, 36.0, 26.0], False)):
        dist = np.linalg.norm(cen - centre, axis=1)
        foot = np.flatnonzero(dist <= 7.5 + C.DIL_MM)
        far = np.flatnonzero(dist > 7.5 + C.FP_EXCL_MM)
        pr = np.random.default_rng(0)
        sets = FT.footprint_sets(cen[far].astype(float), len(foot), pr)
        for s in (a, b):
            p, _ = FT.set_percentile(s[foot].max(), s[far][sets].max(1))
            if expect_high:
                assert p >= 90, p
            else:
                assert p < 99, p
