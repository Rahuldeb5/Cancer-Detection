"""S15 gland shape profile: oracle / unit tests (fast). The Part A known-answer suite is glandshape.known_answer."""
import numpy as np
import pytest

from glandshape import centerline as C
from glandshape import core as K
from glandshape import orient as OR
from glandshape import outline as O
from glandshape import phantom as Ph
from glandshape import profile as P
from glandshape import scores as S


def _run(tube, anterior=K.WORLD_ANTERIOR, ct=None):
    ol = O.outline_from_mask(tube.mask, tube.spacing, tube.affine)
    cl = C.build_centerline(ol, anterior)
    pr = P.compute_profile(ol, cl, ct_sampler=ct(ol) if ct else None)
    return ol, cl, pr


def _mirror(tube, axes):
    """Same physical object on a grid whose array axes `axes` are reversed (affine adjusted)."""
    m = tube.mask
    A = tube.affine.copy()
    for ax in axes:
        m = np.flip(m, ax)
        A[:3, 3] = A[:3, 3] + A[:3, ax] * (tube.mask.shape[ax] - 1)
        A[:3, ax] = -A[:3, ax]
    return Ph.Tube(np.ascontiguousarray(m), tube.spacing, A, tube.curve, tube.s, tube.length, tube.truth)


# ------------------------------------------------------------------ outline
def test_outline_largest_component_hole_fill_and_mapping():
    sp = (0.8, 0.8, 2.5)
    m = np.zeros((120, 100, 40), np.uint8)
    m[20:80, 20:70, 10:30] = 1          # big box
    m[40:50, 40:50, 18:22] = 0          # enclosed hole
    m[95:110, 80:95, 5:10] = 1          # small second component
    A = np.diag([*sp, 1.0]); A[:3, 3] = (-30, 12, 100)
    ol = O.outline_from_mask(m, sp, A)
    assert ol.n_comp == 2
    assert ol.size1 == 60 * 50 * 20 - 10 * 10 * 4 and ol.size2 == 15 * 15 * 5
    assert ol.frac2 == pytest.approx(ol.size2 / ol.size1) and not ol.fragmented
    assert ol.mask.dtype == np.uint8 and set(np.unique(ol.mask)) <= {0, 1}
    # hole filled: the outline volume is the full box (mm^3), within a 1-voxel boundary shell
    assert ol.vol_mm3 == pytest.approx(48 * 40 * 50, rel=0.08)
    # iso -> world: outline centroid equals the box centroid in world
    box = np.argwhere(np.pad(np.ones((60, 50, 20)), 0)) + (20, 20, 10)
    cw = (box @ A[:3, :3].T + A[:3, 3]).mean(0)
    assert np.allclose(ol.iso_to_world(np.argwhere(ol.mask)).mean(0), cw, atol=0.6)
    # bbox + 10 mm margin before resampling
    assert ol.mask.shape[0] >= 48 + 2 * 10 - 1


def test_outline_fragmented_flag_and_empty():
    m = np.zeros((60, 60, 60), np.uint8)
    m[5:25, 5:25, 5:25] = 1
    m[35:50, 35:50, 35:50] = 1          # 3375 / 8000 > 10%
    ol = O.outline_from_mask(m, (1, 1, 1), np.eye(4))
    assert ol.fragmented
    assert O.outline_from_mask(np.zeros((10, 10, 10), np.uint8), (1, 1, 1), np.eye(4)) is None


# ------------------------------------------------------------------ geometry on known tubes
def test_straight_cylinder_radius_area():
    t = Ph.make_tube(Ph.TubeSpec(kappa=[(120.0, 0.0)], r_head=8, r_tail=8, lob_amp=0.0, spacing=(0.8, 0.8, 0.8)))
    ol, cl, pr = _run(t)
    core = (cl.s > 20) & (cl.s < cl.L - 20)
    # a binary 1 mm outline has a +-0.5 voxel staircase per side: single rays err up to ~1 mm, means do not
    assert np.nanmax(np.abs(pr.r[core] - 8)) < 1.2
    assert np.nanmean(np.abs(pr.r[core] - 8)) < 0.4
    assert np.nanmax(np.abs(np.nanmean(pr.r[core], 0) - 8)) < 0.6
    assert np.allclose(pr.A[core], np.pi * 64, rtol=0.05)
    assert not pr.fold.any() and not pr.censored.any()
    # extended to the boundary: 120 mm curve + two round caps of 8 mm
    assert cl.L == pytest.approx(136, abs=3)


def test_orientation_head_end_first_and_theta_sense_under_mirrored_affines():
    spec = Ph.gland_like(3, bumps=[Ph.Bump(s0=70, theta0=2.0, h=6, ell=10)])
    t0 = Ph.make_tube(spec)
    peaks = []
    for t in (t0, _mirror(t0, (0,)), _mirror(t0, (0, 1))):
        ol, cl, pr = _run(t)
        head = Ph.arc_position(t0, ol.iso_to_world(cl.P[:1]))[0]
        assert head < 5, "u = 0 must be the head end (patient right)"
        bl, _ = S.bulge_local(pr.r)
        i, k = np.unravel_index(np.nanargmax(bl), bl.shape)
        d = ol.R @ P.ray_dirs(cl)[i, k]
        peaks.append((Ph.arc_position(t0, ol.iso_to_world(cl.P[i:i + 1]))[0], k, d))
        # theta = 0 is anterior-aligned in world
        assert cl.info["anterior_alignment"] > 0.5
    for s_pk, k, d in peaks:
        assert abs(s_pk - 70) <= 4
        assert d @ t0.truth["bumps"][0]["outward"] > np.cos(np.deg2rad(30))
    # same theta index (+-1 for the different staircase of each grid) whatever the axis order / handedness;
    # a handedness error would mirror theta (k -> 24 - k), e.g. 20 -> 4
    ks = [k for _, k, _ in peaks]
    assert all(min((a - b) % 24, (b - a) % 24) <= 1 for a in ks for b in ks)


def test_fold_over_flags_tight_bend_only():
    # 90 deg bend of radius 10 mm on a tube of radius 16 mm. The smoothed medial axis rounds the corner (turning
    # radius ~17 mm, measured), so the planes meet inside the gland only where the section is wider than that.
    # (A radius-8 tube on a radius-6 bend does NOT fold: its centerline turns at ~14 mm > 8 mm.)
    tight = Ph.make_tube(Ph.TubeSpec(kappa=[(50, 0), (5 * np.pi, 1 / 10.0), (50, 0)], r_head=16, r_tail=16,
                                     lob_amp=0, spacing=(1, 1, 1)))
    ol, cl, pr = _run(tight)
    s_ph = Ph.arc_position(tight, ol.iso_to_world(cl.P))
    assert pr.fold[(s_ph > 45) & (s_ph < 70)].any()
    assert not pr.fold[(s_ph < 30) | (s_ph > 90)].any()
    straight = Ph.make_tube(Ph.TubeSpec(kappa=[(100, 0)], r_head=8, r_tail=8, lob_amp=0, spacing=(1, 1, 1)))
    _, _, pr2 = _run(straight)
    assert not pr2.fold.any()


def test_ray_origin_centerline_point_and_width():
    # straight, noiseless: origin = centerline point, width = 2R on every axis
    t = Ph.make_tube(Ph.TubeSpec(kappa=[(120.0, 0.0)], r_head=8, r_tail=8, lob_amp=0, spacing=(0.8, 0.8, 0.8)))
    ol, cl, pr = _run(t)
    core = (cl.s > 20) & (cl.s < cl.L - 20)
    assert np.allclose(pr.origin[pr.valid], cl.P[pr.valid]) and not pr.origin_fallback.any()
    assert pr.w.shape == (len(cl.s), 12) and np.nanmax(np.abs(pr.w[core] - 16)) < 1.5
    assert np.allclose(pr.w, pr.r[:, :12] + pr.r[:, 12:], equal_nan=True)
    # a one-sided 8 x 10 mm bulge: the width on the bulge axis rises by ~8 mm; the perpendicular axis does not
    t2 = Ph.make_tube(Ph.TubeSpec(kappa=[(120.0, 0.0)], r_head=8, r_tail=8, lob_amp=0, spacing=(0.8, 0.8, 0.8),
                                  bumps=[Ph.Bump(s0=60, theta0=0.0, h=8, ell=10)]))
    ol2, cl2, pr2 = _run(t2)
    i = int(np.argmin(np.abs(Ph.arc_position(t2, ol2.iso_to_world(cl2.P)) - 60)))
    out = ol2.dir_world_to_iso(t2.truth["bumps"][0]["outward"])
    j = int(np.argmax(np.abs(P.ray_dirs(cl2)[i, :12] @ out)))
    assert pr2.w[i, j] == pytest.approx(16 + 8, abs=1.5)
    assert pr2.w[i, (j + 6) % 12] == pytest.approx(16, abs=1.5)
    b = S.bulge_multiscale(pr2.w)
    k = np.unravel_index(np.nanargmax(b["primary"]), pr2.w.shape)
    assert abs(k[0] - i) <= 2 and min((k[1] - j) % 12, (j - k[1]) % 12) <= 1


def test_ray_origin_nearest_inside_when_centerline_point_outside():
    from glandshape.centerline import Centerline
    m = np.zeros((60, 60, 60), np.uint8)
    m[10:50, 10:50, 10:50] = 1
    m[25:35, 25:35, 10:50] = 0                # a hollow core along z; the "centerline" runs through the hole
    ol = O.outline_from_mask(m, (1, 1, 1), np.eye(4))
    z = np.arange(10.0, 40.0, 2.0)
    c = 29.5 - ol.offset                      # the hole's axis (native x = y = 29.5) in iso coords
    P_ = np.stack([np.full_like(z, c[0]), np.full_like(z, c[1]), z - ol.offset[2]], 1)
    n = len(z)
    T = np.tile([0.0, 0.0, 1.0], (n, 1)); N = np.tile([0.0, 1.0, 0.0], (n, 1)); B = np.tile([1.0, 0.0, 0.0], (n, 1))
    cl = Centerline(s=z - z[0], u=(z - z[0]) / (z[-1] - z[0]), P=P_, T=T, N=N, B=B, L=float(z[-1] - z[0]),
                    dense=P_, info={})
    pr = P.compute_profile(ol, cl, do_fold=False)
    assert pr.valid.all() and pr.origin_fallback.all()
    assert np.allclose(pr.origin_shift, 5.5, atol=0.6)     # hole half-width 5 mm (+ sampling)
    # rays now start in the wall: every first exit is within the gland
    assert np.nanmax(pr.r) < 40


def test_footprint_cells_and_chance():
    from glandshape import footprint as F
    t = Ph.make_tube(Ph.TubeSpec(kappa=[(120.0, 0.0)], r_head=10, r_tail=10, lob_amp=0, spacing=(0.8, 0.8, 2.0)))
    ol, cl, pr = _run(t)
    # "lesion": a 6 mm-radius ball inside the tube, centred 5 mm off-axis at phantom s = 60
    i0 = int(np.argmin(np.abs(Ph.arc_position(t, ol.iso_to_world(cl.P)) - 60)))
    d0 = P.ray_dirs(cl)[i0, 3]
    centre_w = ol.iso_to_world(cl.P[i0] + 5.0 * d0)
    ijk = np.argwhere(np.ones(t.mask.shape, bool))
    world = ijk @ t.affine[:3, :3].T + t.affine[:3, 3]
    les = ijk[np.linalg.norm(world - centre_w, axis=1) <= 6.0]
    fp = F.footprint(ol, cl, pr, les)
    st = F.footprint_stops(fp)
    s_ph = Ph.arc_position(t, ol.iso_to_world(cl.P[st]))
    assert s_ph.min() >= 60 - 7.5 and s_ph.max() <= 60 + 7.5      # ball spans +-6 mm (+ native voxel)
    assert fp[i0].all()                                           # every axis through the centre stop hits it
    assert 0 < F.chance_fraction(fp, pr) == pytest.approx(fp.sum() / F.scoreable(pr).sum())   # v5: tips excluded
    assert not F.fold_heavy(fp, pr) and not F.censored_footprint(fp, pr)
    # a lesion outside the gland: no footprint (not evaluable)
    far = les + np.array([0, 0, 40])
    far = far[far[:, 2] < t.mask.shape[2]]
    assert not F.footprint(ol, cl, pr, far).any()


def test_qc_rules():
    from glandshape import qc
    assert qc.outline_drop(None) == "empty"
    m = np.zeros((40, 40, 40), np.uint8); m[10:20, 10:20, 10:20] = 1          # 1 cm^3
    ol = O.outline_from_mask(m, (1, 1, 1), np.eye(4))
    assert qc.outline_drop(ol) == "vol_lt_5cm3"
    m = np.zeros((90, 90, 90), np.uint8); m[5:85, 5:85, 5:85] = 1             # 512 cm^3
    assert qc.outline_drop(O.outline_from_mask(m, (1, 1, 1), np.eye(4))) == "vol_gt_250cm3"
    t = Ph.make_tube(Ph.TubeSpec(kappa=[(120.0, 0.0)], r_head=8, r_tail=8, lob_amp=0, spacing=(1, 1, 1)))
    ol, cl, _ = _run(t)
    assert qc.outline_drop(ol) is None and qc.flags(ol) == []
    d = qc.bbox_diag_mm(ol)
    assert d == pytest.approx(np.sqrt(136 ** 2 + 17 ** 2 + 17 ** 2), abs=3)
    assert qc.centerline_drop(ol, cl.L) is None
    assert qc.centerline_drop(ol, 1.51 * d) == "centerline_too_long"


def test_hu_sampler_matches_world_function():
    t = Ph.make_tube(Ph.TubeSpec(kappa=[(80.0, 0.0)], r_head=8, r_tail=8, lob_amp=0, spacing=(0.7, 0.9, 2.0)))
    A = t.affine
    shape = t.mask.shape
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1)
    world = ijk @ A[:3, :3].T + A[:3, 3]
    ct = (2.0 * world[..., 0] - 3.0 * world[..., 1] + 0.5 * world[..., 2]).astype(np.float32)
    ol, cl, pr = _run(t, ct=lambda ol: P.make_ct_sampler(ol, shape, lambda sl: ct[sl]))
    i = len(cl.s) // 2
    pts = pr.origin[i] + P.T_HU[:, None] * P.ray_dirs(cl)[i, 5]
    w = ol.iso_to_world(pts)
    expect = 2.0 * w[:, 0] - 3.0 * w[:, 1] + 0.5 * w[:, 2]
    inside = np.all((ol.iso_to_native(pts) >= 0) & (ol.iso_to_native(pts) <= np.array(shape) - 1), 1)
    assert np.allclose(pr.hu[i, 5][inside].astype(float), expect[inside], atol=0.5 + 0.002 * np.abs(expect[inside]))


# ------------------------------------------------------------------ aorta test
def test_aorta_posterior_test_and_flip():
    sp = (1.0, 1.0, 1.0)
    m = np.zeros((80, 80, 40), np.uint8)
    m[20:60, 40:50, 15:25] = 1                         # gland
    aorta = np.zeros_like(m)
    aorta[38:44, 15:21, :] = 1                         # posterior (lower y) in RAS
    ol = O.outline_from_mask(m, sp, np.eye(4))
    r = OR.aorta_test(ol, aorta)
    assert not r["y_flipped"] and r["aorta_dy_mm"] < -20 and r["aorta_status"] == "slab"
    assert np.allclose(OR.anterior_world(r), K.WORLD_ANTERIOR)
    # world y mirrored by the affine (voxels unchanged): the aorta now appears anterior
    Af = np.diag([1.0, -1.0, 1.0, 1.0])
    olf = O.outline_from_mask(m, sp, Af)
    rf = OR.aorta_test(olf, aorta)
    assert rf["y_flipped"] and np.allclose(OR.anterior_world(rf), -K.WORLD_ANTERIOR)
    assert OR.aorta_test(ol, None)["aorta_status"] == "missing"


def test_aorta_inconclusive_band():
    m = np.zeros((80, 80, 40), np.uint8)
    m[20:60, 40:50, 15:25] = 1
    ol = O.outline_from_mask(m, (1, 1, 1), np.eye(4))
    # gland centroid y = 44.5; offsets -12 / -6 / +9 / +18 mm
    for y0, call in ((30, "normal"), (36, "inconclusive"), (51, "inconclusive"), (60, "flipped")):
        a = np.zeros_like(m); a[38:44, y0:y0 + 6, :] = 1      # aorta centroid y = y0 + 2.5; gland 44.5
        r = OR.aorta_test(ol, a)
        assert r["aorta_call"] == call, (y0, r)
        assert r["y_flipped"] == (call == "flipped")


def test_aorta_test_accepts_uint8_without_bitwise_not():
    m = np.zeros((40, 40, 20), np.uint8); m[10:30, 20:25, 5:15] = 1
    a = np.zeros_like(m); a[18:22, 5:9, :] = 1
    ol = O.outline_from_mask(m, (1, 1, 1), np.eye(4))
    assert OR.aorta_test(ol, a)["aorta_n_slab"] == 4 * 4 * 10


# ------------------------------------------------------------------ scores
def test_bulge_multiscale_bump_dent_and_long_bump():
    rng = np.random.default_rng(0)
    r = 8 + 0.3 * rng.standard_normal((90, 24))
    r[40:44, 5:8] += 4                       # 8 mm long bump
    r[15:19, 18:21] -= 4                     # dent
    b = S.bulge_multiscale(r)
    assert np.nanmin(b["primary"]) >= 0 and np.nanmax(b["dent"]) <= 0
    i, k = np.unravel_index(np.nanargmax(b["primary"]), r.shape)
    assert 40 <= i <= 43 and 5 <= k <= 7
    i, k = np.unravel_index(np.nanargmin(b["dent"]), r.shape)
    assert 15 <= i <= 18 and 18 <= k <= 20
    assert len(b["mads"]) == 3 and all(0.2 < m < 0.8 for m in b["mads"])
    # 40 mm long (20 stops) bump: v1's +-10 mm median absorbs it, the +-45 mm scale does not
    r2 = 8 + 0.3 * rng.standard_normal((90, 24))
    r2[35:55, 10:13] += 4
    v1, _ = S.bulge_local_v1(r2)
    v2 = S.bulge_multiscale(r2)
    c = (45, 11)
    assert v2["primary"][c] > 5 and v1[c] < 2
    i, k = np.unravel_index(np.nanargmax(v2["primary"]), r2.shape)
    assert 35 <= i <= 54 and 10 <= k <= 12
    assert v2["best_scale"][c] == 2


def test_smooth_map_wraps_theta_and_ignores_nan():
    z = np.zeros((10, 24)); z[5, 0] = 9.0
    sm = S.smooth_map(z)
    assert sm[5, 23] == pytest.approx(1.0) and sm[5, 1] == pytest.approx(1.0) and sm[5, 2] == 0
    z[5, 1] = np.nan
    sm = S.smooth_map(z)
    assert np.isnan(sm[5, 1]) and sm[5, 0] == pytest.approx(9.0 / 8)


def test_atrophy_peaks_at_area_step_and_reference_z():
    A = np.r_[np.full(40, 200.0), np.full(40, 140.0)]
    a = S.atrophy_raw(A)
    # whole-side medians: a clean step gives log(0.7) at EVERY evaluable stop (a plateau, not a peak at the step)
    assert np.allclose(a[5:-5], np.log(0.7))
    assert np.isnan(a[:5]).all() and np.isnan(a[-5:]).all()

    class G:  # normal glands: constant area
        def __init__(self, seed):
            self.u = np.linspace(0, 1, 80)
            self.A = 200 + 5 * np.random.default_rng(seed).standard_normal(80)
            self.w = np.full((80, 12), 16.0)
    ref = S.build_reference([G(s) for s in range(30)])
    z = S.atrophy_z(A, np.linspace(0, 1, 80), ref)
    mag, signed, i = S.case_atrophy(z)
    assert signed < 0 and mag > 5
    bg = S.bulge_global(np.full((80, 12), 16.0), np.linspace(0, 1, 80), ref)
    assert np.allclose(bg[np.isfinite(bg)], 0)


# ------------------------------------------------------------------ Part C definitions (round 4)
def test_partc_axis_arc_and_percentile():
    from glandshape import partc as PC
    assert PC.axis_arc([11, 0, 1]) == (11, 3)
    assert PC.axis_arc([5]) == (5, 1)
    assert PC.axis_arc(range(12)) == (0, 12)
    assert PC.axis_arc([0, 6]) in ((0, 7), (6, 7))
    score = np.zeros((30, 12))
    fp = np.zeros((30, 12), bool)
    fp[10:13, 3:5] = True
    score[11, 4] = 5.0
    r = PC.within_scan_percentile(score, fp)
    assert r["win_stops"] == 3 and r["win_axes"] == 2 and r["percentile"] == 100.0
    # null windows: 28 stop offsets x 12 axis offsets, minus those touching the 3 x 2 footprint (5 x 3 offsets)
    assert r["n_windows"] == 28 * 12 - 5 * 3
    # everything tied -> 50
    r = PC.within_scan_percentile(np.ones((30, 12)), fp)
    assert r["percentile"] == 50.0
    # a higher window elsewhere lowers the percentile; NaN cells are ignored
    score[25, 9] = 9.0
    score[0, :] = np.nan
    r = PC.within_scan_percentile(score, fp)
    assert 90 < r["percentile"] < 100


def test_partc_top_peaks_suppression():
    from glandshape import partc as PC
    s = np.arange(40) * 2.0
    sc = np.zeros((40, 12))
    sc[20, 3] = 10; sc[21, 3] = 9; sc[20, 4] = 8.5          # suppressed neighbours (< 10 mm and < 45 deg)
    sc[20, 6] = 8                                           # 45 deg away: kept
    sc[26, 3] = 7                                           # 12 mm away: kept
    pk = PC.top_peaks(sc, s)
    assert pk == [(20, 3), (20, 6), (26, 3)]
    fp = np.zeros_like(sc, bool); fp[26, 3] = True
    assert PC.top3_hit(sc, s, fp)


def test_partc_uniform_chance_exact_matches_monte_carlo():
    from glandshape import partc as PC
    rng = np.random.default_rng(1)
    n = 18
    s = np.arange(n) * 2.0
    sc_ok = np.ones((n, 12), bool); sc_ok[0, :] = False; sc_ok[5, 7] = False
    fp = np.zeros((n, 12), bool); fp[8:10, 2:4] = True
    exact = PC.chance_top3_uniform(fp, sc_ok, s)
    cells = np.argwhere(sc_ok)
    hits, T = 0, 20000
    for _ in range(T):
        allowed = np.ones(len(cells), bool)
        hit = False
        for _k in range(3):
            idx = np.nonzero(allowed)[0]
            if not len(idx):
                break
            c = cells[rng.choice(idx)]
            hit |= fp[c[0], c[1]]
            allowed &= np.logical_not((np.abs(s[cells[:, 0]] - s[c[0]]) < 10)
                                      & (PC.axis_dist_deg(cells[:, 1], c[1]) < 45))
        hits += hit
    assert exact == pytest.approx(hits / T, abs=0.012)


def test_partc_empirical_chance():
    from glandshape import partc as PC
    u = np.linspace(0, 1, 51)
    fp = np.zeros((51, 12), bool); fp[20:25, [11, 0]] = True   # u 0.40-0.48, axes 11 and 0 (wrap)
    peaks = [[(0.45, 0), (0.9, 5)], [(0.45, 5)], [(0.41, 11)], [(0.2, 0)]]
    assert PC.chance_top3_empirical(fp, u, peaks) == 0.5
    assert PC.gate_chance(0.1, 0.5) == 0.5


# ------------------------------------------------------------------ round 5: extrapolated tip stops
def test_tip_stops_are_the_extensions_and_survive_the_flip():
    from glandshape import footprint as F
    t0 = Ph.make_tube(Ph.TubeSpec(kappa=[(120.0, 0.0)], r_head=12, r_tail=6, lob_amp=0, spacing=(1, 1, 1)))
    for t in (t0, _mirror(t0, (0,))):                    # the mirrored grid reverses the skeleton path order
        ol, cl, pr = _run(t)
        s_ph = Ph.arc_position(t0, ol.iso_to_world(cl.P))
        assert cl.tip.any() and np.array_equal(pr.tip, cl.tip)
        # head-end tip stops sit before the curve start (arc position 0), tail tips after its end
        head_tip = cl.tip & (cl.s < cl.L / 2)
        tail_tip = cl.tip & (cl.s >= cl.L / 2)
        assert head_tip.any() and tail_tip.any()
        assert np.all(s_ph[head_tip] <= 1.0) and np.all(s_ph[tail_tip] >= 119.0)
        assert cl.info["ext_head_mm"] == pytest.approx(cl.s[head_tip].max() + 2, abs=2.1)
        # a 12 mm round head cap is longer than the 6 mm tail cap
        assert cl.info["ext_head_mm"] > cl.info["ext_tail_mm"]
        # excluded from scoring and from footprints; interior untouched
        sc = S.primary_scores(pr)
        assert np.isnan(sc["primary"][cl.tip]).all() and np.isfinite(sc["primary"][np.logical_not(cl.tip)]).any()
        assert not F.scoreable(pr)[cl.tip].any()
