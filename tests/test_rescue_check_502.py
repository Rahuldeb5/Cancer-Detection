"""S14 oracle tests (written and run before any real-data pass)."""
import itertools

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from rescue_check_502 import core as K
from tumorlib.envelope import fill_search_region


def brute_hits(target, target_lo, obj, anchor, domain_xyz):
    """Literal loop: place obj's anchor on each domain voxel, test for any overlapping voxel."""
    tset = {tuple(np.asarray(target_lo) + v) for v in np.argwhere(target)}
    ovox = np.argwhere(obj) - np.asarray(anchor)
    return np.array([any(tuple(a + u) in tset for u in ovox) for a in domain_xyz])


@pytest.mark.parametrize("seed", range(6))
def test_anchor_hits_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    target = rng.random(tuple(rng.integers(2, 6, 3))) > 0.6
    target[0, 0, 0] = True
    obj = rng.random(tuple(rng.integers(1, 5, 3))) > 0.5
    obj[tuple(np.array(obj.shape) - 1)] = True
    anchor = tuple(int(rng.integers(0, s)) for s in obj.shape)  # anchor need not be an obj voxel
    target_lo = tuple(int(x) for x in rng.integers(0, 8, 3))
    domain = np.array(list(itertools.product(range(16), range(16), range(16))))
    R, origin = K.anchor_hits(target, target_lo, obj, anchor)
    expect = brute_hits(target, target_lo, obj, anchor, domain)
    got = np.array([K.hit_at(R, origin, a) for a in domain])
    assert (got == expect).all()
    assert K.placement_fraction(domain, R, origin) == pytest.approx(expect.mean(), abs=0)


def test_placement_fraction_hand_count():
    # domain: a 10-voxel line along x; target: one voxel at x = 4; obj: a 3-voxel bar anchored at its left end
    domain = np.array([[x, 0, 0] for x in range(10)])
    target = np.ones((1, 1, 1), bool)
    obj = np.ones((3, 1, 1), bool)
    R, origin = K.anchor_hits(target, (4, 0, 0), obj, (0, 0, 0))
    # bar [a, a+2] covers x = 4 for a in {2, 3, 4}
    assert K.placement_fraction(domain, R, origin) == pytest.approx(0.3)
    assert [K.hit_at(R, origin, (a, 0, 0)) for a in range(10)] == [a in (2, 3, 4) for a in range(10)]
    # anchored at the right end instead: a in {4, 5, 6}
    R, origin = K.anchor_hits(target, (4, 0, 0), obj, (2, 0, 0))
    assert [K.hit_at(R, origin, (a, 0, 0)) for a in range(10)] == [a in (4, 5, 6) for a in range(10)]
    # empty domain -> NaN; empty target -> never hits
    assert np.isnan(K.placement_fraction(np.zeros((0, 3), int), R, origin))
    R0, o0 = K.anchor_hits(np.zeros((2, 2, 2), bool), (0, 0, 0), obj, (0, 0, 0))
    assert K.placement_fraction(domain, R0, o0) == 0.0


def test_chance_any():
    assert K.chance_any([0.5, 0.5]) == pytest.approx(0.75)
    assert K.chance_any([0.2]) == pytest.approx(0.2)
    assert K.chance_any([]) == 0.0
    assert K.chance_any([0.1, 0.2, 0.3]) == pytest.approx(1 - 0.9 * 0.8 * 0.7)


def _ball(shape, c, r_mm, sp):
    g = np.indices(shape).astype(float)
    return sum(((g[i] - c[i]) * sp[i]) ** 2 for i in range(3)) <= r_mm ** 2


def _metrics(lesion, panc, sp):
    filled = fill_search_region(panc.astype(np.uint8), sp, K.FILL_CLOSE_MM, 0.0).astype(bool)
    dist = distance_transform_edt(~panc, sampling=sp) if panc.any() else None
    m = K.lesion_panc_metrics(lesion, lesion, panc, filled, dist, sp)
    return m, K.panc_category(m["frac_in"], m["frac_filled"], m["dist_mm"])


def test_panc_metrics_phantom():
    sp = (1.0, 1.0, 2.0)                      # anisotropic z
    shape = (80, 80, 40)
    block = np.zeros(shape, bool)
    block[10:70, 10:70, 5:35] = True          # gland block 60 x 60 x 60 mm
    les = _ball(shape, (40, 40, 20), 8.0, sp)  # 16 mm lesion in the middle

    m, c = _metrics(les, block, sp)           # gland grown around the lesion (what S13 saw for detected ones)
    assert m["frac_in"] == 1.0 and m["shell_frac"] == 1.0 and m["dist_mm"] == 0.0 and c == "included"

    m, c = _metrics(les, block & ~les, sp)    # lesion carved out, gland all around: a cavity
    assert m["frac_in"] == 0.0 and m["frac_filled"] == 1.0 and m["shell_frac"] == 1.0 and c == "dent"
    assert m["dist_mm"] == 1.0                # lesion voxels next to the cavity wall are 1 voxel (1 mm in x/y) away

    les2 = _ball(shape, (40, 40, 20), 4.0, sp)  # smaller lesion beside a gland face, gap 2 voxels in x (2 mm)
    gl = np.zeros(shape, bool)
    gl[46:70, 10:70, 5:35] = True             # lesion spans x 36..44 -> nearest gland voxel x = 46 -> 2 mm
    m, c = _metrics(les2, gl, sp)
    assert m["frac_in"] == 0.0 and m["dist_mm"] == pytest.approx(2.0) and c == "adjacent"
    assert 0 < m["shell_frac"] < 0.5          # gland on one side only

    gl2 = np.zeros(shape, bool)
    gl2[:, :, 30:40] = True                   # gland 10 slices away in z: lesion z 18..22 -> gap 8 slices x 2 mm
    m, c = _metrics(les2, gl2, sp)
    assert m["dist_mm"] == pytest.approx(16.0) and c == "away" and m["shell_frac"] == 0.0

    m, c = _metrics(les2, np.zeros(shape, bool), sp)  # no predicted pancreas at all
    assert np.isnan(m["dist_mm"]) and c == "away" and m["frac_in"] == 0.0


def test_duct_placement_boundary_anisotropic():
    """Identity placement of a lesion near an MPD reproduces 'some lesion voxel <= 10 mm' at the boundary."""
    sp = np.array([0.8, 0.8, 2.5])
    shape = (60, 60, 30)
    mpd = np.zeros(shape, bool)
    mpd[30, 10:50, 10] = True                 # duct along y at x 30, z 10
    near = distance_transform_edt(~mpd, sampling=sp) <= K.DUCT_MM
    for dz, expect in ((4, True), (5, False)):   # 4 slices = 10.0 mm (inclusive), 5 slices = 12.5 mm
        les = np.zeros(shape, bool)
        les[29:32, 28:32, 10 + dz:10 + dz + 3] = True
        lo = np.argwhere(les).min(0)
        crop = les[tuple(slice(a, b + 1) for a, b in zip(lo, np.argwhere(les).max(0)))]
        anchor = np.round(np.argwhere(crop).mean(0)).astype(int)
        R, origin = K.anchor_hits(near, (0, 0, 0), crop, anchor)
        assert K.hit_at(R, origin, lo + anchor) is expect


def test_near_region_accepts_uint8_masks():
    """tumorlib.load_mask returns uint8; ~uint8 is a bitwise NOT. near_region must treat it as a mask."""
    sp = (1.0, 1.0, 2.5)
    m = np.zeros((20, 20, 10), np.uint8)
    m[10, 10, 5] = 1
    a, b = K.near_region(m, sp, 5.0), K.near_region(m.astype(bool), sp, 5.0)
    assert (a == b).all() and a[10, 10, 5] and a[15, 10, 5] and not a[16, 10, 5] and a[10, 10, 7] and not a[10, 10, 8]
