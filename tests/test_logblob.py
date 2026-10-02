"""Fast known-answer tests for the S4 scale-space LoG detector (src/log-candidates/logblob.py).

The cohort-scale checks live in src/log-candidates/phantom.py; these are the cheap invariants
that a refactor could silently break. test_nms_points_accepts_the_first_point is a regression
for a real bug: the greedy thinning used all() over an empty "kept" list, which is True, so it
rejected every candidate and the pooled peak list came back empty.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "log-candidates"))
import logblob as LB


def dark_ball(shape, spacing, centre_mm, radius_mm, bg=60.0, contrast=-50.0):
    g = [np.arange(n) * s - c for n, s, c in zip(shape, spacing, centre_mm)]
    d2 = g[0][:, None, None] ** 2 + g[1][None, :, None] ** 2 + g[2][None, None, :] ** 2
    return (bg + contrast * (d2 <= radius_mm ** 2)).astype(np.float32)


# ------------------------------------------------------------------ small pieces
def test_clip_hu_bounds_and_dtype():
    out = LB.clip_hu(np.array([-3000.0, -100.0, 0.0, 300.0, 4000.0]))
    assert out.dtype == np.float32
    assert out.tolist() == [-100.0, -100.0, 0.0, 300.0, 300.0]


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (0.8, 0.8, 2.5), (0.7, 0.7, 5.0)])
def test_ball_offsets_are_exactly_the_voxels_within_the_radius(spacing):
    offs = LB.ball_offsets(5.0, spacing)
    d = np.linalg.norm(offs * np.asarray(spacing), axis=1)
    assert d.max() <= 5.0 + 1e-9
    assert (offs == 0).all(axis=1).sum() == 1                    # the centre is included once
    assert len(np.unique(offs, axis=0)) == len(offs)             # no duplicates
    # nothing within the radius is missing: compare against a brute-force box
    rad = np.ceil(5.0 / np.asarray(spacing)).astype(int)
    box = np.stack(np.meshgrid(*[np.arange(-r, r + 1) for r in rad], indexing="ij"), -1).reshape(-1, 3)
    want = box[np.linalg.norm(box * np.asarray(spacing), axis=1) <= 5.0]
    assert len(offs) == len(want)


def test_robust_scale_matches_the_mad_of_a_known_sample():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])                    # median 0, MAD 1
    assert LB.robust_scale(x) == pytest.approx(LB.MAD_TO_SIGMA)
    assert np.isnan(LB.robust_scale(np.zeros(0)))
    assert LB.robust_scale(np.ones(10)) == LB.EPS                # constant map cannot divide by zero


# ------------------------------------------------------------------ peak thinning
def test_nms_points_accepts_the_first_point():
    """Regression: with nothing kept yet, the first (strongest) point must be accepted."""
    pts = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    kept = LB.nms_points(pts, np.array([1.0, 2.0]), min_sep_mm=5.0, top_k=50)
    assert kept.tolist() == [1, 0]                               # strongest first, both far apart


def test_nms_points_enforces_separation_and_top_k():
    pts = np.array([[0.0, 0, 0], [3.0, 0, 0], [12.0, 0, 0], [24.0, 0, 0]])
    scores = np.array([5.0, 9.0, 7.0, 1.0])
    assert LB.nms_points(pts, scores, 5.0, 50).tolist() == [1, 2, 3]   # [0] is 3 mm from the winner
    assert LB.nms_points(pts, scores, 5.0, 2).tolist() == [1, 2]


def test_local_maxima_nms_keeps_the_stronger_of_a_close_pair():
    """The 3 mm-away weaker peak is suppressed; the far one survives.

    A flat background is a plateau, so every background voxel also satisfies "no neighbour is
    larger" and pads the list with zero-valued peaks. That is harmless -- they sort last and sit
    far below any z threshold -- but it is why this asserts the head of the list, not the set.
    """
    resp = np.zeros((40, 40, 40), np.float32)
    resp[10, 10, 10] = 5.0
    resp[13, 10, 10] = 9.0                                       # 3 mm away: suppressed
    resp[30, 10, 10] = 7.0                                       # far away: kept
    mask = np.ones(resp.shape, bool)
    coords, vals = LB.local_maxima_nms(resp, mask, (1.0, 1.0, 1.0), min_sep_mm=5.0, top_k=50)
    assert coords[:2].tolist() == [[13, 10, 10], [30, 10, 10]]
    assert vals[:2].tolist() == [9.0, 7.0]
    assert [10, 10, 10] not in coords.tolist()                   # suppressed by the 5 mm rule
    assert vals.tolist() == sorted(vals.tolist(), reverse=True)
    assert (vals[2:] == 0.0).all()                               # the rest is the flat plateau


def test_local_maxima_nms_respects_the_mask_and_empty_input():
    resp = np.zeros((20, 20, 20), np.float32)
    resp[5, 5, 5] = 9.0
    mask = np.zeros(resp.shape, bool)
    mask[15, 15, 15] = True                                      # the strong peak is outside the mask
    coords, vals = LB.local_maxima_nms(resp, mask, (1.0, 1.0, 1.0))
    assert coords.tolist() == [[15, 15, 15]]
    assert LB.local_maxima_nms(resp, np.zeros(resp.shape, bool), (1.0, 1.0, 1.0))[0].shape == (0, 3)


# ------------------------------------------------------------------ shape terms
def test_shape_terms_separate_blob_plate_and_tube():
    H = np.zeros((3, 3, 3))
    H[0] = np.diag([4.0, 4.0, 4.0])      # isotropic dark blob
    H[1] = np.diag([4.0, 4.0, 0.01])     # dark tube: one near-zero eigenvalue along the axis
    H[2] = np.diag([4.0, 0.01, 0.01])    # dark sheet: two near-zero
    st = LB.shape_terms(H, sign=1.0)
    assert st["blobness"][0] == pytest.approx(1.0)
    assert st["blobness"][1] < 0.01 and st["Ra"][1] == pytest.approx(1.0)      # tube
    assert st["blobness"][2] < 0.01 and st["Ra"][2] < 0.01                     # sheet
    assert st["n_pos"].tolist() == [3, 3, 3]
    assert st["frob"][0] == pytest.approx(np.sqrt(3 * 16.0))


def test_shape_terms_sign_flip_makes_both_polarities_positive():
    H = np.diag([-4.0, -4.0, -4.0])[None]           # a bright blob: Laplacian eigenvalues negative
    assert LB.shape_terms(H, sign=-1.0)["n_pos"].tolist() == [3]
    assert LB.shape_terms(H, sign=1.0)["n_pos"].tolist() == [0]
    assert LB.shape_terms(H, sign=-1.0)["blobness"][0] == pytest.approx(1.0)


def test_shape_terms_orders_by_magnitude_not_value():
    H = np.diag([9.0, -1.0, 4.0])[None]
    st = LB.shape_terms(H, sign=1.0)
    assert [st["lam1"][0], st["lam2"][0], st["lam3"][0]] == [-1.0, 4.0, 9.0]


def test_shape_terms_handles_an_empty_peak_list():
    st = LB.shape_terms(np.zeros((0, 3, 3)), sign=1.0)
    assert all(v.shape == (0,) for v in st.values())


# ------------------------------------------------------------------ end to end
def test_detect_finds_a_dark_ball_at_about_r_over_sqrt3():
    """With noise present the z scale is meaningful. On a NOISELESS volume the response MAD is
    zero almost everywhere, z collapses to (raw / EPS) for every scale, and the argmax lands on
    the sharp edge instead of the centre -- which is why phantom.py asserts the scale law on the
    raw response and this test adds noise."""
    r, sp = 5.0, (1.0, 1.0, 1.0)
    shape = (60, 60, 60)
    c = (30.4, 29.6, 30.1)
    ct = dark_ball(shape, sp, c, r)
    ct += (8.0 * np.random.default_rng(7).standard_normal(shape)).astype(np.float32)
    mask = np.zeros(shape, bool)
    mask[15:45, 15:45, 15:45] = True
    res = LB.detect(ct, mask, sp, probe_idx={"c": np.array([np.ravel_multi_index((30, 30, 30), shape)])})
    top = [p for p in res["peaks"] if p["polarity"] == "dark"][0]
    assert top["rank"] == 1
    assert np.linalg.norm(np.array([top["ix"], top["iy"], top["iz"]]) - np.array(c)) <= 1.5
    # the raw-response argmax over scales is the ladder step next to r/sqrt(3) = 2.89 mm
    raw = np.array([res["probe"]["c"]["dark"][i] * res["noise"][s]
                    for i, s in enumerate(LB.SIGMAS_MM)])
    want = int(np.argmin(np.abs(np.asarray(LB.SIGMAS_MM) - r / np.sqrt(3))))
    assert abs(int(np.argmax(raw)) - want) <= 1
    assert top["blobness"] > 0.5 and top["n_pos"] == 3


def test_detect_polarity_signs_are_independent():
    sp, shape = (1.0, 1.0, 1.0), (50, 50, 50)
    ct = dark_ball(shape, sp, (18.0, 25.0, 25.0), 4.0)                     # dark blob
    ct += -1.0 * (dark_ball(shape, sp, (34.0, 25.0, 25.0), 4.0) - 60.0)    # bright blob
    ct += (6.0 * np.random.default_rng(3).standard_normal(shape)).astype(np.float32)
    mask = np.zeros(shape, bool)
    mask[12:40, 12:40, 12:40] = True
    res = LB.detect(ct, mask, sp)
    for pol, want_x in (("dark", 18), ("bright", 34)):
        top = [p for p in res["peaks"] if p["polarity"] == pol][0]
        assert abs(top["ix"] - want_x) <= 2, (pol, top["ix"])
        assert top["n_pos"] == 3 and top["blobness"] > 0.5


def test_detect_peaks_stay_inside_the_mask_and_are_rank_ordered():
    rng = np.random.default_rng(0)
    ct = (60.0 + 15.0 * rng.standard_normal((50, 50, 50))).astype(np.float32)
    mask = np.zeros(ct.shape, bool)
    mask[20:30, 20:30, 20:30] = True
    res = LB.detect(ct, mask, (1.0, 1.0, 1.0))
    for pol in LB.POLARITIES:
        pk = [p for p in res["peaks"] if p["polarity"] == pol]
        assert pk and [p["rank"] for p in pk] == list(range(1, len(pk) + 1))
        assert [p["z"] for p in pk] == sorted([p["z"] for p in pk], reverse=True)
        assert all(mask[p["ix"], p["iy"], p["iz"]] for p in pk)
