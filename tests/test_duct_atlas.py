"""S10 duct atlas oracles: synthetic glands in random poses, volume conservation, splat sanity."""
import numpy as np
import pytest

from duct_atlas.case import analyse, new_accumulators
from duct_atlas.frame import build_frame
from duct_atlas.grid import MM_GRID, NORM_GRID, Grid, splat
from duct_atlas.phantom import make_phantom

N_PHANTOMS = 50
SEMI = (75.0, 15.0, 10.0)
# duct radius >= half the 5 mm slice: a thinner duct lying in a slice plane is only voxelized where
# a slice centre falls within r of it (a 2 mm duct lost half its length in one pose) -- correct
# sampling, but it would make the oracle test the voxelizer instead of the frame
DUCT_R = 3.0


def _ellipsoid_u_reference(n=2_000_001):
    """Analytic-ish u of the head/body/tail centroids for an ellipsoid: the volume density along
    the long axis is proportional to (1 - x^2), x in [-1, 1] (units of the semi-axis)."""
    x = np.linspace(-1, 1, n)
    w = 1 - x ** 2
    cdf = np.cumsum(w) / w.sum()
    x1, x99 = np.interp([0.01, 0.99], cdf, x)
    to_u = lambda v: (v - x1) / (x99 - x1)
    out = {}
    for s, m in (("head", x < -1 / 3), ("body", np.abs(x) <= 1 / 3), ("tail", x > 1 / 3)):
        out[s] = to_u((x[m] * w[m]).sum() / w[m].sum())
    # the head's own 1st percentile and the tail's own 99th: just beyond u = 0 and u = 1, because
    # u = 0 / 1 are the ENVELOPE's 1st / 99th percentiles and each end third holds ~26% of it
    f_head = cdf[x < -1 / 3][-1]
    out["head_p1"] = to_u(np.interp(0.01 * f_head, cdf, x))
    out["tail_p99"] = to_u(np.interp(1 - 0.01 * f_head, cdf, x))
    return out


@pytest.fixture(scope="module")
def stack():
    rng = np.random.default_rng(10)
    acc = new_accumulators()
    rows, phantoms = [], []
    for _ in range(N_PHANTOMS):
        ph = make_phantom(rng, semi_axes=SEMI, duct_r=DUCT_R)
        row, pts = analyse(ph.coords, ph.affine, acc, keep_points=True)
        rows.append(row)
        phantoms.append((ph, pts))
    return acc, rows, phantoms


def test_phantoms_all_included(stack):
    _, rows, _ = stack
    assert [r["status"] for r in rows] == ["ok"] * N_PHANTOMS


def test_pc1_recovers_long_axis_with_head_to_tail_sign(stack):
    _, rows, phantoms = stack
    for row, (ph, _) in zip(rows, phantoms):
        e1 = np.array([row["e1_x"], row["e1_y"], row["e1_z"]])
        cos = e1 @ ph.Q[:, 0]          # Q[:,0] points head -> tail by construction
        assert cos > np.cos(np.radians(1.0)), cos


def test_head_lands_at_u0_in_every_pose(stack):
    _, rows, phantoms = stack
    ref = _ellipsoid_u_reference()
    for row, (_, pts) in zip(rows, phantoms):
        for s in ("head", "body", "tail"):
            assert abs(row[f"{s}_u"] - ref[s]) < 0.02, (s, row[f"{s}_u"], ref[s])
        # head end of the gland sits at u ~ 0, tail end at u ~ 1
        assert abs(np.percentile(pts["head"][:, 0], 1) - ref["head_p1"]) < 0.02
        assert abs(np.percentile(pts["tail"][:, 0], 99) - ref["tail_p99"]) < 0.02
        assert row["head_u"] < row["body_u"] < row["tail_u"]


def test_duct_collapses_to_ridge(stack):
    acc, rows, phantoms = stack
    duct = acc["norm"]["duct"]
    ap_c, si_c = NORM_GRID.centers(1), NORM_GRID.centers(2)
    near = (np.abs(ap_c)[:, None] < 0.03) & (np.abs(si_c)[None, :] < 0.03)
    assert duct[:, near].sum() / duct.sum() > 0.99
    # every pose individually: duct at ap~0, si~0, spanning the gland along u
    for row, (_, pts) in zip(rows, phantoms):
        d = pts["duct"]
        half_slice = 2.5 / row["L_mm"]   # 5 mm slices can put the sampled duct up to half a slice off-axis
        assert np.abs(np.median(d[:, 1:], axis=0)).max() < half_slice
        assert np.abs(d[:, 1:]).max() < (DUCT_R + 2.5) / row["L_mm"] + 0.005
        assert np.percentile(d[:, 0], 2) < 0.1 and np.percentile(d[:, 0], 98) > 0.9
    # the ridge is present in every u bin across the duct's span
    prof = duct.sum(axis=(1, 2))
    u_c = NORM_GRID.centers(0)
    assert (prof[(u_c > 0.1) & (u_c < 0.9)] > 0).all()


def test_volume_conservation(stack):
    acc, rows, _ = stack
    total = sum(r["duct_mm3"] for r in rows)
    for g in (NORM_GRID, MM_GRID):
        in_grid = acc[g.name]["duct"].sum()
        out = sum(r.get(f"duct_out_{g.name}_mm3", 0.0) for r in rows)
        assert in_grid + out == pytest.approx(total, rel=1e-9)   # exact bookkeeping
        assert in_grid == pytest.approx(total, rel=0.01)         # and nothing lost off-grid
    tissue = sum(r["tissue_mm3"] for r in rows)
    assert acc["norm"]["tissue"].sum() == pytest.approx(tissue, rel=0.01)


def test_tissue_is_union_without_double_counting(stack):
    _, rows, phantoms = stack
    for row, (ph, _) in zip(rows, phantoms):
        allv = np.concatenate([ph.coords[k] for k in ("head", "body", "tail", "duct")])
        n_union = len(np.unique(allv, axis=0))
        assert row["tissue_mm3"] == pytest.approx(n_union * row["voxel_mm3"])


def test_splat_cube_lands_in_expected_bins():
    g = Grid("t", lo=(0.0, -1.0, -1.0), hi=(1.0, 1.0, 1.0), step=0.1)
    acc = g.zeros()
    # 4x4x4 points evenly inside bin (u 0.5-0.6, ap 0.0-0.1, si -0.2--0.1), weight 2.5 mm^3 each
    o = np.linspace(0.01, 0.09, 4)
    cube = np.array([[0.5 + a, 0.0 + b, -0.2 + c] for a in o for b in o for c in o])
    out = splat(acc, g, cube, 2.5)
    assert out == 0.0
    i, j, k = 5, 10, 8
    assert acc[i, j, k] == pytest.approx(64 * 2.5)
    assert acc.sum() == pytest.approx(64 * 2.5)
    # a 2x2x2 block of bins: one point per bin centre
    acc2 = g.zeros()
    centres = np.array([[0.25 + 0.1 * a, 0.05 + 0.1 * b, 0.05 + 0.1 * c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
    splat(acc2, g, centres, 1.0)
    assert (acc2[2:4, 10:12, 10:12] == 1.0).all() and acc2.sum() == 8
    # points off the grid are reported, not deposited
    acc3 = g.zeros()
    out = splat(acc3, g, np.array([[1.5, 0, 0], [-0.01, 0, 0], [0.5, 0, 0]]), 3.0)
    assert out == pytest.approx(6.0) and acc3.sum() == pytest.approx(3.0)


def test_frame_sign_follows_head_tail_not_eigh():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(5000, 3)) * [30, 5, 3]
    f1 = build_frame(pts, head_centroid=np.array([-20.0, 0, 0]), tail_centroid=np.array([20.0, 0, 0]))
    f2 = build_frame(pts, head_centroid=np.array([20.0, 0, 0]), tail_centroid=np.array([-20.0, 0, 0]))
    assert f1.e1[0] > 0.99 and f2.e1[0] < -0.99
    assert f1.eig_flipped != f2.eig_flipped
    for f in (f1, f2):  # anatomical axes stay world-anterior / world-superior
        assert f.e2 @ [0, 1, 0] > 0.99 and f.e3 @ [0, 0, 1] > 0.99
    assert not f1.leftward and f2.leftward


def test_exclusions():
    rng = np.random.default_rng(1)
    ph = make_phantom(rng, semi_axes=(20.0, 8.0, 6.0))       # ~4 cm^3 < 15 cm^3
    row, _ = analyse(ph.coords, ph.affine)
    assert "env_volume" in row["status"]
    ph = make_phantom(rng, semi_axes=(30.0, 28.0, 27.0))     # near-sphere: PC1 share ~0.35
    row, _ = analyse(ph.coords, ph.affine)
    assert "var_share" in row["status"]
    ph = make_phantom(rng)
    c = dict(ph.coords, tail=np.empty((0, 3), dtype=np.int32))
    row, _ = analyse(c, ph.affine)
    assert row["status"].startswith("empty_tail") or "empty_tail" in row["status"]
    c = dict(ph.coords, duct=ph.coords["duct"][:1])
    acc = new_accumulators()
    row, _ = analyse(c, ph.affine, acc)
    assert "no_duct" in row["status"] and acc["norm"]["tissue"].sum() == 0


@pytest.mark.data
def test_real_case_loads_on_head_grid():
    from duct_atlas.case import process_case
    row, pts = process_case("PanTS_00000003", keep_points=True)
    assert row["status"] in ("ok", "no_duct")
    assert row["shape"] and "msgs" in row
