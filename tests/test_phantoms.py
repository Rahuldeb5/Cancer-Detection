"""Known-answer tests on synthetic volumes (test group c, plus Feret and label oracles)."""
import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt, label
from scipy.spatial.distance import pdist

from tumorlib import envelope as envmod
from tumorlib import io
from tumorlib.lesions import CC_STRUCT, feret_mm, label_lesions
from tumorlib.resample import resample_ct_to_isotropic, resample_mask_to_isotropic


def grid_mm(shape, spacing):
    """Voxel-center coordinates in mm, one array per axis."""
    return np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, spacing)], indexing="ij")


# ------------------------------------------------------------ search region
def ring_with_carves(spacing):
    """Torus in the x-y plane (major 25 mm, tube radius 8 mm) with two carved cavities:
    one fully enclosed in the tube, one notch breaching the outer surface.
    Returns (ring_with_holes, enclosed_hole, notch, centre_mm)."""
    ext = 80.0
    shape = tuple(int(ext / s) for s in spacing)
    x, y, z = grid_mm(shape, spacing)
    c = np.array([ext / 2 + 0.3, ext / 2 - 0.2, ext / 2 + 0.1])  # off-grid centre
    dx, dy, dz = x - c[0], y - c[1], z - c[2]
    rho = np.hypot(dx, dy)
    ring = (rho - 25.0) ** 2 + dz ** 2 <= 8.0 ** 2
    enclosed = (dx - 25.0) ** 2 + dy ** 2 + dz ** 2 <= 4.0 ** 2       # on the tube centreline
    notch = (dx + 33.0) ** 2 + dy ** 2 + dz ** 2 <= 5.0 ** 2          # centred on the outer surface
    enclosed &= ring
    notch &= ring
    return (ring & ~enclosed & ~notch).astype(np.uint8), enclosed, notch, c


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (0.8, 0.8, 2.5), (0.7, 0.7, 5.0)])
def test_search_region_fills_carved_holes(spacing):
    carved, enclosed, notch, c = ring_with_carves(spacing)
    assert enclosed.sum() > 0 and notch.sum() > 0
    assert not carved[enclosed].any() and not carved[notch].any()

    region = envmod.fill_search_region(carved, spacing, close_mm=8, dilate_mm=3).astype(bool)

    assert region[carved.astype(bool)].all(), "region must contain the envelope"
    assert region[enclosed].all(), "enclosed carved cavity not filled"
    assert region[notch].all(), "surface notch not filled"
    # The ring's central opening (17 mm from the tube) is anatomy, not a hole: keep it open.
    centre_idx = tuple(int(round(v / s)) for v, s in zip(c, spacing))
    assert not region[centre_idx]


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (0.8, 0.8, 2.5)])
def test_search_region_fills_cavity_larger_than_closing(spacing):
    """A 12 mm-radius enclosed cavity is too big for the 8 mm closing; only the 3D hole
    fill recovers it (a 24 mm lesion carved from the middle of a thick gland)."""
    shape = tuple(int(100 / s) for s in spacing)
    x, y, z = grid_mm(shape, spacing)
    d = [(x - 50.2) / 40, (y - 49.7) / 30, (z - 50.1) / 25]
    gland = d[0] ** 2 + d[1] ** 2 + d[2] ** 2 <= 1
    cavity = (x - 50.2) ** 2 + (y - 49.7) ** 2 + (z - 50.1) ** 2 <= 12.0 ** 2
    carved = (gland & ~cavity).astype(np.uint8)

    # sanity: the 8 mm closing by itself leaves the cavity open
    grown = distance_transform_edt(~carved.astype(bool), sampling=spacing) <= 8
    closing_only = (distance_transform_edt(grown, sampling=spacing) > 8) | carved.astype(bool)
    assert not closing_only[cavity].all(), "phantom too easy: closing alone fills it"

    region = envmod.fill_search_region(carved, spacing, close_mm=8, dilate_mm=0)
    assert region.astype(bool)[cavity].all()


def test_search_region_on_fake_case_never_reads_lesion(monkeypatch):
    """Head covers a lesion that pancreas.nii.gz has carved out; union + fill recovers it,
    and search_region() never asks for the lesion mask."""
    spacing = (1.0, 1.0, 1.0)
    carved, enclosed, notch, _ = ring_with_carves(spacing)
    head = np.zeros_like(carved)
    head[enclosed] = 1                      # head/body/tail disagree with pancreas.nii.gz
    masks = {"pancreas": carved, "pancreas_head": head,
             "pancreas_body": np.zeros_like(carved), "pancreas_tail": np.zeros_like(carved)}
    requested = []

    def fake_load_mask(case_id, name):
        requested.append(name)
        if name == "pancreatic_lesion":
            raise AssertionError("search region read the lesion mask")
        return masks[name].copy()

    monkeypatch.setattr(io, "load_mask", fake_load_mask)
    monkeypatch.setattr(io, "spacing", lambda case_id: spacing)

    env = envmod.envelope("FAKE")
    assert env[enclosed].all() and not env[notch].any()
    region = envmod.search_region("FAKE").astype(bool)
    assert region[enclosed].all() and region[notch].all()
    assert "pancreatic_lesion" not in requested
    assert set(requested) == set(envmod.ENVELOPE_MASKS)


# ------------------------------------------------------------ resampling
SPACINGS = [(0.8, 0.8, 5.0), (0.7, 0.7, 1.25), (1.0, 1.0, 1.0), (1.5, 1.5, 1.5), (0.93, 0.93, 0.7)]


@pytest.mark.parametrize("spacing", SPACINGS)
def test_sphere_resample_keeps_volume_and_position(spacing):
    shape = tuple(int(70 / s) for s in spacing)
    x, y, z = grid_mm(shape, spacing)
    c = np.array([33.3, 36.1, 34.7])
    sphere = ((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2 <= 12.0 ** 2).astype(np.uint8)

    iso, iso_sp, offset = resample_mask_to_isotropic(sphere, spacing)
    assert iso.dtype == np.uint8
    assert iso_sp == (1.0, 1.0, 1.0)

    vol_native = sphere.sum() * np.prod(spacing)
    vol_iso = iso.sum() * 1.0
    assert abs(vol_iso / vol_native - 1) < 0.02, (vol_native, vol_iso)

    # centroid through the documented offset formula lands where the native centroid is
    p_iso = np.argwhere(iso).mean(0)
    p_native = np.asarray(offset) + p_iso * (1.0 / np.asarray(spacing))
    cen_native = np.argwhere(sphere).mean(0) * np.asarray(spacing)
    assert np.allclose(p_native * np.asarray(spacing), cen_native, atol=0.5)


def test_empty_mask_resample():
    iso, sp, off = resample_mask_to_isotropic(np.zeros((10, 10, 10), np.uint8), (0.8, 0.8, 5))
    assert iso.size == 0 and sp == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("spacing", [(0.8, 0.8, 5.0), (0.7, 0.7, 1.25)])
def test_ct_resample_reproduces_linear_field(spacing):
    """Trilinear interpolation is exact on a linear field: a known-answer oracle."""
    shape = (60, 50, 30)
    x, y, z = grid_mm(shape, spacing)
    ct = (3.0 * x - 2.0 * y + 0.5 * z + 40.0).astype(np.float32)
    crop = (slice(10, 40), slice(5, 45), slice(3, 20))

    out, iso_sp, offset = resample_ct_to_isotropic(ct, spacing, crop)
    assert out.dtype == np.float32 and iso_sp == (1.0, 1.0, 1.0)
    assert offset == (10, 5, 3)

    o = np.argwhere(np.ones(out.shape, bool))
    mm = (np.asarray(offset) + o / np.asarray(spacing)) * np.asarray(spacing)
    expected = 3.0 * mm[:, 0] - 2.0 * mm[:, 1] + 0.5 * mm[:, 2] + 40.0
    assert np.abs(out.ravel() - expected).max() < 1e-2


# ------------------------------------------------------------ Feret
def test_feret_collinear_is_exact():
    """300 collinear voxels: ConvexHull fails, answer must still be exact (299 * 0.8)."""
    coords = np.stack([np.arange(300), np.full(300, 5), np.full(300, 7)], axis=1)
    assert feret_mm(coords, (0.8, 0.9, 5.0)) == pytest.approx(299 * 0.8)


def test_feret_coplanar_matches_brute_force_and_old_fallback_does_not():
    """Single-slice ellipse (coplanar -> degenerate hull). The exact answer is the
    brute-force max over all pairs; the old random-200 fallback underestimates."""
    sp = (0.7, 0.7, 5.0)
    x, y = np.meshgrid(np.arange(80), np.arange(80), indexing="ij")
    ellipse = ((x - 40) / 38.0) ** 2 + ((y - 40) / 4.0) ** 2 <= 1
    coords = np.stack([x[ellipse], y[ellipse], np.full(ellipse.sum(), 3)], axis=1)
    assert len(coords) > 200
    truth = pdist(coords * np.asarray(sp)).max()
    assert feret_mm(coords, sp) == pytest.approx(truth)

    idx = np.random.default_rng(0).choice(len(coords), 200, replace=False)
    old = pdist(coords[idx] * np.asarray(sp)).max()
    assert old < truth  # documents why the fallback was replaced


def test_feret_3d_matches_brute_force():
    sp = (0.8, 0.8, 1.5)
    x, y, z = grid_mm((50, 50, 30), sp)
    blob = ((x - 20) / 12) ** 2 + ((y - 18) / 6) ** 2 + ((z - 20) / 9) ** 2 <= 1
    coords = np.argwhere(blob)
    assert feret_mm(coords, sp) == pytest.approx(pdist(coords * np.asarray(sp)).max())


def test_feret_tiny():
    assert feret_mm(np.array([[1, 2, 3]]), (1, 1, 1)) == 0.0
    assert feret_mm(np.array([[0, 0, 0], [0, 0, 1]]), (1, 1, 2.5)) == pytest.approx(2.5)


# ------------------------------------------------------------ labeling
def test_label_lesions_matches_full_volume_numbering():
    """Labeling the bbox crop must give the same ids as labeling the full volume."""
    rng = np.random.default_rng(1)
    vol = np.zeros((90, 80, 70), np.uint8)
    for _ in range(25):
        c = rng.integers(20, 60, 3)
        r = rng.integers(1, 5, 3)
        vol[c[0] - r[0]:c[0] + r[0] + 1, c[1] - r[1]:c[1] + r[1] + 1, c[2] - r[2]:c[2] + r[2] + 1] = 1
    ref, n_ref = label(vol, structure=CC_STRUCT)
    got, n = label_lesions(vol)
    assert n == n_ref > 1
    assert got.dtype == np.uint16
    assert np.array_equal(got, ref)


def test_label_lesions_26_connectivity_and_empty():
    vol = np.zeros((5, 5, 5), np.uint8)
    vol[1, 1, 1] = vol[2, 2, 2] = 1          # corner-touching only
    assert label_lesions(vol)[1] == 1
    assert label_lesions(np.zeros((4, 4, 4), np.uint8))[1] == 0
