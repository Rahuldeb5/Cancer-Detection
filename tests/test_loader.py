"""Test group b: loader dtypes, encoding correctness, failure modes, peak memory."""
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from tumorlib import io

BIG_CASE = "PanTS_00008854"  # 510 x 431 x 918, the largest volume in the cohort
PEAK_LIMIT_BYTES = 2 * 1024 ** 3
MASKS = ("pancreatic_lesion", "pancreas", "pancreas_head", "pancreas_body", "pancreas_tail")


def peak_rss(snippet: str) -> tuple[int, str]:
    """Run `snippet` in a fresh interpreter; return (peak RSS in bytes, its stdout).

    Reads VmHWM from /proc, not getrusage(): Linux carries ru_maxrss across
    fork+exec, so a child would report the pytest parent's peak."""
    code = "\n".join([
        "import sys",
        f"sys.path.insert(0, {str(io.REPO_ROOT / 'src')!r})",
        "from tumorlib import io, envelope",
        snippet,
        "hwm = [l for l in open('/proc/self/status') if l.startswith('VmHWM')][0]",
        "print('PEAK_KB', int(hwm.split()[1]))",
    ])
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    peak_kb = int(res.stdout.split("PEAK_KB")[-1])
    return peak_kb * 1024, res.stdout


@pytest.mark.data
def test_mask_dtype_is_int8_family_not_float():
    m = io.load_mask(BIG_CASE, "pancreatic_lesion")
    assert m is not None
    assert m.dtype in (np.uint8, np.int8)
    assert set(np.unique(m)) <= {0, 1}
    assert m.shape == (510, 431, 918)


@pytest.mark.data
@pytest.mark.parametrize("case_id", [BIG_CASE, "PanTS_00000003", "PanTS_00000026"])
def test_mask_matches_nibabel_scaled_reading(case_id):
    """Oracle: nibabel's own scaled reading > 0.5. Covers both on-disk encodings
    (raw {0,1} slope 1: PanTS_00000003; raw {-128,127} slope 1/255 inter 0.502:
    PanTS_00000026 and the big case)."""
    for name in MASKS:
        img = nib.load(io.mask_path(case_id, name))
        ref = np.asarray(img.dataobj, dtype=np.float32) > 0.5  # float32 keeps the oracle's memory sane
        got = io.load_mask(case_id, name)
        assert got is not None
        assert np.array_equal(got.astype(bool), ref), (case_id, name, img.dataobj.slope, img.dataobj.inter)


@pytest.mark.data
def test_spacing_from_ct_affine():
    assert io.spacing(BIG_CASE) == pytest.approx((0.93359375, 0.93359375, 0.7))


@pytest.mark.data
def test_ct_crop_is_float32_and_matches_full_read():
    crop = (slice(200, 240), slice(150, 170), slice(400, 410))
    ct = io.load_ct(BIG_CASE, crop)
    assert ct.dtype == np.float32 and ct.shape == (40, 20, 10)
    ref = np.asarray(nib.load(io.ct_path(BIG_CASE)).dataobj[crop], dtype=np.float32)
    assert np.array_equal(ct, ref)


@pytest.mark.data
@pytest.mark.parametrize("what,snippet", [
    ("lesion mask", "m = io.load_mask('PanTS_00008854', 'pancreatic_lesion'); assert m.dtype.itemsize == 1"),
    ("search region", "r = envelope.search_region('PanTS_00008854'); assert r is not None and r.any()"),
    ("full CT float32", "c = io.load_ct('PanTS_00008854'); assert c.dtype.name == 'float32'"),
])
def test_peak_memory_on_largest_case(what, snippet):
    peak, out = peak_rss(snippet)
    print(f"{what}: peak RSS {peak / 1024 ** 3:.2f} GB")
    assert peak < PEAK_LIMIT_BYTES, f"{what}: peak {peak / 1024 ** 3:.2f} GB"


def test_missing_and_lfs_stub_return_none(tmp_path, monkeypatch, capsys):
    seg = tmp_path / "FAKE" / "segmentations"
    seg.mkdir(parents=True)
    (seg / "pancreas.nii.gz").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 12345\n")
    monkeypatch.setattr(io, "MASK_ROOT", tmp_path)
    monkeypatch.setattr(io, "CT_ROOT", tmp_path)
    io.grid.cache_clear()

    assert io.load_mask("FAKE", "pancreas") is None
    assert "git-LFS pointer stub" in capsys.readouterr().out
    assert io.load_mask("FAKE", "pancreas_tail") is None
    assert "missing" in capsys.readouterr().out
    io.grid.cache_clear()
