"""S13 oracles for src/nnunet-multiclass/postprocess/core.py.

1-3: synthetic probability volumes with hand-computed answers (filter survivors, top-k / p_min picks, masks, Dice).
4:   every grid setting on those volumes plus a random one: the atom arithmetic == a dense, independently written
     reference (voxel masks scored by evaluate.py's own lesion_level / overlap_metrics), and score_grid == evaluate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt, gaussian_filter, label as cc_label

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "nnunet-multiclass" / "postprocess"))
sys.path.insert(0, str(REPO / "testing" / "nnunet"))
import core as K  # noqa: E402
import evaluate as EV  # noqa: E402

S26 = np.ones((3, 3, 3), bool)
BG = 1e-6


def build(p, panc, gt, sp):
    gl, _ = cc_label(gt > 0, structure=S26)
    pi = np.flatnonzero(p > K.FLOOR)
    gi = np.flatnonzero(gl)
    return K.build_case(p.shape, sp, pi, p.ravel()[pi], np.flatnonzero(panc), gi, gl.ravel()[gi])


def comp_at(case, t, xyz):
    """0-based index of the component (at t_low t) whose peak voxel is xyz."""
    f = np.ravel_multi_index(xyz, case["shape"])
    return int(np.nonzero(case["comps"][t]["peak_flat"] == f)[0][0])


# ------------------------------------------------------------------ volume 1: filter + ranking
def vol1():
    sp = (1.0, 1.0, 2.0)
    p = np.full((40, 40, 20), BG, np.float32)
    panc = np.zeros(p.shape, bool)
    panc[5:15, 5:35, 5:15] = True                     # x 5..14, z 5..14
    p[8:11, 8:11, 8:11] = 0.9                         # c1: inside
    p[18:20, 20, 10] = 0.25                           # c2: 4 and 5 mm outside in x (0.25 exact in float32)
    p[22:24, 20, 10] = 0.05                           # c3: 8 and 9 mm outside
    p[14:16, 30, 10] = 0.6                            # c4: 1 of 2 voxels inside (50%)
    p[14:17, 25, 10] = 0.7                            # c5: 1 of 3 inside; 1, 2 mm outside
    p[10, 20, 16:18] = 0.2                            # c6: z 16, 17 -> 4, 6 mm outside (z spacing 2)
    peaks = {"c1": (8, 8, 8), "c2": (18, 20, 10), "c3": (22, 20, 10), "c4": (14, 30, 10), "c5": (14, 25, 10),
             "c6": (10, 20, 16)}
    return p, panc, np.zeros(p.shape, np.uint8), sp, peaks


def names(case, t, idx, peaks):
    inv = {comp_at(case, t, v): k for k, v in peaks.items() if case["comps"][t]["peak"].size and
           np.any(case["comps"][t]["peak_flat"] == np.ravel_multi_index(v, case["shape"]))}
    return sorted(inv[i] for i in idx)


def test_vol1_filter_survivors():
    p, panc, gt, sp, peaks = vol1()
    case = build(p, panc, gt, sp)
    t = 0.01
    assert case["comps"][t]["n_vox"].size == 6
    expect = {None: "c1 c2 c3 c4 c5 c6",
              (0.0, "any"): "c1 c4 c5", (0.0, "half"): "c1 c4",
              (5.0, "any"): "c1 c2 c4 c5 c6", (5.0, "half"): "c1 c2 c4 c5 c6",
              (10.0, "any"): "c1 c2 c3 c4 c5 c6", (10.0, "half"): "c1 c2 c3 c4 c5 c6"}
    for filt, exp in expect.items():
        got = names(case, t, np.nonzero(K.filter_pass(case["comps"][t], filt))[0], peaks)
        assert got == exp.split(), (filt, got)
    # c6 at d = 5: exactly 1 of 2 voxels -> kept by 'half' (>= 50%), c5 at d = 0: 1 of 3 -> removed
    c6, c5 = comp_at(case, t, peaks["c6"]), comp_at(case, t, peaks["c5"])
    assert case["comps"][t]["panc_n"][5.0][c6] == 1 and case["comps"][t]["panc_n"][0.0][c5] == 1


def test_vol1_topk_pmin():
    p, panc, gt, sp, peaks = vol1()
    case = build(p, panc, gt, sp)

    def pick(**kw):
        s = K.Setting(t_low=0.01, **kw)
        return names(case, 0.01, K.select(case, s), peaks)

    assert pick(k=1) == ["c1"]
    assert pick(k=2) == ["c1", "c5"]
    assert pick(k=3, p_min=0.3) == ["c1", "c4", "c5"]
    assert pick(k=5, p_min=0.3) == ["c1", "c4", "c5"]
    assert pick(k=5, p_min=0.25) == ["c1", "c2", "c4", "c5"]         # p_min is inclusive (c2 peak exactly 0.25)
    assert pick(k=None, p_min=0.1) == ["c1", "c2", "c4", "c5", "c6"]
    assert pick(k=None) == ["c1", "c2", "c3", "c4", "c5", "c6"]
    assert pick(k=2, filt=(0.0, "any")) == ["c1", "c5"]
    assert pick(k=2, filt=(0.0, "half")) == ["c1", "c4"]
    assert pick(k=2, filt=(5.0, "half"), p_min=0.65) == ["c1", "c5"]
    # first component is kept whatever its peak; p_min only gates the rest
    s = K.Setting(t_low=0.01, k=3, p_min=0.95)
    assert names(case, 0.01, K.select(case, s), peaks) == ["c1"]
    # t_low = 0.5 sees only c1, c5, c4
    assert names(case, 0.5, K.select(case, K.Setting()), peaks) == ["c1", "c4", "c5"]
    assert K.evaluate(case, K.Setting())[0]["n_fp_cc"] == 3            # negative: 3 FP CCs, objective 0
    assert K.evaluate(case, K.Setting())[0]["objective"] == 0.0
    assert K.evaluate(case, K.Setting(filt=(0.0, "half")))[0]["n_fp_cc"] == 2


# ------------------------------------------------------------------ volume 2: masks (B rule, D t_seg) and Dice
def vol2():
    sp = (1.0, 1.0, 1.0)
    p = np.full((30, 30, 30), BG, np.float32)
    p[3:10, 3:10, 3:10] = 0.25                       # X halo
    p[5:8, 5:8, 5:8] = 0.8                           # X core (27)
    p[20:23, 20:23, 20:23] = 0.15                    # Y ring
    p[21, 21, 21] = 0.4                              # Y peak, no voxel > 0.5
    gt = np.zeros(p.shape, np.uint8)
    gt[4:9, 4:9, 4:9] = 1                            # L1 (125), contains X core
    gt[21:24, 21:24, 21:24] = 1                      # L2 (27), overlaps Y in 8 voxels incl. its peak
    panc = np.ones(p.shape, bool)
    return p, panc, gt, sp


@pytest.mark.parametrize("t_seg,x_mask,y_mask,d1,d2", [
    (0.5, 27, 1, 54 / 152, 2 / 28),        # Y: fallback, voxels >= 0.2 = the 0.4 peak only
    (0.3, 27, 1, 54 / 152, 2 / 28),        # halo 0.25 not > 0.3; Y peak 0.4 > 0.3
    (0.2, 343, 1, 250 / 468, 2 / 28),      # halo joins X
    (0.1, 343, 27, 250 / 468, 16 / 54),    # Y ring 0.15 > 0.1
])
def test_vol2_masks(t_seg, x_mask, y_mask, d1, d2):
    p, panc, gt, sp = vol2()
    case = build(p, panc, gt, sp)
    s = K.Setting(t_low=0.1, t_seg=t_seg)
    atoms = case["atoms"][(0.1, t_seg)]
    x, y = comp_at(case, 0.1, (5, 5, 5)), comp_at(case, 0.1, (21, 21, 21))
    assert sum(a for a, _ in atoms[x]) == x_mask and sum(a for a, _ in atoms[y]) == y_mask
    row, les = K.evaluate(case, s)
    assert les[0]["lesion_dice"] == pytest.approx(d1) and les[1]["lesion_dice"] == pytest.approx(d2)
    inter = 27 + min(y_mask, 8) if t_seg > 0.2 else 125 + min(y_mask, 8)
    assert row["dice"] == pytest.approx(2 * inter / (152 + x_mask + y_mask))
    # t_low = 0.5: only X's core, whatever t_seg
    assert K.evaluate(case, K.Setting(t_seg=t_seg))[1][0]["lesion_dice"] == pytest.approx(54 / 152)
    # D family (core > 0.5 only): Y is never kept
    assert K.evaluate(case, K.Setting(t_low=0.1, t_seg=t_seg, require_core=True))[1][1]["detected_any"] == 0


# ------------------------------------------------------------------ volume 3: merging, CC-level FP, ties, empty pancreas
def vol3():
    sp = (1.0, 1.0, 1.0)
    p = np.full((40, 20, 20), BG, np.float32)
    p[5:7, 5:7, 5:7] = 0.9                           # A: 8 voxels
    p[7:15, 6, 6] = 0.05                             # bridge
    p[15:18, 5:8, 5:8] = 0.9                         # B: 27 voxels, same peak as A
    p[30, 10, 10] = 0.95                             # C: 1 voxel, highest peak
    gt = np.zeros(p.shape, np.uint8)
    gt[4:7, 4:7, 4:7] = 1                            # L1 (27) overlaps A in 8
    panc = np.zeros(p.shape, bool)                   # model predicted no pancreas
    return p, panc, gt, sp


def test_vol3_merge_ties_empty_pancreas():
    p, panc, gt, sp = vol3()
    case = build(p, panc, gt, sp)
    # t_low 0.5: A, B, C separate; ties on peak broken by size -> order C, B, A
    r = K.ranked(case["comps"][0.5], None)
    assert [int(case["comps"][0.5]["n_vox"][i]) for i in r] == [1, 27, 8]
    assert K.tie_at_cut(case, K.Setting(k=2)) and not K.tie_at_cut(case, K.Setting(k=1))
    row, les = K.evaluate(case, K.Setting(k=2))                     # C + B: lesion missed
    assert les[0]["detected_any"] == 0 and row["n_fp_cc"] == 2
    # t_low 0.01: A-bridge-B is one component (peak 0.9) but C (0.95) still ranks first
    assert case["comps"][0.01]["n_vox"].size == 2
    row, les = K.evaluate(case, K.Setting(t_low=0.01, k=1))
    assert row["pred_vox"] == 1 and les[0]["detected_any"] == 0
    row, les = K.evaluate(case, K.Setting(t_low=0.01, k=2))         # merged comp: mask = A and B, 2 CCs
    assert row["n_cc"] == 3 and row["n_fp_cc"] == 2                 # B and C touch no lesion
    assert les[0]["lesion_dice"] == pytest.approx(16 / 35)          # Dice vs A only (B does not touch L1)
    assert row["dice"] == pytest.approx(16 / (27 + 36))
    # the model predicted no pancreas: every filter removes everything
    for f in K.FILTERS[1:]:
        assert K.evaluate(case, K.Setting(filt=f, t_low=0.01))[0]["pred_vox"] == 0
    assert np.isinf(case["dist"]).all()


# ------------------------------------------------------------------ dense reference over the whole grid
def reference(p, panc, gt, sp, s: K.Setting):
    """Independent voxel-level implementation of a setting."""
    lab, n = cc_label(p > s.t_low, structure=S26)
    dist = distance_transform_edt(~panc, sampling=sp) if panc.any() else np.full(p.shape, np.inf)
    rows = []
    for c in range(1, n + 1):
        m = lab == c
        pk = float(p[m].max())
        if s.filt is not None:
            d, f = s.filt
            w = int((dist[m] <= d).sum())
            if (f == "any" and w < 1) or (f == "half" and w / m.sum() < 0.5):
                continue
        rows.append((-pk, -int(m.sum()), int(np.flatnonzero((lab == c).ravel() & (p.ravel() == pk)).min()), c, pk))
    rows.sort()
    kept = []
    for i, (_, _, _, c, pk) in enumerate(rows):
        if s.require_core:
            if pk > 0.5:
                kept.append(c)
            continue
        if i > 0 and ((s.k is not None and len(kept) >= s.k) or (s.p_min is not None and pk < s.p_min)):
            break
        kept.append(c)
    mask = np.zeros(p.shape, np.uint8)
    for c, pk in ((c, pk) for *_, c, pk in rows if c in kept):
        m = lab == c
        core = m & (p > s.t_seg)
        mask[core if core.any() else (m & (p >= pk / 2))] = 1
    return mask


def random_vol(seed=0):
    rng = np.random.default_rng(seed)
    sp = (0.8, 0.8, 2.5)
    def field(sig):
        f = gaussian_filter(rng.standard_normal((48, 44, 20)), sig)
        return (f - f.mean()) / f.std()

    p = (1 / (1 + np.exp(-(4.0 * field((2.0, 2.0, 0.8)) - 9.5)))).astype(np.float32)  # sparse cores, faint halos
    panc = field(3.0) > 1.0
    panc[:30] = False                                                                  # one side only
    gt = (field(2.0) > 1.6).astype(np.uint8)
    return p, panc, gt, sp


VOLS = {"vol1": lambda: vol1()[:4], "vol2": vol2, "vol3": vol3, "random": random_vol}


@pytest.mark.parametrize("vname", list(VOLS))
def test_grid_matches_dense_reference(vname):
    p, panc, gt, sp = VOLS[vname]()
    case = build(p, panc, gt, sp)
    settings = K.all_settings()
    fast = K.score_grid(case, settings)
    gl, _ = cc_label(gt > 0, structure=S26)
    n_checked = 0
    for i, s in enumerate(settings):
        row, les = K.evaluate(case, s)
        assert fast[i] == pytest.approx(row["objective"], abs=1e-12), s
        if i % 7 and vname != "random" and i > 40:   # dense reference on a deterministic subset + all of 'random'
            continue
        mask = reference(p, panc, gt, sp, s)
        n_checked += 1
        ov = EV.overlap_metrics(gt, mask)
        assert row["pred_vox"] == ov["pred_voxels"], s
        if gt.any():
            assert row["dice"] == pytest.approx(ov["dice"]), s
            ref = EV.lesion_level(gt, mask, sp)
            assert [r["gt_vox"] for r in ref] == [r["gt_vox"] for r in les]
            for a, b in zip(les, ref):
                assert a["lesion_dice"] == pytest.approx(b["lesion_dice"]), s
                assert a["lesion_iou"] == pytest.approx(b["lesion_iou"]), s
                assert a["detected_any"] == b["detected_any"], s
        ml, mn = cc_label(mask, structure=S26)
        fp = sum(1 for c in range(1, mn + 1) if not (gl[ml == c] > 0).any())
        assert row["n_fp_cc"] == fp and row["n_cc"] == mn, s
    assert n_checked > 50


def test_random_volume_is_nontrivial():
    p, panc, gt, sp = random_vol()
    case = build(p, panc, gt, sp)
    n = {t: case["comps"][t]["n_vox"].size for t in K.T_LOWS}
    assert n[0.5] >= 3 and n[1e-4] >= 3 and case["n_lesions"] >= 3 and panc.sum() > 500
    assert 0 < np.isfinite(case["dist"]).mean() < 1
    fb = sum(1 for t in K.T_LOWS for s in K.T_SEGS if s > t for c in range(n[t])
             if case["comps"][t]["peak"][c] <= s)
    assert fb >= 3   # the fallback path is exercised


def test_all_faint_volume():
    """46 real cases never exceed p = 1e-4: every setting predicts nothing."""
    p = np.full((20, 20, 10), 5e-5, np.float32)
    gt = np.zeros(p.shape, np.uint8)
    gt[5:8, 5:8, 3:5] = 1
    panc = np.zeros(p.shape, bool)
    panc[2:12, 2:12, 2:8] = True
    case = build(p, panc, gt, (1.0, 1.0, 2.0))
    assert case["n_sparse"] == 0
    assert (K.score_grid(case, K.all_settings()) == 0).all()      # tumor+ with no prediction scores 0
    row, les = K.evaluate(case, K.Setting(t_low=1e-4, k=1))
    assert row["pred_vox"] == 0 and les[0]["detected_any"] == 0
    neg = build(p, panc, np.zeros_like(gt), (1.0, 1.0, 2.0))
    assert (K.score_grid(neg, K.all_settings()) == 1).all()       # tumor- with no prediction scores 1


def test_family_grids():
    assert len(K.family("A")) == 7 and len(K.family("B")) == 65 and len(K.family("D")) == 17
    assert len(K.family("ABD")) == 7 * 65 * 4
    for f in ("A", "B", "D", "BD", "AB", "ABD"):
        assert K.family(f)[0] == K.BASELINE          # the no-op is first, so it wins ties
