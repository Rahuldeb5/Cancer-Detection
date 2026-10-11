"""Known-answer tests for S8 (src/multiclass_eval/). Written and passing before any real prediction was read.

Phantom (0.8 x 0.8 x 2.5 mm grid, sizes in mm):
  lesion A  r = 6, predicted by a 0.95 core of r = 4 (inside A)          -> detected, Dice known
  lesion B  r = 3, predicted by a 0.8 blob of r = 2 inside it; B is the "separated" (excluded) lesion
  lesion C  r = 3, lesion p 0.1 but pancreas p 0.9 -> 502 seg == 1 there -> must count as MISSED
  FP blob   r = 2, lesion p 0.7 far from every lesion
502 probabilities are two independent sigmoid regions, seg built exactly as nnU-Net does
(regions_class_order [1, 2]); 501 gets the same lesion probability as a softmax pair. Identical lesion
probability must give identical lesion results through the two code paths.
"""
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from multiclass_eval import core as K
from multiclass_eval import stats as S
from tumorlib.lesions import label_lesions

REPO = Path(__file__).resolve().parents[1]
SP = (0.8, 0.8, 2.5)
SHAPE = (64, 64, 24)
CA, CB, CC, CFP = (20, 20, 30), (36, 20, 30), (20, 40, 30), (42, 44, 30)


def grid():
    return np.stack(np.meshgrid(*[np.arange(n) * s for n, s in zip(SHAPE, SP)], indexing="ij"), -1)


def ball(c, r):
    return np.linalg.norm(grid() - np.asarray(c, float), axis=-1) <= r


def phantom():
    gt = np.zeros(SHAPE, np.uint8)
    for c, r in ((CA, 6), (CB, 3), (CC, 3)):
        gt[ball(c, r)] = 1
    gland = np.zeros(SHAPE, bool)
    gland[10:40, 10:60, 6:18] = True
    p1 = np.full(SHAPE, 1e-4, np.float32)
    p1[ball(CA, 4)] = 0.95
    p1[ball(CB, 2)] = 0.8
    p1[ball(CC, 3)] = 0.1
    p1[ball(CFP, 2)] = 0.7
    pg = np.where(gland, 0.9, 0.01).astype(np.float32)
    seg502 = np.zeros(SHAPE, np.uint8)
    seg502[pg > 0.5] = 1
    seg502[p1 > 0.5] = 2
    seg501 = (p1 > 0.5).astype(np.uint8)
    P502 = np.stack([pg, p1]).transpose(0, 3, 2, 1).copy()            # stored (C, z, y, x)
    P501 = np.stack([1 - p1, p1]).transpose(0, 3, 2, 1).copy()
    labels, n = label_lesions(gt)
    ids = {k: int(labels[tuple(int(round(v / s)) for v, s in zip(c, SP))]) for k, c in zip("ABC", (CA, CB, CC))}
    gt_gland = ((gland | gt.astype(bool))).astype(np.uint8)
    return dict(gt=gt, p1=p1, seg502=seg502, seg501=seg501, P502=P502, P501=P501, labels=labels, n=n, ids=ids,
                gt_gland=gt_gland)


@pytest.fixture(scope="module")
def ph():
    return phantom()


@pytest.fixture(scope="module")
def results(ph):
    sr = np.ones(SHAPE, np.uint8)
    excl = [ph["ids"]["B"]]
    r501 = K.evaluate_case("501", "phantom", 0, ph["seg501"], ph["gt"], ph["P501"], sr, SP, excl)
    r502 = K.evaluate_case("502", "phantom", 0, ph["seg502"], ph["gt"], ph["P502"], sr, SP, excl, ph["gt_gland"])
    return r501, r502


def test_lesion_mask_label_2_only(ph):
    m = K.lesion_mask(ph["seg502"], "502")
    assert set(np.unique(ph["seg502"])) == {0, 1, 2}
    # C sits under label 1 (pancreas only) -> not lesion
    assert not m[ball(CC, 3)].any() and ph["seg502"][ball(CC, 3)].min() == 1
    assert np.array_equal(m, ph["seg501"])


def test_scan_502_consistency(ph):
    v = K.scan_502(ph["P502"], ph["seg502"])
    assert v["argmax_mismatch"] == 0 and v["gland_mismatch"] == 0
    assert abs(v["vol_peak_p"] - 0.95) < 1e-6
    bad = ph["seg502"].copy()
    bad[ball(CC, 3)] = 2           # pretend C was labelled lesion although p1 = 0.1
    assert K.scan_502(ph["P502"], bad)["argmax_mismatch"] == int(ball(CC, 3).sum())


def test_known_answer_lesions(ph, results):
    ids, gt = ph["ids"], ph["gt"]
    nA, nB = int(ball(CA, 6).sum()), int(ball(CB, 3).sum())
    nAc, nBc = int(ball(CA, 4).sum()), int(ball(CB, 2).sum())
    for r in results:
        L = {x["lesion_id"]: x for x in r["ev_lesions"]}
        assert abs(L[ids["A"]]["lesion_dice"] - 2 * nAc / (nA + nAc)) < 1e-12
        assert abs(L[ids["B"]]["lesion_dice"] - 2 * nBc / (nB + nBc)) < 1e-12
        assert L[ids["C"]]["lesion_dice"] == 0 and L[ids["C"]]["detected_any"] == 0
        assert r["case"]["seg_touches_gt"] and r["case"]["seg_any"]
        c05 = [c for c in r["comps"] if c["t"] == 0.5]
        assert len(c05) == 3 and sum(c["touch_ids"] == "" for c in c05) == 1
        assert r["case"]["p05_vs_seg_mismatch"] == 0 and r["case"]["argmax_mismatch"] == 0
        # C appears only once the ladder goes below its 0.1 plateau
        c03 = [c for c in r["comps"] if c["t"] == 0.03]
        assert any(str(ids["C"]) in c["touch_ids"].split(";") for c in c03)
        assert not any(str(ids["C"]) in c["touch_ids"].split(";") for c in c05)
        assert r["ev_case"]["nn_tp"] == nAc + nBc and r["ev_case"]["nn_n_ref"] == int(gt.sum())


def test_two_paths_identical(results):
    r501, r502 = results
    a, b = pd.DataFrame(r501["ev_lesions"]), pd.DataFrame(r502["ev_lesions"])
    pd.testing.assert_frame_equal(a, b)
    for k in ("dice", "iou", "hd95", "nsd_2.0mm", "pred_vol_mm3"):
        assert r501["ev_case"][k] == pytest.approx(r502["ev_case"][k], rel=1e-12)
    ca = pd.DataFrame(r501["comps"]).drop(columns=["peak_raw_logit"])
    cb = pd.DataFrame(r502["comps"]).drop(columns=["peak_raw_logit"])
    pd.testing.assert_frame_equal(ca, cb)
    la = pd.DataFrame(r501["lesions"]).drop(columns=[c for c in pd.DataFrame(r501["lesions"]) if "raw_logit" in c])
    lb = pd.DataFrame(r502["lesions"]).drop(columns=[c for c in pd.DataFrame(r502["lesions"]) if "raw_logit" in c])
    pd.testing.assert_frame_equal(la, lb, rtol=1e-6)


def test_exclusion_known_answer(ph, results):
    nA, nC, nAc, nFP = (int(ball(c, r).sum()) for c, r in ((CA, 6), (CC, 3), (CA, 4), (CFP, 2)))
    for r in results:
        e = r["excl"]
        assert e["action"] == "modified" and e["n_pred_comps_removed"] == 1
        assert e["ev_case"]["dice"] == pytest.approx(2 * nAc / (nA + nC + nAc + nFP))
        assert e["seg_touches_gt"]
        assert abs(e["ev_case"]["max_tumor_prob"] - 0.95) < 1e-6
        assert len(e["ev_lesions"]) == 2


def test_exclusion_all_excluded_dropped(ph):
    sr = np.ones(SHAPE, np.uint8)
    r = K.evaluate_case("501", "phantom", 0, ph["seg501"], ph["gt"], ph["P501"], sr, SP, [1, 2, 3])
    assert r["excl"]["action"] == "dropped"


def test_gland_counts(ph, results):
    g = results[1]["gland"]
    pred = ph["seg502"] > 0
    gt = ph["gt_gland"].astype(bool)
    assert g["tp"] == int((pred & gt).sum()) and g["n_ref"] == int(gt.sum()) and g["n_pred"] == int(pred.sum())


def _write(arr, path):
    nib.save(nib.Nifti1Image(arr.astype(np.uint8), np.diag([*SP, 1.0])), str(path))


def test_exclusion_matches_prepare_excluded_eval(ph, tmp_path):
    """The real src/nnunet/prepare_excluded_eval.py (SimpleITK, z,y,x) vs exclusion_variant (nibabel)."""
    pred = ph["seg501"].copy()
    # extra predicted component bridging A and B: survives (touches kept A) minus its voxels inside B
    bridge = np.zeros(SHAPE, bool)
    bridge[25:46, 23:27, 11:13] = True
    pred[bridge] = 1
    for d in ("pred", "gt"):
        (tmp_path / d).mkdir()
    _write(pred, tmp_path / "pred" / "X_1.nii.gz")
    _write(ph["gt"], tmp_path / "gt" / "X_1.nii.gz")
    b = ph["ids"]["B"]
    pd.DataFrame([dict(case_id="X_1", lesion_id=b, n_components_in_case=ph["n"],
                       n_vox=int((ph["labels"] == b).sum()))]).to_csv(tmp_path / "ex.csv", index=False)
    subprocess.run([sys.executable, str(REPO / "src/nnunet/prepare_excluded_eval.py"), "--pred_dir", str(tmp_path / "pred"),
                    "--gt_dir", str(tmp_path / "gt"), "--work_dir", str(tmp_path / "w"), "--excluded_csv",
                    str(tmp_path / "ex.csv")], check=True, capture_output=True)
    ref_p = np.asarray(nib.load(tmp_path / "w/pred/X_1.nii.gz").dataobj)
    ref_g = np.asarray(nib.load(tmp_path / "w/gt/X_1.nii.gz").dataobj)
    g2, p2, _, n_rm = K.exclusion_variant(ph["labels"], pred, [b])
    assert np.array_equal(ref_g > 0, g2 > 0) and np.array_equal(ref_p > 0, p2 > 0)
    assert n_rm == 0  # B's own blob is now part of the bridge component, which touches A
    assert not p2[ph["labels"] == b].any() and p2[bridge & ~(ph["labels"] == b)].all()


def test_metrics_match_evaluate_py(ph, tmp_path):
    """The real testing/nnunet/evaluate.py (SimpleITK, z,y,x, full volume) vs evaluate_metrics (nibabel, crop)."""
    for d in ("pred", "gt"):
        (tmp_path / d).mkdir()
    _write(ph["seg501"], tmp_path / "pred" / "X_1.nii.gz")
    _write(ph["gt"], tmp_path / "gt" / "X_1.nii.gz")
    subprocess.run([sys.executable, str(REPO / "testing/nnunet/evaluate.py"), "--pred_dir", str(tmp_path / "pred"),
                    "--gt_dir", str(tmp_path / "gt"), "--out_dir", str(tmp_path / "o")], check=True, capture_output=True)
    ref_c = pd.read_csv(tmp_path / "o/per_case_metrics.csv").iloc[0]
    ref_l = pd.read_csv(tmp_path / "o/per_lesion_metrics.csv")
    row, les = K.evaluate_metrics(ph["gt"], ph["seg501"], SP, np.nan)
    for k in ("dice", "iou", "hd95", "assd", "nsd_1.0mm", "nsd_2.0mm", "nsd_3.0mm", "pred_vol_mm3", "ref_vol_mm3",
              "rel_vol_diff", "missed", "n_gt_lesions", "largest_lesion_diam_mm"):
        assert row[k] == pytest.approx(ref_c[k], rel=1e-6, abs=1e-12), k  # header spacing is float32
    mine = pd.DataFrame(les).sort_values(["gt_vox", "lesion_dice"]).reset_index(drop=True)  # B, C tie on gt_vox
    ref_l = ref_l.sort_values(["gt_vox", "lesion_dice"]).reset_index(drop=True)
    for k in ("gt_vox", "diam_mm", "bin", "lesion_dice", "lesion_iou", "detected_any", "detected_iou10"):
        assert np.allclose(mine[k], ref_l[k], rtol=1e-6), k
    # and the ids are tumorlib's
    assert [r["gt_vox"] for r in les] == [int((ph["labels"] == j).sum()) for j in range(1, ph["n"] + 1)]


# ------------------------------------------------------------------ bootstrap
def _toy(seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(40):
        for _ in range(rng.integers(1, 4)):
            a = rng.random()
            rows.append(dict(case=f"c{c}", a=a, b=min(1.0, a + rng.normal(0.05, 0.1))))
    return pd.DataFrame(rows)


def test_stats_identical_models_zero():
    df = _toy()
    df["b"] = df["a"]
    cases = df.case.unique()
    r = S.paired_ratio(df, "case", "a", "b", cases, S.case_weights(len(cases), 200))
    assert r["delta"] == 0 and r["ci_lo"] == 0 and r["ci_hi"] == 0 and r["p_boot"] == 1.0


def test_stats_constant_shift():
    df = _toy()
    df["b"] = df["a"] + 0.1
    cases = df.case.unique()
    r = S.paired_ratio(df, "case", "a", "b", cases, S.case_weights(len(cases), 200))
    assert r["delta"] == pytest.approx(0.1) and r["ci_lo"] == pytest.approx(0.1) and r["ci_hi"] == pytest.approx(0.1)


def test_stats_matches_naive_loop():
    df = _toy()
    cases = df.case.unique()
    W = S.case_weights(len(cases), 50, seed=3)
    r = S.paired_ratio(df, "case", "a", "b", cases, W)
    by = {c: g for c, g in df.groupby("case")}
    deltas = []
    for w in W:  # literally rebuild each resample
        draw = pd.concat([by[c] for c, k in zip(cases, w) for _ in range(int(k))])
        deltas.append(draw.b.mean() - draw.a.mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    assert r["ci_lo"] == pytest.approx(lo) and r["ci_hi"] == pytest.approx(hi)
    assert r["delta"] == pytest.approx(df.b.mean() - df.a.mean())
