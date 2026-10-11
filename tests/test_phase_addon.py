"""Known-answer tests for S8b (src/multiclass_eval/phase_addon.py), written before the real run."""
import numpy as np
import pandas as pd
import pytest

from multiclass_eval import phase_addon as A


def test_phase_stratum_split():
    est = pd.Series(["arterial", "arterial", "venous", "non-contrast", "uncertain", "unmeasurable"])
    early = pd.Series([True, False, False, False, False, False])
    assert A.phase_stratum(est, early).tolist() == ["early arterial", "late arterial", "venous", "non-contrast",
                                                    "uncertain", "unmeasurable"]


def test_thickness_bins_left_closed():
    assert A.thickness_bin(pd.Series([0.8, 2.49, 2.5, 4.99, 5.0, 7.5])).tolist() == \
        ["<2.5", "<2.5", "2.5-5", "2.5-5", ">=5", ">=5"]


def test_cell_weights_known_answer():
    # positives 50/50 in A/B plus 10 in C (no negatives); negatives 2 in A, 8 in B, 5 in D (no positives)
    pos = pd.Series(["A"] * 45 + ["B"] * 45 + ["C"] * 10)
    neg = pd.Series(["A"] * 2 + ["B"] * 8 + ["D"] * 5)
    w, t = A.cell_weights(pos, neg)
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w[:2], 0.25) and np.allclose(w[2:10], 0.0625) and np.allclose(w[10:], 0.0)
    assert not t.loc["C", "covered"] and t.loc["C", "n_neg"] == 0 and t.loc["C", "n_pos"] == 10
    y = np.r_[[1, 1], [0] * 8, [1] * 5]          # A all 1, B all 0, D ignored
    assert w @ y == pytest.approx(0.5)            # = 0.5 * mean_A + 0.5 * mean_B over covered cells
    assert A.ess(w) == pytest.approx(1 / (2 * 0.25 ** 2 + 8 * 0.0625 ** 2))


def test_ess_uniform():
    assert A.ess(np.full(40, 1 / 40)) == pytest.approx(40)


def test_weights_identity_when_mixes_match():
    pos = pd.Series(["A"] * 30 + ["B"] * 70)
    neg = pd.Series(["A"] * 3 + ["B"] * 7)
    w, _ = A.cell_weights(pos, neg)
    assert np.allclose(w, 0.1)


def test_weighted_bootstrap_matches_point_and_identity():
    rng = np.random.default_rng(0)
    pos = np.array(["A"] * 60 + ["B"] * 40)
    neg = np.array(["A"] * 30 + ["B"] * 30)
    y1 = rng.integers(0, 2, 60).astype(float)
    reps = A.weighted_bootstrap(pos, neg, {"a": y1, "b": y1.copy()}, b=300)
    assert np.allclose(reps["a"], reps["b"])                       # identical outcomes -> delta 0 in every replicate
    w, _ = A.cell_weights(pd.Series(pos), pd.Series(neg))
    assert abs(np.median(reps["a"]) - w @ y1) < 0.05               # replicates centred on the point estimate
    # a replicate weight vector always sums to 1: a constant outcome gives exactly that constant
    assert np.allclose(A.weighted_bootstrap(pos, neg, {"c": np.ones(60)}, b=50)["c"], 1.0)


def test_peak_distance_10mm_boundary():
    sp = (0.8, 0.8, 2.5)
    les = np.zeros((60, 60, 20), bool)
    les[20:25, 20:25, 8:10] = True                 # lesion edge at x index 24 -> 19.2 mm
    # x index 37 -> 29.6 mm: 10.4 mm from the edge (far); index 36 -> 9.6 mm (near); along z: 9 -> +5 slices = 12.5 mm
    d = A.peak_distances_mm(np.array([[37, 22, 9], [36, 22, 9], [22, 22, 14]]), les, sp)
    assert d[0] == pytest.approx(10.4) and d[1] == pytest.approx(9.6) and d[2] == pytest.approx(12.5)
    assert (d > A.FAR_MM).tolist() == [True, False, True]
