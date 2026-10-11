"""Known-answer tests for S11 (src/rescue_check/). Written and passing before any real data was read.

The central oracle is a tiny synthetic per-lesion table whose raw columns are chosen so each signal's
hit is known by construction; running it through the same signal functions and overlap counter as the
real analysis must reproduce the hand-counted Venn cells, 0/1/2/3 counts, unique catches and gate.
"""
import numpy as np
import pandas as pd
import pytest

from rescue_check import core as K


# ------------------------------------------------------------------ synthetic 10-lesion table
# lesion: expected (nnunet, normative, duct) with None = not evaluable
EXPECT = {
    1: (True, False, False),
    2: (True, True, False),
    3: (False, True, False),
    4: (False, None, True),
    5: (True, True, True),
    6: (False, False, False),
    7: (False, None, None),
    8: (False, False, True),
    9: (False, True, True),
    10: (False, False, False),
}


def synthetic() -> pd.DataFrame:
    rows = {
        #     hit_t, rank_t, pct_A, cand_rank_A, mpd_present, mpd_dilated, head_cal, dist_mpd
        1: (True, 2, 50.0, 11, True, False, np.nan, 30.0),
        2: (True, 3, 95.0, 1, True, True, np.nan, 15.0),        # flagged case but 15 mm away
        3: (False, np.nan, 80.0, 10, True, False, 2.0, 5.0),     # rank 10 is in (<=); duct not flagged
        4: (False, np.nan, np.nan, 2, True, True, np.nan, 3.0),  # S9 could not score -> NA despite rank 2
        5: (True, 1, 99.0, 1, True, False, 4.0, 0.0),            # flagged by head-end only
        6: (True, 4, 10.0, np.nan, True, False, np.nan, 2.0),    # touched but rank 4 > 3
        7: (False, np.nan, np.nan, np.nan, False, False, np.nan, np.nan),
        8: (False, np.nan, 20.0, 50, True, False, 3.21, 10.0),   # both boundaries are inclusive
        9: (False, np.nan, 70.0, 5, True, True, np.nan, 9.9),
        10: (False, np.nan, 30.0, 12, True, True, np.nan, 10.5),
    }
    d = pd.DataFrame.from_dict(rows, orient="index", columns=[
        "hit_t", "rank_t", "pct_A", "cand_rank_A", "mpd_present", "mpd_dilated", "head_cal", "dist_mpd"])
    d = d.reset_index(names="lesion_id").reset_index(drop=True)
    d["sig_nnunet"] = K.nnunet_signal(d.hit_t, d.rank_t)
    d["sig_normative"] = K.normative_signal(d.pct_A, d.cand_rank_A)
    d["duct_case_flag"] = K.duct_case_flag(d.mpd_dilated, d.head_cal)
    d["sig_duct"] = K.duct_signal(d.duct_case_flag, d.dist_mpd, d.mpd_present)
    return d


def test_signals_reproduce_known_hits():
    d = synthetic()
    for r in d.itertuples():
        got = tuple(None if pd.isna(v) else bool(v) for v in (r.sig_nnunet, r.sig_normative, r.sig_duct))
        assert got == EXPECT[r.lesion_id], (r.lesion_id, got)


def test_overlap_counts_match_hand_count():
    ov = K.overlap_counts(synthetic())
    assert ov["n"] == 10
    assert ov["by_n_signals"] == {0: 3, 1: 4, 2: 2, 3: 1}
    assert ov["cells"] == {"none": 3, "nnunet": 1, "normative": 1, "duct": 2, "nnunet+normative": 1,
                           "nnunet+duct": 0, "normative+duct": 1, "nnunet+normative+duct": 1}
    assert ov["union_k"] == 7 and ov["unique"] == {"nnunet": 1, "normative": 1, "duct": 2}
    assert ov["any_caught"] == {"nnunet": 3, "normative": 4, "duct": 4}
    assert sum(ov["cells"].values()) == 10


def test_per_signal_rates_exclude_not_evaluable():
    d = synthetic()
    s9, du, nn = K.rate(d.sig_normative), K.rate(d.sig_duct), K.rate(d.sig_nnunet)
    assert (s9["n"], s9["k"], s9["n_not_evaluable"]) == (8, 4, 2)
    assert (du["n"], du["k"], du["n_not_evaluable"]) == (9, 4, 1)
    assert (nn["n"], nn["k"], nn["n_not_evaluable"]) == (10, 3, 0)


def test_gate_pass_and_fail():
    g = K.gate(K.overlap_counts(synthetic()))
    assert g["union_ok"] and g["unique_ok"] and g["passed"] and g["best_unique_signal"] == "duct"
    assert abs(g["union_ci_lo"] - 0.3968) < 1e-3
    # 10 lesions, 2 caught by nnunet only: union 20% fails; unique 20% alone is not enough
    d = pd.DataFrame({"sig_nnunet": pd.array([True, True] + [False] * 8, dtype="boolean"),
                      "sig_normative": pd.array([False] * 10, dtype="boolean"),
                      "sig_duct": pd.array([pd.NA] * 10, dtype="boolean")})
    g = K.gate(K.overlap_counts(d))
    assert not g["union_ok"] and g["unique_ok"] and not g["passed"]
    # union 40% (CI lo >= 20% needs n large enough) but every catch shared by two signals -> unique fails
    n = 100
    both = [True] * 40 + [False] * 60
    d = pd.DataFrame({"sig_nnunet": pd.array(both, dtype="boolean"),
                      "sig_normative": pd.array(both, dtype="boolean"),
                      "sig_duct": pd.array([False] * n, dtype="boolean")})
    g = K.gate(K.overlap_counts(d))
    assert g["union_ok"] and not g["unique_ok"] and not g["passed"]


def test_wilson_known_values():
    lo, hi = K.wilson(7, 10)
    assert abs(lo - 0.3968) < 1e-3 and abs(hi - 0.8922) < 1e-3
    assert K.wilson(0, 10)[0] == 0.0 and np.isnan(K.wilson(0, 0)[0])


# ------------------------------------------------------------------ component ranking / hits
def comps_fixture() -> pd.DataFrame:
    return pd.DataFrame([
        ("A", 1, 0.9, ""),
        ("A", 2, 0.3, "1"),
        ("A", 3, 0.3, "2"),      # tie with comp 2
        ("A", 4, 0.05, "1"),
        ("C", 1, 0.7, np.nan),   # wrong place only
    ], columns=["case_id", "comp_id", "peak_p", "touch_ids"])


LESIONS = pd.DataFrame([("A", 1), ("A", 2), ("A", 3), ("B", 1), ("C", 1)], columns=["case_id", "lesion_id"])


def test_lesion_hits_pessimistic_rank_and_ties():
    h = K.lesion_hits(comps_fixture(), LESIONS).set_index(["case_id", "lesion_id"])
    assert h.loc[("A", 1), "hit"] and h.loc[("A", 1), "rank"] == 3 and h.loc[("A", 1), "rank_min"] == 2
    assert abs(h.loc[("A", 1), "comp_peak_p"] - 0.3) < 1e-12
    assert h.loc[("A", 2), "rank"] == 3
    assert not h.loc[("A", 3), "hit"] and np.isnan(h.loc[("A", 3), "rank"])
    assert not h.loc[("B", 1), "hit"] and h.loc[("B", 1), "n_comps_case"] == 0
    assert h.loc[("A", 1), "n_comps_case"] == 4


def test_lesion_hits_rejects_unknown_lesions():
    bad = comps_fixture()
    bad.loc[0, "touch_ids"] = "9"
    with pytest.raises(AssertionError):
        K.lesion_hits(bad, LESIONS)


def test_case_fp_all_and_top_k():
    c = comps_fixture()
    assert K.case_fp(c, ["A", "B", "C"]).to_dict() == {"A": 1, "B": 0, "C": 1}
    assert K.case_fp(c, ["A", "B", "C"], top_k=2).to_dict() == {"A": 1, "B": 0, "C": 1}
    assert K.case_fp(c[c.case_id == "A"].iloc[1:], ["A"], top_k=3).to_dict() == {"A": 0}


def test_case_prediction_category():
    cat = K.case_prediction_category(comps_fixture(), LESIONS).tolist()
    assert cat == ["touches_this", "touches_this", "other_lesion_found", "no_prediction", "wrong_place_only"]


def test_evidence_order_and_thick_bins():
    d = synthetic().rename(columns={"dist_mpd": "dist_to_mpd_mm"})
    d["peak_p_dil2"] = np.linspace(0.001, 0.01, len(d))
    d["diam_mm"] = 25.0
    o = K.evidence_order(d)
    assert o.lesion_id.iloc[0] == 5 and o.n_signals.is_monotonic_decreasing
    assert set(o.lesion_id.iloc[-3:]) == {6, 7, 10}
    assert K.thick_bin(pd.Series([1.0, 2.5, 4.9, 5.0, 7.5])).tolist() == ["<2.5", "2.5-5", "2.5-5", ">=5", ">=5"]
