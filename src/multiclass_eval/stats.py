"""Paired case-level bootstrap (pure). The resampling unit is the case, because lesions in one scan share the
scan's quality, phase, thickness and the model's behaviour there; resampling lesions would understate the CI.

A statistic is a ratio of per-case sums, mean = sum_cases(num) / sum_cases(den), so a bootstrap replicate is
just a reweighting of cases: W (B x n_cases) multinomial counts, stat_b = (W @ num) / (W @ den). This is exactly
"resample cases with replacement, pool their lesions, take the mean" (test_stats_matches_naive_loop).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

B = 1000
SEED = 0
ALPHA = 0.05


def case_weights(n_cases: int, b: int = B, seed: int = SEED) -> np.ndarray:
    """(b, n_cases) float matrix: how often each case is drawn in each replicate."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_cases, np.full(n_cases, 1.0 / n_cases), size=b).astype(np.float64)


def paired_ratio(df: pd.DataFrame, case_col: str, a_col: str, b_col: str, cases: np.ndarray,
                 W: np.ndarray) -> dict:
    """Mean of a_col, b_col and (b - a) over the rows of df, with case-bootstrap percentile CIs.
    `cases` fixes the case order of W's columns; cases with no rows in df get weight but contribute 0/0
    (they still matter: they change how many rows each replicate holds, as they would in a real resample)."""
    idx = pd.Index(cases)
    g = df.groupby(case_col)
    num_a = g[a_col].sum().reindex(idx, fill_value=0.0).to_numpy(float)
    num_b = g[b_col].sum().reindex(idx, fill_value=0.0).to_numpy(float)
    den = g.size().reindex(idx, fill_value=0).to_numpy(float)
    n = int(den.sum())
    out = dict(n=n, n_cases=int((den > 0).sum()))
    if n == 0:
        return out | dict(a=np.nan, b=np.nan, delta=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_boot=np.nan)
    wd = W @ den
    with np.errstate(invalid="ignore", divide="ignore"):
        da = (W @ num_a) / wd
        db = (W @ num_b) / wd
    dd = db - da
    ok = np.isfinite(dd)
    lo, hi = np.percentile(dd[ok], [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    a, b = num_a.sum() / n, num_b.sum() / n
    # two-sided bootstrap p for delta = 0 (percentile method; resolution 1/B)
    p = min(1.0, 2 * min((dd[ok] <= 0).mean(), (dd[ok] >= 0).mean())) if (b - a) != 0 else 1.0
    return out | dict(a=a, b=b, delta=b - a, ci_lo=float(lo), ci_hi=float(hi), p_boot=float(p),
                      n_boot_valid=int(ok.sum()))
