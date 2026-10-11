"""Constants fixed a priori (2026-10-03, before any S11 result was computed) and pure functions.

Nothing here does I/O. Every operating point and gate threshold below is from the S11 brief and must
not be changed after seeing results.

Signals (each evaluated per lesion, key (case_id, lesion_id)):
  1. faint nnU-Net guess: a component of p > t touches the lesion (t in NN_THRESHOLDS). Operating point:
     t = 0.01 AND that component ranks <= 3 among the case's t = 0.01 components by peak probability.
  2. normative model (S9): score A matched-radius candidate rank <= 10. Lesions S9 could not score
     (pct_A is NaN) are not evaluable.
  3. duct: case flagged by (mpd_dilated OR MPD head-end dilation) AND lesion within 10 mm of the MPD.
     Lesions in cases without an MPD mask are not evaluable.

Not-evaluable lesions are excluded from that signal's own hit rate, and count as NOT caught in the
union / overlap / gate (conservative for the gate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ fixed a priori
NN_THRESHOLDS = (0.1, 0.01, 0.001)
NN_OP_T = 0.01
NN_OP_RANK = 3
S9_TOP_N = 10
DUCT_DIST_MM = 10.0
MPD_HEAD_DILATE_MM = 3.21     # src/duct-cutoff/component_features.py DILATE_MM["mpd"]
DETECT_T = 0.5

GATE_UNION_MIN = 0.30
GATE_UNION_CI_LO_MIN = 0.20
GATE_UNIQUE_MIN = 0.10        # caught ONLY by this signal / all missed lesions in the group

SIGNALS = ("nnunet", "normative", "duct")
BINS = ("<5", "5-10", "10-20", "20-40", ">=40")
PRIMARY_BIN = "20-40"
CONTEXT_BINS = ("10-20", ">=40")


# ------------------------------------------------------------------ statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k/n. (nan, nan) for n == 0."""
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1.0 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(hits: pd.Series) -> dict:
    """Hit rate over the evaluable entries of a nullable-boolean series (NA = not evaluable)."""
    ev = hits.dropna().astype(bool)
    n, k = int(len(ev)), int(ev.sum())
    lo, hi = wilson(k, n)
    return dict(n=n, k=k, rate=k / n if n else np.nan, ci_lo=lo, ci_hi=hi, n_not_evaluable=int(hits.isna().sum()))


# ------------------------------------------------------------------ nnU-Net components
def split_ids(s) -> list[int]:
    return [int(x) for x in str(s).split(";")] if isinstance(s, str) and s else []


def rank_components(comps: pd.DataFrame) -> pd.DataFrame:
    """Add `rank` (1 = highest peak_p in its case) and `rank_min`. `rank` is pessimistic under ties:
    the number of the case's components with peak_p >= this one. `rank_min` is the optimistic
    (competition) rank, kept only to count how often ties matter."""
    c = comps.copy()
    g = c.groupby("case_id")["peak_p"]
    c["rank"] = g.rank(method="max", ascending=False).astype(int)
    c["rank_min"] = g.rank(method="min", ascending=False).astype(int)
    return c


def lesion_hits(comps: pd.DataFrame, lesions: pd.DataFrame) -> pd.DataFrame:
    """Per lesion at one threshold: hit = any component touches it; rank = best (smallest) pessimistic
    rank among the touching components; rank_min = the same with optimistic tie ranks; peak of that
    component; n_comps_case = components in the case at this threshold.

    comps: one threshold, columns case_id, comp_id, peak_p, touch_ids. lesions: case_id, lesion_id."""
    c = rank_components(comps)
    rows = []
    for r in c.itertuples(index=False):
        for j in split_ids(r.touch_ids):
            rows.append((r.case_id, j, r.rank, r.rank_min, r.peak_p))
    t = pd.DataFrame(rows, columns=["case_id", "lesion_id", "rank", "rank_min", "comp_peak_p"])
    if len(t):
        t = t.sort_values(["rank", "comp_peak_p"], ascending=[True, False])
        best = t.groupby(["case_id", "lesion_id"], as_index=False).agg(
            rank=("rank", "first"), comp_peak_p=("comp_peak_p", "first"), rank_min=("rank_min", "min"))
    else:
        best = t
    n_case = c.groupby("case_id").size().rename("n_comps_case")
    out = lesions[["case_id", "lesion_id"]].merge(best, on=["case_id", "lesion_id"], how="left", validate="1:1")
    unknown = set(zip(best.case_id, best.lesion_id)) - set(zip(lesions.case_id, lesions.lesion_id))
    assert not unknown, f"components touch lesions not in the lesion table: {sorted(unknown)[:5]}"
    out["hit"] = out["rank"].notna()
    out["n_comps_case"] = out.case_id.map(n_case).fillna(0).astype(int)
    return out


def case_fp(comps: pd.DataFrame, case_ids, top_k: int | None = None) -> pd.Series:
    """FP components per case (touch rule: a component touching no GT lesion), optionally only among
    the case's top_k components by peak_p (pessimistic ranks). Cases with no components get 0."""
    c = rank_components(comps) if len(comps) else comps.assign(rank=[])
    if top_k is not None:
        c = c[c["rank"] <= top_k]
    fp = c[[not split_ids(s) for s in c.touch_ids]] if len(c) else c
    return fp.groupby("case_id").size().reindex(list(case_ids), fill_value=0)


def case_prediction_category(comps_t05: pd.DataFrame, lesions: pd.DataFrame) -> pd.Series:
    """For each lesion, what its case predicted at t = 0.5 (the saved segmentation):
    'no_prediction' (no component), 'other_lesion_found' (some component touches a different lesion of
    the case), 'wrong_place_only' (components exist, none touches any GT lesion), or 'touches_this'."""
    n = comps_t05.groupby("case_id").size()
    touched = {}
    for r in comps_t05.itertuples(index=False):
        touched.setdefault(r.case_id, set()).update(split_ids(r.touch_ids))
    out = []
    for cid, lid in zip(lesions.case_id, lesions.lesion_id):
        t = touched.get(cid, set())
        if n.get(cid, 0) == 0:
            out.append("no_prediction")
        elif lid in t:
            out.append("touches_this")
        elif t:
            out.append("other_lesion_found")
        else:
            out.append("wrong_place_only")
    return pd.Series(out, index=lesions.index, name="case_pred_t05")


# ------------------------------------------------------------------ signals at the operating points
def _nullable(values, evaluable) -> pd.Series:
    s = pd.Series(np.asarray(values, bool), dtype="boolean")
    s[~np.asarray(evaluable, bool)] = pd.NA
    return s


def nnunet_signal(hit_t: pd.Series, rank_t: pd.Series) -> pd.Series:
    """Signal 1 at the operating point: touched at NN_OP_T by a component of case rank <= NN_OP_RANK.
    Always evaluable (every lesion has a probability map)."""
    v = hit_t.fillna(False).astype(bool).to_numpy() & (rank_t.fillna(np.inf).to_numpy() <= NN_OP_RANK)
    return pd.Series(v, dtype="boolean")


def normative_signal(pct_a: pd.Series, cand_rank_a: pd.Series) -> pd.Series:
    """Signal 2: matched-radius score-A candidate rank <= S9_TOP_N; NA where S9 could not score."""
    v = cand_rank_a.fillna(np.inf).to_numpy() <= S9_TOP_N
    return _nullable(v, pct_a.notna().to_numpy())


def duct_case_flag(mpd_dilated: pd.Series, head_caliber_mm: pd.Series) -> pd.Series:
    """Case-level duct flag: mpd_dilated OR MPD head-end caliber >= MPD_HEAD_DILATE_MM (NaN head = no)."""
    he = head_caliber_mm.fillna(-np.inf).to_numpy() >= MPD_HEAD_DILATE_MM
    return pd.Series(mpd_dilated.fillna(False).astype(bool).to_numpy() | he)


def duct_signal(case_flag: pd.Series, dist_to_mpd_mm: pd.Series, mpd_present: pd.Series) -> pd.Series:
    """Signal 3: case flag AND lesion within DUCT_DIST_MM of the MPD; NA where the case has no MPD mask."""
    v = case_flag.to_numpy(bool) & (dist_to_mpd_mm.fillna(np.inf).to_numpy() <= DUCT_DIST_MM)
    return _nullable(v, mpd_present.astype(bool).to_numpy())


# ------------------------------------------------------------------ overlap and gate
def caught(df: pd.DataFrame, signals=SIGNALS) -> pd.DataFrame:
    """Boolean caught-by matrix: not evaluable (NA) counts as not caught."""
    return pd.DataFrame({s: df[f"sig_{s}"].fillna(False).astype(bool).to_numpy() for s in signals}, index=df.index)


def overlap_counts(df: pd.DataFrame, signals=SIGNALS) -> dict:
    """Venn cells, 0/1/2/3 counts, union and per-signal unique catches over the rows of df."""
    c = caught(df, signals)
    n = len(c)
    nsig = c.sum(axis=1)
    cells = {}
    for mask in range(1 << len(signals)):
        on = [s for i, s in enumerate(signals) if mask >> i & 1]
        sel = np.ones(n, bool)
        for i, s in enumerate(signals):
            sel &= c[s].to_numpy() == bool(mask >> i & 1)
        cells["+".join(on) if on else "none"] = int(sel.sum())
    unique = {s: int((c[s] & (nsig == 1)).sum()) for s in signals}
    k_union = int((nsig > 0).sum())
    lo, hi = wilson(k_union, n)
    return dict(n=n, by_n_signals={k: int((nsig == k).sum()) for k in range(len(signals) + 1)},
                cells=cells, union_k=k_union, union_rate=k_union / n if n else np.nan,
                union_ci=(lo, hi), unique=unique,
                any_caught={s: int(c[s].sum()) for s in signals})


def gate(ov: dict) -> dict:
    """Fixed gate on the missed primary-bin lesions. Returns the parts and the verdict."""
    n = ov["n"]
    uniq = {s: (k / n if n else np.nan) for s, k in ov["unique"].items()}
    best = max(uniq, key=lambda s: uniq[s])
    union_ok = ov["union_rate"] >= GATE_UNION_MIN and ov["union_ci"][0] >= GATE_UNION_CI_LO_MIN
    unique_ok = uniq[best] >= GATE_UNIQUE_MIN
    return dict(union_rate=ov["union_rate"], union_ci_lo=ov["union_ci"][0], union_ok=bool(union_ok),
                unique_frac=uniq, best_unique_signal=best, unique_ok=bool(unique_ok),
                passed=bool(union_ok and unique_ok))


# ------------------------------------------------------------------ review list ordering
def evidence_order(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-specified ordering for the visual-review list (within the rows given, e.g. missed 20-40 mm).
    n_signals (desc), then `graded` = mean of the available graded strengths, each in [0, 1]:
      nnU-Net  percentile rank of peak_p_dil2 within the rows given
      S9       pct_A / 100
      duct     1 - min(dist_to_mpd_mm, 30) / 30 if the case is duct-flagged, else 0 (NA without MPD mask)
    then diam_mm (desc). Strongest first."""
    d = df.copy()
    d["n_signals"] = caught(d).sum(axis=1)
    g_nn = d["peak_p_dil2"].rank(pct=True)
    g_s9 = d["pct_A"] / 100.0
    g_du = np.where(d["duct_case_flag"].astype(bool),
                    1.0 - np.minimum(d["dist_to_mpd_mm"].to_numpy(float), 30.0) / 30.0, 0.0)
    g_du = pd.Series(g_du, index=d.index).where(d["mpd_present"].astype(bool))
    d["graded"] = pd.concat([g_nn, g_s9, g_du], axis=1).mean(axis=1, skipna=True)
    return d.sort_values(["n_signals", "graded", "diam_mm"], ascending=[False, False, False])


def thick_bin(mm: pd.Series) -> pd.Series:
    """Slice-spacing bins at the S1 cut points: <2.5, 2.5-5, >=5 mm."""
    return pd.cut(mm, [-np.inf, 2.5, 5.0, np.inf], right=False, labels=["<2.5", "2.5-5", ">=5"]).astype(str)
