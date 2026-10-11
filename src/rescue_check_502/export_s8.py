"""S14 step 0: S8's Dataset502 per-case pickles -> an S11 --pred_source directory, so S11's
rescue_check.nnunet_table runs on 502 unchanged.

Writes work/rescue_check_502/pred_source/
  components/components_t{t}.csv  t in {0.5, 0.1, 0.01, 0.001}: case_id, t, comp_id, peak_p, touch_ids (S2 schema;
                                  S8's comps are S2's analyze() output, 26-connected over the whole volume)
  per_lesion.csv                  case_id, lesion_id, gt_vox, peak_p_dil2 (peak p in lesion + 2 mm), max_t_hit_touch

Usage (PC, repo root):
  PYTHONPATH=src .venv/bin/python -m rescue_check_502.export_s8
  PYTHONPATH=src .venv/bin/python -m rescue_check.nnunet_table --pred_source work/rescue_check_502/pred_source --tag ds502
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
S8 = REPO / "work/nnunet_multiclass/cases/502"
OUT = REPO / "work/rescue_check_502/pred_source"
TS = (0.5, 0.1, 0.01, 0.001)


def main() -> None:
    comps, les = [], []
    files = sorted(S8.glob("PanTS_*.pkl"))
    assert len(files) == 1308, len(files)
    for f in files:
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        assert r["model"] == "502"
        comps.extend(r["comps"])
        les.extend(r["lesions"])
    c = pd.DataFrame(comps)
    lt = pd.DataFrame(les)
    (OUT / "components").mkdir(parents=True, exist_ok=True)
    for t in TS:
        ct = c[np.isclose(c.t, t, rtol=0, atol=1e-12)][["case_id", "t", "comp_id", "peak_p", "touch_ids"]].copy()
        ct["t"] = t
        ct.to_csv(OUT / "components" / f"components_t{t:g}.csv", index=False, float_format="%.9g")
        print(f"components_t{t:g}.csv: {len(ct)} components in {ct.case_id.nunique()} cases")

    # max_t_hit_touch: highest S2-ladder t (all 9 rungs) at which some component touches the lesion (NaN = never)
    hit = []
    for r in c.itertuples(index=False):
        for j in (str(r.touch_ids).split(";") if isinstance(r.touch_ids, str) and r.touch_ids else []):
            hit.append((r.case_id, int(j), r.t))
    h = pd.DataFrame(hit, columns=["case_id", "lesion_id", "t"]).groupby(["case_id", "lesion_id"]).t.max()
    pl = lt[["case_id", "lesion_id", "gt_vox", "peak_p_dil2"]].copy()
    pl["max_t_hit_touch"] = [h.get((a, b), np.nan) for a, b in zip(pl.case_id, pl.lesion_id)]
    assert len(pl) == 1235 and not pl.duplicated(["case_id", "lesion_id"]).any()
    pl.to_csv(OUT / "per_lesion.csv", index=False, float_format="%.9g")
    print(f"per_lesion.csv: {len(pl)} lesions -> {OUT}")


if __name__ == "__main__":
    main()
