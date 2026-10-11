"""S15 Part B: S12's contrast-phase estimate on the 300 S9 normal-bank CTs (replaces their noisy metadata phase).

S12's code is used unchanged: phase.measure.measure_case (aorta / portal-vein HU, CT grid, tumorlib loaders) on each
bank CT, then the S12 rule refitted exactly as phase.analyze does (3-component GMM on the 1308 cohort cases'
(aorta_hu, pv_hu), seed 0, n_init 20, from work/phase/measure.csv) and APPLIED to the bank (predict only; the bank
does not change the fit). Labels: core.assign (posterior < 0.8 -> uncertain), pv_enh_frac, early arterial.

Usage: PYTHONPATH=src .venv/bin/python -m glandshape.bank_phase [--workers 4]
Output: work/gland_shape/partB/bank_phase.csv and bank_phase.txt
"""
from __future__ import annotations

import os
from pathlib import Path

BANK = Path.home() / "research/datasets/pants/ct_normal_bank"
os.environ["TUMORLIB_CT_ROOT"] = str(BANK)  # before tumorlib is imported: bank CTs are not in ct_staging

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io as sio  # noqa: E402
import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phase import analyze as A12  # noqa: E402
from phase import core as K12  # noqa: E402
from phase import measure as M12  # noqa: E402
from tumorlib import io as tio  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "work/gland_shape/partB"


def bank_ids() -> list[str]:
    return sorted(p.name for p in BANK.iterdir() if (p / "ct.nii.gz").exists())


def _one(cid):
    buf = sio.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            row = M12.measure_case(cid)
        except Exception as e:
            row = {"case_id": cid, "error": f"{type(e).__name__}: {e}"}
    row["log"] = " | ".join(l for l in buf.getvalue().splitlines() if "postcava" not in l)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    assert tio.CT_ROOT == BANK, tio.CT_ROOT
    OUT.mkdir(parents=True, exist_ok=True)
    ids = bank_ids()
    with mp.Pool(a.workers) as pool:
        rows = pool.map(_one, ids, chunksize=4)
    b = pd.DataFrame(rows)

    # S12 rule, refitted on the cohort exactly as phase.analyze, applied to the bank
    m = pd.read_csv(A12.MEAS)
    ok_c = m.aorta_hu.notna() & m.pv_hu.notna()
    gmm = A12.fit_gmm(m.loc[ok_c, ["aorta_hu", "pv_hu"]].to_numpy())
    names = K12.name_clusters(gmm.means_)
    # regression: the refit reproduces S12's labels on the cohort
    lab_c, _, _ = K12.assign(gmm.predict_proba(m.loc[ok_c, ["aorta_hu", "pv_hu"]].to_numpy()), names)
    s12 = pd.read_csv(REPO / "results/phase_estimate/phase_estimate.csv").set_index("case_id").est_phase
    agree = float(np.mean(lab_c == s12.loc[m.loc[ok_c, "case_id"]].to_numpy()))
    assert agree == 1.0, agree

    for c in ("aorta_hu", "pv_hu"):
        if c not in b:
            b[c] = np.nan
    ok = b.aorta_hu.notna() & b.pv_hu.notna()
    b["est_phase"], b["posterior"] = "unmeasurable", np.nan
    if ok.any():
        X = b.loc[ok, ["aorta_hu", "pv_hu"]].to_numpy()
        lab, pmax, _ = K12.assign(gmm.predict_proba(X), names)
        b.loc[ok, "est_phase"], b.loc[ok, "posterior"] = lab, pmax
        b.loc[ok, "pv_enh_frac"] = K12.pv_enhancement_fraction(b.loc[ok, "pv_hu"], gmm.means_, names)
    b["early_arterial"] = (b.est_phase == "arterial") & (b.get("pv_enh_frac", np.nan) < K12.EARLY_ART_F)
    b["phase5"] = np.where(b.early_arterial, "early arterial",
                           np.where(b.est_phase == "arterial", "late arterial", b.est_phase))
    meta = pd.read_csv(REPO / "results/PanTS_metadata_new.csv", usecols=["PanTS ID", "ct phase"]).rename(
        columns={"PanTS ID": "case_id", "ct phase": "metadata_phase"})
    b = b.merge(meta, on="case_id", how="left")
    b.to_csv(OUT / "bank_phase.csv", index=False)

    coh = s12.reset_index().rename(columns={"est_phase": "e"})
    L = [f"S15 Part B bank phase (S12 rule; cohort refit reproduces S12 labels: {agree:.0%} of "
         f"{int(ok_c.sum())} cases)",
         f"bank cases {len(b)}, measurable {int(ok.sum())}, errors {int(b.get('error', pd.Series(dtype=str)).notna().sum())}",
         "bank est_phase (5 strata): " + str(b.phase5.value_counts().to_dict()),
         "cohort est_phase (S12, carried over): " + str(coh.e.value_counts().to_dict()),
         "bank metadata phase vs estimate:",
         pd.crosstab(b.metadata_phase.fillna("missing"), b.phase5).to_string(),
         f"bank slice thickness median {b.slice_thickness_mm.median():.2f} mm"]
    txt = "\n".join(L)
    (OUT / "bank_phase.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
