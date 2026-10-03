"""PHASE 0: the normal-bank candidate pool, its filters, and the stratified 300-case sample.

Filters, applied in this order (counts reported at each step):
  1. PanTS case NOT in the 1308-case cohort (the cohort is evaluation data only, never bank)
  2. metadata tumor? == 0
  3. no "Pancreas lesions" section in the structured report (S1's rule: a line starting
     "Pancreas lesions:")
  4. pancreatic_lesion mask EMPTY, read with tumorlib.io.load_mask (a None = unreadable mask fails)
  5. non-empty 4-mask gland envelope. pancreas.nii.gz is an un-pulled LFS stub for every non-cohort
     case, while head/body/tail are real; head|body|tail non-empty already proves the envelope is
     non-empty. Cases where all three are empty are listed as "undetermined" and only enter the pool
     if their pancreas.nii.gz (pulled for them alone) is non-empty.
  6. patient overlap with the cohort: the metadata has no patient/study identifier, so this filter
     cannot be applied (reported as unverifiable).

Sample: 300 cases, seed 42, stratified so the (raw phase x slice-thickness bin) mix matches the 981
cohort POSITIVES (largest-remainder rounding). Thickness for pool cases comes from max(metadata
spacing): on the 1308 cohort cases this reproduces S1's affine-derived thickness bin in 1306/1308.
The true value is recomputed from each extracted CT's affine afterwards and the achieved mix reported.

Usage (repo root, PC):
  PYTHONPATH=src .venv/bin/python -m normative.pool checks    # mask checks over the pool (resumable)
  PYTHONPATH=src .venv/bin/python -m normative.pool sample    # filters + sample + member lists
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time

import numpy as np
import pandas as pd

from normative import common as C
from tumorlib import io as tio

CHECKS = C.WORK / "pool" / "pool_mask_checks.csv"
SECTIONS = ("pancreas_head", "pancreas_body", "pancreas_tail")


def report_has_lesion_section(reports: pd.Series) -> pd.Series:
    return reports.fillna("").str.contains(r"^Pancreas lesions:", regex=True, flags=__import__("re").M)


def mask_check(job: tuple[str, tuple[int, int, int]]) -> dict:
    """Lesion emptiness + head/body/tail non-emptiness for one non-cohort case (no CT exists yet,
    so the grid check inside load_mask is skipped; the metadata shape is checked instead)."""
    case_id, shape = job
    row = dict(case_id=case_id)
    with C.quiet() as log:
        les = tio.load_mask(case_id, "pancreatic_lesion")
        row["lesion_readable"] = les is not None
        row["lesion_vox"] = int(les.sum()) if les is not None else -1
        row["shape_ok"] = les is not None and tuple(les.shape) == tuple(shape)
        del les
        hbt = 0
        ok = True
        for s in SECTIONS:
            m = tio.load_mask(case_id, s)
            if m is None:
                ok = False
                continue
            row[f"{s}_vox"] = int(m.sum())
            hbt += row[f"{s}_vox"]
            del m
    row["sections_readable"] = ok
    row["hbt_vox"] = hbt
    row["log"] = " | ".join(l for l in log.getvalue().splitlines() if "missing" not in l or "ct.nii" not in l)[:300]
    return row


def candidates_meta() -> pd.DataFrame:
    m = C.metadata()
    cohort = set(tio.case_ids())
    m["in_cohort"] = m.case_id.isin(cohort)
    m["report_section"] = report_has_lesion_section(m["structured report"])
    return m


def run_checks() -> None:
    m = candidates_meta()
    pool = m[~m.in_cohort & (m["tumor?"] == 0)]
    done = set(pd.read_csv(CHECKS).case_id) if CHECKS.exists() else set()
    jobs = [(c, s) for c, s in zip(pool.case_id, pool.shape_meta) if c not in done]
    print(f"{len(pool)} non-cohort tumor?=0 cases, {len(done)} already checked, {len(jobs)} to go", flush=True)
    CHECKS.parent.mkdir(parents=True, exist_ok=True)
    t0, buf = time.time(), []
    with mp.Pool(C.NUM_WORKERS, maxtasksperchild=50) as p:
        for i, row in enumerate(p.imap_unordered(mask_check, jobs, chunksize=4), 1):
            buf.append(row)
            if len(buf) >= 200 or i == len(jobs):
                pd.DataFrame(buf).to_csv(CHECKS, mode="a", header=not CHECKS.exists(), index=False)
                buf = []
                print(f"  {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)


def largest_remainder(weights: pd.Series, n: int) -> pd.Series:
    raw = weights / weights.sum() * n
    out = np.floor(raw).astype(int)
    rem = n - int(out.sum())
    out[(raw - out).sort_values(ascending=False, kind="stable").index[:rem]] += 1
    return out


def sample() -> None:
    m = candidates_meta()
    chk = pd.read_csv(CHECKS).drop_duplicates("case_id")
    counts = [("PanTS cases", len(m))]
    s = m[~m.in_cohort]
    counts.append(("1. not in the 1308 cohort", len(s)))
    s = s[s["tumor?"] == 0]
    counts.append(("2. metadata tumor? == 0", len(s)))
    marg_report = int(s.report_section.sum())
    s = s[~s.report_section]
    counts.append(("3. no 'Pancreas lesions' report section", len(s)))
    assert set(s.case_id) <= set(chk.case_id), "run `pool checks` first"
    s = s.merge(chk, on="case_id", how="left", validate="1:1")
    s = s[s.lesion_readable & s.shape_ok & (s.lesion_vox == 0)]
    counts.append(("4. pancreatic_lesion mask readable, on-grid and EMPTY", len(s)))
    undetermined = s[~s.sections_readable | (s.hbt_vox == 0)]
    s = s[s.sections_readable & (s.hbt_vox > 0)]
    counts.append(("5. non-empty envelope (head|body|tail non-empty)", len(s)))
    s = s[s["ct phase"].notna()]
    counts.append(("   ... with a known contrast phase", len(s)))

    # marginal counts over all non-cohort tumor?=0 cases, for context
    allneg = m[~m.in_cohort & (m["tumor?"] == 0)].merge(chk, on="case_id", how="left")
    marg = dict(report_section=marg_report,
                lesion_nonempty=int((allneg.lesion_vox > 0).sum()),
                lesion_unreadable=int((~allneg.lesion_readable.astype(bool)).sum()),
                hbt_empty=int((allneg.hbt_vox == 0).sum()),
                undetermined_after_4=len(undetermined))

    # ---- strata of the cohort positives
    sp = pd.read_csv(C.S1_CASES)[["case_id", "slice_thickness_mm"]]
    pos = m[m.in_cohort & (m["tumor?"] == 1)].merge(sp, on="case_id", validate="1:1")
    pos["tbin"] = C.thick_bin(pos.slice_thickness_mm).values
    pos["phase"] = pos["ct phase"].fillna("NA")
    target = pos.groupby(["phase", "tbin"], observed=True).size()
    target = target[target.index.get_level_values(0) != "NA"]   # 1 positive has no phase
    quota = largest_remainder(target.astype(float), C.N_BANK)

    s["tbin"] = C.thick_bin(s.max_spacing_meta).values
    s["phase"] = s["ct phase"]
    avail = s.groupby(["phase", "tbin"], observed=True).size()
    rng = np.random.default_rng(C.SEED)
    picks, short = [], []
    for key, q in quota.items():
        cell = s[(s.phase == key[0]) & (s.tbin == key[1])].sort_values("case_id")
        if len(cell) < q:
            short.append((key, int(q), len(cell)))
        k = min(int(q), len(cell))
        if k:
            picks.append(cell.iloc[rng.choice(len(cell), k, replace=False)])
    assert not short, f"strata with too few pool cases: {short}"
    bank = pd.concat(picks).sort_values("case_id")
    assert len(bank) == C.N_BANK and not bank.case_id.isin(set(tio.case_ids())).any()

    out = C.WORK / "pool"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(counts, columns=["filter", "n_pass"]).to_csv(out / "filter_counts.csv", index=False)
    undetermined[["case_id", "hbt_vox", "sections_readable"]].to_csv(out / "undetermined_envelope.csv", index=False)
    tab = pd.DataFrame({"cohort_pos": target, "quota": quota, "pool_available": avail.reindex(quota.index)})
    tab.to_csv(out / "strata_quota.csv")
    bank[["case_id", "ct phase", "max_spacing_meta", "tbin", "shape_meta", "site", "sex", "age",
          "study year"]].to_csv(out / "bank_cases.csv", index=False)
    (out / "bank_ids.txt").write_text("\n".join(bank.case_id) + "\n")

    # LFS include lists, one structure per pull (gotcha 7)
    for st in C.BANK_STRUCTURES:
        inc = ",".join(f"mask_only/{c}/segmentations/{st}.nii.gz" for c in bank.case_id)
        assert len(inc.encode()) < C.LFS_INCLUDE_MAX, (st, len(inc))
        (out / f"lfs_include_{st}.txt").write_text(inc)

    print("filter counts:")
    for k, v in counts:
        print(f"  {v:>6}  {k}")
    print("marginal (over all non-cohort tumor?=0):", marg)
    print("6. patient overlap: no patient/study identifier exists in the metadata -> unverifiable")
    print("\nphase x thickness quota (cohort positives -> bank) and pool availability:")
    print(tab.to_string())
    print(f"\nbank: {len(bank)} cases -> {out / 'bank_ids.txt'}")
    meta = C.metadata().set_index("case_id")
    est = meta.loc[bank.case_id, "shape_meta"].map(np.prod).sum()
    print(f"bank voxels {est / 1e9:.2f} G")
    pd.Series(marg).to_csv(out / "marginal_counts.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["checks", "sample"])
    a = ap.parse_args()
    {"checks": run_checks, "sample": sample}[a.step]()


if __name__ == "__main__":
    sys.exit(main())
