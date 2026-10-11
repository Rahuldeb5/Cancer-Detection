"""Chunked, resumable atlas run over PanTS cases.

  PYTHONPATH=src .venv/bin/python -m duct_atlas.run [--limit N] [--cases ID ...] [--ids-file F --tag T]
                                                     [--workers 4] [--chunk 50]

Cases = every PanTS_* folder under tumorlib.io.MASK_ROOT, sorted (9901). They are cut into fixed
chunks of the sorted list; each worker process handles one chunk, accumulating dense atlas arrays
(both grids, 5 arrays each) and writing them to work/duct_atlas/chunks/<start>_<end>.npz as sparse
(flat index, value) pairs plus a per-case CSV. A chunk whose files exist is skipped, so an
interrupted run resumes. --limit N runs the first N cases (chunk names include the range, so a
--limit run never collides with full-run chunks unless the ranges are identical, in which case
the work is reused). --ids-file runs a subset (one id per line) in chunks tagged <tag>_, e.g. the
tumor+ cases for the split figure: the atlas is a sum, so tumor- = full - tumor+. Finally all chunks of this run are summed into work/duct_atlas/<tag>/:
atlas.npz (dense arrays) and cases.csv.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from duct_atlas.case import ACC_KEYS, new_accumulators, process_case
from duct_atlas.grid import GRIDS
from tumorlib import io

WORK = io.REPO_ROOT / "work" / "duct_atlas"
CHUNK_DIR = WORK / "chunks"


def all_case_ids() -> list[str]:
    return sorted(p.name for p in io.MASK_ROOT.iterdir() if p.name.startswith("PanTS_"))


def chunk_paths(start: int, end: int, tag: str = "") -> tuple[Path, Path]:
    stem = f"{tag}{start:05d}_{end:05d}"
    return CHUNK_DIR / f"{stem}.npz", CHUNK_DIR / f"{stem}.csv"


def run_chunk(job: tuple[int, int, list[str], str]) -> tuple[str, int, float, float]:
    start, end, ids, tag = job
    npz, csv = chunk_paths(start, end, tag)
    t0 = time.time()
    acc = new_accumulators()
    rows = []
    for cid in ids:
        t = time.time()
        try:
            row, _ = process_case(cid, acc)
        except Exception as e:  # one bad case must not kill a chunk; it is listed in cases.csv
            row = {"case_id": cid, "status": "error", "msgs": f"{type(e).__name__}: {e}"}
        row["sec"] = time.time() - t
        rows.append(row)
    sparse = {}
    for g in acc:
        for k in ACC_KEYS:
            flat = acc[g][k].reshape(-1)
            nz = np.flatnonzero(flat)
            sparse[f"{g}__{k}__idx"] = nz.astype(np.int64)
            sparse[f"{g}__{k}__val"] = flat[nz]
    tmp = npz.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **sparse)
    pd.DataFrame(rows).to_csv(csv, index=False)
    tmp.rename(npz)  # npz appears last: its presence marks the chunk complete
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return npz.name, len(ids), time.time() - t0, peak_mb


def merge(jobs, out_dir: Path) -> tuple[dict, pd.DataFrame]:
    dense = {g.name: {k: g.zeros() for k in ACC_KEYS} for g in GRIDS}
    frames = []
    for start, end, _, tag in jobs:
        npz, csv = chunk_paths(start, end, tag)
        with np.load(npz) as z:
            for g in dense:
                for k in ACC_KEYS:
                    dense[g][k].reshape(-1)[z[f"{g}__{k}__idx"]] += z[f"{g}__{k}__val"]
        frames.append(pd.read_csv(csv))
    cases = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "atlas.npz", **{f"{g}__{k}": a for g, d in dense.items() for k, a in d.items()})
    cases.to_csv(out_dir / "cases.csv", index=False)
    return dense, cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cases", nargs="+", help="explicit case ids (own chunk, tag 'x_')")
    ap.add_argument("--ids-file", help="subset of case ids, one per line (needs --tag)")
    ap.add_argument("--tag", help="chunk/merge name for --ids-file")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--out", default=None, help="merge dir name under work/duct_atlas (default: full / limitN / cases)")
    a = ap.parse_args()
    assert a.workers <= 4, "PC has 15 GB RAM: <= 4 workers"

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    if a.cases:
        jobs = [(0, len(a.cases), list(a.cases), "x_" + "_".join(c[-5:] for c in a.cases[:3]) + "_")]
        out = a.out or "cases"
    elif a.ids_file:
        assert a.tag, "--ids-file needs --tag"
        ids = sorted(x.strip() for x in open(a.ids_file) if x.strip())
        jobs = [(s, min(s + a.chunk, len(ids)), ids[s:s + a.chunk], f"{a.tag}_") for s in range(0, len(ids), a.chunk)]
        out = a.out or a.tag
    else:
        ids = all_case_ids()
        if a.limit:
            ids = ids[:a.limit]
        jobs = [(s, min(s + a.chunk, len(ids)), ids[s:s + a.chunk], "") for s in range(0, len(ids), a.chunk)]
        out = a.out or (f"limit{a.limit}" if a.limit else "full")
    todo = [j for j in jobs if not chunk_paths(j[0], j[1], j[3])[0].exists()]
    n_cases = sum(len(j[2]) for j in jobs)
    print(f"{n_cases} cases in {len(jobs)} chunks; {len(jobs) - len(todo)} already done; {a.workers} workers", flush=True)

    t0 = time.time()
    if todo:
        with mp.get_context("fork").Pool(min(a.workers, len(todo)), maxtasksperchild=1) as pool:
            for i, (name, n, sec, peak) in enumerate(pool.imap_unordered(run_chunk, todo), 1):
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] {name}: {n} cases {sec:.0f}s ({sec / n:.2f} s/case), worker peak RSS "
                      f"{peak:.0f} MB; elapsed {el / 60:.1f} min, ETA {el / i * (len(todo) - i) / 60:.1f} min", flush=True)
    dense, cases = merge(jobs, WORK / out)
    print(f"merged -> {WORK / out}; status counts:\n{cases['status'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
