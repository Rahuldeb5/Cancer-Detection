"""Centerline + cutoff detection over GT duct masks; writes CSVs to results/duct_cutoff/.

Run from the repo root (paths are relative to it):
    python src/duct-cutoff/main.py --limit 12                  # smoke test
    python src/duct-cutoff/main.py --set neg                   # null test on tumor-negatives
    python src/duct-cutoff/main.py --set all --workers 6       # everything
    python src/duct-cutoff/main.py --set all --profiles        # also save per-duct caliber profiles

Outputs (results/duct_cutoff/):
    cutoff_per_duct.csv   one row per (case, duct): centerline QC, caliber stats, cutoff
    cutoff_per_case.csv   one row per case: per-duct detections + double-duct pairing
    profiles/*.npz        (--profiles only) caliber + world-mm path for plotting/debugging
Then prints a summary: usable-duct-length table, negatives-vs-positives detection rates.
"""
import argparse
import multiprocessing as mp
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from centerline import (
    CenterlineParams, CutoffParams, caliber_profile, detect_cutoff, extract_centerline,
    iso_to_world_mm, mask_centroid_mm, orient_head_to_tail,
)
from loader import case_ids, load_ducts_isotropic, load_pancreas_regions

QUALITY_CSV = Path("results/geometry_results/duct_mask_quality.csv")
RESULTS_DIR = Path("results/duct_cutoff")

STEP_MM = 1.0
END_MARGIN_MM = 5.0
DOUBLE_DUCT_DIST_MM = 30.0
DUCTS = ("mpd", "cbd")
# Session 2 Youden's-J thresholds (native-grid global percentiles) -- a first guess only;
# this pipeline's calibers differ, so re-derive from the negatives before trusting them.
CUTOFF_PARAMS = {
    "mpd": CutoffParams(dilate_thresh_mm=3.21, step_mm=STEP_MM),
    "cbd": CutoffParams(dilate_thresh_mm=8.05, step_mm=STEP_MM),
}
CENTERLINE_PARAMS = CenterlineParams(step_mm=STEP_MM, end_margin_mm=END_MARGIN_MM)


def duct_row(case_id: str, duct: str, tumor: float, **fields) -> dict:
    return {"case_id": case_id, "duct": duct, "tumor_present": tumor, **fields}


def analyze_duct(case_id: str, duct: str, tumor: float, iso: dict, head_c, tail_c,
                 save_profiles: bool) -> dict:
    mask = iso[duct]
    cp, cut_p = CENTERLINE_PARAMS, CUTOFF_PARAMS[duct]
    native_sp = iso["native_spacing"]
    base = dict(
        native_sp_max_mm=max(native_sp), native_sp_min_mm=min(native_sp),
        mask_vox_iso=int(mask.sum()),
    )

    res = extract_centerline(mask, iso["spacing"], cp)
    base.update(
        status=res.status, n_components=res.n_components,
        largest_comp_vox=max(res.component_sizes) if res.component_sizes else 0,
        n_dropped_components=res.n_dropped_components,
        n_unbridged_components=res.n_unbridged_components,
        n_bridged_gaps=res.n_bridged_gaps,
        max_gap_mm=max(res.gap_lengths_mm) if res.gap_lengths_mm else 0.0,
        total_len_mm=res.total_len_mm,
    )
    if res.status != "ok":
        return duct_row(case_id, duct, tumor, detected=False, **base)

    if head_c is None:
        base["status"] = "no_regions"   # cannot orient without pancreas head/tail masks
        return duct_row(case_id, duct, tumor, detected=False, **base)

    res = orient_head_to_tail(
        res, iso[f"{duct}_offset"], iso["spacing"], native_sp, iso["affine"], head_c,
        tail_centroid_mm=tail_c if duct == "mpd" else None,   # CBD runs perpendicular to head->tail
    )
    base.update(status=res.status, flipped=res.flipped, orientation_margin_mm=res.orientation_margin_mm)
    if res.status != "ok":
        return duct_row(case_id, duct, tumor, detected=False, **base)

    cal = caliber_profile(mask, res, iso["spacing"], STEP_MM)
    fin = cal[np.isfinite(cal)]
    k = int(round(10 / STEP_MM))
    base.update(
        n_samples=len(cal), n_finite=len(fin),
        caliber_p50_mm=float(np.median(fin)) if fin.size else np.nan,
        caliber_p90_mm=float(np.percentile(fin, 90)) if fin.size else np.nan,
        caliber_max_mm=float(fin.max()) if fin.size else np.nan,
        caliber_head_mm=float(np.nanmedian(cal[:k])) if np.isfinite(cal[:k]).any() else np.nan,
        caliber_tail_mm=float(np.nanmedian(cal[-k:])) if np.isfinite(cal[-k:]).any() else np.nan,
        frac_dilated=float((fin >= cut_p.dilate_thresh_mm).mean()) if fin.size else np.nan,
        len_evaluable=bool(res.total_len_mm >= END_MARGIN_MM + cut_p.persistence_mm + 1),
    )

    cut = detect_cutoff(cal, res.bridged, cut_p, END_MARGIN_MM)
    base.update(
        detected=cut.detected, score=cut.score, arc_mm=cut.arc_mm,
        cutoff_frac=cut.arc_mm / res.total_len_mm if cut.detected else np.nan,
        scale_mm=cut.scale_mm, up_caliber_mm=cut.up_caliber_mm, down_caliber_mm=cut.down_caliber_mm,
        ratio=cut.ratio, persistence_mm=cut.persistence_mm, n_candidates=cut.n_candidates,
        on_bridge=cut.on_bridge,
        dist_to_duct_end_mm=min(cut.arc_mm, res.total_len_mm - cut.arc_mm) if cut.detected else np.nan,
    )
    if cut.detected:
        idx = min(int(round(cut.arc_mm / STEP_MM)), len(res.path_xyz) - 1)
        x, y, z = iso_to_world_mm(res.path_xyz[[idx]], iso[f"{duct}_offset"], iso["spacing"],
                                  native_sp, iso["affine"])[0]
        base.update(cutoff_x_mm=x, cutoff_y_mm=y, cutoff_z_mm=z)

    if save_profiles:
        world = iso_to_world_mm(res.path_xyz, iso[f"{duct}_offset"], iso["spacing"], native_sp, iso["affine"])
        out = RESULTS_DIR / "profiles"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"{case_id}_{duct}.npz", caliber=cal, path_world_mm=world,
                            bridged=res.bridged, step_mm=STEP_MM)
    return duct_row(case_id, duct, tumor, **base)


def process_case(args: tuple) -> list[dict]:
    case_id, tumor, save_profiles = args
    try:
        iso = load_ducts_isotropic(case_id)
        if iso is None:
            return [duct_row(case_id, d, tumor, status="load_failed", detected=False) for d in DUCTS]

        head_c = tail_c = None
        if iso["mpd"].any() or iso["cbd"].any():
            regions = load_pancreas_regions(case_id, iso["native_shape"], iso["affine"])
            if regions is not None:
                head_c = mask_centroid_mm(regions["head"], iso["affine"])
                tail_c = mask_centroid_mm(regions["tail"], iso["affine"])
                if tail_c is None:
                    head_c = None

        rows = []
        for duct in DUCTS:
            try:
                rows.append(analyze_duct(case_id, duct, tumor, iso, head_c, tail_c, save_profiles))
            except Exception as e:  # one bad duct must not kill a ~1300-case run
                traceback.print_exc()
                rows.append(duct_row(case_id, duct, tumor, status=f"error:{type(e).__name__}", detected=False))
        return rows
    except Exception as e:
        traceback.print_exc()
        return [duct_row(case_id, d, tumor, status=f"error:{type(e).__name__}", detected=False) for d in DUCTS]


def per_case_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per case, with double-duct pairing: both ducts detected AND their
    cutoff points within DOUBLE_DUCT_DIST_MM (world mm)."""
    rows = []
    for case_id, g in df.groupby("case_id", sort=True):
        m = g[g.duct == "mpd"].iloc[0]
        c = g[g.duct == "cbd"].iloc[0]
        both = bool(m.detected) and bool(c.detected)
        dist = np.nan
        if both:
            dist = float(np.linalg.norm(
                m[["cutoff_x_mm", "cutoff_y_mm", "cutoff_z_mm"]].to_numpy(float)
                - c[["cutoff_x_mm", "cutoff_y_mm", "cutoff_z_mm"]].to_numpy(float)))
        rows.append(dict(
            case_id=case_id, tumor_present=m.tumor_present,
            mpd_status=m.status, cbd_status=c.status,
            mpd_detected=bool(m.detected), cbd_detected=bool(c.detected),
            any_detected=bool(m.detected) or bool(c.detected),
            double_duct=both and dist <= DOUBLE_DUCT_DIST_MM, coupling_dist_mm=dist,
        ))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, cases: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    print("\n" + "=" * 78 + "\nSUMMARY\n" + "=" * 78)
    print(f"{df.case_id.nunique()} cases, tumor+ = {int((cases.tumor_present == 1).sum())}, "
          f"tumor- = {int((cases.tumor_present == 0).sum())}")

    for duct in DUCTS:
        d = df[df.duct == duct]
        print(f"\n--- {duct.upper()} ---")
        print("status:", d.status.value_counts().to_dict())
        ok = d[d.total_len_mm.notna()]
        if len(ok):
            print(f"centerline length mm (n={len(ok)}): "
                  + " ".join(f"p{q}={ok.total_len_mm.quantile(q / 100):.0f}" for q in (10, 25, 50, 75, 90)))
            need = END_MARGIN_MM + CUTOFF_PARAMS[duct].persistence_mm + 1
            print(f"  usable-length table (of ALL {len(d)} cases in this set): "
                  + "  ".join(f">={t}mm:{(ok.total_len_mm >= t).sum() / len(d):.0%}" for t in (need, 30, 45, 60, 100)))
            print(f"  n_components median={ok.n_components.median():.0f}  "
                  f"unbridged>0: {(ok.n_unbridged_components > 0).mean():.0%}  "
                  f"bridged>0: {(ok.n_bridged_gaps > 0).mean():.0%}  "
                  f"dropped>0: {(ok.n_dropped_components > 0).mean():.0%}")
        o = d[d.status == "ok"]
        if len(o):
            print(f"caliber (oriented+ok, n={len(o)}): p50 median={o.caliber_p50_mm.median():.2f}  "
                  f"max median={o.caliber_max_mm.median():.2f}  head={o.caliber_head_mm.median():.2f}  "
                  f"tail={o.caliber_tail_mm.median():.2f}   (expect MPD head>tail)")
            print(f"orientation: flipped {o.flipped.mean():.0%}, margin median {o.orientation_margin_mm.median():.0f}mm")
            print("detection rate (denominator = all cases in class | = evaluable-length cases):")
            for t, name in ((0, "tumor-"), (1, "tumor+")):
                allc = d[d.tumor_present == t]
                ev = allc[(allc.status == "ok") & (allc.len_evaluable == True)]  # noqa: E712
                print(f"  {name}: {int(allc.detected.sum())}/{len(allc)} = {allc.detected.mean():.1%}   |   "
                      f"{int(ev.detected.sum())}/{len(ev)} = {ev.detected.mean() if len(ev) else float('nan'):.1%}")

    print("\n--- per case ---")
    for t, name in ((0, "tumor-"), (1, "tumor+")):
        c = cases[cases.tumor_present == t]
        print(f"{name} (n={len(c)}): any_detected={c.any_detected.mean():.1%}  "
              f"mpd={c.mpd_detected.mean():.1%}  cbd={c.cbd_detected.mean():.1%}  double_duct={c.double_duct.mean():.1%}")
    print("\nNull-test readout: tumor+ detection should be >=2-3x tumor-; if they match, the detector "
          "is measuring anatomy, not obstruction. Sensitivity ceiling = tumor+ any_detected above.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("neg", "pos", "all"), default="all")
    ap.add_argument("--limit", type=int, default=None, help="first N cases (after --set filter)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--profiles", action="store_true", help="save per-duct caliber profiles (.npz)")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    if QUALITY_CSV.exists():
        q = pd.read_csv(QUALITY_CSV)[["case_id", "tumor_present"]]
    else:
        print(f"{QUALITY_CSV} missing -- using fold-file ids, tumor labels unknown")
        q = pd.DataFrame({"case_id": case_ids(), "tumor_present": np.nan})
    if args.set != "all":
        q = q[q.tumor_present == (1 if args.set == "pos" else 0)]
    q = q.sort_values("case_id")
    if args.limit:
        q = q.head(args.limit)

    print(f"{len(q)} cases ({args.set}), {args.workers} workers")
    jobs = [(c, t, args.profiles) for c, t in zip(q.case_id, q.tumor_present)]
    rows: list[dict] = []
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_case, jobs, chunksize=2), 1):
            rows.extend(r)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} cases done", flush=True)

    df = pd.DataFrame(rows).sort_values(["case_id", "duct"]).reset_index(drop=True)
    cases = per_case_table(df)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "cutoff_per_duct.csv", index=False)
    cases.to_csv(args.out / "cutoff_per_case.csv", index=False)
    print(f"wrote {args.out / 'cutoff_per_duct.csv'} ({len(df)} rows), {args.out / 'cutoff_per_case.csv'} ({len(cases)} rows)")
    summarize(df, cases)


if __name__ == "__main__":
    main()
