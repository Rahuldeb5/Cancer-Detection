"""Build results/master_lesion_table/master_lesions.csv: one row per GT lesion component
(1235), key (case_id, lesion_id), joining every per-lesion result the project has so far,
then run the oracles and the S1 audits.

Inputs
  work/master_lesion_table/{case_spacing,lesion_recount}.csv   case_pass.py (straight from masks)
  results/attenuation_results/attenuation_labels.csv            dHU, v2 class, MPD distance
  results/attenuation_results/lesion_dice_attenuation.csv       nnU-Net lesion Dice (server GT)
  results/nnunet_fold*/per_lesion_metrics.csv                   detected_any (no lesion_id: cross-check only)
  results/nnunet_fold*/per_case_metrics.csv                     case fold membership
  results/lesion_data/lesion_duct_position/lesion_duct_position.csv   published region (1142)
  results/lesion_data/lesion_location_results/excluded_lesions.csv    the 70 separated lesions
  src/lesion-sectioning/work/lesion_inclusion.csv               tier for all 1235
  src/lesion-sectioning/work/lesion_pca_position.csv            u = PC1 position (1205)
  src/lesion-sectioning/work/section_boundaries.csv             PC1 fallback for region
  results/PanTS_metadata_new.csv                                phase, tumor?, reports

Every join is on (case_id, lesion_id) and is followed by a gt_vox equality assert.
Audit tables go to work/master_lesion_table/audit_*.csv; numbers print to stdout.

Usage (from repo root, after case_pass.py): .venv/bin/python3 src/master-lesion-table/build_table.py
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path("work/master_lesion_table")
OUT_DIR = Path("results/master_lesion_table")
LS_WORK = Path("src/lesion-sectioning/work")
K = ["case_id", "lesion_id"]

TINY_VOL_MM3 = 8.0  # e.g. 2x2x2 mm; flagged, never dropped
DIAM_EDGES = [0, 5, 10, 20, 40, np.inf]
DIAM_LABELS = ["<5", "5-10", "10-20", "20-40", ">=40"]
TEST_ID_MIN = 9001  # PanTS official test split = PanTS_00009001..00009901 (ImageTe tarball)


def ids_from(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def check_join(df: pd.DataFrame, col: str, name: str) -> None:
    """gt_vox equality after a join (rows the source does not cover are skipped, and counted)."""
    have = df[col].notna()
    bad = df[have & (df[col] != df["gt_vox"])]
    assert bad.empty, f"gt_vox mismatch vs {name}:\n{bad[K + ['gt_vox', col]].head()}"
    print(f"  gt_vox == {name:<34} {int(have.sum()):>4} rows checked, 0 mismatches")


def load_sources() -> pd.DataFrame:
    rec = pd.read_csv(WORK / "lesion_recount.csv")
    cases = pd.read_csv(WORK / "case_spacing.csv")
    df = rec.merge(cases, on="case_id", how="left", validate="m:1")
    df["vol_mm3"] = df.gt_vox * df.spacing_x_mm * df.spacing_y_mm * df.spacing_z_mm

    at = pd.read_csv("results/attenuation_results/attenuation_labels.csv")
    ld = pd.read_csv("results/attenuation_results/lesion_dice_attenuation.csv").rename(columns={"case": "case_id"})
    inc = pd.read_csv(LS_WORK / "lesion_inclusion.csv")
    ex = pd.read_csv("results/lesion_data/lesion_location_results/excluded_lesions.csv")
    dp = pd.read_csv("results/lesion_data/lesion_duct_position/lesion_duct_position.csv")
    dpw = pd.read_csv(LS_WORK / "lesion_duct_position.csv")  # unplotted version, still has n_vox
    pca = pd.read_csv(LS_WORK / "lesion_pca_position.csv")

    for name, src in [("attenuation_labels", at), ("lesion_dice_attenuation", ld), ("lesion_inclusion", inc),
                      ("excluded_lesions", ex), ("lesion_duct_position", dp), ("lesion_pca_position", pca)]:
        assert not src.duplicated(K).any(), f"duplicate key in {name}"

    df = df.merge(at.rename(columns={"n_vox": "gv_atten", "vol_mm3": "vol_mm3_atten",
                                     "n_components": "nc_atten"}), on=K, how="left", validate="1:1")
    df = df.merge(ld[K + ["fold", "gt_vox", "diam_mm", "lesion_dice"]].rename(
        columns={"fold": "fold_dice", "gt_vox": "gv_dice", "diam_mm": "diam_mm_dice"}), on=K, how="left", validate="1:1")
    df = df.merge(inc[K + ["n_vox", "fold", "tier", "outside_separated", "outside_lt50", "location", "diam_mm"]].rename(
        columns={"n_vox": "gv_incl", "fold": "fold_incl", "diam_mm": "diam_mm_incl"}), on=K, how="left", validate="1:1")
    df = df.merge(ex[K + ["n_vox", "fold"]].rename(columns={"n_vox": "gv_excl", "fold": "fold_excl"}),
                  on=K, how="left", validate="1:1")
    df = df.merge(dpw[K + ["n_vox"]].rename(columns={"n_vox": "gv_dpw"}), on=K, how="left", validate="1:1")
    df = df.merge(dp[K + ["section", "section_source", "pc1_norm"]].rename(columns={"pc1_norm": "pc1_dp"}),
                  on=K, how="left", validate="1:1")
    df = df.merge(pca[K + ["pc1_norm"]], on=K, how="left", validate="1:1")
    return df, cases


def oracles(df: pd.DataFrame, cases: pd.DataFrame) -> None:
    print("\n=== ORACLES ===")
    folds = {k: ids_from(Path(f"src/data/fold_{k}_ids.txt")) for k in range(1, 6)}
    all_ids = [i for v in folds.values() for i in v]
    assert len(all_ids) == len(set(all_ids)) == 1308, "a case sits in more than one fold file"
    print(f"  1308 cases, each in exactly one fold_k_ids.txt")

    assert len(df) == 1235 and not df.duplicated(K).any()
    print(f"  {len(df)} rows, (case_id, lesion_id) unique")

    for col, name in [("gv_atten", "attenuation_labels.n_vox"), ("gv_dice", "lesion_dice_attenuation.gt_vox"),
                      ("gv_incl", "lesion_inclusion.n_vox"), ("gv_excl", "excluded_lesions.n_vox"),
                      ("gv_dpw", "work/lesion_duct_position.n_vox")]:
        check_join(df, col, name)
    for col in ("gv_atten", "gv_dice", "gv_incl"):
        assert df[col].notna().all(), f"{col} does not cover all 1235"

    # per_lesion_metrics.csv has no lesion_id: compare the per-case multiset of gt_vox instead
    pl = pd.concat([pd.read_csv(f).assign(fold=int(re.search(r"fold(\d)", f)[1]))
                    for f in sorted(glob.glob("results/nnunet_fold*/per_lesion_metrics.csv"))])
    a = df.groupby("case_id")["gt_vox"].apply(sorted)
    b = pl.groupby("case")["gt_vox"].apply(sorted)
    assert a.index.equals(b.index.rename("case_id")) and (a == b.reindex(a.index).values).all()
    print(f"  gt_vox == per_lesion_metrics (per-case multiset, {len(pl)} rows, no lesion_id to key on)")

    # detected_any: == (lesion_dice > 0) by construction (Dice uses the predicted components that
    # touch the lesion). Check against evaluate.py's own flag per case: multiset of (gt_vox, detected)
    mine = df.assign(d=(df.lesion_dice > 0).astype(int)).groupby("case_id").apply(
        lambda g: sorted(zip(g.gt_vox, g.d)), include_groups=False)
    theirs = pl.groupby("case").apply(lambda g: sorted(zip(g.gt_vox, g.detected_any)), include_groups=False)
    assert (mine == theirs.reindex(mine.index).values).all()
    print("  detected_any (lesion_dice>0) == per_lesion_metrics.detected_any for every case")

    # fold: fold file mapping == every source that stores a fold
    for col in ("fold_dice", "fold_incl", "fold_excl"):
        have = df[col].notna()
        assert (df.loc[have, col] == df.loc[have, "fold"]).all(), col
    pc = pd.concat([pd.read_csv(f).assign(fold=int(re.search(r"fold(\d)", f)[1]))
                    for f in sorted(glob.glob("results/nnunet_fold*/per_case_metrics.csv"))])
    assert pc.case.is_unique and (pc.set_index("case").fold == cases.set_index("case_id").fold.reindex(pc.case)).all()
    print("  fold (fold_k -> k-1) == lesion_dice_attenuation, lesion_inclusion, excluded_lesions, per_case_metrics")

    assert (df.n_components == df.nc_atten).all()
    meta = pd.read_csv("results/PanTS_metadata_new.csv").set_index("PanTS ID")
    assert (cases.set_index("case_id").mask_nonempty == (meta.loc[cases.case_id, "tumor?"] == 1).values).all()
    print("  tumor? == (lesion mask non-empty) for 1308/1308")

    dv = (df.vol_mm3 - df.vol_mm3_atten).abs().max()
    dd = (df.diam_mm - df.diam_mm_dice).abs()
    print(f"  vol_mm3 (affine voxel_sizes) vs attenuation_labels vol (header zooms): max |diff| {dv:.2e} mm3")
    print(f"  diam_mm recount vs lesion_dice_attenuation: max |diff| {dd.max():.2e} mm "
          f"({int((dd > 1e-3).sum())} rows > 1e-3)")
    print(f"  cases with header zooms != affine voxel sizes: {int((~cases.header_zooms_match).sum())}; "
          f"lesion affine != CT affine: {int((~cases.lesion_affine_matches_ct).sum())}")


def build(df: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    meta = pd.read_csv("results/PanTS_metadata_new.csv").set_index("PanTS ID")
    df["tiny_flag"] = df.vol_mm3 < TINY_VOL_MM3
    df["diam_bin"] = pd.cut(df.diam_mm, DIAM_EDGES, right=False, labels=DIAM_LABELS)
    df["n_lesions_in_case"] = df.groupby("case_id").gt_vox.transform("size")
    # largest = most voxels; one exact tie (PanTS_00009056, 732 vs 732 vox) is broken by Feret diameter
    top = df.sort_values(["gt_vox", "diam_mm"], ascending=False).drop_duplicates("case_id").set_index(K).index
    df["largest_in_case"] = df.set_index(K).index.isin(top)
    assert df.groupby("case_id").largest_in_case.sum().eq(1).all()
    df["ct_phase"] = meta.loc[df.case_id, "ct phase"].values
    df["official_split"] = np.where(df.case_id.str[-8:].astype(int) >= TEST_ID_MIN, "test", "train")
    df["detected_any"] = df.lesion_dice > 0
    df["excluded"] = df.outside_separated.astype(bool)
    assert df.excluded.sum() == 70 and df.gv_excl.notna().sum() == 70 and (df.excluded == df.gv_excl.notna()).all()

    # region: ring vote from the mask recount (duct_position.py rule); where the ring touches
    # no section, fall back to PC1 position vs the pooled median section boundaries
    # (plot_duct_position.py rule, which produced the published section column)
    b = pd.read_csv(LS_WORK / "section_boundaries.csv")
    hb, bt = b.boundary_head_body.median(), b.boundary_body_tail.median()
    df["u"] = df.pc1_norm
    un = df.region == "unassigned"
    fb = np.select([df.u < hb, df.u < bt], ["head", "body"], "tail")
    df["region_source"] = np.where(un, np.where(df.u.notna(), "pc1_boundary", "none"), "mask_ring")
    df.loc[un & df.u.notna(), "region"] = fb[un & df.u.notna()]
    df.loc[df.region_source == "none", "region"] = np.nan

    have = df.section.notna()
    agree = (df.loc[have, "region"] == df.loc[have, "section"]).mean()
    src_agree = (df.loc[have, "region_source"].str.replace("mask_ring", "mask") == df.loc[have, "section_source"]).mean()
    u_diff = (df.u - df.pc1_dp).abs().max()
    print(f"\n  region == published lesion_duct_position.section on {have.sum()} rows: {agree:.4f}; "
          f"source agrees: {src_agree:.4f}; |u - published pc1_norm| max {u_diff:.1e} (3-dp rounding)")
    assert agree == 1.0

    cols = ["case_id", "lesion_id", "fold", "official_split", "ct_phase",
            "spacing_x_mm", "spacing_y_mm", "spacing_z_mm", "slice_axis", "slice_thickness_mm",
            "gt_vox", "vol_mm3", "diam_mm", "diam_bin", "tiny_flag",
            "n_lesions_in_case", "largest_in_case",
            "delta_hu", "attenuation", "lesion_dice", "detected_any",
            "region", "region_source", "u", "tier", "excluded", "mpd_present", "dist_to_mpd_mm"]
    out = df[cols].rename(columns={"delta_hu": "dHU", "attenuation": "attenuation_v2"})
    return out.sort_values(K).reset_index(drop=True)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def audits(t: pd.DataFrame, cases: pd.DataFrame) -> None:
    meta = pd.read_csv("results/PanTS_metadata_new.csv")
    meta["id_num"] = meta["PanTS ID"].str[-8:].astype(int)
    cohort = set(cases.case_id)
    mc = meta[meta["PanTS ID"].isin(cohort)]

    print("\n=== AUDIT a: cohort counts ===")
    pos, neg = ids_from(Path("src/data/train_pos_ids.txt")), ids_from(Path("src/data/train_neg_ids.txt"))
    raw_lines = {f: Path(f"src/data/{f}").read_text().count("\n") for f in ("train_pos_ids.txt", "train_neg_ids.txt")}
    rows = [
        ("fold_{1..5}_ids.txt union", len(cohort), "", ""),
        ("metadata tumor? over the 1308", "", int((mc["tumor?"] == 1).sum()), int((mc["tumor?"] == 0).sum())),
        ("per_case_metrics gt_tumor", "", *[int(v) for v in pd.concat(
            [pd.read_csv(f) for f in glob.glob("results/nnunet_fold*/per_case_metrics.csv")]).gt_tumor.value_counts().sort_index(ascending=False)]),
        ("train_pos/neg_ids.txt, parsed", "", len(pos), len(neg)),
        ("train_pos/neg_ids.txt, `wc -l` (no trailing newline)", "", raw_lines["train_pos_ids.txt"], raw_lines["train_neg_ids.txt"]),
        ("metadata tumor? over PanTS IDs 1-9000 (official train)", 9000,
         int((meta[meta.id_num <= 9000]["tumor?"] == 1).sum()), int((meta[meta.id_num <= 9000]["tumor?"] == 0).sum())),
        ("metadata tumor? over PanTS IDs 9001-9901 (official test)", 901,
         int((meta[meta.id_num > 9000]["tumor?"] == 1).sum()), int((meta[meta.id_num > 9000]["tumor?"] == 0).sum())),
        ("cohort cases with ID > 9000 (official TEST split)", int((mc.id_num > 9000).sum()),
         int(((mc.id_num > 9000) & (mc["tumor?"] == 1)).sum()), int(((mc.id_num > 9000) & (mc["tumor?"] == 0)).sum())),
        ("testing/PanTS/old/pants_utils.py TRAIN_NEG_COUNT", "", "", 309),
    ]
    a = pd.DataFrame(rows, columns=["source", "n_cases", "positives", "negatives"])
    print(a.to_string(index=False)); a.to_csv(WORK / "audit_a_cohort_counts.csv", index=False)

    print("\n=== AUDIT b: slice thickness over the 1308 (mm, along the slice axis) ===")
    st = cases.slice_thickness_mm
    q = st.quantile([0, .05, .10, .25, .50, .75, .90, .95, 1.0])
    print(q.round(3).to_string())
    print(f"  mean {st.mean():.3f}; >=2.5 mm: {int((st >= 2.5).sum())} ({pct((st >= 2.5).mean())}); "
          f">=5 mm: {int((st >= 5).sum())} ({pct((st >= 5).mean())})")
    print(f"  slice axis position: {cases.slice_axis.value_counts().sort_index().to_dict()}")
    print(f"  spacing_z_mm (last voxel axis) median {cases.spacing_z_mm.median():.3f} vs slice-axis median {st.median():.3f}")
    ms = mc.set_index("PanTS ID").loc[cases.case_id, "spacing"].str.strip("()").str.split(",", expand=True).astype(float)
    last = ms.iloc[:, 2].values
    print(f"  metadata 'spacing' 3rd entry median {np.median(last):.3f}; "
          f"== affine voxel sizes (as a set) for {int(np.all(np.isclose(np.sort(ms.values, 1), np.sort(cases[['spacing_x_mm','spacing_y_mm','spacing_z_mm']].values, 1), atol=1e-3), 1).sum())}/1308")
    by = cases.assign(pos=cases.mask_nonempty).groupby("pos").slice_thickness_mm.median()
    print(f"  median by status: tumor- {by[False]:.3f}, tumor+ {by[True]:.3f}")
    les_st = t.slice_thickness_mm
    print(f"  per LESION (n=1235): median {les_st.median():.3f}; >=2.5 {pct((les_st >= 2.5).mean())}; >=5 {pct((les_st >= 5).mean())}")
    st_tab = pd.DataFrame({"stat": list(q.index.map(lambda v: f"p{int(v * 100)}")) + ["mean", "n>=2.5", "n>=5"],
                           "value": list(q.values) + [st.mean(), (st >= 2.5).sum(), (st >= 5).sum()]})
    st_tab.to_csv(WORK / "audit_b_slice_thickness.csv", index=False)
    print("  value counts:", st.round(3).value_counts().head(10).to_dict())

    print("\n=== AUDIT e: size distribution with / without tiny lesions ===")
    for name, s in [("all 1235", t), ("tiny_flag=False", t[~t.tiny_flag]), ("tiny only", t[t.tiny_flag])]:
        d = s.diam_mm
        print(f"  {name:<16} n={len(s):>4}  diam p5={d.quantile(.05):.2f}  p10={d.quantile(.10):.2f}  "
              f"median={d.median():.2f}  bins={s.diam_bin.value_counts().reindex(DIAM_LABELS).to_dict()}")
    tiny = t[t.tiny_flag]
    print(f"  tiny: vol_mm3 max {tiny.vol_mm3.max():.2f}, gt_vox range {tiny.gt_vox.min()}-{tiny.gt_vox.max()}, "
          f"diam max {tiny.diam_mm.max():.2f}, largest_in_case {int(tiny.largest_in_case.sum())}, "
          f"in {tiny.case_id.nunique()} cases, excluded {int(tiny.excluded.sum())}, detected {int(tiny.detected_any.sum())}")
    print(f"  tiny tier: {tiny.tier.value_counts().to_dict()}")

    print("\n=== AUDIT f: lesions < 10 mm, isolated (largest in case) vs secondary component ===")
    s = t[t.diam_mm < 10]
    f = s.groupby(["diam_bin", "largest_in_case"], observed=True).agg(
        n=("lesion_id", "size"), tiny=("tiny_flag", "sum"), detected=("detected_any", "sum"),
        excluded=("excluded", "sum"), mean_dice=("lesion_dice", "mean")).reset_index()
    print(f.to_string(index=False)); f.to_csv(WORK / "audit_f_small_isolated_vs_secondary.csv", index=False)
    iso = s[s.largest_in_case]
    print(f"  <10 mm total {len(s)}: largest-in-case {len(iso)} (sole lesion in case: {int((iso.n_lesions_in_case == 1).sum())}), "
          f"secondary {int((~s.largest_in_case).sum())}")
    nt = s[~s.tiny_flag]
    print(f"  <10 mm without tiny: {len(nt)}; largest-in-case {int(nt.largest_in_case.sum())}")
    print(f"  cases whose LARGEST lesion is <10 mm: {t[t.largest_in_case & (t.diam_mm < 10)].case_id.nunique()}")
    print(f"  10-20 mm: {int((t.diam_bin == '10-20').sum())}, largest-in-case {int(((t.diam_bin == '10-20') & t.largest_in_case).sum())}")


REPORT_LESION = re.compile(
    r"Pancreas lesion (\d+):\s*\n(?:Location: pancreas ([a-z/]+)\.\s*\n)?Size: ([\d.]+) x ([\d.]+) cm.*?"
    r"Volume: ([\d.]+) cc\.\s*\nEnhancement relative to pancreas: (\w+)", re.S)
DIAGNOSIS_TERMS = (r"adenocarcinoma|pdac|carcinoma|cyst|ipmn|neuroendocrine|pnet|metasta|pancreatitis|"
                   r"malignan|benign|mucinous|serous|patholog|biopsy|diagnos|lymphoma")


def audit_reports(cases: pd.DataFrame) -> None:
    meta = pd.read_csv("results/PanTS_metadata_new.csv")
    print("\n=== AUDIT c: metadata columns / diagnosis field ===")
    print("  columns:", list(meta.columns))
    rep = meta["structured report"].fillna("")
    lines = rep.str.split("\n").explode().str.strip().str.replace(r"-?[\d.]+", "#", regex=True)
    panc = lines[lines.str.contains("ancrea|Location|Size:|Enhancement", regex=True)]
    print(f"  distinct pancreas-related report line templates (numbers masked): {panc.nunique()}")
    hits = rep.str.contains(DIAGNOSIS_TERMS, case=False, regex=True)
    print(f"  reports containing any diagnosis term ({DIAGNOSIS_TERMS}): {int(hits.sum())} / {len(rep)}")
    print(f"  cases with no report: {meta.loc[meta['structured report'].isna(), 'PanTS ID'].tolist()}")

    print("\n=== AUDIT d: report 'Pancreas lesions' section vs tumor? ===")
    meta["report_lesion_section"] = rep.str.contains(r"^Pancreas lesions:", flags=re.M, regex=True)
    coh = meta[meta["PanTS ID"].isin(set(cases.case_id))]
    for name, d in [("cohort 1308", coh), ("all PanTS 9901", meta)]:
        print(f"  {name}:\n" + pd.crosstab(d["tumor?"], d.report_lesion_section).to_string())
    rows = []
    for _, r in coh[(coh["tumor?"] == 0) & coh.report_lesion_section].iterrows():
        text = r["structured report"]
        in_imp = bool(re.search(r"pancrea\w* mass", text.split("IMPRESSION:")[-1]))
        for g in REPORT_LESION.findall(text):
            rows.append({"case_id": r["PanTS ID"], "report_lesion": int(g[0]), "location": g[1] or "(none given)",
                         "long_cm": float(g[2]), "short_cm": float(g[3]), "vol_cc": float(g[4]),
                         "enhancement": g[5], "ct_phase": r["ct phase"], "in_impression": in_imp})
    d = pd.DataFrame(rows).sort_values("long_cm", ascending=False)
    d.to_csv(WORK / "audit_d_negatives_report_lesions.csv", index=False)
    print(f"  negatives with a report lesion: {d.case_id.nunique()} cases, {len(d)} lesions; "
          f"long axis median {d.long_cm.median():.1f} cm, >=1 cm: {int((d.long_cm >= 1).sum())}, "
          f">=2 cm: {int((d.long_cm >= 2).sum())}; named in IMPRESSION: {d[d.in_impression].case_id.nunique()} cases")
    print(d.to_string(index=False))


def main() -> None:
    df, cases = load_sources()
    oracles(df, cases)
    t = build(df, cases)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT_DIR / "master_lesions.csv", index=False, float_format="%.6g")
    print(f"\nwrote {OUT_DIR / 'master_lesions.csv'}  shape {t.shape}")
    print("  missing per column:", {c: int(v) for c, v in t.isna().sum().items() if v})
    print("  region:", t.region.value_counts(dropna=False).to_dict(), " source:", t.region_source.value_counts().to_dict())
    print("  tier:", t.tier.value_counts().to_dict(), " phase:", t.ct_phase.value_counts(dropna=False).to_dict())
    print("  attenuation_v2:", t.attenuation_v2.value_counts().to_dict(), " split:", t.official_split.value_counts().to_dict())
    audits(t, cases)
    audit_reports(cases)


if __name__ == "__main__":
    main()
