"""S3 step 3: how the iso fraction moves from v2 to v3, and what it depends on.

Joins work/attenuation_v3/measure_v3.csv onto the master lesion table (gt_vox asserted equal),
writes the deliverable CSV results/attenuation_v3/attenuation_labels_v3.csv and the analysis
tables to work/attenuation_v3/analysis.md (numbers for the README come from there).

    .venv/bin/python src/attenuation-v3/analyze_v3.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import attn_v3 as av  # noqa: E402

MASTER = REPO / "results" / "master_lesion_table" / "master_lesions.csv"
MEAS = REPO / "work" / "attenuation_v3" / "measure_v3.csv"
OUT_CSV = REPO / "results" / "attenuation_v3" / "attenuation_labels_v3.csv"
OUT_MD = REPO / "work" / "attenuation_v3" / "analysis.md"

V2_SHORT = {"hypoattenuating": "hypo", "isoattenuating": "iso", "hyperattenuating": "hyper", "unknown": "unknown"}
# dHU column -> short name, primary first
VARIANTS = {
    "dHU_v2": "v2",
    "dHU_whole_global": "whole|global",
    "dHU_core_global": "core|global (v3)",
    "dHU_coretrim_global": "coretrim|global",
    "dHU_core_local": "core|local",
    "dHU_coretrim_local": "coretrim|local",
}
CLASSES = ["hypo", "iso", "hyper", "unknown"]


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([str(df.index.name or "")] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                 else ("" if pd.isna(v) else str(v)) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def iso_table(d: pd.DataFrame, by: str, cols=VARIANTS, cutoff=av.CUTOFF_HU) -> pd.DataFrame:
    """Rows = levels of `by`; columns = n and iso fraction per dHU variant (unknown excluded
    from the denominator of each variant)."""
    out = {}
    for lvl, g in d.groupby(by, observed=True, dropna=False):
        row = {"n": len(g)}
        for c, name in cols.items():
            v = g[c].dropna()
            row[name] = float((v.abs() <= cutoff).mean()) if len(v) else np.nan
        out[lvl] = row
    t = pd.DataFrame(out).T
    t.index.name = by
    t["n"] = t["n"].astype(int)
    return t


def main():
    m = pd.read_csv(MASTER)
    v3 = pd.read_csv(MEAS)
    d = m.merge(v3, on=["case_id", "lesion_id"], how="left", suffixes=("", "_v3"), validate="1:1")
    missing = d.gt_vox_v3.isna().sum()
    assert missing == 0, f"{missing} master lesions have no v3 row"
    assert (d.gt_vox == d.gt_vox_v3).all(), "gt_vox mismatch after join"
    d = d.drop(columns="gt_vox_v3").rename(columns={"dHU": "dHU_v2"})
    d["v2"] = d.attenuation_v2.map(V2_SHORT)
    d["v3"] = av.classify(d.dHU_core_global.to_numpy())
    d["v3_local"] = av.classify(d.dHU_core_local.to_numpy())
    d["v3_trim"] = av.classify(d.dHU_coretrim_global.to_numpy())
    # v2 recomputed from dHU must equal the stored v2 label (same +-10, |dHU|=10 -> iso rule)
    assert (av.classify(d.dHU_v2.to_numpy()) == d.v2).all()

    d["thick_bin"] = pd.cut(d.slice_thickness_mm, [0, 2.0, 4.0, 99], labels=["<=2 mm", "2-4 mm", ">4 mm"])
    d["size_bin"] = pd.Categorical(d.diam_bin, ["<5", "5-10", "10-20", "20-40", ">=40"], ordered=True)
    d["phase"] = d.ct_phase.fillna("missing")

    # ---------------------------------------------------------- deliverable CSV
    keep = ["case_id", "lesion_id", "gt_vox",
            "n_core", "core_fallback", "core_min_depth_mm", "max_depth_mm",
            "tumor_whole_median", "tumor_core_median", "tumor_core_trim",
            "ref_global_median", "ref_global_trim", "n_ref_global",
            "ref_local_median", "ref_local_trim", "n_ref_local",
            "dHU_v2", "dHU_whole_global", "dHU_core_global", "dHU_coretrim_global",
            "dHU_core_local", "dHU_coretrim_local", "sigma_hu", "n_sigma", "cnr",
            "attenuation_v2", "attenuation_v3", "attenuation_v3_local", "attenuation_v3_trim"]
    out = d.assign(attenuation_v2=d.v2, attenuation_v3=d.v3,
                   attenuation_v3_local=d.v3_local, attenuation_v3_trim=d.v3_trim)[keep]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.4g")

    # ---------------------------------------------------------- analysis tables
    S = []
    S.append(f"# S3 analysis ({len(d)} lesions)\n")
    S.append("## Class counts per variant (+-10 HU)\n")
    cnt = pd.DataFrame({name: av.classify(d[c].to_numpy()) for c, name in VARIANTS.items()})
    cnt = cnt.apply(lambda s: s.value_counts()).reindex(CLASSES).fillna(0).astype(int)
    cnt.index.name = "class"
    S.append(md(cnt) + "\n")

    for label, sub in [("all", d), ("excluding separated", d[~d.excluded])]:
        S.append(f"## Transition v2 -> v3 ({label}, n={len(sub)})\n")
        ct = pd.crosstab(sub.v2, sub.v3).reindex(index=CLASSES, columns=CLASSES).fillna(0).astype(int)
        ct.index.name = "v2 \\ v3"
        S.append(md(ct) + "\n")
    ch = d[d.v2 != d.v3]
    S.append(f"Changed class: {len(ch)} / {len(d)}. Direction counts:\n")
    S.append(md(ch.groupby(["v2", "v3"]).size().rename("n").to_frame()) + "\n")

    S.append("## Decomposition of the shift (median of per-lesion differences, HU)\n")
    dd = pd.DataFrame({
        "reference change (whole|global - v2)": d.dHU_whole_global - d.dHU_v2,
        "tumor core (core|global - whole|global)": d.dHU_core_global - d.dHU_whole_global,
        "local vs global ref (core|local - core|global)": d.dHU_core_local - d.dHU_core_global,
    })
    dd = dd.groupby(d.size_bin, observed=True).median()
    S.append(md(dd, "{:.1f}") + "\n")

    for lab, sub in [("all", d), ("excluding separated", d[~d.excluded])]:
        S.append(f"## Iso fraction by contrast phase ({lab})\n")
        S.append(md(iso_table(sub, "phase")) + "\n")
    S.append("## Iso fraction by slice thickness\n")
    S.append(md(iso_table(d, "thick_bin")) + "\n")
    S.append("## Iso fraction by size (Feret)\n")
    S.append(md(iso_table(d, "size_bin")) + "\n")
    S.append("## Venous-phase only, by thickness and by size\n")
    ven = d[d.phase == "Venous"]
    S.append(md(iso_table(ven, "thick_bin")) + "\n")
    S.append(md(iso_table(ven, "size_bin")) + "\n")

    S.append("## Cutoff sensitivity: iso fraction at +-5 / +-10 / +-15 HU\n")
    cohorts = {
        "all": d,
        "venous": ven,
        "venous >=10 mm": ven[ven.diam_mm >= 10],
        "venous >=10 mm, not separated": ven[(ven.diam_mm >= 10) & ~ven.excluded],
        "arterial >=10 mm": d[(d.phase == "Arterial") & (d.diam_mm >= 10)],
        "non-contrast >=10 mm": d[(d.phase == "Non-contrast") & (d.diam_mm >= 10)],
    }
    rows = {}
    for cname, g in cohorts.items():
        for cut in (5, 10, 15):
            r = {"n": len(g)}
            for c, name in VARIANTS.items():
                v = g[c].dropna()
                r[name] = float((v.abs() <= cut).mean())
            rows[f"{cname} @ {cut}"] = r
    t = pd.DataFrame(rows).T
    t.index.name = "cohort @ cutoff"
    t["n"] = t["n"].astype(int)
    S.append(md(t) + "\n")

    S.append("## Headline: venous-phase lesions >= 10 mm, +-10 HU, 95% Wilson CI\n")
    hl = {}
    for cname in ("venous >=10 mm", "venous >=10 mm, not separated"):
        g = cohorts[cname]
        for c, name in VARIANTS.items():
            v = g[c].dropna()
            k = int((v.abs() <= 10).sum())
            lo, hi = wilson(k, len(v))
            hl[(cname, name)] = {"n": len(v), "iso": k, "frac": k / len(v), "ci_lo": lo, "ci_hi": hi}
    h = pd.DataFrame(hl).T
    h.index = [f"{a} / {b}" for a, b in h.index]
    h.index.name = "cohort / variant"
    h[["n", "iso"]] = h[["n", "iso"]].astype(int)
    S.append(md(h) + "\n")

    S.append("## Noise and CNR\n")
    S.append(md(d.groupby("thick_bin", observed=True).sigma_hu.describe()[["count", "25%", "50%", "75%"]], "{:.1f}") + "\n")
    S.append(md(d.groupby("v3").cnr.describe()[["count", "25%", "50%", "75%"]], "{:.2f}") + "\n")
    se = 1.2533 * d.sigma_hu / np.sqrt(d.n_core)  # white-noise SE of a median; optimistic for correlated noise
    iso3 = d.v3 == "iso"
    S.append(f"v3-iso lesions whose |dHU| is within 2 x (white-noise SE of the core median) of 0: "
             f"{int((iso3 & (d.dHU_core_global.abs() <= 2 * se)).sum())} / {int(iso3.sum())}\n")
    S.append(f"v3-iso lesions with |dHU| <= 5 HU: {int((iso3 & (d.dHU_core_global.abs() <= 5)).sum())} / {int(iso3.sum())}\n")

    S.append("## Core / reference QC\n")
    S.append(f"core fallback (fewer than {av.CORE_MIN_VOX} voxels at depth >= {av.CORE_DEPTH_MM} mm): "
             f"{int(d.core_fallback.sum())} / {len(d)}\n")
    S.append(md(d.groupby("size_bin", observed=True).agg(
        n=("n_core", "size"), fallback=("core_fallback", "mean"), n_core_median=("n_core", "median"),
        n_ref_local_median=("n_ref_local", "median"), local_nan=("dHU_core_local", lambda s: int(s.isna().sum()))),
        "{:.2f}") + "\n")
    S.append(f"global reference NaN: {int(d.dHU_core_global.isna().sum())}; "
             f"local reference NaN: {int(d.dHU_core_local.isna().sum())} "
             f"(of which separated: {int((d.dHU_core_local.isna() & d.excluded).sum())})\n")
    S.append(f"missing duct masks: {int((d.missing_ducts.fillna('') != '').sum()) if 'missing_ducts' in d else 'n/a'}\n")

    OUT_MD.write_text("\n".join(S))
    print("\n".join(S))
    print(f"wrote {OUT_CSV} and {OUT_MD}")


if __name__ == "__main__":
    main()
