"""S15 round 5, Part B rerun (PC): insertion and held-out top-cell checks with the v5 primary score.

Same held-out 49 kept glands (bank_split.csv + bank_pass1.csv), same insertion points (bank_profile.insertion_point,
rng(1500 + index), computed on the unmodified gland exactly as in Part B), same sphere sizes. Score = v5 primary
(raw-width multi-scale, extrapolated tip stops excluded); the residual score against the bank reference is secondary.
Also writes the held-out glands' top-3 peaks under the v5 primary for Part C's empirical chance.

Usage: PYTHONPATH=src .venv/bin/python -m glandshape.bank_round5   -> work/gland_shape/partB/{bank_round5.txt,
       insertion_v5.csv, heldout_top_cells_v5.csv, heldout_peaks_v5.json}
"""
from __future__ import annotations

import json
import multiprocessing as mp

import numpy as np
import pandas as pd

from glandshape import bank_profile as BP  # sets the bank CT root before tumorlib loads
from glandshape import footprint as FP
from glandshape import partc as PC
from glandshape import scores as S

HB_BAND = BP.HB_BAND


def one(args):
    cid, gi = args
    ref = BP.load_reference()
    out = {"case_id": cid, "rows": [], "tops": []}
    try:
        xyz, shape, g = BP.load_pred_coords(cid)
        aorta = BP._aorta(cid)
        base = BP.run_pipeline(xyz, shape, g, aorta)
        pr = base["pr"]
        sc = S.primary_scores(pr, ref)
        for name in ("primary", "residual"):
            m = sc[name]
            i, j = np.unravel_index(np.nanargmax(m), m.shape)
            band = (pr.u >= HB_BAND[0]) & (pr.u <= HB_BAND[1])
            out["tops"].append(dict(case_id=cid, score=name, u=float(pr.u[i]), axis=int(j), z=float(m[i, j]),
                                    band_share=float(np.isfinite(m[band]).sum() / np.isfinite(m).sum()),
                                    n_tip=int(pr.tip.sum())))
        out["peaks"] = [(float(pr.u[i]), int(j)) for i, j in PC.top_peaks(sc["primary"], pr.s)]
        pt, i0, k0 = BP.insertion_point(base, gi)
        out["ins_u"] = float(pr.u[i0])
        for diam in BP.SIZES_MM:
            ball = BP.ball_coords(pt, diam, shape, g)
            res = BP.run_pipeline(np.unique(np.vstack([xyz, ball]), axis=0), shape, g, aorta)
            row = dict(case_id=cid, kind=f"sphere{diam:.0f}", drop=res["drop"] or "", ins_u=out["ins_u"])
            if not res["drop"]:
                p2, cl2 = res["pr"], res["cl"]
                fp = FP.footprint(res["outline"], cl2, p2, ball)
                s2 = S.primary_scores(p2, ref)
                a, b = PC.within_scan_percentile(s2["primary"], fp), PC.within_scan_percentile(s2["residual"], fp)
                row.update(fp_cells=int(fp.sum()), pct_primary=a["percentile"], pct_residual=b["percentile"],
                           n_windows=a["n_windows"], top3_primary=PC.top3_hit(s2["primary"], p2.s, fp),
                           top3_residual=PC.top3_hit(s2["residual"], p2.s, fp),
                           chance_uniform=PC.chance_top3_uniform(fp, np.isfinite(s2["primary"]), p2.s))
            out["rows"].append(row)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main():
    split = pd.read_csv(BP.OUT / "bank_split.csv")
    q = pd.read_csv(BP.OUT / "bank_pass1.csv").merge(split, on="case_id")
    q["drop"] = q["drop"].fillna("")
    kept = q[q["drop"] == ""]
    ho = list(kept[kept.role == "heldout"].case_id)          # same order as bank_profile -> same insertion seeds
    assert len(ho) == 49, len(ho)
    with mp.Pool(4) as pool:
        res = pool.map(one, [(c, gi) for gi, c in enumerate(ho)], chunksize=1)
    errs = [(o["case_id"], o["error"]) for o in res if "error" in o]
    ins = pd.DataFrame([r for o in res for r in o["rows"]])
    tops = pd.DataFrame([r for o in res for r in o["tops"]])
    ins.to_csv(BP.OUT / "insertion_v5.csv", index=False)
    tops.to_csv(BP.OUT / "heldout_top_cells_v5.csv", index=False)
    (BP.OUT / "heldout_peaks_v5.json").write_text(json.dumps({o["case_id"]: o["peaks"] for o in res if "peaks" in o}))
    # same insertion points as Part B (regression)
    old = pd.read_csv(BP.OUT / "insertion.csv")
    L = [f"S15 round 5 Part B rerun: {len(ho)} held-out glands, errors {errs}",
         f"tip stops per gland median {tops.n_tip.median():.0f}"]
    for name, d in tops.groupby("score"):
        L.append(f"held-out top cell, {name} (n={len(d)}): u deciles "
                 f"{np.histogram(d.u, bins=np.linspace(0, 1, 11))[0].tolist()}; u < 0.1: {(d.u < 0.1).mean():.0%}; "
                 f"in head-body band {HB_BAND}: {((d.u >= HB_BAND[0]) & (d.u <= HB_BAND[1])).mean():.0%} vs band share "
                 f"{d.band_share.median():.0%}; axis counts {np.bincount(d.axis, minlength=12).tolist()}")
    for diam in BP.SIZES_MM:
        d = ins[(ins.kind == f"sphere{diam:.0f}") & (ins["drop"] == "")]
        o = old[old.kind == f"sphere{diam:.0f}"]
        L.append(f"sphere {diam:.0f} mm (n={len(d)}, with a percentile {int(d.pct_primary.notna().sum())}): primary "
                 f"(v5) percentile median {d.pct_primary.median():.1f} (IQR {d.pct_primary.quantile(.25):.1f}-"
                 f"{d.pct_primary.quantile(.75):.1f}); residual {d.pct_residual.median():.1f}; top-3 hit primary "
                 f"{d.top3_primary.mean():.0%} / residual {d.top3_residual.mean():.0%} vs uniform chance median "
                 f"{d.chance_uniform.median():.1%} | Part B (v4) primary {o.pct_primary.median():.1f}, raw {o.pct_raw.median():.1f}")
    d20 = ins[(ins.kind == "sphere20") & (ins["drop"] == "")]
    med = d20.pct_primary.median()
    L.append(f"INSERTION CHECK (20 mm median >= 95): {med:.1f} -> {'PASS' if med >= 95 else 'FAIL'}")
    txt = "\n".join(L)
    (BP.OUT / "bank_round5.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
