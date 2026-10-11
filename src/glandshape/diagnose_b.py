"""S15 Part B diagnostic (read-only; changes no definition): why the 20 mm insertion percentile is low.

For each held-out kept bank gland: rerun the 20 mm insertion exactly as bank_profile.pass2, then list the null
windows (partc definition) that score >= the inserted sphere, and where they lie in u (end caps u < 0.1 or > 0.9).
Also reported, as a counterfactual only: the percentile if null windows touching u < 0.1 or u > 0.9 were not counted.

Usage: PYTHONPATH=src .venv/bin/python -m glandshape.diagnose_b   -> work/gland_shape/partB/diagnose_b.txt
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pandas as pd

from glandshape import bank_profile as BP  # sets the bank CT root before tumorlib loads
from glandshape import footprint as FP
from glandshape import partc as PC
from glandshape import scores as S

CAP_U = 0.1


def one(args):
    cid, gi = args
    ref = BP.load_reference()
    xyz, shape, g = BP.load_pred_coords(cid)
    aorta = BP._aorta(cid)
    base = BP.run_pipeline(xyz, shape, g, aorta)
    pt, i0, _ = BP.insertion_point(base, gi)
    ball = BP.ball_coords(pt, 20.0, shape, g)
    res = BP.run_pipeline(np.unique(np.vstack([xyz, ball]), axis=0), shape, g, aorta)
    pr = res["pr"]
    fp = FP.footprint(res["outline"], res["cl"], pr, ball)
    out = dict(case_id=cid, ins_u=float(base["pr"].u[i0]))
    for name, sc in (("primary", S.bulge_residual(pr.w, pr.u, ref)["primary"]),
                     ("raw", S.bulge_multiscale(pr.w)["primary"])):
        st = np.nonzero(fp.any(1))[0]
        h = int(st.max() - st.min() + 1)
        _, wa = PC.axis_arc(np.nonzero(fp.any(0))[0])
        m, o = PC._sliding_any_max(sc, fp, h, wa)
        ok = np.logical_not(o) & np.isfinite(m)
        L = float(np.nanmax(sc[fp & np.isfinite(sc)]))
        nw = ok.shape[0]
        cap = (pr.u[:nw] < CAP_U) | (pr.u[np.arange(nw) + h - 1] > 1 - CAP_U)   # window touches an end cap
        has = ok.any(1)
        wmax = np.nanmax(np.where(ok, m, -np.inf), axis=1)
        beat = has & (wmax >= L)
        nullv = m[ok]
        pct = 100 * (np.sum(nullv < L) + 0.5 * np.sum(nullv == L)) / len(nullv)
        nc = m[ok & np.logical_not(cap)[:, None]]
        pct_nocap = 100 * (np.sum(nc < L) + 0.5 * np.sum(nc == L)) / len(nc) if len(nc) else np.nan
        rows = np.nonzero(has)[0]
        out.update({f"{name}_pct": pct, f"{name}_pct_nocap": pct_nocap, f"{name}_n_win_rows": len(rows),
                    f"{name}_beat_rows": int(beat.sum()), f"{name}_beat_rows_cap": int((beat & cap).sum()),
                    f"{name}_win_stops": h, f"{name}_win_axes": wa, f"{name}_lesion_z": L})
    return out


def main():
    split = pd.read_csv(BP.OUT / "bank_split.csv")
    q = pd.read_csv(BP.OUT / "bank_pass1.csv").merge(split, on="case_id")
    ho = list(q[(q["drop"].fillna("") == "") & (q.role == "heldout")].case_id)
    with mp.Pool(4) as pool:
        rows = pool.map(one, [(c, gi) for gi, c in enumerate(ho)])
    d = pd.DataFrame(rows)
    d.to_csv(BP.OUT / "diagnose_b.csv", index=False)
    L = [f"S15 Part B diagnostic: 20 mm insertion, {len(d)} held-out glands (window rows = stop offsets)"]
    for name in ("primary", "raw"):
        L.append(f"{name}: median percentile {d[name + '_pct'].median():.1f} (counterfactual without end-cap windows "
                 f"u<{CAP_U} or >{1 - CAP_U}: {d[name + '_pct_nocap'].median():.1f}); window size median "
                 f"{d[name + '_win_stops'].median():.0f} stops x {d[name + '_win_axes'].median():.0f} axes; "
                 f"window rows per gland median {d[name + '_n_win_rows'].median():.0f}; rows beating the sphere: "
                 f"total {int(d[name + '_beat_rows'].sum())}, of which touch an end cap "
                 f"{int(d[name + '_beat_rows_cap'].sum())}; glands with >= 1 beating row "
                 f"{int((d[name + '_beat_rows'] > 0).sum())}; sphere score median {d[name + '_lesion_z'].median():.2f}")
    txt = "\n".join(L)
    (BP.OUT / "diagnose_b.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
