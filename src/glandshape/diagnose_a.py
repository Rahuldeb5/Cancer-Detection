"""S15 Part A diagnostics of the known-answer failures (synthetic only; nothing here changes a frozen parameter).

(a) 20 mm-long bulges: why the window peak misses (|ds| vs angle).
(b) Longer bulges (30, 40 mm; the S15 target lesions are 20-40 mm) under the frozen +-10 mm median window.
(c) Bend: centerline offset from the true curve at the bend, and where the bent tube's max bulge points.
(d) Atrophy: the whole-side plateau, and (option only, NOT adopted) +-20 mm windowed sides on the same tubes.

This is the v1 RECORD: work/gland_shape/partA/diagnose_a_v1.txt was produced with the v1 tool (+-10 mm signed score,
centerline ray origin) from known_answer_v1.pkl. Parts (a), (c-noisy) and (d) read that pickle; parts (b) and
(c-noiseless) re-voxelize tubes and would now run the CURRENT (v2) tool, so a rerun does not reproduce them.

Usage: PYTHONPATH=src .venv/bin/python -m glandshape.diagnose_a   -> work/gland_shape/partA/diagnose_a.txt
"""
from __future__ import annotations

import multiprocessing as mp
import pickle

import numpy as np

from glandshape import known_answer as KA
from glandshape import phantom as Ph
from glandshape import scores as S


def _load():
    with open(KA.OUT / "known_answer_v1.pkl", "rb") as f:
        return pickle.load(f)


def _at_v1(res, s0, outward):
    """v1 AT rule: within 4 mm along the centerline AND ray within 45 deg of the outward direction."""
    near = np.abs(res["s_ph"] - s0) <= KA.AT_S_MM
    return near[:, None] & (res["dirs_w"] @ outward >= np.cos(np.deg2rad(KA.AT_DEG)))


def diag_bulge20(R, grid, lines):
    for h in KA.HEIGHTS:
        G = [r for r in R if r["grid"] == grid and r["kind"] == "bulge" and r["ell"] == 20.0 and r["h"] == h]
        fails = []
        for r in G:
            s0, outw = KA._bump_truth(r)
            at = _at_v1(r, s0, outw)
            win = (np.abs(r["s_ph"] - s0) <= 20)[:, None] & np.isfinite(r["bl"])
            pk = KA._argmax(r["bl"], win)
            g = KA._argmax(r["bl"])
            if not (at[pk] and at[g]):
                ang = np.degrees(np.arccos(np.clip(r["dirs_w"][pk] @ outw, -1, 1)))
                gs = r["s_ph"][g[0]] - s0
                fails.append(f"seed {r['seed']} pos {r['pos']}: window peak ds {r['s_ph'][pk[0]] - s0:+.1f} mm "
                             f"angle {ang:.0f} deg z {r['bl'][pk]:.2f}; global max ds {gs:+.1f} mm z {r['bl'][g]:.2f}")
        lines.append(f"  {grid} h {h:.0f} len 20: {len(fails)}/{len(G)} replicates miss located-or-top")
        lines += ["    " + f for f in fails]


def _bulge_job(args):
    grid, seed, h, ell, pos = args
    r = KA.job("bulge", grid, seed, h=h, ell=ell, pos=pos)
    r.pop("tube")
    return r


def diag_long(lines, workers=4):
    grid = "primary_0.8x0.8x2.5"
    jobs = [(grid, sd, h, ell, pos) for h in (5.0, 8.0) for ell in (30.0, 40.0) for pos in KA.POS for sd in range(10)]
    with mp.Pool(workers) as pool:
        R = pool.map(_bulge_job, jobs, chunksize=2)
    for h in (5.0, 8.0):
        for ell in (30.0, 40.0):
            G = [r for r in R if r["h"] == h and r["ell"] == ell]
            loc, top, zat = [], [], []
            for r in G:
                s0, outw = KA._bump_truth(r)
                at = _at_v1(r, s0, outw)
                win = (np.abs(r["s_ph"] - s0) <= ell / 2 + 10)[:, None] & np.isfinite(r["bl"])
                pk, g = KA._argmax(r["bl"], win), KA._argmax(r["bl"])
                loc.append(bool(at[pk])); top.append(bool(at[g]))
                zat.append(np.nanmax(np.where(at, r["bl"], np.nan)))
            lines.append(f"  {grid} h {h:.0f} len {ell:.0f} (n={len(G)}): located {np.mean(loc):.0%} top-ranked "
                         f"{np.mean(top):.0%}; median best z AT bulge {np.median(zat):.2f}")


def diag_bend(R, lines):
    for grid in KA.GRIDS:
        for bent in (False, True):
            r = [x for x in R if x["grid"] == grid and x["kind"] == "bend" and x.get("lob") == 0.0 and x["bent"] == bent][0]
            spec = Ph.bend_tube(0, bent=bent, spacing=KA.GRIDS[grid], lob_amp=0.0)
            tube = Ph.make_tube(spec)
            reg = (r["s_ph"] >= KA.BEND_LO) & (r["s_ph"] <= KA.BEND_HI)
            pk = KA._argmax(r["bl"], reg[:, None] & np.isfinite(r["bl"]))
            # true centre of curvature direction (inner side) at the bend midpoint
            s_mid = Ph.BEND_STRAIGHT_MM + Ph.BEND_ARC_MM / 2
            i = int(np.argmin(np.abs(tube.s - s_mid)))
            j0, j1 = max(i - 25, 0), min(i + 25, len(tube.s) - 1)
            acc = tube.curve[j1] - 2 * tube.curve[i] + tube.curve[j0]
            inner = acc / np.linalg.norm(acc) if np.linalg.norm(acc) > 1e-9 else np.zeros(3)
            ang = np.degrees(np.arccos(np.clip(r["dirs_w"][pk] @ inner, -1, 1))) if inner.any() else np.nan
            lines.append(f"  {grid} {'bent' if bent else 'straight'} noiseless: max bulge_local in bend window "
                         f"{r['bl'][pk]:.2f} at phantom s {r['s_ph'][pk[0]]:.1f} mm, angle to INNER side {ang:.0f} deg")
        for bent in (False, True):
            rs = [x for x in R if x["grid"] == grid and x["kind"] == "bend" and x.get("lob") is None and x["bent"] == bent]
            sts = [np.nanmax(np.where(((x["s_ph"] >= KA.BEND_LO) & (x["s_ph"] <= KA.BEND_HI))[:, None], x["bl"], np.nan))
                   for x in rs]
            lines.append(f"    {grid} {'bent' if bent else 'straight'} noisy (n={len(rs)}): bend-window max "
                         f"median {np.median(sts):.2f}, P90 {np.percentile(sts, 90):.2f}, max {np.max(sts):.2f}")


def windowed_atrophy(A, half_mm=20.0):
    h = int(round(half_mm / 2.0))
    n = len(A)
    out = np.full(n, np.nan)
    for i in range(n):
        hd, tl = A[max(0, i - h):i], A[i + 1:i + 1 + h]
        hd, tl = hd[np.isfinite(hd)], tl[np.isfinite(tl)]
        if len(hd) >= 5 and len(tl) >= 5 and np.median(hd) > 0 and np.median(tl) > 0:
            out[i] = np.log(np.median(tl) / np.median(hd))
    return out


def diag_atrophy(R, lines):
    for grid in KA.GRIDS:
        G = [r for r in R if r["grid"] == grid]
        refs = [r for r in G if r["kind"] == "normal" and r.get("role") == "ref"]
        nulls = [r for r in G if r["kind"] == "normal" and r.get("role") == "null"]
        steps = [r for r in G if r["kind"] == "step"]
        for name, fn in (("whole-side (frozen)", S.atrophy_raw), ("+-20 mm windows (option, not adopted)", windowed_atrophy)):
            nb = S.N_UBINS
            pool = [[] for _ in range(nb)]
            for r in refs:
                a = fn(r["A"])
                for b, v in zip(S.u_bin(r["u"]), a):
                    if np.isfinite(v):
                        pool[b].append(v)
            med = np.array([np.median(v) if v else np.nan for v in pool])
            mad = np.array([S._mad(np.asarray(v)) if v else np.nan for v in pool])

            def z(r):
                b = S.u_bin(r["u"])
                return (fn(r["A"]) - med[b]) / np.maximum(mad[b], 1e-6)
            thr = np.nanpercentile([S.case_atrophy(z(r))[0] for r in nulls], 95)
            fl, lo, off = [], [], []
            for r in steps:
                mag, sg, i = S.case_atrophy(z(r))
                L = KA._truth(r)[1]
                d = r["s_ph"][i] - 0.5 * L
                fl.append(mag > thr and sg < 0); lo.append(abs(d) <= 10); off.append(d)
            lines.append(f"  {grid} {name}: flagged {np.mean(fl):.0%}, peak within 10 mm {np.mean(lo):.0%}, "
                         f"median offset {np.median(off):+.1f} mm")
        # plateau illustration: deviation from the reference median along u for one step tube
        r = steps[0]
        a = S.atrophy_raw(r["A"])
        lines.append(f"  {grid} one step tube: raw whole-side log ratio at u=0.2/0.4/0.5/0.6/0.8: " +
                     " ".join(f"{a[np.argmin(np.abs(r['u'] - q))]:+.2f}" for q in (0.2, 0.4, 0.5, 0.6, 0.8)))


def main():
    R = _load()
    lines = ["S15 Part A diagnostics (synthetic only)", "(a) 20 mm-long bulge misses"]
    for g in KA.GRIDS:
        diag_bulge20(R, g, lines)
    lines.append("(b) longer bulges under the frozen +-10 mm window (not gated)")
    diag_long(lines)
    lines.append("(c) bend")
    diag_bend(R, lines)
    lines.append("(d) atrophy")
    diag_atrophy(R, lines)
    txt = "\n".join(lines)
    (KA.OUT / "diagnose_a.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
