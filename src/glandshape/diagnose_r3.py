"""S15 Part A round 3 diagnostics (synthetic only; reads work/gland_shape/partA/known_answer.pkl; changes nothing).

(a) Long bulges: where the window peak / whole-map top cell lands along the centerline relative to the bulge
    (|ds| <= 4 mm is the gate; also reported: inside the middle half of the bulge |ds| <= length/4, inside the
    bulge |ds| <= length/2), and the crest flatness of the phantom's elliptical profile at |ds| = 4 / 8 mm.
(b) Bend: is the observed exceedance consistent with no bend effect? Bent and straight tubes share a noise seed, so
    the null swaps the bent/straight label within each pair (10,000 random swaps) and recomputes the exceedance of
    the straight-tube P95 exactly as the gate does.

Usage: PYTHONPATH=src .venv/bin/python -m glandshape.diagnose_r3   -> work/gland_shape/partA/diagnose_r3.txt
"""
from __future__ import annotations

import pickle

import numpy as np

from glandshape import known_answer as KA


def main():
    with open(KA.OUT / "known_answer_v3.pkl", "rb") as f:
        R = pickle.load(f)
    L = ["S15 Part A round 3 diagnostics (synthetic only)", "(a) long bulges: top cell |ds| (gate 4 mm)"]
    for grid in KA.GRIDS:
        for h in KA.HEIGHTS:
            for ell in (30.0, 40.0):
                G = [r for r in R if r["grid"] == grid and r["kind"] == "bulge" and r["h"] == h and r["ell"] == ell]
                ds = []
                for r in G:
                    s0, _ = KA._bump_truth(r)
                    g = KA._argmax(r["bl"])
                    ds.append(abs(r["s_ph"][g[0]] - s0))
                ds = np.array(ds)
                crest = [np.sqrt(1 - (x / (ell / 2)) ** 2) for x in (4.0, 8.0)]
                L.append(f"  {grid} h {h:.0f} len {ell:.0f}: top cell |ds| <= 4 {np.mean(ds <= 4):.0%}, "
                         f"<= len/4 ({ell / 4:.0f} mm) {np.mean(ds <= ell / 4):.0%}, <= len/2 {np.mean(ds <= ell / 2):.0%}; "
                         f"median {np.median(ds):.1f}, max {ds.max():.1f} mm; phantom height at |ds| 4 / 8 mm = "
                         f"{crest[0]:.2f} / {crest[1]:.2f} of h ({h * (1 - crest[0]):.2f} / {h * (1 - crest[1]):.2f} mm "
                         f"below the apex vs +-1 mm lobulation)")
    L.append("(b) bend: paired label-swap null of the exceedance fraction")
    rng = np.random.default_rng(0)
    for grid in KA.GRIDS:
        def stat(r):
            reg = ((r["s_ph"] >= KA.BEND_LO) & (r["s_ph"] <= KA.BEND_HI))[:, None] & np.isfinite(r["bl"])
            return float(np.nanmax(np.where(reg, r["bl"], np.nan)))
        B = {(r["seed"], r["bent"]): stat(r) for r in R
             if r["grid"] == grid and r["kind"] == "bend" and r.get("lob") is None}
        seeds = sorted({k[0] for k in B})
        st = np.array([B[(s, False)] for s in seeds])
        bt = np.array([B[(s, True)] for s in seeds])

        def exceed(a, b):
            return np.mean(b > np.percentile(a, 95))
        obs = exceed(st, bt)
        null = []
        for _ in range(10_000):
            sw = rng.random(len(seeds)) < 0.5
            a, b = np.where(sw, bt, st), np.where(sw, st, bt)
            null.append(exceed(a, b))
        null = np.array(null)
        L.append(f"  {grid}: observed {obs:.1%}; null mean {null.mean():.1%}, P(null >= observed) "
                 f"{np.mean(null >= obs - 1e-12):.3f}, P(null > 10%) {np.mean(null > 0.10 + 1e-12):.3f}; "
                 f"paired median difference bent - straight {np.median(bt - st):+.2f} z")
    txt = "\n".join(L)
    (KA.OUT / "diagnose_r3_v3.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
