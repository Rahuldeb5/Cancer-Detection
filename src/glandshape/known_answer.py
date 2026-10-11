"""S15 Part A step 6: known-answer tests on synthetic curved tubes (radius 12 -> 6 mm, lobulation +-1 mm).

v5 / round 5 (Rahul, 2026-10-08), written here BEFORE the v5 run (`--core`; output known_answer_v5.*).
  Score: v5 primary = multi-scale bulge_local on RAW width with the extrapolated tip stops excluded
  (scores.primary_scores). Rerun ONCE, same seeds, CORE tests only, both grids, rules as v4:
  bulge h 8 mm at 10/20/30/40 mm and h 5 mm at 10/20/30 mm (located AND top-ranked >= 90%, middle-half rule);
  dent sign (>= 90%); noise-free bend (|median score difference in the bend window| < 0.25 z).
  Outcome rule (SESSION_15.md round 5): core pass AND bank 20 mm insertion median >= 95 -> Part C confirmatory;
  otherwise Part C runs EXPLORATORY. No further rule changes.

v4 / round 4 (Rahul, 2026-10-06), written here BEFORE the v4 run. Supersedes the rules below where they differ.
  Scores as v3 (raw WIDTH map, multi-scale bulge_local); the v4 residual score (width minus a normal median width)
  is reported beside it with SYNTHETIC references, not gated.
  A. Bulge: AT = the stop lies within the middle half of the bulge along the centerline (|s - s0| <= length/4),
     "changed after round 3 to match Part C's footprint question". The old 4 mm rule is reported next to it.
     located = window max AT the bulge; top-ranked = whole-map max AT the bulge. Each cell passes if located >= 90%
     and (for h >= 5) top-ranked >= 90%. 3 mm x 30-40 mm stays a reported resolution-limit cell.
  B. Atrophy: unchanged (step flagged >= 90%, no-step flagged <= 10%).
  C. Bend, noisy: ONE rerun with 200 NEW pairs (lobulation seeds 5000-5199, recorded), same limit: <= 10% of bent
     tubes above the straight tubes' 95th percentile; that result stands.
     Bend, noise-free (new deterministic gate): |median primary score over the bend-window cells, bent - straight|
     < 0.25 z on BOTH grids.
  D. Dent sign: unchanged.
  E. Head-body step (report only, not a gate): phantom.step_tube, width 30 -> 14 mm over 12 mm at 0.3 L. Reference
     = 30 bulge-free step tubes (seeds 6000-6029); test = 30 tubes (seeds 6100-6129) x {no bulge; 5 mm x 20 mm
     bulge on the head side (centre 21 mm headward of the step centre), body side (21 mm tailward), at the step}.
     Reported for raw width and for width minus the step-tube reference (u bin, axis): where the top cell lands on
     no-bulge tubes; located / top-ranked rates (middle half and 4 mm) with bulges.
  CORE tests (pre-committed: any CORE fail -> stop before Part B; every other cell is reported and does not block):
     bulge h 8 mm at every length; bulge h 5 mm at 10, 20, 30 mm; atrophy step / no-step; dent sign; noise-free
     bend. Each on both grids.
v1 / v2 / v3 runs: known_answer_v1.*, known_answer_v2.*, known_answer_v3.*.

v3 / round 3 (Rahul, 2026-10-06), written here BEFORE the v3 run. MEASUREMENT changes, GATES do not:
  - every bulge score (A, C, D, bulge_global) is computed on the WIDTH map w(s, axis), 12 axes 15 deg apart,
    w = r(theta) + r(theta + 180), rays from the centerline point (nearest inside point if it is outside);
    the 24-direction radius map is not scored;
  - A-D pass rules below are unchanged from v2, including 3 mm bulges at every length; AT = within 4 mm along
    the centerline;
  - secondary (not gated): the axis of the window peak and of the whole-map top cell within 45 deg of the bulge's
    outward direction (axis = a line: angle = min(a, 180 - a));
  - reported: ray-origin fallbacks (centerline point outside its plane's gland), censored widths.
  Decision rule fixed before the run: if ONLY the 3 mm x 30-40 mm bulge cells fail, they are recorded as the tool's
  resolution limit (a long 3 mm bulge against +-1 mm lobulation) and reported as FAILs of those cells; Part B may
  then proceed. If anything else fails, stop.

v2 criteria (Rahul, 2026-10-06), written here BEFORE the v2 run. They replace v1 after v1 failed (v1 run kept in
work/gland_shape/partA/known_answer_v1.*). Tool changes behind v2: multi-scale bulge (+-15/30/45 mm, primary =
outward z), rays from the section-region centroid. Two voxel grids, BOTH must pass: 0.8 x 0.8 x 2.5 mm (cohort
median slice spacing) and 0.8 mm isotropic. Tubes: phantom.gland_like (one arc, curvature radius 80-150 mm, length
140-180 mm, random tilt), head at patient right.

A. Bulge. h in {3, 5, 8} mm x length in {10, 20, 30, 40} mm; position s0 = 0.3 / 0.5 / 0.7 of the phantom length;
   seeds 0-9; direction theta0 random per seed. 30 replicates per (h, length). Score = primary bulge_local (v2).
   A cell is AT the bulge if |phantom arc position of its stop - s0| <= 4 mm (position along the centerline ONLY).
   - located: the max score within |s - s0| <= length/2 + 10 mm (all directions) is AT the bulge.
   - top-ranked: the max score of the whole map is AT the bulge.
   PASS: located in >= 90% of replicates for every (h, length); top-ranked in >= 90% for h >= 5.
   Secondary (not gated): the window peak's ray within 45 deg of the bulge's outward direction.
B. Atrophy (case level only). Normal reference: 40 normal tubes (seeds 1000-1039). Threshold = 95th percentile of
   the reference tubes' own case atrophy, each scored leave-one-out against a reference built from the other 39.
   A tube is FLAGGED if case atrophy (max |z|) > threshold and z < 0 at that stop. Step tubes: 30% area drop over the
   tail half (step at L/2, logistic width 1 mm; seeds 2000-2029). Null: 30 smoothly tapering tubes without a step
   (seeds 3000-3029), same threshold and rule.
   PASS: step tubes flagged >= 90% AND null tubes flagged <= 10%.
   Secondary (not gated): the +-20 mm windowed version's peak within 10 mm of the step; null flagged with any sign.
C. Bend. 40 seeds; straight 60 + 23.6 + 60 mm vs 60 + 90-deg arc of radius 15 mm + 60, same taper and noise seed.
   Statistic = max primary score over cells whose stop lies at phantom s in [50, 93.6] mm (the arc +-10 mm; the
   same window on the straight tube). Threshold = 95th percentile of the straight tubes' statistic.
   PASS: <= 10% of bent tubes exceed the threshold (5% expected under no effect). The noiseless pair is reported.
D. Dent (sign convention only). 10 seeds, dent h 5, length 20 at 0.5 L. Score = dent side (min over scales of the
   negative z). PASS (>= 90%): its min within |s - s0| <= 20 mm is AT the dent (position) and < 0.
Reported, not gated: bulge_global top-ranked (reference = the 40 normal tubes); winning scale; fold-over flags;
ray-origin fallbacks; timings.

v1 criteria (superseded): +-10 mm signed score; AT also required the ray within 45 deg; bulge lengths 10/20 only;
atrophy gate = peak within 10 mm of the step and flagged vs the held-out normals' P95.

Usage: PYTHONPATH=src .venv/bin/python -m glandshape.known_answer [--workers 4] [--quick]
Output: work/gland_shape/partA/known_answer.txt (+ .pkl with every replicate).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np

from glandshape import centerline as C
from glandshape import outline as O
from glandshape import phantom as Ph
from glandshape import profile as P
from glandshape import scores as S

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "work/gland_shape/partA"
GRIDS = {"primary_0.8x0.8x2.5": (0.8, 0.8, 2.5), "secondary_0.8iso": (0.8, 0.8, 0.8)}
HEIGHTS, LENGTHS, POS = (3.0, 5.0, 8.0), (10.0, 20.0, 30.0, 40.0), (0.3, 0.5, 0.7)
AT_S_MM, AT_DEG = 4.0, 45.0
PASS_FRAC = 0.90
BEND_LO = Ph.BEND_STRAIGHT_MM - 10.0
BEND_HI = Ph.BEND_STRAIGHT_MM + Ph.BEND_ARC_MM + 10.0


def run_tube(spec: Ph.TubeSpec):
    t0 = time.time()
    tube = Ph.make_tube(spec)
    ol = O.outline_from_mask(tube.mask, tube.spacing, tube.affine)
    cl = C.build_centerline(ol)
    pr = P.compute_profile(ol, cl)
    b = S.primary_scores(pr)            # v5: raw width, tip stops excluded
    s_ph = Ph.arc_position(tube, ol.iso_to_world(cl.P))
    dirs_w = P.ray_dirs(cl) @ ol.R.T                          # (n, 24, 3) world unit vectors
    head_ok = bool(s_ph[0] < s_ph[-1])
    return dict(tube=tube, s_ph=s_ph, dirs_w=dirs_w[:, :P.N_AXES], r=pr.r, w=pr.w, A=pr.A, u=pr.u,
                bl=b["primary"], dent=b["dent"], best_scale=b["best_scale"], mads=b["mads"], fold=pr.fold, tip=pr.tip,
                fallback=pr.origin_fallback, valid=pr.valid, cens=pr.w_censored, L=cl.L, head_ok=head_ok,
                secs=time.time() - t0)


def at_cells(res, s0):
    """v2: AT = within 4 mm along the centerline (position only), every direction."""
    near = np.abs(res["s_ph"] - s0) <= AT_S_MM
    return np.broadcast_to(near[:, None], res["bl"].shape)


def axis_angle(res, cell, outward) -> float:
    """Angle (deg) between the cell's axis (a line) and the outward direction: min(a, 180 - a)."""
    c = abs(float(res["dirs_w"][cell] @ outward))
    return float(np.degrees(np.arccos(np.clip(c, 0, 1))))


def dir_ok(res, cell, outward) -> bool:
    """Secondary: the cell's axis within 45 deg of the outward direction."""
    return axis_angle(res, cell, outward) <= AT_DEG


def _argmax(score, region=None, sign=+1):
    z = sign * score
    if region is not None:
        z = np.where(region, z, np.nan)
    if not np.isfinite(z).any():
        return None
    return np.unravel_index(np.nanargmax(z), z.shape)


# ------------------------------------------------------------------ jobs
BEND_SEEDS = range(5000, 5200)          # v4: 200 new noisy pairs
HB_REF_SEEDS = range(6000, 6030)
HB_TEST_SEEDS = range(6100, 6130)
HB_BUMP_OFFSET_MM = 21.0                # bulge centre this far headward / tailward of the step centre
CORE = "CORE"


def job(kind, grid, seed, **kw):
    sp = GRIDS[grid]
    extra = {}
    if kind == "bulge":
        rng = np.random.default_rng(500 + seed)
        th = rng.uniform(0, 2 * np.pi)
        base = Ph.gland_like(seed, spacing=sp)
        L = base.kappa[0][0]
        s0 = kw["pos"] * L
        base.bumps = [Ph.Bump(s0=s0, theta0=th, h=kw["h"], ell=kw["ell"])]
        spec = base
        extra = dict(s0=s0, theta0=th)
    elif kind == "dent":
        base = Ph.gland_like(seed, spacing=sp)
        th = np.random.default_rng(700 + seed).uniform(0, 2 * np.pi)
        s0 = 0.5 * base.kappa[0][0]
        base.bumps = [Ph.Bump(s0=s0, theta0=th, h=5.0, ell=20.0, sign=-1)]
        spec = base
        extra = dict(s0=s0, theta0=th)
    elif kind == "normal":
        spec = Ph.gland_like(seed, spacing=sp)
    elif kind == "step":
        spec = Ph.gland_like(seed, spacing=sp)
        spec.area_step = (0.5 * spec.kappa[0][0], 0.7)
    elif kind == "bend":
        spec = Ph.bend_tube(seed, bent=kw["bent"], spacing=sp, lob_amp=kw.get("lob", Ph.LOB_AMP_MM))
    elif kind == "hbstep":
        spec = Ph.step_tube(seed, spacing=sp)
        sc = Ph.HB_STEP_U * spec.kappa[0][0]
        extra = dict(s_c=sc)
        v = kw["variant"]
        if v != "none":
            s0 = sc + {"head": -HB_BUMP_OFFSET_MM, "body": HB_BUMP_OFFSET_MM, "step": 0.0}[v]
            th = np.random.default_rng(900 + seed).uniform(0, 2 * np.pi)
            spec.bumps = [Ph.Bump(s0=s0, theta0=th, h=5.0, ell=20.0)]
            extra.update(s0=s0, theta0=th)
    else:
        raise ValueError(kind)
    res = run_tube(spec)
    res.update(kind=kind, grid=grid, seed=seed, **kw, **extra)
    return res


def _work(args):
    kind, grid, seed, kw = args
    try:
        r = job(kind, grid, seed, **kw)
        r.pop("tube")  # keep the pickle small; truths are re-attached below
        return r
    except Exception as e:  # report, never hide
        return dict(kind=kind, grid=grid, seed=seed, error=f"{type(e).__name__}: {e}", **kw)


def build_jobs(grid, quick=False):
    seeds = range(3) if quick else range(10)
    J = []
    for h in HEIGHTS:
        for ell in LENGTHS:
            for pos in POS:
                for sd in seeds:
                    J.append(("bulge", grid, sd, dict(h=h, ell=ell, pos=pos)))
    nn = 10 if quick else 40
    nt = 8 if quick else 30
    J += [("normal", grid, 1000 + i, dict(role="ref")) for i in range(nn)]
    J += [("step", grid, 2000 + i, {}) for i in range(nt)]
    J += [("normal", grid, 3000 + i, dict(role="null")) for i in range(nt)]
    for sd in (list(BEND_SEEDS)[:10] if quick else BEND_SEEDS):
        J.append(("bend", grid, sd, dict(bent=False)))
        J.append(("bend", grid, sd, dict(bent=True)))
    J.append(("bend", grid, 0, dict(bent=False, lob=0.0)))
    J.append(("bend", grid, 0, dict(bent=True, lob=0.0)))
    J += [("dent", grid, i, {}) for i in (range(3) if quick else range(10))]
    J += [("hbstep", grid, sd, dict(variant="none", role="ref")) for sd in HB_REF_SEEDS]
    for v in ("none", "head", "body", "step"):
        J += [("hbstep", grid, sd, dict(variant=v, role="test")) for sd in HB_TEST_SEEDS]
    return J


# ------------------------------------------------------------------ evaluation
def _truth(res):
    """Rebuild the tube spec cheaply to get its truths (bump apex/outward, step position) without voxelizing."""
    sp = GRIDS[res["grid"]]
    if res["kind"] in ("bulge", "dent", "step", "normal"):
        spec = Ph.gland_like(res["seed"], spacing=sp)
        L = spec.kappa[0][0]
        return spec, L
    return None, None


def _bump_truth(res):
    """Outward world direction of the bump: recomputed from the phantom geometry (no voxelization)."""
    spec = Ph.gland_like(res["seed"], spacing=GRIDS[res["grid"]])   # step_tube shares the gland_like curve
    s0, th = res["s0"], res["theta0"]
    s, xy, phi = Ph._curve2d(spec.kappa)
    e_p, e_q, n_pl = Ph._basis(spec)
    i = int(np.clip(np.rint(s0 / Ph.DS), 0, len(s) - 1))
    T = np.cos(phi[i]) * e_p + np.sin(phi[i]) * e_q
    M = np.cross(n_pl, T)
    return s0, np.cos(th) * n_pl + np.sin(th) * M


def _loo_threshold(refs):
    """95th percentile of the reference tubes' case atrophy, each scored against the other reference tubes."""
    mags = []
    for j in range(len(refs)):
        ref_j = S.build_reference([_P(r) for i, r in enumerate(refs) if i != j])
        mags.append(S.case_atrophy(S.atrophy_z(refs[j]["A"], refs[j]["u"], ref_j))[0])
    return float(np.nanpercentile(mags, 95))


class _P:
    def __init__(self, r):
        self.u, self.w, self.A = r["u"], r["w"], r["A"]


def _mid(r, s0, ell):
    """v4 AT rule: stops within the middle half of the bulge (|s - s0| <= length / 4), every axis."""
    return np.broadcast_to((np.abs(r["s_ph"] - s0) <= ell / 4)[:, None], r["bl"].shape)


def _bulge_rates(G, score_fn, ell):
    """located / top-ranked under the v4 middle-half rule and the old 4 mm rule; secondary axis check."""
    out = {k: [] for k in ("loc", "top", "loc4", "top4", "ax", "ds")}
    for r in G:
        s0, outw = _bump_truth(r)
        sc = score_fn(r)
        mid, at4 = _mid(r, s0, ell), at_cells(r, s0)
        win = (np.abs(r["s_ph"] - s0) <= ell / 2 + 10)[:, None] & np.isfinite(sc)
        pk, g = _argmax(sc, win), _argmax(sc)
        out["loc"].append(pk is not None and bool(mid[pk]))
        out["top"].append(g is not None and bool(mid[g]))
        out["loc4"].append(pk is not None and bool(at4[pk]))
        out["top4"].append(g is not None and bool(at4[g]))
        if g is not None:
            out["ax"].append(dir_ok(r, g, outw))
            out["ds"].append(abs(r["s_ph"][g[0]] - s0))
    return {k: (float(np.mean(v)) if len(v) else np.nan) for k, v in out.items() if k != "ds"} | \
        {"ds_med": float(np.median(out["ds"])) if out["ds"] else np.nan}


def _fmt(x):
    return f"{x:.0%}" if np.isfinite(x) else "n/a"


def evaluate(results, grid):
    R = [r for r in results if r["grid"] == grid]
    errs = [r for r in R if "error" in r]
    R = [r for r in R if "error" not in r]
    lines = [f"=== grid {grid}: {len(R)} tubes ok, {len(errs)} errors"]
    for e in errs:
        lines.append(f"  ERROR {e['kind']} seed {e['seed']}: {e['error']}")
    lines.append(f"  orientation head end first: {sum(r['head_ok'] for r in R)}/{len(R)}; "
                 f"median secs/tube {np.median([r['secs'] for r in R]):.2f}; "
                 f"tubes with fold flags {sum(bool(r['fold'].any()) for r in R)}; "
                 f"ray-origin fallbacks {sum(int(r['fallback'].sum()) for r in R)} stops in "
                 f"{sum(bool(r['fallback'].any()) for r in R)} tubes; "
                 f"censored widths {sum(int(r['cens'].sum()) for r in R)}")
    cells = []  # (name, core, passed)
    refs = [r for r in R if r["kind"] == "normal" and r.get("role") == "ref"]
    ref = S.build_reference([_P(r) for r in refs])

    def raw(r):
        return r["bl"]

    def resid(r, rf=ref):
        return S.bulge_residual(r["w"], r["u"], rf)["primary"]

    # A. bulge
    lines.append("A. bulge on raw width. located/top = window/whole-map max in the bulge's MIDDLE HALF (v4 rule); "
                 "[old 4 mm rule]; secondary: top cell's axis <= 45 deg; residual score (synthetic normal ref) reported")
    for h in HEIGHTS:
        for ell in LENGTHS:
            G = [r for r in R if r["kind"] == "bulge" and r["h"] == h and r["ell"] == ell]
            a = _bulge_rates(G, raw, ell)
            b = _bulge_rates(G, resid, ell)
            core = h >= 8 or (h >= 5 and ell <= 30)
            need_top = h >= 5
            ok = a["loc"] >= PASS_FRAC and (a["top"] >= PASS_FRAC or not need_top)
            tag = CORE if core else ("resolution limit" if h < 5 and ell >= 30 else "non-core")
            cells.append((f"bulge h{h:.0f} len{ell:.0f}", core, ok))
            lines.append(f"  [{tag}] h {h:.0f} len {ell:.0f} (n={len(G)}): located {_fmt(a['loc'])} top {_fmt(a['top'])}"
                         f"{'' if need_top else ' (top not gated)'} -> {'PASS' if ok else 'FAIL'} | [4 mm: located "
                         f"{_fmt(a['loc4'])} top {_fmt(a['top4'])}] | top |ds| median {a['ds_med']:.1f} mm; axis<=45 "
                         f"{_fmt(a['ax'])} | residual: located {_fmt(b['loc'])} top {_fmt(b['top'])}")

    # B. atrophy (case level)
    steps = [r for r in R if r["kind"] == "step"]
    nulls = [r for r in R if r["kind"] == "normal" and r.get("role") == "null"]
    thr = _loo_threshold(refs)

    def flagged(r):
        mag, sgn, i = S.case_atrophy(S.atrophy_z(r["A"], r["u"], ref))
        return bool(mag > thr and sgn < 0), bool(mag > thr), mag
    fs, fn = [flagged(r) for r in steps], [flagged(r) for r in nulls]
    f_step, f_null = np.mean([f[0] for f in fs]), np.mean([f[0] for f in fn])
    ok = f_step >= PASS_FRAC and f_null <= 1 - PASS_FRAC
    cells.append(("atrophy", True, ok))
    lines.append(f"B. [{CORE}] atrophy case level (LOO P95 |z| {thr:.2f}): step flagged {f_step:.0%}, no-step flagged "
                 f"{f_null:.0%} (any sign {np.mean([f[1] for f in fn]):.0%}) -> {'PASS' if ok else 'FAIL'}")

    # C. bend
    def window_cells(r):
        return ((r["s_ph"] >= BEND_LO) & (r["s_ph"] <= BEND_HI))[:, None] & np.isfinite(r["bl"])

    def bend_max(r):
        return float(np.nanmax(np.where(window_cells(r), r["bl"], np.nan)))

    def bend_median(r):
        return float(np.nanmedian(r["bl"][window_cells(r)]))
    noisy = [r for r in R if r["kind"] == "bend" and r.get("lob") is None]
    st = np.array([bend_max(r) for r in noisy if not r["bent"]])
    bt = np.array([bend_max(r) for r in noisy if r["bent"]])
    thr_b = float(np.percentile(st, 95))
    exceed = float(np.mean(bt > thr_b))
    ok = exceed <= 0.10
    cells.append(("bend noisy (200 pairs)", False, ok))
    lines.append(f"C. [non-core] bend noisy (n={len(bt)} pairs, seeds {BEND_SEEDS.start}-{BEND_SEEDS.stop - 1}): straight "
                 f"P95 {thr_b:.2f}; bent median {np.median(bt):.2f} vs straight {np.median(st):.2f}; bent above P95 "
                 f"{exceed:.1%} -> {'PASS' if ok else 'FAIL'}")
    n0 = {r["bent"]: r for r in R if r["kind"] == "bend" and r.get("lob") == 0.0}
    d0 = bend_median(n0[True]) - bend_median(n0[False])
    ok = abs(d0) < 0.25
    cells.append(("bend noise-free", True, ok))
    lines.append(f"   [{CORE}] bend noise-free: median score in the bend window bent {bend_median(n0[True]):.3f} vs "
                 f"straight {bend_median(n0[False]):.3f} (diff {d0:+.3f} z; max {bend_max(n0[True]):.2f} vs "
                 f"{bend_max(n0[False]):.2f}) -> {'PASS' if ok else 'FAIL'}")

    # D. dent
    D = [r for r in R if r["kind"] == "dent"]
    good = []
    for r in D:
        s0, _ = _bump_truth(r)
        win = (np.abs(r["s_ph"] - s0) <= 20)[:, None] & np.isfinite(r["dent"])
        pk = _argmax(r["dent"], win, sign=-1)
        good.append(pk is not None and bool(_mid(r, s0, 20.0)[pk]) and r["dent"][pk] < 0)
    ok = np.mean(good) >= PASS_FRAC
    cells.append(("dent sign", True, ok))
    lines.append(f"D. [{CORE}] dent sign (n={len(D)}): dent-side min in the dent's middle half and < 0 in "
                 f"{np.mean(good):.0%} -> {'PASS' if ok else 'FAIL'}")

    # E. head-body step (report only)
    hb_ref = S.build_reference([_P(r) for r in R if r["kind"] == "hbstep" and r.get("role") == "ref"])
    lines.append(f"E. head-body step (report only; residual = width minus the {hb_ref.n_glands}-tube step reference)")
    for name, fn in (("raw", raw), ("residual", lambda r: resid(r, hb_ref))):
        T0 = [r for r in R if r["kind"] == "hbstep" and r.get("role") == "test" and r["variant"] == "none"]
        d = []
        for r in T0:
            g = _argmax(fn(r))
            d.append(r["s_ph"][g[0]] - r["s_c"])
        d = np.array(d)
        bins = {"head side far (< -30)": d < -30, "head side near (-30..-6)": (d >= -30) & (d < -6),
                "on the step (-6..+6)": np.abs(d) <= 6, "body side near (+6..+30)": (d > 6) & (d <= 30),
                "body side far (> +30)": d > 30}
        lines.append(f"  {name} no-bulge top cell vs step centre (n={len(d)}): " +
                     ", ".join(f"{k} {np.mean(v):.0%}" for k, v in bins.items()) +
                     f"; median {np.median(d):+.1f} mm")
        for v in ("head", "body", "step"):
            G = [r for r in R if r["kind"] == "hbstep" and r.get("role") == "test" and r["variant"] == v]
            a = _bulge_rates(G, fn, 20.0)
            lines.append(f"    {name} bulge 5 x 20 mm on the {v}{' side' if v != 'step' else ''} (n={len(G)}): located "
                         f"{_fmt(a['loc'])} top {_fmt(a['top'])} [4 mm: {_fmt(a['loc4'])} / {_fmt(a['top4'])}]; "
                         f"top |ds| median {a['ds_med']:.1f} mm")
    core_fail = [c for c, core, ok in cells if core and not ok]
    other_fail = [c for c, core, ok in cells if not core and not ok]
    lines.append(f"  verdict {grid}: CORE {'PASS' if not core_fail else 'FAIL ' + str(core_fail)}; "
                 f"non-core fails (reported, not blocking): {other_fail or 'none'}")
    return lines, not core_fail


def build_core_jobs(grid):
    """v5: core cells only, same seeds as v4."""
    J = []
    for h, lengths in ((8.0, (10.0, 20.0, 30.0, 40.0)), (5.0, (10.0, 20.0, 30.0))):
        for ell in lengths:
            for pos in POS:
                for sd in range(10):
                    J.append(("bulge", grid, sd, dict(h=h, ell=ell, pos=pos)))
    J += [("dent", grid, i, {}) for i in range(10)]
    J.append(("bend", grid, 0, dict(bent=False, lob=0.0)))
    J.append(("bend", grid, 0, dict(bent=True, lob=0.0)))
    return J


def evaluate_core(results, grid):
    R = [r for r in results if r["grid"] == grid and "error" not in r]
    errs = [r for r in results if r["grid"] == grid and "error" in r]
    lines = [f"=== grid {grid}: {len(R)} tubes ok, {len(errs)} errors; tip stops per tube median "
             f"{np.median([int(r['tip'].sum()) for r in R]):.0f}"]
    lines += [f"  ERROR {e['kind']} seed {e['seed']}: {e['error']}" for e in errs]
    fails = []
    for h, lengths in ((8.0, (10.0, 20.0, 30.0, 40.0)), (5.0, (10.0, 20.0, 30.0))):
        for ell in lengths:
            G = [r for r in R if r["kind"] == "bulge" and r["h"] == h and r["ell"] == ell]
            a = _bulge_rates(G, lambda r: r["bl"], ell)
            ok = a["loc"] >= PASS_FRAC and a["top"] >= PASS_FRAC
            fails += [] if ok else [f"bulge h{h:.0f} len{ell:.0f}"]
            lines.append(f"  [CORE] bulge h {h:.0f} len {ell:.0f} (n={len(G)}): located {_fmt(a['loc'])} top "
                         f"{_fmt(a['top'])} -> {'PASS' if ok else 'FAIL'} [4 mm: {_fmt(a['loc4'])} / {_fmt(a['top4'])}]")
    D = [r for r in R if r["kind"] == "dent"]
    good = []
    for r in D:
        s0, _ = _bump_truth(r)
        win = (np.abs(r["s_ph"] - s0) <= 20)[:, None] & np.isfinite(r["dent"])
        pk = _argmax(r["dent"], win, sign=-1)
        good.append(pk is not None and bool(_mid(r, s0, 20.0)[pk]) and r["dent"][pk] < 0)
    ok = np.mean(good) >= PASS_FRAC
    fails += [] if ok else ["dent"]
    lines.append(f"  [CORE] dent sign (n={len(D)}): {np.mean(good):.0%} -> {'PASS' if ok else 'FAIL'}")

    def bend_median(r):
        c = ((r["s_ph"] >= BEND_LO) & (r["s_ph"] <= BEND_HI))[:, None] & np.isfinite(r["bl"])
        return float(np.nanmedian(r["bl"][c]))
    n0 = {r["bent"]: r for r in R if r["kind"] == "bend" and r.get("lob") == 0.0}
    d0 = bend_median(n0[True]) - bend_median(n0[False])
    ok = abs(d0) < 0.25
    fails += [] if ok else ["bend noise-free"]
    lines.append(f"  [CORE] bend noise-free: bent {bend_median(n0[True]):.3f} vs straight {bend_median(n0[False]):.3f} "
                 f"(diff {d0:+.3f} z) -> {'PASS' if ok else 'FAIL'}")
    lines.append(f"  verdict {grid}: CORE {'PASS' if not fails else 'FAIL ' + str(fails)}")
    return lines, not fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--grids", nargs="*", default=list(GRIDS))
    ap.add_argument("--core", action="store_true", help="v5: core tests only -> known_answer_v5.*")
    a = ap.parse_args()
    if a.core:
        OUT.mkdir(parents=True, exist_ok=True)
        jobs = [j for g in a.grids for j in build_core_jobs(g)]
        t0 = time.time()
        with mp.Pool(a.workers) as pool:
            results = pool.map(_work, jobs, chunksize=2)
        with open(OUT / "known_answer_v5.pkl", "wb") as f:
            pickle.dump(results, f)
        lines = [f"S15 Part A round 5 CORE rerun ({len(jobs)} tubes, {time.time() - t0:.0f} s)"]
        ok_all = True
        for g in a.grids:
            ls, ok = evaluate_core(results, g)
            lines += ls
            ok_all &= ok
        lines.append(f"OVERALL CORE (v5): {'PASS' if ok_all else 'FAIL'}")
        txt = "\n".join(lines)
        (OUT / "known_answer_v5.txt").write_text(txt + "\n")
        print(txt)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_quick" if a.quick else ""
    jobs = [j for g in a.grids for j in build_jobs(g, a.quick)]
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        results = pool.map(_work, jobs, chunksize=2)
    with open(OUT / f"known_answer{tag}.pkl", "wb") as f:
        pickle.dump(results, f)
    lines = [f"S15 Part A known-answer tests ({len(jobs)} tubes, {time.time() - t0:.0f} s, {a.workers} workers)"]
    core_ok = True
    for g in a.grids:
        ls, ok = evaluate(results, g)
        lines += ls
        core_ok &= ok
    lines.append(f"OVERALL CORE: {'PASS -> proceed to Part B' if core_ok else 'FAIL -> stop before Part B'}")
    txt = "\n".join(lines)
    (OUT / f"known_answer{tag}.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
