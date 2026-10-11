"""Synthetic curved tubes for the S15 known-answer tests (Part A step 6).

A tube is a dense planar centre curve (0.2 mm steps; heading integrated from a curvature profile) embedded in 3D,
with radius R(s, theta) = taper(s) + lobulation(s, theta), then scaled by sqrt(area factor) after an optional area
step. Lobulation: Gaussian noise on an (s, theta) grid, smoothed (sigma 4 mm along s, 30 deg around) and scaled so
max |lobulation| = LOB_AMP_MM (+-1 mm). Ends are round caps. Bulges / dents are ellipsoids centred on the tube
surface at (s0, theta0): semi-axes ell/2 along the tangent, ell/2 around, h radially (so it sticks out h mm),
unioned (bulge) or subtracted (dent).

The phantom has its own frame (planar curve: N = plane normal, B = in-plane normal) and its own geometry code: only
world-space truths (apex point, outward direction, arc length on the phantom curve) are compared with the tool.
The tube is voxelized on an anisotropic RAS grid (head at +x, i.e. patient right; tail toward patient left).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

DS = 0.2                     # dense curve step (mm)
LOB_AMP_MM = 1.0
LOB_SIGMA_S_MM = 4.0
LOB_SIGMA_TH_DEG = 30.0
LOB_DS = 0.5                 # lobulation grid step (mm)
LOB_NTH = 72                 # lobulation grid: 5 deg
STEP_WIDTH_MM = 1.0          # logistic width of the area step


@dataclass
class Bump:
    s0: float                # arc length on the phantom curve (mm from the head end of the curve)
    theta0: float            # phantom-frame angle (rad)
    h: float                 # radial extent beyond the surface (mm)
    ell: float               # length along the tube (mm)
    sign: int = +1           # +1 bulge, -1 dent


@dataclass
class TubeSpec:
    kappa: list              # [(length_mm, curvature 1/mm), ...] piecewise-constant curvature segments
    seed: int = 0
    r_head: float = 12.0
    r_tail: float = 6.0
    r_knots: tuple | None = None       # optional ((s_mm, R_mm), ...) piecewise-linear base radius (overrides the taper)
    lob_amp: float = LOB_AMP_MM
    bumps: list = field(default_factory=list)
    area_step: tuple | None = None     # (s_step mm, area factor)
    spacing: tuple = (0.8, 0.8, 2.5)
    tilt_deg: float = 20.0             # tail end raised toward +z (superior)
    plane_deg: float = 30.0            # curve plane: in-plane normal tilted from +y toward +z


@dataclass
class Tube:
    mask: np.ndarray
    spacing: np.ndarray
    affine: np.ndarray
    curve: np.ndarray        # (m, 3) world mm, head -> tail
    s: np.ndarray            # (m,) arc length
    length: float
    truth: dict


def _curve2d(kappa):
    lens = np.array([seg[0] for seg in kappa], float)
    L = lens.sum()
    s = np.arange(0.0, L + 1e-9, DS)
    k = np.zeros_like(s)
    edges = np.r_[0, np.cumsum(lens)]
    for (ln, kv), a, b in zip(kappa, edges[:-1], edges[1:]):
        k[(s >= a) & (s < b + 1e-9)] = kv
    phi = np.r_[0.0, np.cumsum(0.5 * (k[1:] + k[:-1]) * DS)]
    xy = np.r_[[[0.0, 0.0]], np.cumsum(DS * np.stack([np.cos(phi[:-1]), np.sin(phi[:-1])], 1), 0)]
    return s, xy[: len(s)], phi


def _basis(spec: TubeSpec):
    t = np.deg2rad(spec.tilt_deg)
    e_p = np.array([-np.cos(t), 0.0, np.sin(t)])                 # along the gland: toward patient left
    q = np.deg2rad(spec.plane_deg)
    e_q = np.array([0.0, np.cos(q), np.sin(q)])
    e_q -= (e_q @ e_p) * e_p
    e_q /= np.linalg.norm(e_q)
    n_pl = np.cross(e_p, e_q)
    return e_p, e_q, n_pl


def lobulation(L: float, seed: int, amp: float):
    rng = np.random.default_rng(seed)
    ns = int(np.ceil(L / LOB_DS)) + 1
    g = rng.standard_normal((ns, LOB_NTH))
    g = gaussian_filter(g, (LOB_SIGMA_S_MM / LOB_DS, LOB_SIGMA_TH_DEG / (360 / LOB_NTH)), mode=("nearest", "wrap"))
    m = np.abs(g).max()
    return g * (amp / m) if m > 0 and amp > 0 else g * 0.0


def make_tube(spec: TubeSpec) -> Tube:
    s, xy, phi = _curve2d(spec.kappa)
    L = float(s[-1])
    e_p, e_q, n_pl = _basis(spec)
    curve0 = xy[:, :1] * e_p + xy[:, 1:] * e_q                  # origin at the head end
    T = np.cos(phi)[:, None] * e_p + np.sin(phi)[:, None] * e_q
    M = np.cross(n_pl[None, :], T)                               # in-plane normal (theta = 90 deg)
    lob = lobulation(L, spec.seed, spec.lob_amp)

    def radius(si, th):
        if spec.r_knots is not None:
            ks, kr = np.array(spec.r_knots, float).T
            base = np.interp(si, ks, kr)
        else:
            base = spec.r_head + (spec.r_tail - spec.r_head) * np.clip(si / L, 0, 1)
        i = np.clip(np.rint(si / LOB_DS).astype(int), 0, lob.shape[0] - 1)
        j = np.rint((np.mod(th, 2 * np.pi)) / (2 * np.pi) * LOB_NTH).astype(int) % LOB_NTH
        R = base + lob[i, j]
        if spec.area_step is not None:
            s_st, fac = spec.area_step
            w = 1.0 / (1.0 + np.exp(-(si - s_st) / STEP_WIDTH_MM))
            R = R * (1.0 + (np.sqrt(fac) - 1.0) * w)
        return R

    def frame_dir(i, th):
        return np.cos(th) * n_pl + np.sin(th) * M[i]

    # bump geometry (world, origin = curve0[0] for now)
    bumps = []
    for b in spec.bumps:
        i = int(np.clip(np.rint(b.s0 / DS), 0, len(s) - 1))
        d = frame_dir(i, b.theta0)
        R0 = float(radius(np.array([s[i]]), np.array([b.theta0]))[0])
        c = curve0[i] + R0 * d
        ax = np.stack([T[i], d, np.cross(T[i], d)], 0)
        semi = np.array([b.ell / 2, b.h, b.ell / 2])
        bumps.append((c, ax, semi, b.sign, d, curve0[i] + (R0 + b.sign * b.h) * d))

    # voxel grid, RAS, axis-aligned
    sp = np.asarray(spec.spacing, float)
    r_big = max(spec.r_head, max((k[1] for k in spec.r_knots), default=0.0) if spec.r_knots else 0.0)
    margin = r_big + spec.lob_amp + max([b.h for b in spec.bumps], default=0) + 8.0
    lo = curve0.min(0) - margin
    hi = curve0.max(0) + margin
    shape = np.ceil((hi - lo) / sp).astype(int) + 1
    affine = np.eye(4)
    affine[:3, :3] = np.diag(sp)
    affine[:3, 3] = lo
    ii = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3)
    pw = ii * sp + lo

    tree = cKDTree(curve0)
    rmax = r_big * max(1.0, np.sqrt(spec.area_step[1]) if spec.area_step else 1.0) + spec.lob_amp + 1.0
    dist, j = tree.query(pw, distance_upper_bound=rmax + 1.0)
    near = np.isfinite(dist)
    inside = np.zeros(len(pw), bool)
    jn = j[near]
    v = pw[near] - curve0[jn]
    tau = np.einsum("nk,nk->n", v, T[jn])
    rad = v - tau[:, None] * T[jn]
    rho = np.linalg.norm(rad, axis=1)
    th = np.arctan2(np.einsum("nk,nk->n", rad, M[jn]), np.einsum("nk,nk->n", rad, np.broadcast_to(n_pl, rad.shape)))
    R = radius(s[jn], th)
    cap = ((jn == 0) & (tau < 0)) | ((jn == len(s) - 1) & (tau > 0))
    ok = np.where(cap, np.linalg.norm(v, axis=1) <= R, rho <= R)
    inside[np.nonzero(near)[0]] = ok

    for c, ax, semi, sign, _, _ in bumps:
        q = (pw - c) @ ax.T / semi
        e = (q ** 2).sum(1) <= 1.0
        inside = inside | e if sign > 0 else inside & np.logical_not(e)

    mask = inside.reshape(tuple(shape)).astype(np.uint8)
    truth = {"bumps": [{"apex": apex, "outward": d, "s0": b.s0, "h": b.h, "ell": b.ell, "sign": b.sign}
                       for (c, ax, semi, sign, d, apex), b in zip(bumps, spec.bumps)],
             "area_step": spec.area_step, "head": curve0[0], "tail": curve0[-1]}
    return Tube(mask=mask, spacing=sp, affine=affine, curve=curve0, s=s, length=L, truth=truth)


def arc_position(tube: Tube, world_pts: np.ndarray) -> np.ndarray:
    """Phantom arc length (mm) of the nearest centre-curve point to each world point."""
    _, j = cKDTree(tube.curve).query(np.atleast_2d(world_pts))
    return tube.s[j]


# ---------------------------------------------------------------- tube families used by the tests
def gland_like(seed: int, **kw) -> TubeSpec:
    """Gently curved gland-like tube: one arc, curvature radius 80-150 mm, length 140-180 mm."""
    rng = np.random.default_rng(10_000 + seed)
    L = rng.uniform(140, 180)
    Rc = rng.uniform(80, 150)
    return TubeSpec(kappa=[(L, 1.0 / Rc)], seed=seed, tilt_deg=rng.uniform(10, 30),
                    plane_deg=rng.uniform(15, 45), **kw)


HB_STEP_U = 0.3            # head-body step centre (fraction of the phantom length)
HB_STEP_LEN_MM = 12.0      # width 30 -> 14 mm over 12 mm
HB_R_HEAD, HB_R_BODY, HB_R_TAIL = 15.0, 7.0, 6.0


def step_tube(seed: int, **kw) -> TubeSpec:
    """Head-body step (round 4): the gland_like curve and noise, radius 15 mm (width 30) in the head, dropping
    linearly to 7 mm (width 14) over 12 mm centred at 0.3 L, then tapering to 6 mm at the tail. No bulge unless
    `bumps` is given."""
    spec = gland_like(seed, **kw)
    L = spec.kappa[0][0]
    sc = HB_STEP_U * L
    spec.r_knots = ((0.0, HB_R_HEAD), (sc - HB_STEP_LEN_MM / 2, HB_R_HEAD), (sc + HB_STEP_LEN_MM / 2, HB_R_BODY),
                    (L, HB_R_TAIL))
    return spec


BEND_STRAIGHT_MM = 60.0
BEND_RADIUS_MM = 15.0
BEND_ARC_MM = BEND_RADIUS_MM * np.pi / 2


def bend_tube(seed: int, bent: bool, **kw) -> TubeSpec:
    """Straight 60 mm + (90 deg arc of radius 15 mm, or the same length straight) + straight 60 mm."""
    k = 1.0 / BEND_RADIUS_MM if bent else 0.0
    return TubeSpec(kappa=[(BEND_STRAIGHT_MM, 0.0), (BEND_ARC_MM, k), (BEND_STRAIGHT_MM, 0.0)], seed=seed, **kw)
