"""Scale-normalized LoG blob detection: pure array functions, no I/O.

phantom.py and case_pass.py call exactly these functions, so the phantom exercises the
code the cohort run uses. Arrays are in nibabel (x, y, z) order; `spacing` is mm per axis
(the cohort run always passes a 1 mm isotropic grid, so sigma in mm == sigma in voxels).

Detector (fixed a priori, before any evaluation lesion was looked at)
  response   R_s = s^2 * laplacian(G_s * f), the scale-normalized LoG. A dark (hypodense)
             blob is a Laplacian *maximum*, a bright one a minimum, so the two polarities
             are +R and -R. The s^2 factor is what makes the peak response of a ball of
             radius r land at s = r/sqrt(3) (checked in phantom.py).
  noise      per (case, scale): 1.4826 * MAD of R_s over the search region. Dividing by it
             gives z, which is what ranks peaks *across* scales: s^2-normalization equalizes
             the signal of an ideal ball, not the noise, which falls off with s.
  peaks      3D local maxima of z (26-neighbourhood) inside the search region, greedily
             thinned to a 5 mm minimum separation, top K per scale; then all scales pooled
             and thinned again to give the case's scale-space candidate list.
  shape      Hessian eigenvalues of the same scale-normalized smoothed image at the peak,
             sign-flipped for bright polarity so "blob-like" always means all three > 0, and
             sorted |l1| <= |l2| <= |l3|. blobness = |l1|/|l3| is ~1 for a ball and ~0 for a
             tube (whose Hessian has one near-zero eigenvalue along the axis).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

SIGMAS_MM = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
HU_CLIP = (-100.0, 300.0)      # clip only; no z-score (per-case intensity normalization would
                               # erase the HU contrast the detector is supposed to see)
MIN_SEP_MM = 5.0
TOP_K = 50
POLARITIES = ("dark", "bright")
POLARITY_SIGN = {"dark": 1.0, "bright": -1.0}
TRUNCATE = 4.0                 # gaussian_filter kernel radius = TRUNCATE * sigma
MAD_TO_SIGMA = 1.4826
MAX_CAND = 500_000             # cap on local maxima fed to the greedy thinning
EPS = 1e-12


def clip_hu(ct: np.ndarray) -> np.ndarray:
    """HU clipped to HU_CLIP as float32. Clipping bounds air/bone so a single rib or gas
    bubble cannot dominate the MAD or produce the top blob response."""
    return np.clip(np.asarray(ct, dtype=np.float32), *HU_CLIP)


def _deriv(ct: np.ndarray, sigma_vox, order) -> np.ndarray:
    """Gaussian derivative of `ct`. mode="nearest" matches resample_ct_to_isotropic's edge
    handling, so the crop border adds no fake step."""
    return gaussian_filter(ct, sigma_vox, order=order, mode="nearest", truncate=TRUNCATE)


def robust_scale(values: np.ndarray) -> float:
    """1.4826 * MAD. NaN for an empty input, EPS-floored so a constant map cannot divide by 0."""
    if values.size == 0:
        return np.nan
    mad = float(np.median(np.abs(values - np.median(values))))
    return max(MAD_TO_SIGMA * mad, EPS)


def ball_offsets(radius_mm: float, spacing) -> np.ndarray:
    """(N, 3) integer voxel offsets whose voxel centres lie within radius_mm."""
    sp = np.asarray(spacing, dtype=np.float64)
    rad = np.ceil(radius_mm / sp).astype(int)
    g = np.stack(np.meshgrid(*[np.arange(-r, r + 1) for r in rad], indexing="ij"), -1)
    d2 = ((g * sp) ** 2).sum(-1)
    return g[d2 <= radius_mm ** 2].reshape(-1, 3)


def _paint(blocked: np.ndarray, p, offs: np.ndarray) -> None:
    q = offs + np.asarray(p)
    ok = np.all((q >= 0) & (q < np.asarray(blocked.shape)), axis=1)
    q = q[ok]
    blocked[q[:, 0], q[:, 1], q[:, 2]] = True


def local_maxima_nms(resp: np.ndarray, mask: np.ndarray, spacing, min_sep_mm: float = MIN_SEP_MM,
                     top_k: int = TOP_K, max_cand: int = MAX_CAND):
    """Top-`top_k` local maxima of `resp` inside `mask`, at least min_sep_mm apart.

    A candidate is a voxel that is >= all 26 neighbours (maximum_filter size 3). Candidates
    are then taken in descending order, each accepted one blocking a min_sep_mm ball, which is
    the standard greedy non-maximum suppression: it keeps the strongest peak of every cluster.

    ">=" means an exactly flat region counts as a plateau of maxima, so a case with a constant
    block (e.g. gas clipped to the HU floor) can have its list padded with zero-response peaks.
    They sort last and fall far below any usable z threshold, so they are left in rather than
    special-cased; every case therefore returns exactly top_k peaks.
    Returns (coords (N, 3) int array, values (N,)) sorted by value descending.
    """
    m = mask.astype(bool, copy=False)
    if not m.any():
        return np.zeros((0, 3), int), np.zeros(0, np.float32)
    mx = maximum_filter(resp, size=3, mode="nearest")
    cand = m & (resp >= mx)
    del mx
    flat = np.flatnonzero(cand)
    if flat.size == 0:
        return np.zeros((0, 3), int), np.zeros(0, np.float32)
    vals = resp.ravel()[flat]
    order = np.argsort(-vals, kind="stable")[:max_cand]
    coords_all = np.stack(np.unravel_index(flat, resp.shape), axis=1)
    blocked = np.zeros(resp.shape, bool)
    offs = ball_offsets(min_sep_mm, spacing)
    keep = []
    for k in order:
        p = coords_all[k]
        if blocked[p[0], p[1], p[2]]:
            continue
        keep.append(k)
        if len(keep) >= top_k:
            break
        _paint(blocked, p, offs)
    keep = np.asarray(keep, dtype=int)
    return coords_all[keep], vals[keep]


def nms_points(coords_mm: np.ndarray, scores: np.ndarray, min_sep_mm: float = MIN_SEP_MM,
               top_k: int = TOP_K) -> np.ndarray:
    """Indices kept by greedy min-separation thinning of an already-small point set
    (used to pool the per-scale peak lists into one scale-space candidate list)."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    kept: list[int] = []
    for i in order:
        p = coords_mm[i]
        if any(float(np.linalg.norm(p - coords_mm[j])) < min_sep_mm for j in kept):
            continue
        kept.append(int(i))
        if len(kept) >= top_k:
            break
    return np.asarray(kept, dtype=int)


def shape_terms(hess: np.ndarray, sign: float) -> dict[str, np.ndarray]:
    """Eigenvalue shape terms for stacked symmetric Hessians `hess` (N, 3, 3).

    `sign` is +1 for dark and -1 for bright, so a blob of the detector's own polarity has all
    three eigenvalues positive whichever polarity it is. Eigenvalues are then ordered by
    magnitude, |l1| <= |l2| <= |l3|:
      blobness = |l1|/|l3|   1 = isotropic (ball), 0 = tube or sheet   <- the blob-vs-tube term
      Ra       = |l2|/|l3|   1 = tube, 0 = sheet                       (Frangi's R_A)
      Rb       = |l1|/sqrt(|l2 l3|)  Frangi's R_B, 1 = ball
      frob     = sqrt(sum l^2), the response magnitude
      n_pos    = how many of the three have the blob-expected sign (3 = fully blob-like)
    """
    if len(hess) == 0:
        z = np.zeros(0)
        return {k: z.copy() for k in ("lam1", "lam2", "lam3", "blobness", "Ra", "Rb", "frob", "n_pos")}
    ev = np.linalg.eigvalsh(sign * np.asarray(hess, dtype=np.float64))   # ascending by value
    idx = np.argsort(np.abs(ev), axis=1, kind="stable")
    ev = np.take_along_axis(ev, idx, axis=1)
    a = np.abs(ev)
    return {"lam1": ev[:, 0], "lam2": ev[:, 1], "lam3": ev[:, 2],
            "blobness": a[:, 0] / np.maximum(a[:, 2], EPS),
            "Ra": a[:, 1] / np.maximum(a[:, 2], EPS),
            "Rb": a[:, 0] / np.maximum(np.sqrt(a[:, 1] * a[:, 2]), EPS),
            "frob": np.sqrt((ev ** 2).sum(1)),
            "n_pos": (ev > 0).sum(1)}


def detect(ct: np.ndarray, mask: np.ndarray, spacing=(1.0, 1.0, 1.0), sigmas=SIGMAS_MM,
           min_sep_mm: float = MIN_SEP_MM, top_k: int = TOP_K, polarities=POLARITIES,
           probe_idx: dict[str, np.ndarray] | None = None) -> dict:
    """Run the whole detector on one already-cropped, already-isotropic CT block.

    `ct` must be HU (clip_hu is applied here), `mask` is the search region: peaks are only
    taken inside it and the noise MAD is measured over it. `probe_idx` maps a name to flat
    voxel indices (e.g. one lesion's 2 mm dilation) whose per-scale max z is recorded; that
    is evaluation-only bookkeeping and never influences the detection.

    Returns dict with
      scale_peaks  one record per (polarity, sigma) peak
      peaks        the pooled scale-space candidate list, per polarity, rank 1..top_k
      smax         {polarity: float32 max-over-scales z map}
      argscale     {polarity: uint8 index into `sigmas` of the winning scale}
      noise        {sigma_mm: MAD-based scale of R_s over the mask}
      probe        {name: {polarity: [max z per scale]}}
    """
    f = clip_hu(ct)
    sp = np.asarray(spacing, dtype=np.float64)
    m = mask.astype(bool, copy=False)
    smax = {p: np.full(f.shape, -np.inf, np.float32) for p in polarities}
    argscale = {p: np.zeros(f.shape, np.uint8) for p in polarities}
    noise: dict[float, float] = {}
    scale_peaks: list[dict] = []
    probe_idx = probe_idx or {}
    probe = {name: {p: [] for p in polarities} for name in probe_idx}

    for si, s_mm in enumerate(sigmas):
        sv = s_mm / sp
        hxx = _deriv(f, sv, (2, 0, 0))
        hyy = _deriv(f, sv, (0, 2, 0))
        hzz = _deriv(f, sv, (0, 0, 2))
        z = hxx + hyy
        z += hzz
        z *= s_mm ** 2                                  # scale-normalized LoG, in place
        nz = robust_scale(z[m])
        noise[float(s_mm)] = nz
        z /= nz                                         # z-units: response / per-case noise
        picks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for pol in polarities:
            r = z if POLARITY_SIGN[pol] > 0 else -z
            upd = r > smax[pol]
            smax[pol][upd] = r[upd]
            argscale[pol][upd] = si
            del upd
            picks[pol] = local_maxima_nms(r, m, sp, min_sep_mm, top_k)
            for name, idx in probe_idx.items():
                probe[name][pol].append(float(r.ravel()[idx].max()) if idx.size else np.nan)
            if POLARITY_SIGN[pol] < 0:
                del r
        del z
        # Hessian at this scale's peaks: sample the diagonals already in memory, free them,
        # then compute the three off-diagonals one at a time.
        cc = {pol: picks[pol][0] for pol in polarities}
        comp = {}
        for key, vol in (("xx", hxx), ("yy", hyy), ("zz", hzz)):
            comp[key] = {pol: vol[tuple(cc[pol].T)] * s_mm ** 2 for pol in polarities}
        del hxx, hyy, hzz
        for key, order in (("xy", (1, 1, 0)), ("xz", (1, 0, 1)), ("yz", (0, 1, 1))):
            vol = _deriv(f, sv, order)
            comp[key] = {pol: vol[tuple(cc[pol].T)] * s_mm ** 2 for pol in polarities}
            del vol
        for pol in polarities:
            coords, vals = picks[pol]
            n = len(coords)
            H = np.empty((n, 3, 3), np.float64)
            H[:, 0, 0], H[:, 1, 1], H[:, 2, 2] = comp["xx"][pol], comp["yy"][pol], comp["zz"][pol]
            H[:, 0, 1] = H[:, 1, 0] = comp["xy"][pol]
            H[:, 0, 2] = H[:, 2, 0] = comp["xz"][pol]
            H[:, 1, 2] = H[:, 2, 1] = comp["yz"][pol]
            st = shape_terms(H, POLARITY_SIGN[pol])
            for i in range(n):
                scale_peaks.append(dict(polarity=pol, sigma_mm=float(s_mm), scale_rank=i + 1,
                                        z=float(vals[i]), score_raw=float(vals[i] * nz),
                                        noise_mad=nz, hu=float(f[tuple(coords[i])]),
                                        ix=int(coords[i][0]), iy=int(coords[i][1]), iz=int(coords[i][2]),
                                        **{k: float(v[i]) if k != "n_pos" else int(v[i])
                                           for k, v in st.items()}))

    peaks: list[dict] = []
    for pol in polarities:
        sub = [p for p in scale_peaks if p["polarity"] == pol]
        if sub:
            cm = np.array([[p["ix"], p["iy"], p["iz"]] for p in sub], float) * sp
            kept = nms_points(cm, np.array([p["z"] for p in sub]), min_sep_mm, top_k)
            for rank, k in enumerate(kept, 1):
                peaks.append({**sub[k], "rank": rank})
    return dict(scale_peaks=scale_peaks, peaks=peaks, smax=smax, argscale=argscale,
                noise=noise, probe=probe, sigmas=tuple(float(s) for s in sigmas))
