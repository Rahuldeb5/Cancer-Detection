"""Ordered centerline + cutoff detection for one duct mask (MPD or CBD).

Input is a bool mask already resampled to isotropic spacing (loader.py);
nothing here touches the filesystem, so phantoms can call it directly.

Pipeline, in order (each is its own function so it can be tested alone):
    label_components -> skeletonize -> skeleton_graph -> bridge_components
    -> (later) order trunk -> resample by arc length -> caliber -> detect

Convention: s = 0 at the head/ampullary end, s increasing toward the tail.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field, replace

import networkx as nx
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import splev, splprep
from scipy.ndimage import binary_dilation, distance_transform_edt, label, map_coordinates
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize

# 13 of the 26 neighbor offsets -- one per +/- pair, so each edge is added once
HALF_OFFSETS = [o for o in itertools.product((-1, 0, 1), repeat=3) if o > (0, 0, 0)]


@dataclass
class CenterlineParams:
    max_bridge_mm: float = 8.0   # max endpoint-to-endpoint gap to bridge
    min_component_vox: int = 20  # noise floor; counted, never silently dropped
    spur_len_mm: float = 5.0     # off-trunk pieces shorter than this are spurs
    end_margin_mm: float = 5.0   # no cutoff allowed this close to either duct end
    step_mm: float = 1.0         # arc-length resampling step (extract_centerline uses this)


@dataclass
class CutoffParams:
    # per-duct, NOT module constants: MPD and CBD have different normal calibers
    dilate_thresh_mm: float = 3.21   # MPD; CBD ~8.05 (Session 2, Youden's J)
    window_mm: tuple[float, ...] = (5.0, 8.0, 12.0, 16.0)  # multi-scale step widths
    ratio_thresh: float = 2.0
    persistence_mm: float = 15.0
    step_mm: float = 1.0         # arc-length resampling step
    down_floor_mm: float = 1.0   # ratio denominator floor: <1 mm is unresolvable at 1 mm voxels
    min_window_frac: float = 0.6  # a window needs this fraction of finite (non-bridged) samples


@dataclass
class CenterlineResult:
    status: str = "ok"           # ok | no_mask | too_small | loop_in_skeleton | ...
    n_components: int = 0
    component_sizes: list[int] = field(default_factory=list)
    n_bridged_gaps: int = 0
    gap_lengths_mm: list[float] = field(default_factory=list)
    n_dropped_components: int = 0   # < min_component_vox, excluded before skeletonizing
    n_unbridged_components: int = 0  # real pieces left OUT of the trunk (gap > max_bridge_mm) -- >0 means a truncated duct
    n_branch_points: int = 0        # not computed yet -- wire up when you write Phantom A3/A4
    flipped: bool = False                   # orient_head_to_tail reversed the path
    orientation_margin_mm: float = float("nan")  # how decisively the head end was identified
    total_len_mm: float = float("nan")
    path_xyz: np.ndarray | None = None      # (n, 3) ordered iso-voxel coords, head -> tail
    bridged: np.ndarray | None = None       # (n,) bool: sample lies on a bridge edge


@dataclass
class CutoffResult:
    detected: bool = False
    score: float = float("nan")
    arc_mm: float = float("nan")
    scale_mm: float = float("nan")           # which window won
    up_caliber_mm: float = float("nan")
    down_caliber_mm: float = float("nan")
    ratio: float = float("nan")
    persistence_mm: float = float("nan")
    n_candidates: int = 0
    on_bridge: bool = False



def label_components(mask: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    labeled, n = label(mask, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labeled.ravel())[1:]

    return labeled, n, sizes


def skeleton_of(mask: np.ndarray) -> np.ndarray | None:
    if not mask.any():
        return None

    return skeletonize(mask)


def skeleton_graph(skel: np.ndarray, spacing: tuple[float, float, float]) -> nx.Graph:

    coords = np.argwhere(skel)
    index = {tuple(c): i for i, c in enumerate(coords)}

    G = nx.Graph()
    G.add_nodes_from(range(len(coords)))
    for i, c in enumerate(coords):
        for o in HALF_OFFSETS:
            j = index.get(tuple(c+o))
            if j is not None:
                G.add_edge(i, j, w=np.linalg.norm(np.array(o) * spacing), bridged=False)

    return G


def double_sweep(G: nx.Graph) -> tuple[int, int]:
    u = next(iter(G))
    d = nx.single_source_dijkstra_path_length(G, u, weight='w')
    a = max(d, key=d.get)
    d = nx.single_source_dijkstra_path_length(G, a, weight="w")
    b = max(d, key=d.get)

    return a, b


def bridge_components(G: nx.Graph, coords: np.ndarray, max_bridge_mm: float, spacing: tuple[float, float, float]) -> list[float]:
    """Join separate components at their double_sweep endpoints only (2 per
    component) if the gap <= max_bridge_mm, greedily shortest-first, never
    linking two endpoints already connected. Adds edges with bridged=True,
    w=gap. Mutates G; returns the gap lengths (mm)."""
    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return []

    endpoints = []
    node_to_comp = {}

    for comp_idx, comp_nodes in enumerate(components):
        G_comp = G.subgraph(comp_nodes)
        a, b = double_sweep(G_comp)

        for node in {a, b}:
            endpoints.append(node)
            node_to_comp[node] = comp_idx

    spacing_arr = np.array(spacing)
    ep_coords = np.array([coords[node] * spacing_arr for node in endpoints])

    dist_matrix = cdist(ep_coords, ep_coords)

    valid_pairs = []
    n_endpoints = len(endpoints)
    for i in range(n_endpoints):
        for j in range(i + 1, n_endpoints):
            u = endpoints[i]
            v = endpoints[j]
            d = dist_matrix[i, j]
            
            if node_to_comp[u] != node_to_comp[v] and d <= max_bridge_mm:
                valid_pairs.append((d, u, v))

    valid_pairs.sort(key=lambda x: x[0])

    gap_lengths_mm = []
    for d, u, v in valid_pairs:
        if not nx.has_path(G, u, v):
            G.add_edge(u, v, w=d, bridged=True)
            gap_lengths_mm.append(float(d))

    return gap_lengths_mm
    

def order_trunk(G: nx.Graph) -> tuple[list[int], float]:
    """Ordered node path between the two farthest-apart nodes of G, via
    shortest_path (edges weighted by true mm). Call this on a SINGLE connected
    component's subgraph -- extract_centerline picks that component first."""
    a, b = double_sweep(G)
    path = nx.shortest_path(G, a, b, weight="w")
    return path, nx.path_weight(G, path, weight="w")


def extract_centerline(mask: np.ndarray, spacing: tuple[float, float, float],
                       p: CenterlineParams = CenterlineParams()) -> CenterlineResult:
    """mask -> label -> skeleton -> graph -> bridge -> ordered, arc-resampled path."""
    if not mask.any():
        return CenterlineResult(status="no_mask")

    labeled, n, sizes = label_components(mask)
    keep_labels = [i + 1 for i, sz in enumerate(sizes) if sz >= p.min_component_vox]
    n_dropped = n - len(keep_labels)
    mask_filtered = np.isin(labeled, keep_labels)

    skel = skeleton_of(mask_filtered)
    if skel is None or not skel.any():
        return CenterlineResult(status="too_small", n_components=n,
                                component_sizes=list(sizes), n_dropped_components=n_dropped)

    # skeleton_graph() computes its own np.argwhere(skel) internally, and node
    # ids are indices into that array -- this recomputes it identically (same
    # skel, not mutated in between) so path/coords stay aligned below.
    coords = np.argwhere(skel)
    G = skeleton_graph(skel, spacing)

    gap_lengths = bridge_components(G, coords, p.max_bridge_mm, spacing)

    comps = list(nx.connected_components(G))

    def comp_len(nodes):
        a, b = double_sweep(G.subgraph(nodes))
        return nx.shortest_path_length(G.subgraph(nodes), a, b, weight="w")

    trunk_nodes = max(comps, key=comp_len)
    trunk = G.subgraph(trunk_nodes)

    # 26-connectivity gives 3-cycles from diagonal steps on ANY clean line --
    # only a cycle whose true physical perimeter is large is a real loop.
    for cycle in nx.cycle_basis(trunk):
        cyc_len = sum(
            trunk[cycle[i]][cycle[(i + 1) % len(cycle)]]["w"]
            for i in range(len(cycle))
        )
        if cyc_len > 10.0:
            return CenterlineResult(status="loop_in_skeleton", n_components=n,
                                    component_sizes=list(sizes), n_dropped_components=n_dropped,
                                    n_bridged_gaps=len(gap_lengths), gap_lengths_mm=gap_lengths)

    path, _ = order_trunk(trunk)
    if len(path) < 2:
        return CenterlineResult(status="too_small", n_components=n,
                                component_sizes=list(sizes), n_dropped_components=n_dropped)

    spacing_arr = np.array(spacing)
    raw_pts_mm = coords[path] * spacing_arr                          # (m, 3) mm, still jagged
    raw_bridged_edge = np.array(
        [trunk[path[i]][path[i + 1]]["bridged"] for i in range(len(path) - 1)]
    )
    raw_seg_len = np.linalg.norm(np.diff(raw_pts_mm, axis=0), axis=1)
    raw_chord = np.concatenate([[0.0], np.cumsum(raw_seg_len)])      # true arc length so far, mm
    total_len_mm = float(raw_chord[-1])

    # parameterize the spline BY raw arc length (not splprep's own guess) so
    # resampled points and the raw per-segment bridge flags share one scale
    u = raw_chord / total_len_mm
    k = min(3, len(path) - 1)
    s = len(path) * (0.5 ** 2)   # ~0.5mm avg deviation tolerance -- verify this in your phantom test
    tck, _ = splprep(raw_pts_mm.T, u=u, s=s, k=k)

    u_dense = np.linspace(0, 1, max(2000, len(path) * 4))
    dense = np.array(splev(u_dense, tck)).T
    dense_arc = u_dense * total_len_mm

    arc_targets = np.arange(0.0, total_len_mm, p.step_mm)
    resampled_mm = np.column_stack(
        [np.interp(arc_targets, dense_arc, dense[:, ax]) for ax in range(3)]
    )
    path_xyz = resampled_mm / spacing_arr    # back to iso-voxel index units, for map_coordinates later

    # bridged flag per resampled sample: which raw segment did this arc
    # position fall into, and was that segment a bridge edge
    seg_idx = np.clip(np.searchsorted(raw_chord, arc_targets, side="right") - 1,
                      0, len(raw_bridged_edge) - 1)
    bridged = raw_bridged_edge[seg_idx]

    return CenterlineResult(
        status="ok",
        n_components=n,
        component_sizes=list(sizes),
        n_dropped_components=n_dropped,
        n_unbridged_components=len(comps) - 1,
        n_bridged_gaps=len(gap_lengths),
        gap_lengths_mm=gap_lengths,
        total_len_mm=total_len_mm,
        path_xyz=path_xyz,
        bridged=bridged,
    )


# ---- orientation: s = 0 at the head/ampullary end ---------------------------

def iso_to_world_mm(pts_iso: np.ndarray, crop_offset: tuple, spacing_iso: tuple,
                    native_spacing: tuple, affine: np.ndarray) -> np.ndarray:
    """Points in a loader.py isotropic crop (index units) -> world mm.
    Inverse of what resample_mask_to_isotropic did: output index o was read
    from crop index o * spacing_iso / native_spacing, and the crop starts at
    crop_offset in the native array."""
    native_idx = np.asarray(crop_offset) + np.asarray(pts_iso) * (
        np.asarray(spacing_iso) / np.asarray(native_spacing)
    )
    return native_idx @ affine[:3, :3].T + affine[:3, 3]


def mask_centroid_mm(mask: np.ndarray, affine: np.ndarray) -> np.ndarray | None:
    """Centroid of a NATIVE-grid bool mask in world mm; None if empty."""
    if not mask.any():
        return None
    return np.argwhere(mask).mean(axis=0) @ affine[:3, :3].T + affine[:3, 3]


def orient_head_to_tail(res: CenterlineResult, crop_offset: tuple, spacing_iso: tuple,
                        native_spacing: tuple, affine: np.ndarray,
                        head_centroid_mm: np.ndarray,
                        tail_centroid_mm: np.ndarray | None = None,
                        min_margin_mm: float = 10.0) -> CenterlineResult:
    """Reverse path_xyz/bridged if needed so index 0 is the head/ampullary end.

    Primary signal: the end nearer the pancreas-head centroid is s = 0. That
    holds for the MPD (head end vs tail end) and the CBD (ampullary end sits in
    the head; the other end is up in the hepatoduodenal ligament).
    Second signal, MPD only: projection onto the head->tail centroid axis. Pass
    tail_centroid_mm=None for the CBD -- it runs mostly perpendicular to that
    axis, so the projection is meaningless there.

    Status becomes "orient_ambiguous" (path still oriented by the primary
    signal, so diagnostics stay sensible) if the two signals disagree or the
    two ends are within min_margin_mm of equally far from the head centroid.
    Callers should exclude non-"ok" cases and count them.
    """
    if res.status != "ok" or res.path_xyz is None:
        return res

    ends = iso_to_world_mm(res.path_xyz[[0, -1]], crop_offset, spacing_iso, native_spacing, affine)
    d = np.linalg.norm(ends - head_centroid_mm, axis=1)
    flip = bool(d[0] > d[1])
    margin = float(abs(d[0] - d[1]))

    status = res.status
    if margin < min_margin_mm:
        status = "orient_ambiguous"
    if tail_centroid_mm is not None:
        axis = tail_centroid_mm - head_centroid_mm
        proj = (ends - head_centroid_mm) @ axis / np.linalg.norm(axis)
        if bool(proj[0] > proj[1]) != flip:
            status = "orient_ambiguous"

    return replace(
        res,
        status=status,
        path_xyz=res.path_xyz[::-1].copy() if flip else res.path_xyz,
        bridged=res.bridged[::-1].copy() if flip else res.bridged,
        flipped=flip,
        orientation_margin_mm=margin,
    )


def caliber_profile(mask: np.ndarray, centerline: CenterlineResult,
                    spacing: tuple[float, float, float], step_mm: float,
                    edt_offset_mm: float = 0.0, bridge_pad_mm: float = 3.0) -> np.ndarray:
    """Caliber (mm) at each centerline sample; sample i is at arc position i * step_mm.

    `mask` must be the exact isotropic array given to extract_centerline (path_xyz
    is in its index units). Returns an empty array unless centerline.status is
    "ok", so unoriented / failed centerlines never reach detect_cutoff.

    edt_offset_mm is subtracted from 2*EDT. Default 0: on phantoms plain 2*EDT is
    already near the true diameter, and Session 2's half-voxel correction
    over-subtracts ~1 mm. Calibrate it on your own phantoms.

    Samples on a bridged span, plus bridge_pad_mm either side, are NaN -- EDT
    drops near a cut face and there is no real duct on the bridge itself.
    """
    if centerline.status != "ok" or centerline.path_xyz is None:
        return np.array([])

    edt = distance_transform_edt(mask, sampling=spacing)
    radius = map_coordinates(edt, centerline.path_xyz.T, order=1, mode="nearest")
    caliber = 2 * radius - edt_offset_mm

    pad = int(round(bridge_pad_mm / step_mm))
    caliber[binary_dilation(centerline.bridged, structure=np.ones(2 * pad + 1, bool))] = np.nan
    return caliber


def _window_medians(x: np.ndarray, w: int, min_frac: float) -> np.ndarray:
    """med[k] = nanmedian(x[k-w : k]) for k = 0 .. len(x)+w  (length n+w+1).
    NaN where fewer than min_frac*w of the window's samples are finite, which
    also NaNs windows hanging off either end of the array."""
    pad = np.concatenate([np.full(w, np.nan), x, np.full(w, np.nan)])
    win = sliding_window_view(pad, w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slice
        med = np.nanmedian(win, axis=1)
    med[np.isfinite(win).sum(axis=1) < min_frac * w] = np.nan
    return med


def detect_cutoff(caliber: np.ndarray, bridged: np.ndarray,
                  p: CutoffParams = CutoffParams(),
                  end_margin_mm: float = 5.0) -> CutoffResult:
    """Abrupt-narrowing cutoff on a head->tail oriented caliber profile.

    Flow is tail -> head, so UPSTREAM is the tail side (larger s). At candidate
    index i, up = median(caliber[i+1 : i+1+w]) (tail side), down = median(
    caliber[i-w : i]) (head side). A cutoff is dilated-then-narrow going toward
    the head: up >= dilate_thresh, up/down >= ratio_thresh, and the dilated
    condition holds continuously for >= persistence_mm past i. Tried at every
    window in p.window_mm; the best (highest log2 ratio) passing scale wins per
    index. NaN (bridged / unmeasured) samples never count as narrow or dilated.

    Returns detected=False (all else NaN) for an empty profile -- callers should
    check the centerline status to tell "no cutoff" from "could not evaluate".
    """
    n = len(caliber)
    if n == 0:
        return CutoffResult()
    step = p.step_mm

    margin = int(np.ceil(end_margin_mm / step))
    valid = np.zeros(n, bool)
    valid[margin:n - margin] = True
    valid &= np.isfinite(caliber)

    # dilated = 5 mm-smoothed caliber over threshold; NaN -> not dilated (breaks the run)
    smooth = _window_medians(caliber, 5, p.min_window_frac)[3:3 + n]
    with np.errstate(invalid="ignore"):
        dilated = smooth >= p.dilate_thresh_mm
    run = np.zeros(n + 1, int)
    for i in range(n - 1, -1, -1):
        run[i] = run[i + 1] + 1 if dilated[i] else 0
    persist = run[1:] * step          # persist[i]: dilated stretch just past i, mm

    best = np.full(n, -np.inf)
    best_w = np.zeros(n, int)
    best_up = np.full(n, np.nan)
    best_down = np.full(n, np.nan)
    best_ratio = np.full(n, np.nan)
    for w_mm in p.window_mm:
        w = max(2, int(round(w_mm / step)))
        med = _window_medians(caliber, w, p.min_window_frac)
        down, up = med[:n], med[w + 1:w + 1 + n]
        ratio = up / np.maximum(down, p.down_floor_mm)
        with np.errstate(invalid="ignore"):
            ok = valid & (up >= p.dilate_thresh_mm) & (ratio >= p.ratio_thresh) & (persist >= p.persistence_mm)
        score = np.where(ok, np.log2(np.where(ok, ratio, 1.0)), -np.inf)
        better = score > best
        best[better], best_w[better] = score[better], w
        best_up[better], best_down[better], best_ratio[better] = up[better], down[better], ratio[better]

    cand = np.flatnonzero(np.isfinite(best))
    if cand.size == 0:
        return CutoffResult(detected=False)

    i = int(np.argmax(best))
    w = best_w[i]
    gap = int(round(5.0 / step))       # candidates > 5 mm apart are separate peaks
    return CutoffResult(
        detected=True,
        score=float(best[i]),
        arc_mm=float(i * step),
        scale_mm=float(w * step),
        up_caliber_mm=float(best_up[i]),
        down_caliber_mm=float(best_down[i]),
        ratio=float(best_ratio[i]),
        persistence_mm=float(persist[i]),
        n_candidates=int(1 + np.sum(np.diff(cand) > gap)),
        on_bridge=bool(np.asarray(bridged)[max(0, i - w):min(n, i + w + 1)].any()),
    )
