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
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from scipy.ndimage import label
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


@dataclass
class CutoffParams:
    # per-duct, NOT module constants: MPD and CBD have different normal calibers
    dilate_thresh_mm: float = 3.21   # MPD; CBD ~8.05 (Session 2, Youden's J)
    window_mm: tuple[float, ...] = (5.0, 8.0, 12.0, 16.0)  # multi-scale step widths
    ratio_thresh: float = 2.0
    persistence_mm: float = 15.0
    step_mm: float = 1.0         # arc-length resampling step


@dataclass
class CenterlineResult:
    status: str = "ok"           # ok | no_mask | too_small | loop_in_skeleton | ...
    n_components: int = 0
    component_sizes: list[int] = field(default_factory=list)
    n_bridged_gaps: int = 0
    gap_lengths_mm: list[float] = field(default_factory=list)
    n_branch_points: int = 0
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
    

def order_trunk(G: nx.Graph) -> list[int]:
    """shortest_path(G, a, b, weight='w') between the double_sweep endpoints."""
    a, b = double_sweep(G)
    return nx.shortest_path(G, a, b, weight='w')


def extract_centerline(mask: np.ndarray, spacing: tuple[float, float, float],
                       p: CenterlineParams = CenterlineParams()) -> CenterlineResult:
    """mask -> label -> skeleton -> graph -> bridge -> ordered, arc-resampled path."""
    raise NotImplementedError


def caliber_profile(mask: np.ndarray, centerline: CenterlineResult,
                    spacing: tuple[float, float, float], step_mm: float) -> np.ndarray:
    """2 * EDT sampled at centerline points (map_coordinates, order=1);
    NaN on bridged samples."""
    raise NotImplementedError


def detect_cutoff(caliber: np.ndarray, bridged: np.ndarray,
                  p: CutoffParams = CutoffParams(),
                  end_margin_mm: float = 5.0) -> CutoffResult:
    """Windowed up/down median step statistic over several scales, gated by
    dilate_thresh, ratio, persistence, end margins and bridges."""
    raise NotImplementedError
