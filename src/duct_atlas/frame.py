"""Per-case gland frame in world mm (RAS+).

  c   = centroid of the envelope voxel centres (envelope = head | body | tail; never the duct,
        so the frame is independent of what the atlas plots)
  e1  = PC1 of the envelope voxel centres, sign fixed so (tail centroid - head centroid) . e1 > 0
  e2  = world +y (anterior) with its e1 component removed
  e3  = world +z (superior) with its e1 and e2 components removed
PCA axes 2/3 are not used: they are not anatomical in ~35% of PanTS glands (carried over).

Coordinates of a world point p:
  normalized  u = ((p-c).e1 - p1) / L,  ap = (p-c).e2 / L,  si = (p-c).e3 / L
  physical    u_mm = (p-c).e1 - p1,     ap_mm = (p-c).e2,   si_mm = (p-c).e3
with p1, p99 the 1st/99th percentiles of the envelope's e1 projections and L = p99 - p1
(same convention as src/lesion-sectioning/pancreas_map.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WORLD_X = np.array([1.0, 0.0, 0.0])
WORLD_Y = np.array([0.0, 1.0, 0.0])  # RAS+: +y anterior
WORLD_Z = np.array([0.0, 0.0, 1.0])  # RAS+: +z superior
PCTILE_CLIP = (1.0, 99.0)


@dataclass(frozen=True)
class Frame:
    c: np.ndarray        # (3,) envelope centroid, world mm
    e1: np.ndarray       # (3,) head -> tail
    e2: np.ndarray       # (3,) anterior, perpendicular to e1
    e3: np.ndarray       # (3,) superior, perpendicular to e1 and e2
    p1: float            # 1st percentile of envelope projections on e1 (mm, relative to c)
    p99: float
    var_share: float     # PC1 eigenvalue / sum of eigenvalues
    eig_flipped: bool    # eigh's PC1 had to be negated by the head->tail rule (eigh's sign is arbitrary)
    leftward: bool       # e1 points toward patient left (-x in RAS+), the usual head->tail direction

    @property
    def L(self) -> float:
        return float(self.p99 - self.p1)

    def rotate(self, world_mm: np.ndarray) -> np.ndarray:
        """(N,3) world mm -> (N,3) [proj_e1, ap_mm, si_mm] relative to c (no scaling)."""
        rel = np.asarray(world_mm, dtype=np.float64) - self.c
        return rel @ np.stack([self.e1, self.e2, self.e3], axis=1)

    def coords_mm(self, world_mm: np.ndarray) -> np.ndarray:
        """(N,3) [u_mm, ap_mm, si_mm]."""
        r = self.rotate(world_mm)
        r[:, 0] -= self.p1
        return r

    def coords_norm(self, world_mm: np.ndarray) -> np.ndarray:
        """(N,3) [u, ap, si] in gland lengths."""
        return self.coords_mm(world_mm) / self.L


def voxel_to_world(idx: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """(N,3) voxel indices (array order) -> (N,3) world mm via the 4x4 affine."""
    idx = np.asarray(idx, dtype=np.float64)
    return idx @ affine[:3, :3].T + affine[:3, 3]


def _orthonormal(v: np.ndarray, *against: np.ndarray) -> np.ndarray:
    for a in against:
        v = v - (v @ a) * a
    n = np.linalg.norm(v)
    if n < 1e-6:
        raise ValueError("world axis is parallel to PC1; anatomical axis undefined")
    return v / n


def build_frame(env_mm: np.ndarray, head_centroid: np.ndarray, tail_centroid: np.ndarray) -> Frame:
    """Frame from the envelope voxel centres (N,3 world mm) and the head/tail mask centroids."""
    env_mm = np.asarray(env_mm, dtype=np.float64)
    c = env_mm.mean(axis=0)
    rel = env_mm - c
    cov = rel.T @ rel / max(len(rel) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)  # ascending
    e1 = evecs[:, -1].copy()
    var_share = float(evals[-1] / evals.sum()) if evals.sum() > 0 else float("nan")

    flipped = bool(e1 @ (np.asarray(tail_centroid) - np.asarray(head_centroid)) < 0)
    if flipped:
        e1 = -e1
    e2 = _orthonormal(WORLD_Y, e1)
    e3 = _orthonormal(WORLD_Z, e1, e2)

    proj = rel @ e1
    p1, p99 = np.percentile(proj, PCTILE_CLIP)
    return Frame(c=c, e1=e1, e2=e2, e3=e3, p1=float(p1), p99=float(p99), var_share=var_share,
                 eig_flipped=flipped, leftward=bool(e1 @ -WORLD_X > 0))
