"""Atlas grids and the voxel-centre splat.

Splat = nearest-bin deposit: every foreground voxel centre adds its full voxel volume (mm^3) to
the one bin containing it. No trilinear spreading, so the mass in the grid is exactly the mass of
the voxels whose centres fall inside it; what falls outside is returned so conservation can be
checked (sum(grid) + outside == n_voxels * voxel_volume).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AXES = ("u", "ap", "si")


@dataclass(frozen=True)
class Grid:
    name: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    step: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(round((h - l) / self.step)) for l, h in zip(self.lo, self.hi))

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def edges(self, axis: int) -> np.ndarray:
        return self.lo[axis] + self.step * np.arange(self.shape[axis] + 1)

    def centers(self, axis: int) -> np.ndarray:
        e = self.edges(axis)
        return 0.5 * (e[:-1] + e[1:])

    def bin_index(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(flat indices of in-grid points, boolean in-grid mask) for (N,3) coords."""
        coords = np.asarray(coords, dtype=np.float64)
        ijk = np.floor((coords - np.asarray(self.lo)) / self.step).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < np.asarray(self.shape)), axis=1)
        return np.ravel_multi_index(tuple(ijk[ok].T), self.shape), ok

    def zeros(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=np.float64)


NORM_GRID = Grid("norm", lo=(-0.15, -0.4, -0.4), hi=(1.15, 0.4, 0.4), step=0.01)
MM_GRID = Grid("mm", lo=(-40.0, -80.0, -80.0), hi=(280.0, 80.0, 80.0), step=1.0)
GRIDS = (NORM_GRID, MM_GRID)


def splat(acc: np.ndarray, grid: Grid, coords: np.ndarray, weight: float) -> float:
    """Add `weight` (mm^3) per point into acc (grid.shape) in place; return the mass outside the grid."""
    if len(coords) == 0:
        return 0.0
    flat, ok = grid.bin_index(coords)
    if flat.size:
        idx, counts = np.unique(flat, return_counts=True)
        acc.reshape(-1)[idx] += counts * float(weight)
    return float((~ok).sum()) * float(weight)
