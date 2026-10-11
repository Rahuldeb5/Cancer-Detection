"""Synthetic glands for the oracle tests.

An ellipsoid (semi-axes a > b > c mm) defined in its own "gland" coordinates g, with
head = g_x < -a/3, body = middle third, tail = g_x > a/3, and a duct = cylinder of radius r around
the g_x axis (|g_x| <= duct_frac * a). The gland is placed in the world with a random 3D rotation
Q and translation t, and voxelized on a grid whose affine has anisotropic spacing with randomly
permuted and flipped voxel axes - the layouts real PanTS affines come in.

World point of voxel index i:  p = A i + o;  gland coords:  g = Q^T (p - t).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Phantom:
    coords: dict[str, np.ndarray]   # head/body/tail/duct -> (N,3) voxel indices
    affine: np.ndarray
    shape: tuple[int, int, int]
    Q: np.ndarray                   # gland -> world rotation; Q[:,0] is the head->tail axis
    t: np.ndarray


def random_affine(rng, spacing=(0.8, 0.8, 5.0)) -> np.ndarray:
    """3x3 linear part: voxel axes = a random permutation of world axes, random signs."""
    perm = rng.permutation(3)
    signs = rng.choice([-1.0, 1.0], size=3)
    A = np.zeros((3, 3))
    for vox_axis, world_axis in enumerate(perm):
        A[world_axis, vox_axis] = signs[vox_axis] * spacing[vox_axis]
    return A


def make_phantom(rng, semi_axes=(75.0, 15.0, 10.0), duct_r=2.0, duct_frac=0.9,
                 spacing=(0.8, 0.8, 5.0), rotation: np.ndarray | None = None) -> Phantom:
    a, b, c = semi_axes
    Q = Rotation.random(random_state=rng).as_matrix() if rotation is None else rotation
    t = rng.uniform(-200, 200, size=3)
    A = random_affine(rng, spacing)

    # index-space bbox of the gland's world bbox, padded by 2 voxels
    ext = np.sqrt(((Q * np.array([a, b, c])) ** 2).sum(axis=1))
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * ext + t
    idx = np.linalg.solve(A, corners.T).T
    imin = np.floor(idx.min(axis=0)) - 2
    shape = tuple((np.ceil(idx.max(axis=0)) + 2 - imin + 1).astype(int))
    o = A @ imin                     # world point of new index 0 (old index imin)
    affine = np.eye(4)
    affine[:3, :3], affine[:3, 3] = A, o

    ii = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), axis=-1).reshape(-1, 3)
    g = (ii @ A.T + o - t) @ Q
    inside = ((g / np.array([a, b, c])) ** 2).sum(axis=1) <= 1.0
    head = inside & (g[:, 0] < -a / 3)
    tail = inside & (g[:, 0] > a / 3)
    body = inside & ~head & ~tail
    duct = (np.hypot(g[:, 1], g[:, 2]) <= duct_r) & (np.abs(g[:, 0]) <= duct_frac * a)
    coords = {k: ii[m].astype(np.int32) for k, m in (("head", head), ("body", body), ("tail", tail), ("duct", duct))}
    return Phantom(coords=coords, affine=affine, shape=shape, Q=Q, t=t)
