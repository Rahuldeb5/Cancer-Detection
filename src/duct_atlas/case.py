"""One case: masks -> voxel coordinates -> gland frame -> splat into the atlas accumulators.

Masks are held as (N,3) voxel-index arrays, not dense volumes: each mask is loaded with
tumorlib.io.load_mask (two on-disk encodings; never `raw != 0`), np.nonzero'd and freed, so peak
memory is one full-size mask (~2 bytes/voxel inside load_mask), not four. The only dense array
built afterwards is a bit-flag volume over the bounding box of all four masks (the crop), used to
form the unions without double counting voxels that sit in two masks (e.g. duct inside head).

All voxel -> world conversion uses the pancreas_head mask's affine for every mask, including the
duct: a few duct masks carry corrupted affines on an otherwise aligned voxel grid (CONTEXT gotcha 4).
Shapes of all four masks must agree, else the case is skipped and listed.
"""
from __future__ import annotations

import contextlib
import io as _stdio

import nibabel as nib
import numpy as np

from duct_atlas.frame import Frame, build_frame, voxel_to_world
from duct_atlas.grid import GRIDS, splat
from tumorlib import io

STRUCTS = {"head": "pancreas_head", "body": "pancreas_body", "tail": "pancreas_tail", "duct": "pancreatic_duct"}
BITS = {"head": 1, "body": 2, "tail": 4, "duct": 8}
ENV_BITS = BITS["head"] | BITS["body"] | BITS["tail"]
ACC_KEYS = ("duct", "tissue", "head", "body", "tail")

MIN_DUCT_MM3 = 8.0
ENV_CM3_RANGE = (15.0, 300.0)
MIN_VAR_SHARE = 0.5


def new_accumulators() -> dict[str, dict[str, np.ndarray]]:
    return {g.name: {k: g.zeros() for k in ACC_KEYS} for g in GRIDS}


# ------------------------------------------------------------------ core (no disk access)
def _section_bounds(u: dict[str, np.ndarray]) -> tuple[float, float]:
    """Per-case head|body and body|tail boundaries on u: midpoint between adjacent sections'
    5th/95th percentiles (same rule as src/lesion-sectioning/duct_position.py)."""
    if any(len(u[s]) == 0 for s in ("head", "body", "tail")):
        return float("nan"), float("nan")
    ext = {s: np.percentile(u[s], [5, 95]) for s in ("head", "body", "tail")}
    return 0.5 * (ext["head"][1] + ext["body"][0]), 0.5 * (ext["body"][1] + ext["tail"][0])


def analyse(coords: dict[str, np.ndarray], affine: np.ndarray, acc=None, keep_points: bool = False):
    """Frame + atlas contribution of one case.

    coords: {"head","body","tail","duct"} -> (N,3) voxel indices, all on the grid of `affine`.
    acc:    accumulators from new_accumulators(), updated in place only if the case is included.
    Returns (row: dict, points: dict | None). points (if keep_points) holds normalized coords of
    tissue/duct voxels and a section code per tissue voxel, for QC plots.
    """
    vox = float(abs(np.linalg.det(affine[:3, :3])))
    n = {k: len(coords[k]) for k in STRUCTS}
    row = {"voxel_mm3": vox, "duct_mm3": n["duct"] * vox,
           **{f"{k}_mm3": n[k] * vox for k in ("head", "body", "tail")}}
    reasons = []
    if row["duct_mm3"] < MIN_DUCT_MM3:
        reasons.append("no_duct")
    for s in ("head", "tail"):
        if n[s] == 0:
            reasons.append(f"empty_{s}")
    if reasons and ("empty_head" in reasons or "empty_tail" in reasons):
        row["status"] = ";".join(reasons)
        return row, None

    # bit-flag crop over the bbox of all four masks -> unions without double counting
    allc = np.concatenate([coords[k] for k in STRUCTS if n[k]], axis=0)
    lo = allc.min(axis=0)
    shape = tuple(allc.max(axis=0) - lo + 1)
    del allc
    flags = np.zeros(shape, dtype=np.uint8)
    for k, b in BITS.items():
        if n[k]:
            c = coords[k] - lo
            flags[c[:, 0], c[:, 1], c[:, 2]] |= b
    env_idx = np.argwhere(flags & ENV_BITS) + lo
    tissue_idx = np.argwhere(flags) + lo
    duct_in_env = int(np.count_nonzero((flags & BITS["duct"]) & ((flags & ENV_BITS) > 0)))
    del flags

    env_cm3 = len(env_idx) * vox / 1000.0
    row["env_cm3"] = env_cm3
    row["tissue_mm3"] = len(tissue_idx) * vox
    row["duct_in_env_frac"] = duct_in_env / n["duct"] if n["duct"] else float("nan")
    if not ENV_CM3_RANGE[0] <= env_cm3 <= ENV_CM3_RANGE[1]:
        reasons.append("env_volume")

    head_c = voxel_to_world(coords["head"], affine).mean(axis=0)
    tail_c = voxel_to_world(coords["tail"], affine).mean(axis=0)
    fr = build_frame(voxel_to_world(env_idx, affine), head_c, tail_c)
    del env_idx
    row.update(L_mm=fr.L, var_share=fr.var_share, eig_flipped=fr.eig_flipped, leftward=fr.leftward,
               e1_x=fr.e1[0], e1_y=fr.e1[1], e1_z=fr.e1[2])
    if not fr.var_share >= MIN_VAR_SHARE:
        reasons.append("var_share")

    u = {s: fr.coords_norm(voxel_to_world(coords[s], affine))[:, 0] if n[s] else np.empty(0)
         for s in ("head", "body", "tail")}
    for s in u:
        row[f"{s}_u"] = float(u[s].mean()) if len(u[s]) else float("nan")
    row["b_head_body"], row["b_body_tail"] = _section_bounds(u)
    if n["duct"]:
        du = fr.coords_norm(voxel_to_world(coords["duct"], affine))
        row.update(duct_u_p5=float(np.percentile(du[:, 0], 5)), duct_u_p50=float(np.median(du[:, 0])),
                   duct_u_p95=float(np.percentile(du[:, 0], 95)),
                   duct_ap_med=float(np.median(du[:, 1])), duct_si_med=float(np.median(du[:, 2])))
    row["status"] = ";".join(reasons) if reasons else "ok"

    points = None
    if keep_points:
        points = {"tissue": fr.coords_norm(voxel_to_world(tissue_idx, affine))}
        for s in ("head", "body", "tail", "duct"):
            points[s] = fr.coords_norm(voxel_to_world(coords[s], affine)) if n[s] else np.empty((0, 3))

    if acc is not None and row["status"] == "ok":
        sources = {"duct": coords["duct"], "tissue": tissue_idx,
                   "head": coords["head"], "body": coords["body"], "tail": coords["tail"]}
        for k, idx in sources.items():
            if len(idx) == 0:
                continue
            w = voxel_to_world(idx, affine)
            for g in GRIDS:
                pts = fr.coords_norm(w) if g.name == "norm" else fr.coords_mm(w)
                out = splat(acc[g.name][k], g, pts, vox)
                row[f"{k}_out_{g.name}_mm3"] = out
    return row, points


# ------------------------------------------------------------------ disk loading
def _quiet(fn, *a):
    """Call fn, returning (result, printed lines). tumorlib.io prints one 'missing .../ct.nii.gz'
    line per call for every non-cohort case (no CT on this PC); that is expected and dropped."""
    buf = _stdio.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a)
    lines = [ln for ln in buf.getvalue().splitlines() if ln and "ct.nii.gz" not in ln]
    return out, lines


def _coords(mask: np.ndarray) -> np.ndarray:
    return np.stack(np.nonzero(mask), axis=1).astype(np.int32)


def load_coords(case_id: str):
    """(coords dict, head affine, info dict) or (None, None, info) with info["status"] set.

    The duct is loaded first; a case with < MIN_DUCT_MM3 of duct stops there (status no_duct)
    without reading the gland masks.
    """
    info: dict = {"case_id": case_id}
    msgs: list[str] = []
    hdr = {}
    for k, name in STRUCTS.items():
        try:
            hdr[k] = nib.load(io.mask_path(case_id, name))
        except Exception as e:
            info.update(status="load_error", msgs=f"{name}: {type(e).__name__}: {e}")
            return None, None, info
    affine = hdr["head"].affine.copy()
    shapes = {k: tuple(h.shape[:3]) for k, h in hdr.items()}
    info["shape"] = "x".join(map(str, shapes["head"]))
    if len(set(shapes.values())) > 1:
        info.update(status="shape_mismatch", msgs=str(shapes))
        return None, None, info
    for k in ("body", "tail", "duct"):
        info[f"{k}_affine_differs"] = not np.allclose(hdr[k].affine, affine, atol=1e-3)
    g, lines = _quiet(io.grid, case_id)
    msgs += lines
    info["head_vs_ct_affine"] = "no_ct" if g is None else ("same" if np.allclose(g[1], affine, atol=1e-3) else "differs")
    info["spacing"] = ",".join(f"{s:.3g}" for s in nib.affines.voxel_sizes(affine))
    vox = float(abs(np.linalg.det(affine[:3, :3])))

    coords = {}
    for k in ("duct", "head", "body", "tail"):
        m, lines = _quiet(io.load_mask, case_id, STRUCTS[k])
        msgs += lines
        if m is None:
            info.update(status="load_error", msgs=" | ".join(msgs))
            return None, None, info
        coords[k] = _coords(m)
        del m
        if k == "duct" and len(coords["duct"]) * vox < MIN_DUCT_MM3:
            info.update(status="no_duct", duct_mm3=len(coords["duct"]) * vox, voxel_mm3=vox, msgs=" | ".join(msgs))
            return None, None, info
    info["msgs"] = " | ".join(msgs)
    return coords, affine, info


def process_case(case_id: str, acc=None, keep_points: bool = False):
    """load_coords + analyse -> (row, points)."""
    coords, affine, info = load_coords(case_id)
    if coords is None:
        return info, None
    row, points = analyse(coords, affine, acc, keep_points)
    return {**info, **row}, points
