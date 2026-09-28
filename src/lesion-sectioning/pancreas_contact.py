"""Per-lesion-component contact with the pancreas, against the pancreas ENVELOPE
(pancreas.nii.gz | pancreas_head | pancreas_body | pancreas_tail).

Why an envelope and a contact measure instead of plain overlap with
pancreas.nii.gz: in many PanTS cases pancreas.nii.gz has the lesion carved out
of it (0 voxels of overlap even for a 40mm tumor sitting inside the gland),
while pancreas_head/body/tail still cover it -- and the two mask families
disagree by tens of thousands of voxels elsewhere. Overlap alone therefore
mixes "carved out of the organ mask" with "actually detached from the pancreas".
Contact fixes that: a carved-out lesion is ringed by pancreas voxels, a lesion
that really sits elsewhere is not.

Columns written (lesion_id numbering identical to main.py / join_lesion_attenuation.py):
  frac_envelope        fraction of lesion voxels inside the envelope
  dist_to_envelope_mm  min distance from any lesion voxel to an envelope voxel
  contact_1vox         fraction of the 1-voxel (26-conn) shell around the lesion inside the envelope
  contact_3vox         same for a 3-voxel shell (survives thin gaps / 1-2 voxel mask misregistration)

Usage (from repo root): .venv/bin/python3 src/lesion-sectioning/pancreas_contact.py
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, distance_transform_edt, generate_binary_structure, label as cc_label

MASK_ROOT = Path("/home/rahuldeb5/research/datasets/pants/masks/mask_only")
FOLD_DIR = Path("src/data")
RESULTS_DIR = Path("src/lesion-sectioning/work")
NUM_WORKERS = 4

CC_STRUCT = generate_binary_structure(3, 3)
SEG_NAMES = ("pancreas", "pancreas_head", "pancreas_body", "pancreas_tail")


def case_ids() -> list[str]:
    ids: set[str] = set()
    for i in range(1, 6):
        for line in (FOLD_DIR / f"fold_{i}_ids.txt").read_text().splitlines():
            line = line.strip()
            if line:
                ids.add(line)
    return sorted(ids)


def load_bool(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(path)
    raw = np.nan_to_num(np.asanyarray(img.dataobj), nan=0.0)
    return raw > 0.5, img.header.get_zooms()[:3]


def shell_contact(comp_crop: np.ndarray, env_crop: np.ndarray, n_iter: int) -> float:
    grown = binary_dilation(comp_crop, structure=CC_STRUCT, iterations=n_iter)
    shell = grown & ~comp_crop
    n = int(shell.sum())
    return float((shell & env_crop).sum() / n) if n else np.nan


def process_case(case_id: str) -> list[dict]:
    seg_dir = MASK_ROOT / case_id / "segmentations"
    lesion_path = seg_dir / "pancreatic_lesion.nii.gz"
    if not lesion_path.exists():
        return []
    lesion, zooms = load_bool(lesion_path)
    if not lesion.any():
        return []

    env = np.zeros_like(lesion)
    for name in SEG_NAMES:
        p = seg_dir / f"{name}.nii.gz"
        if not p.exists():
            print(f"{case_id}: missing {p}")
            return []
        m, _ = load_bool(p)
        if m.shape != lesion.shape:
            print(f"{case_id}: shape mismatch on {name}")
            return []
        env |= m

    labeled, n = cc_label(lesion, structure=CC_STRUCT)

    # EDT scratch scales with the full input, so crop to lesion|envelope first
    # (always contains every lesion voxel, so distances are unchanged).
    roi = np.argwhere(lesion | env)
    lo, hi = roi.min(axis=0), roi.max(axis=0) + 1
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    env_crop = env[sl]
    dist_from_env = distance_transform_edt(~env_crop, sampling=zooms) if env_crop.any() else None

    rows = []
    pad = 4  # > 3-voxel shell
    for k in range(1, n + 1):
        comp = labeled == k
        n_vox = int(comp.sum())

        idx = np.argwhere(comp)
        clo = np.maximum(idx.min(axis=0) - pad, 0)
        chi = np.minimum(idx.max(axis=0) + 1 + pad, lesion.shape)
        csl = tuple(slice(l, h) for l, h in zip(clo, chi))

        rows.append({
            "case_id": case_id,
            "lesion_id": k,
            "frac_envelope": float((comp & env).sum() / n_vox),
            "dist_to_envelope_mm": float(dist_from_env[comp[sl]].min()) if dist_from_env is not None else np.nan,
            "contact_1vox": shell_contact(comp[csl], env[csl], 1),
            "contact_3vox": shell_contact(comp[csl], env[csl], 3),
        })
    return rows


def main() -> None:
    ids = case_ids()
    print(f"{len(ids)} case ids to load, {NUM_WORKERS} workers")
    with mp.Pool(NUM_WORKERS) as pool:
        results = pool.map(process_case, ids)
    rows = [r for rs in results for r in rs]
    print(f"{len(rows)} lesion rows total")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "lesion_pancreas_contact.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
