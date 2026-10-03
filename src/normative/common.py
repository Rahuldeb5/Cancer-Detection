"""Paths, a-priori constants and small helpers shared by every S9 module.

Everything numeric in here was fixed before any cohort lesion was scored (design freeze,
2026-10-02). Changing one after seeing cohort results means using nnU-Net fold 0 as the dev
fold and reporting folds 1-4 separately (see the session brief).
"""
from __future__ import annotations

import ast
import contextlib
import io as _io
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work" / "normative"
RESULTS = REPO / "results" / "normative_pilot"
METADATA = REPO / "results" / "PanTS_metadata_new.csv"
MASTER = REPO / "results" / "master_lesion_table" / "master_lesions.csv"
ATTEN3 = REPO / "results" / "attenuation_v3" / "attenuation_labels_v3.csv"
S2_LESIONS = REPO / "results" / "nnunet_subthreshold" / "per_lesion_subthreshold.csv"
S4_LESIONS = REPO / "results" / "log_candidates" / "separability_by_lesion.csv"
S1_CASES = REPO / "work" / "master_lesion_table" / "case_spacing.csv"

DATA = Path.home() / "research" / "datasets" / "pants"
IMAGES = DATA / "images"
MASK_REPO = DATA / "masks"
# Bank CTs live apart from the cohort's ct_staging so nothing that globs ct_staging (extract_cts.sh's
# own count, ad-hoc `ls | wc -l`) ever sees a non-cohort case.
BANK_CT_ROOT = DATA / "ct_normal_bank"

SEED = 42
N_BANK = 300
NUM_WORKERS = 4

# ---- PHASE 0
THICK_EDGES = [0.0, 1.5001, 2.5001, 5.0001, np.inf]      # same edges as S4 (5.0 mm is in "2.5-5")
THICK_LABELS = ["<=1.5", "1.5-2.5", "2.5-5", ">5"]
BANK_STRUCTURES = ("pancreas", "veins", "duodenum", "superior_mesenteric_artery", "aorta")
LFS_INCLUDE_MAX = 131072                                  # CONTEXT gotcha 7

# ---- PHASE 1
ISO_MM = 1.0
HU_CLIP = (-100.0, 300.0)
RADII_MM = (6.0, 10.0)
GRID_MM = 4
SHELL_MM = 2.0
RING_MM = 5.0
GRAD_SIGMA_MM = 1.0           # Gaussian derivative scale for the gradient-magnitude map
ENTROPY_BIN_HU = 25.0         # 16 bins over the clipped [-100, 300] range
LOG_TRUNCATE = 4.0            # same kernel extent as S4
PAD_MM = 30.0                 # crop margin > max(r)+ring (15 mm) and > 4*sigma_LoG(10) = 23 mm
CONTEXT_STRUCTURES = ("veins", "duodenum", "superior_mesenteric_artery", "aorta")
CONTEXT_MAX_MM = 30.0         # distances beyond the crop margin are not reliable -> capped here

# ---- PHASE 2
PCA_VAR = 0.95
MIN_STRATUM = 2000
K_MEANS = 16
KNN_K = 5
VAR_SHRINK_N = 10             # cluster variance shrunk toward the stratum variance with this weight

# ---- PHASES 3-4
DIL_MM = 2.0                  # hit = centre/peak inside the lesion dilated 2 mm (S2/S4 rule)
FP_EXCL_MM = 10.0             # S4 source (b) exclusion around every lesion
N_RANDOM_SETS = 200           # random same-footprint sets per lesion (S4 used 200 patches)
CAND_SEP_MM = 5.0
CAND_TOP_K = 50
TOP_N_GRID = (1, 2, 3, 5, 10, 20, 50)
Z_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
INSERT_DIAMS_MM = (10.0, 15.0, 20.0)
INSERT_DHU = (-30.0, -15.0, 15.0)
INSERT_EDGE_MM = 1.0
N_INSERT_CASES = 50
BANK_SIZE_CURVE = (100, 200)

ENHANCED = ("Venous", "Arterial", "Delay")
S4_TARGET_FRAC_BEATEN = 0.765  # carried over from S4 (recomputed from its CSV in evaluate.py)
S4_TARGET_TOP10 = 0.028        # carried over from S4: 3/106, Wilson CI 1.0-8.0%


def thick_bin(s) -> pd.Series:
    return pd.cut(pd.Series(s, dtype=float), THICK_EDGES, labels=THICK_LABELS, right=True)


def phase_group(phase) -> str:
    """Model phase group. Delay (4 cohort positives, 68 PanTS scans) is folded into venous: both are
    portal/late enhancement and Delay alone could never fill a stratum. Unknown -> 'unknown', which
    the scorer handles by pooling over all phase groups."""
    if not isinstance(phase, str):
        return "unknown"
    return {"Non-contrast": "noncontrast", "Arterial": "arterial",
            "Venous": "venous", "Delay": "venous"}.get(phase, "unknown")


def thick_group(t: float) -> str:
    """Model thickness group (coarser than the sampling bins): <=2.5 / >2.5 mm."""
    if not np.isfinite(t):
        return "unknown"
    return "thin" if t <= 2.5001 else "thick"


def metadata() -> pd.DataFrame:
    m = pd.read_csv(METADATA).rename(columns={"PanTS ID": "case_id"})
    m["max_spacing_meta"] = m.spacing.map(lambda s: max(ast.literal_eval(s)))
    m["shape_meta"] = m["shape"].map(lambda s: tuple(int(v) for v in ast.literal_eval(s)))
    return m


def slice_thickness(affine: np.ndarray) -> tuple[float, int]:
    """(spacing along the slice axis, slice axis): S1's rule, the voxel axis whose direction is most
    aligned with world superior-inferior."""
    import nibabel as nib
    vs = nib.affines.voxel_sizes(affine)
    ax = int(np.argmax(np.abs(affine[2, :3]) / vs))
    return float(vs[ax]), ax


@contextlib.contextmanager
def quiet():
    """Swallow tumorlib's per-case prints (e.g. 'missing ct' for non-cohort cases with no CT yet)."""
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf
