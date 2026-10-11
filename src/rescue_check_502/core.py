"""S14 constants (fixed 2026-10-05, before any S14 result was computed) and pure functions. No I/O.

Model and missed set
  model        Dataset502 + S13's D: each saved object (component of p > 0.5) regrown to p > 0.1 inside its
               t_low 0.1 footprint. One fixed rule for every fold (S13's LOFO picked it in 4/5 folds; fold 3
               picked the t_low 1e-4 variant, not used here).
  missed       no RAW 502 component (saved segmentation, seg == 2) touches the GT lesion. D's extra touches
               are grazes (S13: median Dice 0.008), so they never turn a lesion into "detected".
  detection    every rate is reported twice: touch (raw 502, the missed/detected split) and lesion Dice >= 0.1
               under 502 + D.

Descriptors
  phase        S12 est_phase with S8b's arterial split (early / late), + "unmeasurable" kept as its own level
  thickness    S8's bins <2.5, 2.5-5, >=5 mm
  satellite    not the largest lesion in its case (master largest_in_case == False)
  pred pancreas  502 npz channel 0 > 0.5 (S13 sparse cache). Per lesion:
                 frac_in      share of lesion voxels inside it
                 frac_filled  share inside its closing (8 mm) + 3D hole fill (tumorlib fill_search_region,
                              dilate 0): a cavity or narrow notch the model carved out counts as filled
                 shell_frac   share of the lesion's outer shell (0 < d <= SHELL_MM from the lesion, minus every GT
                              lesion) inside the predicted pancreas
                 dist_mm      min over lesion voxels of the distance to the nearest predicted-pancreas voxel
                 category     included (frac_in >= 0.5) / dent (frac_filled >= 0.5) / adjacent (dist <= ADJ_MM) /
                              away; first matching rule wins

Signals (S11's operating points, unchanged; S9 dropped because S11 showed its top-10 rate equals chance)
  nnunet  a t = 0.01 component of case rank <= 3 (by peak p, pessimistic ties) touches the lesion
  duct    case flag (mpd_dilated OR MPD head-end caliber >= 3.21 mm) AND the lesion within 10 mm of the MPD;
          not evaluable without an MPD mask

Chance under uniform placement (domain = tumorlib search region: GT envelope, 8 mm closing, hole fill, 3 mm
dilation; never reads the lesion; every voxel equally likely)
  nnunet  each of the case's rank <= 3 components is translated so that its peak voxel lands on a random domain
          voxel, independently; chance = 1 - prod_i (1 - f_i), f_i = share of domain voxels at which component i
          overlaps the lesion. Shapes are the real components, not spheres.
  duct    the lesion is translated so that its centroid voxel lands on a random domain voxel; g = share of domain
          voxels at which some lesion voxel is <= 10 mm from the MPD; chance = case flag x g (the flag is
          case-level and does not move).
  signal - chance  mean over evaluable lesions of (hit - chance), case-bootstrap 95% CI (S8's W, B 1000, seed 0)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

# ------------------------------------------------------------------ fixed a priori
D_T_LOW = 0.1
D_T_SEG = 0.1
DICE_DET = 0.1            # "Dice >= 0.1 per lesion" detection
SHELL_MM = 3.0
ADJ_MM = 3.0
FRAC_IN = 0.5
FILL_CLOSE_MM = 8.0       # = tumorlib.envelope.search_region's closing
DUCT_MM = 10.0            # = rescue_check.core.DUCT_DIST_MM
FOCUS_BINS = ("20-40", "10-20")
REPORT_BINS = ("10-20", "20-40", ">=40")
PHASE_LEVELS = ("non-contrast", "early arterial", "late arterial", "venous", "uncertain", "unmeasurable")
THICK_LABELS = ("<2.5", "2.5-5", ">=5")
PANC_CATS = ("included", "dent", "adjacent", "away")


# ------------------------------------------------------------------ placement chance
def anchor_hits(target: np.ndarray, target_lo, obj: np.ndarray, obj_anchor) -> tuple[np.ndarray, np.ndarray]:
    """Where can `obj` be placed so that it overlaps `target`?

    target: bool array (a crop), whose [0,0,0] sits at global voxel `target_lo`.
    obj: bool array (its own bbox crop); obj_anchor: the voxel of obj (in obj-crop coordinates) that is placed.
    Returns (R, origin): R[k] is True iff placing obj_anchor at global voxel a = origin + k makes some obj voxel
    coincide with a target voxel. Exact (integer overlap counts from an FFT correlation, thresholded at 0.5).

    Derivation: obj voxel u lands at a + u - obj_anchor. conv(T, flip(O))[k] = sum_u T[k - (S-1) + u] O[u], so
    k - (S-1) = a - obj_anchor - target_lo."""
    t = np.asarray(target, bool)
    o = np.asarray(obj, bool)
    if not t.any() or not o.any():
        return np.zeros((1, 1, 1), bool), np.zeros(3, int)
    full = fftconvolve(t.astype(np.float64), o[::-1, ::-1, ::-1].astype(np.float64), mode="full")
    origin = np.asarray(target_lo, int) + np.asarray(obj_anchor, int) - (np.asarray(o.shape) - 1)
    return full > 0.5, origin


def placement_fraction(domain_xyz: np.ndarray, R: np.ndarray, origin: np.ndarray) -> float:
    """Share of the domain voxels (N x 3 global coordinates) at which the placement hits (R from anchor_hits)."""
    n = len(domain_xyz)
    if n == 0:
        return np.nan
    k = np.asarray(domain_xyz, int) - origin
    inb = np.all((k >= 0) & (k < np.asarray(R.shape)), axis=1)
    return float(R[tuple(k[inb].T)].sum()) / n


def hit_at(R: np.ndarray, origin: np.ndarray, a) -> bool:
    """R evaluated at one global anchor (identity placement check)."""
    k = np.asarray(a, int) - origin
    return bool(np.all(k >= 0) and np.all(k < np.asarray(R.shape)) and R[tuple(k)])


def near_region(mask: np.ndarray, sp, mm: float) -> np.ndarray:
    """{x : dist(x, mask) <= mm} on mask's own crop. Casts to bool first: tumorlib masks are uint8, and ~uint8 is a
    bitwise NOT (0 -> 255), which would make the distance transform see no mask at all."""
    return distance_transform_edt(~np.asarray(mask, bool), sampling=sp) <= mm


def chance_any(fs) -> float:
    """P(at least one of independent placements hits) = 1 - prod(1 - f). Empty -> 0."""
    fs = np.asarray(list(fs), float)
    return float(1.0 - np.prod(1.0 - fs)) if fs.size else 0.0


# ------------------------------------------------------------------ predicted-pancreas lesion metrics
def lesion_panc_metrics(lesion: np.ndarray, any_lesion: np.ndarray, panc: np.ndarray, filled: np.ndarray,
                        dist_panc: np.ndarray | None, sp) -> dict:
    """All arrays are the same crop (bool), large enough to hold the lesion plus SHELL_MM on every side.
    dist_panc: distance (mm) to the nearest predicted-pancreas voxel on that crop (None = no predicted pancreas)."""
    les = np.asarray(lesion, bool)
    n = int(les.sum())
    d_les = distance_transform_edt(~les, sampling=sp)
    shell = (d_les > 0) & (d_les <= SHELL_MM) & ~np.asarray(any_lesion, bool)
    ns = int(shell.sum())
    return dict(frac_in=float((les & panc).sum()) / n,
                frac_filled=float((les & filled).sum()) / n,
                shell_frac=float((shell & panc).sum()) / ns if ns else np.nan,
                dist_mm=float(dist_panc[les].min()) if dist_panc is not None else np.nan)


def panc_category(frac_in: float, frac_filled: float, dist_mm: float) -> str:
    if frac_in >= FRAC_IN:
        return "included"
    if frac_filled >= FRAC_IN:
        return "dent"
    if np.isfinite(dist_mm) and dist_mm <= ADJ_MM:
        return "adjacent"
    return "away"


def phase_level(est_phase: str, early_arterial: bool) -> str:
    """S8b phase_stratum for one case (arterial -> early / late), other labels unchanged."""
    if est_phase == "arterial":
        return "early arterial" if early_arterial else "late arterial"
    return est_phase
