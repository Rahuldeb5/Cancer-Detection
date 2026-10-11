"""Every S15 parameter, frozen before any real gland was profiled (2026-10-05).

Brief-given values are marked (brief). The others are implementation choices made a priori, also frozen.
Nothing here may be changed after a lesion has been scored.

v2 (2026-10-06, Rahul, after the Part A known-answer failure; synthetic only, no cohort lesion scored):
multi-scale bulge (+-15/30/45 mm), rays from the section-region centroid, gland QC rules, aorta inconclusive band.
v3 (2026-10-06, Rahul, round 3 after the v2 synthetic results): WIDTH (origin-independent) replaces radius as the
primary measurement; ray origin back to the centerline point (nearest inside point if outside); fold-over stops kept
and flagged; width with a capped half-ray is censored (flagged).
v4 (2026-10-06, Rahul, round 4): primary score on the residual width (bank reference); located = top cell in the
bulge's middle half; Part C windows / top-3 / chance definitions fixed in glandshape.partc.
v1 values that were replaced are kept below, marked "v1 (superseded)".
"""
import numpy as np

# ---------------------------------------------------------------- outline
PANC_THRESHOLD = 0.5          # (brief) 502 npz channel 0 > 0.5
CROP_MARGIN_MM = 10.0         # (brief) bbox + 10 mm, cropped BEFORE resampling (gotcha 6)
ISO_MM = 1.0                  # (brief) 1 mm isotropic
FRAG_RATIO = 0.10             # (brief) report / drop when 2nd component > 10% of the 1st

# ---------------------------------------------------------------- centerline
SPLINE_RMS_MM = 1.5           # splprep smoothing s = N * SPLINE_RMS_MM**2 (allowed RMS deviation from the skeleton path)
DENSE_STEP_MM = 0.25          # dense evaluation of the centerline curve
END_TANGENT_MM = 5.0          # end tangent = chord over the last 5 mm of the spline
MAX_TRIM_MM = 10.0            # an end outside the mask is trimmed inward by at most this much
MAX_EXTEND_MM = 40.0          # extension along the end tangent to the gland boundary, at most this much
STOP_MM = 2.0                 # (brief) stops every 2 mm of arc length
TANGENT_HALF_MM = 1.0         # tangent at a stop = central difference over +-1 mm of the dense curve

# ---------------------------------------------------------------- cross-sections and rays
PLANE_STEP_MM = 0.5           # (brief) in-plane sampling
RAY_MAX_MM = 40.0             # (brief) max ray length
N_DIRS = 24                   # (brief) 15 deg apart
THETAS = np.arange(N_DIRS) * (2 * np.pi / N_DIRS)
RAY_STEP_MM = 0.1             # ray marching step; exit refined by linear interpolation
HU_STEP_MM = 1.0              # HU stored along each ray at 0, 1, ..., 40 mm
RAY_ORIGIN = "centerline"     # v3 (round 3): rays start at the centerline point; if it is outside the gland in its
                              # plane, at the nearest inside sample (flagged). v2 (superseded): "centroid" of the
                              # section region (split bulge signal to the opposite side; seams). v1: centerline point
                              # with no fallback (stop invalid)
PRIMARY_SCORE_V5 = "raw"      # v5 (round 5): primary = multi-scale bulge_local on RAW width; residual secondary
EXCLUDE_TIP_STOPS = True      # v5: stops on the centerline extension past the skeleton ends are excluded from every
                              # score, window, footprint and chance (no other stop is excluded by u)
PRIMARY_SCORE = "residual"    # v4 (superseded by v5) (round 4), Part B onward: primary = multi-scale bulge_local on width minus the
                              # normal median width (u bin 0.05, 12 axes) of the 250 reference bank glands; the
                              # raw-width score is secondary
PRIMARY_MEASURE = "width"     # v3: width w = r(theta) + r(theta + 180) on 12 axes; the 24-direction radius map is a
                              # secondary output (which side), never a score. v1/v2: radius
FOLD_TOL_MM = 0.5             # a section point lies "in" another stop's plane if within this distance

# ---------------------------------------------------------------- scores
LOCAL_HALF_MM = 10.0          # v1 (superseded): single running median over s +- 10 mm at the same theta
BULGE_SCALES_MM = (15.0, 30.0, 45.0)  # v2 (Rahul): running-median half-windows; z per scale with the gland's own
                              # MAD at that scale; primary = max over scales of the positive z, then smoothed
SMOOTH_STOPS, SMOOTH_DIRS = 3, 3   # (brief) score smoothing window (nan-mean, theta wraps, s truncates); v3: on
                              # the width map the second dimension is the 12 axes (3 axes = 45 deg, wraps)
MAD_SCALE = 1.4826            # MAD -> SD-equivalent (a per-gland constant: changes no within-scan rank)
U_BIN = 0.05                  # (brief) normal reference bins in u
ATROPHY_MIN_SIDE = 5          # atrophy(s) needs >= 5 finite stops (10 mm) on each side, else NaN
ATROPHY_WIN_HALF_MM = 20.0    # v2: secondary localizer only (+-20 mm windowed sides); the case score is whole-side

# ---------------------------------------------------------------- orientation
AORTA_SLAB_MARGIN_MM = 0.0    # aorta-posterior test uses aorta voxels within the gland's world-z range (+ margin)
AORTA_INCONCLUSIVE_MM = 10.0  # v2 (Rahul): |aorta - gland y| <= 10 mm = inconclusive, treated as not flipped

# ---------------------------------------------------------------- gland QC (v2, Rahul, fixed before any profile score)
QC_VOL_MIN_CM3 = 5.0          # drop: outline volume < 5 cm^3
QC_VOL_MAX_CM3 = 250.0        # drop: outline volume > 250 cm^3
QC_CL_DIAG_RATIO = 1.5        # drop: centerline length > 1.5 x the outline's bounding-box diagonal (mm)
                              # drop: empty outline, centerline or profile raises. Fragmented (2nd > 10% of 1st): KEEP
                              # the largest component and flag

# world axes (nibabel affines map to RAS+)
WORLD_LEFT = np.array([-1.0, 0.0, 0.0])
WORLD_ANTERIOR = np.array([0.0, 1.0, 0.0])
