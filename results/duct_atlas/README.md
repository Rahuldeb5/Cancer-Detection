# Pancreatic-duct atlas (S10, 2026-10-03)

**Question.** Align every PanTS pancreas to one common frame and stack the `pancreatic_duct` masks. Where does the
annotated duct sit in the gland, and how consistent is its position?

**Read this first.** PanTS duct annotation is fragmentary. Per included case, the duct's 5th–95th percentile extent
along the gland has a median of 27.6 mm (0.20 gland lengths). This matches the carried-over median duct centreline
of ~28 mm, roughly half the gland length or less. How much duct gets annotated also tracks scan quality. So the
darkness of every panel here is **annotation coverage × anatomy**, not anatomy alone. A pale region means "rarely
annotated"; it does not mean "no duct there".

## Files

| file | content |
|---|---|
| `duct_atlas.png` | normalized frame: every gland scaled to its own length |
| `duct_atlas_mm.png` | physical frame: same rotation, mm, no scaling |
| `duct_atlas_by_tumor.png` | optional split by `tumor?` from `results/PanTS_metadata_new.csv`, normalized frame |

Each main figure has two columns. **Axial** is u vs ap, summed along si, anterior up. **Coronal** is u vs si, summed
along ap, superior up. The rows are:

- **A**: tissue tinted by its dominant section (head, body or tail), with the duct overlaid in black.
- **B**: the duct alone. The colorbar is duct mm³ per bin, summed over cases.
- **C**: duct rate = projected duct mm³ / projected tissue mm³, left blank where tissue is below 1% of its max.

Duct opacity is `clip((v/vmax)^0.5, 0, 1)`, where vmax is the 99.5th percentile of the nonzero displayed values (the
value is printed on each panel). All panels use display-only Gaussian smoothing with σ = 1.5 bins. Raw arrays are in
`work/duct_atlas/full/` (`atlas.npz`, 3D grids; `projections.npz`, the unsmoothed 2D projections; `cases.csv`, one
row per case).

## How it was computed

Code is in `src/duct_atlas/` and the tests are in `tests/test_duct_atlas.py`.

- **Masks.** For each case, `pancreas_head`, `pancreas_body`, `pancreas_tail` and `pancreatic_duct` are loaded with
  `tumorlib.io.load_mask`. All four use the **head mask's affine and voxel grid**, and their shapes must match.
- **Frame.** The envelope is head ∪ body ∪ tail.
  - Centre c = envelope centroid.
  - e1 = PC1 of the envelope voxel centres, signed so that tail centroid − head centroid points along +e1.
  - e2 = world +y (anterior) made orthogonal to e1. e3 = world +z (superior) made orthogonal to e1 and e2.
  - u = (projection − p1)/L, where p1 and p99 are the envelope's 1st/99th percentile projections and L = p99 − p1.
    ap and si are offsets from c, also in units of L.
  - The duct is never used to build the frame.
- **Splat.** Every foreground voxel centre adds its voxel volume (mm³) to one bin. Normalized grid: u from −0.15 to
  1.15, ap and si from −0.4 to 0.4, bin 0.01. mm grid: 1 mm bins.
  - Tissue = head ∪ body ∪ tail ∪ duct.
  - The duct label is **disjoint** from head, body and tail in every case (0 overlapping voxels). The duct is
    carved out of the section masks, so it has to be added back to get tissue.
- **Section boundaries.** Per case, the midpoint between adjacent sections' 5th/95th percentiles on u (same rule as
  the lesion map). The figures show the median over included cases.

**Oracles, run before any real data, all pass.**
1. 50 synthetic ellipsoid glands in random 3D poses, with 0.8×0.8×5 mm voxels and permuted/flipped affines:
   - PC1 is within 1° of the true long axis in every pose.
   - Head, body and tail centroids land within 0.02 of their analytic u values, so the head is at u≈0 every time.
   - Over 99% of stacked duct mass falls within |ap|, |si| < 0.03, so the duct collapses to a thin ridge.
2. Volume conservation: atlas sum + off-grid mass equals the per-case duct mm³ exactly.
3. A known cube of points lands in the expected bins with the expected mm³.

On real data, conservation holds to within 1%: the duct total in the atlas is 0.9967 × the per-case sum on the
normalized grid (0.33% falls outside it) and 0.9997 on the mm grid.

## Cases

All 9901 case folders were checked. No head, body, tail or duct mask is a git-LFS stub, so nothing was pulled. There
were no shape mismatches and no load errors.

| | n |
|---|---|
| no duct ≥ 8 mm³ (2944 empty; 520 with 0–8 mm³) | 3464 |
| duct ≥ 8 mm³ | 6437 |
| excluded: empty tail mask | 278 |
| excluded: empty head mask | 35 |
| excluded: head and tail both empty | 2 |
| excluded: envelope < 15 cm³ (52) or > 300 cm³ (2) | 54 |
| excluded: PC1 variance share < 0.5 | 7 |
| **included** | **6061** |

The full list of excluded case IDs is in `work/duct_atlas/full/cases.csv`, column `status`. The 7 PC1 exclusions are
PanTS_00000877, 00000973, 00002324, 00002621, 00004056, 00009367 and 00009373.

**Cohort cross-check.** Of the 1308 cohort cases, `results/geometry_results/duct_caliber.csv` marks MPD present in
1063. This run finds a duct ≥ 8 mm³ in 1031 of them, a difference of −32. All 32 are cohort MPDs of 2–7.6 mm³, below
the 8 mm³ floor. Every case counted here is also MPD-present there. Of the 1031, 999 are included.

## Frame statistics (6061 included)

- **Gland length L (mm)**, percentiles 1/5/25/50/75/95/99: 77.9 / 99.1 / 121 / **136** / 150 / 174 / 193.
- **PC1 variance share**, same percentiles: 0.665 / 0.738 / 0.811 / **0.846** / 0.875 / 0.912 / 0.936.
- **PC1 sign flip.** `eigh` returned the tail→head direction, so the head/tail rule negated it, in 20 of the 6122
  frames built. Since `eigh`'s sign is arbitrary, this count says nothing about the data. The meaningful count:
  head→tail points to the patient's *right* in 10 frames, 9 of them included (PanTS_00000704, 00001481, 00002182,
  00002545, 00005205, 00005381, 00006012, 00007345, 00007757). Not investigated. A mirror in x does not change u, ap
  or si.
- **Section order.** Head < tail on u holds in 6061/6061 cases. Head < body < tail holds in 6036 of the 6051 cases
  that have a body mask; 10 included cases have an empty body. The 15 violators are listed in
  `work/duct_atlas/full/report.txt`; in each, the body centroid sits beyond the tail or before the head. They are
  left in, because the frame only uses head and tail.
- **Median section boundaries.** head|body at u = 0.364, which is −0.016 vs the lesion map's 0.38. body|tail at
  0.670, which is −0.010 vs 0.68. Both are within the 0.02 tolerance. In mm: 49.8 and 91.2.

## What the atlas shows

- **A ridge exists.** In both views the stacked duct forms one continuous, curved band that follows the gland's own
  bend. In the axial view it climbs from posterior at the head end (ap ≈ −0.09 at u 0–0.1) to its most anterior
  point in the neck/body (ap ≈ +0.12 at u 0.4–0.5), then descends to ap ≈ −0.12 at the tail end. Coronally it rises
  from si ≈ −0.05 to +0.09 at u ≈ 0.35, then descends to −0.09.
- **Position within the gland.** Between u = 0.2 and 0.8, the duct's mass centroid sits 0.026–0.040 gland lengths
  (~3.5–5.5 mm at the median L) anterior to the tissue centroid at the same u, and 0.01–0.05 superior to it. At the
  two ends (u < 0.1 and u > 0.9) it sits posterior to the tissue centroid instead (−0.03 and −0.04).
- **Where it fades.** Duct mass is concentrated toward the head: 58% of included duct mm³ lies in the head, which
  holds 46% of the tissue; the body has 29% vs 25%; the tail 13% vs 26%. The share of cases whose duct (p5–p95)
  covers a given u:

  | u | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | cases covering | 33% | **56%** | 51% | 39% | 29% | 20% | 14% | 8% | 4% | 1% |

  Within the gland, row C (duct rate) peaks in the head/neck at u ≈ 0.25–0.4, slightly anterior and superior.
- **The head end.** In the axial view the band does not widen evenly. It splits into two lobes. The posterior lobe
  is centred near u ≈ 0.1, ap ≈ −0.1. The main ridge rises anteriorly from u ≈ 0.2. At u = 0.15–0.25 the duct's ap
  profile is bimodal, with peaks at ap −0.07 and +0.05. The coronal view shows a single band. This is a statement
  about where annotated duct voxels fall; the figure does not say what anatomy the lobes are.
- **Physical frame.** Without scaling, the head and neck look the same. The body and tail smear out because gland
  lengths differ (L from 78 to 193 mm, 1st to 99th percentile).
- **Tumor split (optional figure).**
  - Tumor+ glands (N = 843) carry far more annotated duct: median 405 mm³ per case vs 108 mm³ in tumor− (N = 5218).
    Median p5–p95 extent is 46 mm vs 25 mm.
  - In tumor+, duct mass reaches well into body and tail (head/body/tail shares 42/37/21%). In tumor−, 72% of duct
    mass is in the head, which is where the posterior head lobe comes from.
  - With these annotation-quality differences, the split shows *where duct gets annotated* in each group. It does not
    show a difference in duct anatomy.

## Caveats

- **Opacity = annotation coverage × anatomy** (see the top). The figures cannot separate the two.
- **Mirrored affines (y axis).** Every affine disagreement in PanTS here is an anterior↔posterior flip, with spacing
  unchanged:
  - **Duct vs head affine, 21 cases** (15 of them included). This is handled: both masks use the head affine.
  - **Gland affine vs CT, 8 cohort cases.** In these the gland masks themselves are flipped.
  - **Anatomical audit.** Across the 1608 cases with a real aorta mask, the aorta is a median 35.8 mm *posterior* to
    the gland. In 13 cases it is 15–76 mm *anterior*, so these glands are mirrored in world space. That includes 4
    whose masks match the CT.
  - **Impact.** Among included cases that could be tested, 8/1200 (0.7%) are mirrored, and their ap sign is
    inverted. Most included cases have no aorta mask, so their mirrors cannot be detected. Roughly 40 mirrored cases
    are expected among the 6061, under 1% of the mass, which cannot visibly change the figure. Per-case `ap` values
    should not be trusted without this check. Details: `work/duct_atlas/aorta_flip_check.csv`,
    `src/duct_atlas/flip_audit.py`.
- **Outside the gland.** 0.33% of duct mass falls outside the normalized grid, and some duct sits beyond u < 0 or
  u > 1. These are ducts annotated past the gland envelope. At the low-tissue rim, row C is noisy, e.g. the dark edge
  pixels at the tail tip.
- **Section labels are imperfect** in a few cases (15 out of order, 10 with an empty body), and PanTS head/body/tail
  are not consistent with `pancreas.nii.gz` (known since S1).

Runtime: 22.9 min on the PC, 4 workers, worker peak RSS ≤ 602 MB. The worst case, PanTS_00008854 (510×431×918),
took 3.8 s at 460 MB peak RSS.
