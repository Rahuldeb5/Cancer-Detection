# S15: gland shape profile on Dataset502's predicted pancreas — EXPLORATORY

> **Status: EXPLORATORY ("tool failed its insertion check").** Under the outcome rule fixed before Part C
> (notes/LoG_notes/SESSION_15.md, round 5), the tool failed two checks on normal glands and synthetic tubes:
> - the core noise-free bend test on the 0.8 mm iso grid (−0.498 z vs a two-sided limit of 0.25; the bent tube
>   scored *lower*);
> - the insertion check (20 mm sphere on 49 held-out normal glands: median within-scan percentile 87.8 vs ≥ 95).
>
> These results cannot pass a gate and are not used for the case-level classifier. Every number below was computed
> in S15 (2026-10-08) unless marked *carried over*.

**Question.** Does a tumor that Dataset502 misses still show as a bulge of 502's own predicted gland (channel 0 > 0.5)?
The target is S14's group of 48 missed 20-40 mm index lesions that 502 swallows into its gland, not GT-separated.
S15 addresses at most 48 of the 116 missed 20-40 mm lesions; the rest are satellites or lie outside / beside the
gland (S14, *carried over*).

**Answer (exploratory).** No.
- **Missed targets bulge at chance level:** top-3 hit rate 39% (Wilson CI 26-54%) vs 41% chance.
- **Detected lesions of the same size bulge above chance:** 68% vs 50%.
- The targets' median within-scan percentile (78.8) looks moderate, but it comes from lesions in the head, where
  normal glands also score high. Targets in the head: 52% top-3 vs 51% chance. Targets outside the head: 18% vs 23%.

## What was computed

Per cohort case (1308; 1284 glands kept by the QC below; out-of-fold 502 outlines):
1. **Outline:** largest component, hole-filled, 1 mm isotropic.
2. **Centerline:** head→tail, signed from the CT affine, not from GT masks; stops every 2 mm.
3. **Cross-sections:** perpendicular sections; width on 12 axes (w = r(θ) + r(θ + 180°), first exit, 40 mm cap).
4. **Primary score:** multi-scale bulge on raw width. At each running-median window (±15 / ±30 / ±45 mm) the residual
   is divided by the gland's own MAD; the score is the maximum positive z over the three, smoothed over 3 stops × 3
   axes. Extrapolated tip stops are excluded.
5. **Secondary scores:** the same rule on width minus the median of 249 normal bank glands; case-level atrophy.

GT lesions are used only for evaluation:
- **Footprint:** the (stop, axis) cells where either half of the axis's diameter passes through the lesion.
- **Within-scan percentile:** against every window of the footprint's box size that does not overlap the footprint.
- **Top-3 hit:** one of the gland's 3 non-max-suppressed peaks lies in the footprint.
- **Chance:** the larger of (a) exact uniform placement of the 3 suppressed peaks and (b) the empirical rate on the
  49 held-out normal glands.

All definitions were fixed before this run (`src/glandshape/partc.py`, `footprint.py`).

## Scoreboard (v5 primary score)

| group | n | evaluable | median percentile (95% CI) | top-3 hit (Wilson CI) | chance | hit / chance |
|---|---|---|---|---|---|---|
| **Target**: missed 20-40 mm, included (S14) | 48 | 46 | 78.8 (65.7-90.7) | 39% (26-54%) | 41% | 0.96 |
| target without fold-heavy footprints | 32 | 30 | 65.6 (49.5-79.4) | 23% (12-41%) | 33% | 0.72 |
| target, no flag (fold-heavy, fragmented, censored) | 30 | 29 | 65.7 (45.0-80.0) | 24% (12-42%) | 33% | 0.74 |
| target, enhanced | 30 | 28 | 88.8 (78.0-96.0) | 50% (33-67%) | 41% | 1.21 |
| target, non-contrast | 18 | 18 | 61.2 (25.8-79.2) | 22% (9-45%) | 40% | 0.56 |
| target, iso | 27 | 25 | 89.6 (65.1-99.9) | 44% (27-63%) | 50% | 0.89 |
| target, not iso | 21 | 21 | 67.2 (51.7-87.8) | 33% (17-55%) | 30% | 1.10 |
| Secondary: missed 10-20 mm, included | 46 | 45 | 66.7 (49.8-86.2) | 27% (16-41%) | 21% | 1.26 |
| Reference: detected 20-40 mm, included | 461 | 455 | 96.2 (92.3-100.0) | 68% (63-72%) | 50% | 1.36 |
| Control: detected ≥ 40 mm | 253 | 246 | 100.0 (100.0-100.0) | 85% (80-89%) | 73% | 1.15 |

Gates as written, for the record only (exploratory, so they cannot pass):

| gate | value | status |
|---|---|---|
| control median ≥ 90 | 100.0 | met |
| target median ≥ 80 | 78.8 | not met |
| target top-3 ≥ 2 × chance | 39% vs 82% | not met |
| target top-3 ≥ 25% absolute | 39% | met |

**By position:**

| group | head (footprint starts at u < 0.3) | elsewhere |
|---|---|---|
| target | n = 29: percentile 93.3, top-3 52% vs chance 51% | n = 17: 65.6, 18% vs 23% |
| reference | 100.0, 79% vs 58% | 59.7, 42% vs 31% |
| control | 100.0, 91% vs 81% | 70.6, 70% vs 55% |

**Secondary residual score:** target 78.3 / 35% (chance 41%); reference 92.7 / 60%; control 100 / 81%.

**Echo check.**
- The primary score correlates with 502's lesion probability in the same cells: per-case Spearman ρ median 0.285
  (IQR 0.130-0.428) over 961 tumor+ glands.
- Among the 46 targets, the lesion's score correlates with the faint 502 probability in its footprint (ρ 0.458;
  median footprint probability 0.008).
- In detected lesions the probability is saturated (median 1.0), so ρ is 0.12 (reference) and 0.01 (control).

**Case level (reported only).**
- AUC tumor+ vs tumor−: max primary score 0.566 (95% CI 0.530-0.604), residual 0.530, atrophy 0.560.
- Within slice-thickness bins the AUCs are 0.54-0.59.
- Negatives are thinner-sliced: median slice 1.5 vs 2.5 mm.
- Atrophy flags 7.0% of tumor+ vs 3.6% of tumor− glands.

## Counts

- **Glands:** 1284 kept. Dropped 24: 12 empty, 7 under 5 cm³, 5 over 250 cm³.
- **Tip exclusion (round 5, item 4):**
  - target: 10 of 46 lose ≥ 1 footprint cell; none lose all.
  - secondary: 11 of 46 lose ≥ 1 cell; 1 loses all and becomes not evaluable (45 left).
- **Not evaluable (no footprint cell, footprint only on tip stops, or gland dropped):** evaluable share by tier is
  inside 94.8%, abutting 75.3%, embedded 66.7%, separated 24.3%.

## Caveats

- **Exploratory:** see the status box.
- **The normal head scores as a bulge.** The ±30/45 mm windows read the wide head next to the narrow neck as
  outward. On held-out normal glands, 41% of top cells lie at u 0.1-0.2. Chance is measured on those normal glands,
  so it absorbs this. But it means a head-lesion "hit" mostly reflects head shape.
- **Chance is high for 20-40 mm lesions** (median 41%). A footprint of that size usually spans all 12 axes over
  ~10 stops, so the 3 peaks have a large target. Percentiles rest on ~50-90 windows per gland.
- **n = 46 is a pilot.** CIs are wide; a pass would only have justified a larger test.
- **Fold-over:** 1020 of 1284 glands have fold-over stops, mostly at the head-neck bend. Fold-heavy targets are
  mostly head lesions.
- **Normal reference:** 300 S9 bank CTs run through the 502 5-fold ensemble (phase re-estimated with S12's rule:
  venous 152, non-contrast 63, arterial 74).

## Files

- `per_lesion_shape.csv` — 1235 lesions (S14 / master-table keys), group flags, evaluability, footprint size and
  flags, percentile, top-3 hit, both chance versions, residual-score columns, max 502 lesion probability in the
  footprint, and `status` = EXPLORATORY.
- Code: `src/glandshape/` (`partc_run.py` produces this folder).
- Work files: `work/gland_shape/partC/` (cases.csv, lesions_joined.csv, group_table.csv, partc_summary.txt,
  qc/ with 12 QC PNGs).
- Profiles (npz with HU along rays): `~/research/gland_profiles/{cohort,bank}/`, outside the repo.
