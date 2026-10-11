# S8b: phase and negative-matching add-on to S8 (Dataset501 vs Dataset502)

**Answer: the S8 conclusions survive phase and negative matching.**

1. **502's gain holds in every scan-derived phase stratum with ≥20 lesions**, except late-arterial 10-20 mm (n = 29, CI spans 0).
2. **Phase changes the attenuation picture.** Truly unenhanced scans hold 23% of the lesions (not a third), and 55% of those lesions are iso (not ~50%). Pancreatic-phase (late arterial) scans have the fewest iso lesions: 20%.
3. **Matching the negatives to the tumor+ phase × thickness mix makes specificity slightly *worse* for both models**, and does not change the 501 vs 502 gap:

   | | unweighted | weighted |
   |---|---|---|
   | 501 specificity | 0.223 | 0.178 |
   | 502 specificity | 0.722 | 0.703 |
   | 501 FP per negative | 2.06 | 2.34 |
   | 502 FP per negative | 0.33 | 0.38 |

   Positives' own non-tumor region, which needs no matching, gives the same ranking: 1.57 vs 0.29 FP components per case.

All numbers are 5-fold out-of-fold and were computed in S8b, except rows marked `source = S8` or `S3 (reproduced)`. Those were recomputed only to check they match and are not new results.

## What was already in S8, and what is new

| item | already in S8 / S3 | new here |
|---|---|---|
| 1. phase strata at 10-20 and 20-40 mm | est_phase non-contrast, venous, uncertain and unmeasurable, both lesion sets, same bootstrap (`results/nnunet_multiclass/summary_vs_baseline.csv`) | early vs late arterial split (the S8 rows are reproduced exactly with the same bootstrap draws) |
| 2. iso fraction by phase | metadata phase only (S3) | estimated phase, both 3-class and 5-stratum |
| 3. negatives | unweighted Sp 0.223 / 0.722, FP per negative 2.06 / 0.33 | reweighted values with CI, ESS, cell table, positives' far-from-tumor FP by stratum |

## How it was computed

```
PYTHONPATH=src .venv/bin/python -m pytest tests/test_phase_addon.py -q        # 7 known-answer tests
PYTHONPATH=src .venv/bin/python -m multiclass_eval.phase_addon --workers 4    # ~2 min on the PC
```

**Code:** `src/multiclass_eval/phase_addon.py`. Full log: `work/nnunet_multiclass_phase/summary.txt`.

**Inputs.** Only S8's existing per-case pickles, S8's CSVs, `phase_estimate.csv`, `attenuation_labels_v3.csv` and the master table. **No case pass was rerun and no probability file was read.**
- **One extra input:** the GT lesion masks, read-only through `tumorlib.load_mask` on the PC, for 717 tumor+ cases. Item 3's "> 10 mm from every lesion" test needs lesion geometry, and the pickles store only each component's peak voxel and which lesions it overlaps.
- The masks' voxel and lesion counts equal the master table in every case.

**Phase strata.**
- S12 `est_phase`, with arterial split by `early_arterial`. That is `pv_enh_frac < 0.5`, the constant frozen in `src/phase/core.py`; it was re-derived and asserted for all 1308 cases.
- Early arterial = portal vein less than halfway to venous enhancement. Late arterial = pancreatic phase.
- `unmeasurable` (no vessel HU; 17 cases) is reported but is not one of the five strata.

**Thickness bins.** <2.5, 2.5-<5 and ≥5 mm (slice spacing along the slice axis).

**Bootstrap.**
- Item 1 uses S8's exact case weights (`stats.case_weights`, seed 0). The S8 rows therefore reproduce to the 6th digit.
- Item 3 resamples positives (seed 0) and negatives (seed 1) independently and recomputes the weights in each of the 1000 replicates, so the CI includes the uncertainty in the weights.

**Reweighting.**
- Each negative in a phase × thickness cell gets weight P(tumor+ case in that cell) / (negatives in that cell), normalized over the cells that have at least one negative.
- So the weighted negatives have the tumor+ *case* mix.
- ESS = (Σw)² / Σw².

**"Far" FP on positives.** A component of the saved lesion segmentation that:
- touches no GT lesion, and
- has its peak-probability voxel more than 10 mm from every GT lesion voxel (Euclidean, voxel centres, mm).

Non-touching components whose peak is within 10 mm are counted separately as "near". They are rare: 0.033 per case for 501 and 0.048 for 502. Far + near reproduces S8's FP per tumor+ case (1.60 and 0.33).

**Small strata.** Strata or cells with fewer than 20 lesions (item 1) or fewer than 20 cases on a side (item 3) are flagged in the `note` column and **not interpreted** below.

## Item 1: phase strata, lesion set = all

Columns are Dice and detect@any, 501 → 502, with the delta and its 95% CI.

| bin | stratum | n lesions (cases) | Dice | ΔDice [CI] | detect@any | Δdetect [CI] | recovered / lost |
|---|---|---|---|---|---|---|---|
| 10-20 | non-contrast | 78 (72) | 0.034 → 0.166 | +0.132 [+0.076, +0.192] | 21.8 → 38.5% | +16.7 [+2.9, +30.6] | 21 / 8 |
| 10-20 | early arterial | 13 (12) | 0.003 → 0.155 | *n < 20, not interpreted* | 7.7 → 46.2% | | 5 / 0 |
| 10-20 | **late arterial** | 29 (25) | 0.123 → 0.184 | +0.061 [−0.023, +0.167] | 31.0 → 41.4% | +10.3 [−8.3, +30.3] | 5 / 2 |
| 10-20 | venous | 143 (121) | 0.098 → 0.234 | +0.136 [+0.093, +0.179] | 30.1 → 55.9% | +25.9 [+16.6, +36.1] | 46 / 9 |
| 10-20 | uncertain | 10 (10) | 0.004 → 0.027 | *n < 20, not interpreted* | 10.0 → 30.0% | | 2 / 0 |
| 20-40 | non-contrast | 120 (109) | 0.137 → 0.336 | +0.199 [+0.156, +0.246] | 34.2 → 70.0% | +35.8 [+26.4, +45.4] | 46 / 3 |
| 20-40 | early arterial | 37 (30) | 0.268 → 0.388 | +0.120 [+0.047, +0.206] | 59.5 → 70.3% | +10.8 [+2.6, +21.9] | 5 / 1 |
| 20-40 | late arterial | 41 (37) | 0.221 → 0.410 | +0.189 [+0.107, +0.285] | 43.9 → 75.6% | +31.7 [+18.4, +46.2] | 13 / 0 |
| 20-40 | venous | 374 (351) | 0.395 → 0.526 | +0.131 [+0.105, +0.160] | 69.5 → 85.3% | +15.8 [+11.3, +20.3] | 71 / 12 |
| 20-40 | uncertain | 21 (21) | 0.378 → 0.501 | +0.123 [−0.010, +0.252] | 71.4 → 85.7% | +14.3 [+0.0, +31.8] | 3 / 0 |

`unmeasurable` has 2 (10-20 mm) and 4 (20-40 mm) lesions and is not interpreted. The `excl_separated` rows are in the CSV; they move in the same direction (e.g. late-arterial 10-20 mm +0.065 [−0.024, +0.175], n = 27).

**Readings:**
- **Late-arterial 10-20 mm is the one interpretable stratum where 502's gain is not significant.** It is also where 501 was already joint-best at 10-20 mm (31.0% detected; venous 30.1%). Pancreatic phase gives the best tumor-to-gland contrast, so there was less for the gland context to add. n = 29, so this is a weak signal.
- **At 20-40 mm, late arterial gains most among the contrast phases** (+31.7 points detection, 13 recovered, 0 lost). Early arterial gains least (+10.8). Non-contrast still gains the most overall.

## Item 2: v3 iso fraction by phase (Wilson 95% CI)

**All 1235 lesions** (iso overall 37.3%). Each cell is iso fraction (iso / n) [95% CI], share of all lesions.

| metadata phase | iso fraction |
|---|---|
| Non-contrast | 48.7% (201/413) [43.9, 53.5], 33% of lesions |
| Arterial | 34.2% (89/260) [28.7, 40.2], 21% |
| Venous | 30.3% (169/557) [26.7, 34.3], 45% |
| Delay | 2/4, 0.3% (not interpreted) |

| estimated phase | iso fraction |
|---|---|
| **non-contrast** | **55.1%** (156/283) [49.3, 60.8], **23%** of lesions |
| early arterial | 34.5% (41/119) [26.5, 43.4], 10% |
| **late arterial** | **19.8%** (20/101) [13.2, 28.6], 8% |
| venous | 33.1% (222/670) [29.7, 36.8], 54% |
| uncertain | 30.2% (16/53) [19.5, 43.5], 4% |
| unmeasurable | 6/9 (not interpreted) |

**Lesions ≥10 mm** (n = 1149, iso overall 38.1%):
- Estimated phase: non-contrast **61.5%** (144/234) [55.2, 67.5]; early arterial 35.2% (37/105); late arterial **19.1%** (18/94) [12.5, 28.3]; venous 33.4% (221/661) [29.9, 37.1]; uncertain 29.2% (14/48).
- Metadata (reproduces S3): Non-contrast 52.4% (355), Arterial 33.7% (246), Venous 30.8% (543).

**Where the metadata "Non-contrast" lesions really sit:** non-contrast 127/228 iso; venous 50/120 iso (42%); arterial 9/39; uncertain 10/20; unmeasurable 5/6.

**Suggested replacement for the CONTEXT line** "non-contrast scans are 1/3 of lesions and ~50% iso" (CONTEXT.md was not edited):

> By S12 estimated phase, non-contrast scans hold 23% of lesions and 55% of them are iso (≥10 mm: 61.5%, n = 234). Venous 33% (≥10 mm 33.4%, n = 661). Pancreatic phase (late arterial) has the lowest iso, 20% (n = 101). The metadata-based "1/3 of lesions, ~50% iso" included 120 lesions on scans that are actually venous-enhanced.

**Readings:**
- The metadata label understated how iso unenhanced scans are, because enhanced scans were mixed into it.
- The venous figure barely moves: 30.3% metadata, 33.1% estimated.
- Late arterial comes closest to the 5-15% literature range (*carried over* from CONTEXT, not verified): its CI [13.2, 28.6] just reaches the top of that range. That fits the idea that part of "iso" is a phase effect. It does not explain venous iso of about 33%.

## Item 3: negatives reweighted to the tumor+ phase × thickness mix

**Primary strata:** S12 est_phase (5 levels) × 3 thickness bins = 14 occupied cells.
- **ESS 186.9 of 327 negatives.**
- **Uncovered cells: none.** Every tumor+ cell has at least one negative, so nothing had to be dropped.
- Thin coverage, though, in two cells:
  - **non-contrast ≥5 mm:** 19 tumor+ cases vs 3 negatives (weight 2.1× per negative)
  - **venous 2.5-5 mm:** 357 tumor+ cases vs 36 negatives (3.3×). This cell drives the reweighting.
- 1 negative (unmeasurable ≥5 mm) sits in a cell with no tumor+ case and gets weight 0.

| per negative case | 501 unweighted | 501 weighted [CI] | 502 unweighted | 502 weighted [CI] | weighted Δ (502 − 501) [CI] |
|---|---|---|---|---|---|
| specificity (no lesion voxel) | 0.223 | **0.178** [0.131, 0.234] | 0.722 | **0.703** [0.633, 0.772] | +0.525 [+0.431, +0.614] |
| fraction with any lesion voxel | 0.777 | 0.822 [0.766, 0.869] | 0.278 | 0.297 [0.228, 0.367] | −0.525 |
| FP components | 2.06 | **2.34** [2.04, 2.68] | 0.33 | **0.38** [0.28, 0.48] | −1.96 [−2.29, −1.66] |

"Fraction with any lesion voxel" is 1 − specificity by definition, because evaluate.py's case detection is "predicted volume > 0". It is listed because it was asked for.

**Secondary strata** (5 phase strata + unmeasurable, × thickness; 17 cells): ESS 182.5, again no uncovered cells. Results: 501 Sp 0.181 and FP/neg 2.31; 502 Sp 0.702 and FP/neg 0.38. Practically identical to the primary strata.

**Reading.** The unmatched negatives made both models look *slightly better* than they are on a tumor+-like scan mix. Negatives are thinner-sliced, and thin slices give fewer FPs. The correction is small (≤0.05 Sp) and the same direction for both models, so the 501 vs 502 gap is unchanged.

### FP components on positives' non-tumor region (no matching needed)

| per tumor+ case (981) | 501 | 502 | Δ [CI] |
|---|---|---|---|
| far FP components (peak > 10 mm from every lesion) | 1.57 | **0.29** | −1.28 [−1.39, −1.18] |
| near FP components (non-touching, peak ≤ 10 mm) | 0.033 | 0.048 | +0.015 [−0.002, +0.035] |
| cases with ≥1 far FP component | 69.2% | 22.3% | |

**By cell, cells with ≥20 cases on both sides.** Columns are FP components per case: positives' far region vs negatives, 501 then 502.

| cell | positives (far) 501 / 502 | negatives 501 / 502 |
|---|---|---|
| venous 2.5-5 mm (357 / 36) | 1.41 / 0.26 | 2.61 / 0.42 |
| venous <2.5 mm (114 / 76) | 1.48 / 0.24 | 1.89 / 0.30 |
| venous ≥5 mm (104 / 24) | 2.38 / 0.37 | 3.17 / 0.58 |
| non-contrast <2.5 mm (159 / 75) | 0.99 / 0.16 | 1.12 / 0.31 |
| arterial <2.5 mm (113 / 49) | 1.54 / 0.39 | 1.69 / 0.16 |
| late arterial <2.5 mm (33 / 30) | 1.82 / 0.30 | 1.83 / 0.13 |

All other cells have fewer than 20 cases on a side; they are in the CSV, flagged.

**Readings:**
- **Both FP measures give the same ranking**: 502 has about 5–6× fewer FP components than 501 on both denominators, and fewer in every interpretable cell.
- **≥5 mm slices give more FPs than <2.5 mm slices** in all three main phases (non-contrast, arterial, venous), for both models and on both denominators. Most ≥5 mm cells have fewer than 20 cases, so this is descriptive. The 2.5-5 mm bin is not consistently in between.
- **Within a matched cell, negatives usually carry *more* FPs than positives' far region.** This holds in all 6 interpretable cells for 501 (late arterial <2.5 mm is a tie, 1.83 vs 1.82). For 502 it holds in 4 of 6; the exceptions are the two arterial cells, where 502's negatives have fewer. (The late-arterial row is from the 5-stratum scheme and overlaps the arterial row.) So for nnU-Net, negatives are if anything a *pessimistic* stand-in: they overstate FPs relative to the rest of a tumor+ scan. That is the opposite of S4's LoG finding (*carried over*: negatives understated false peaks by about 1/3).
- **Two possible causes, neither tested.**
  - The 10 mm exclusion removes the peri-tumoral region, which is near the gland where FPs concentrate. That makes positives' far region an easier region.
  - The models may put their probability mass on the true tumor when one exists, and on the next-best spot when none does.

## Caveats

- **Strata are not independent of thickness.** Estimated non-contrast is 84% ≤2 mm (S12). So item 1's phase strata partly reflect thickness, and the two cannot be separated with these n.
- **The estimated phase is itself a model.** 53 scans are `uncertain`, and arterial is a continuum (S12). The early/late split is a fixed cut, not a discovered boundary.
- **Weighted-negative CIs are wide** (ESS 187), and cells with 3–5 negatives carry up to 2.1× weight.
- **The far-FP denominator differs from a negative's.** It is the whole scan minus 10 mm around every lesion, so the per-case rates are not over identical volumes.
- **Many strata, unadjusted CIs.** Treat the stratum rows as descriptive.
- **The 30 negatives with report-described lesions (S1) are still counted as negatives.**

## Files

| file | rows | content |
|---|---|---|
| `phase_addon.csv` | 228 | Long table. Columns: `item` (1/2/3), `section`, `metric`, `lesion_set`, `stratum_var`, `stratum`, `n`, `n_cases`, `value` (single-valued stats: iso fraction, ESS, counts, weights), `baseline` (501), `multiclass` (502), `delta`, `ci_lo`, `ci_hi`, `p_boot`, `source` (`S8b` / `S8` / `S3 (reproduced)`), `note` (n<20 flags, recovered/lost counts, ESS, per-model CIs). Sections: `phase_strata`, `iso_by_phase`, `reweighting`, `reweighting_cells`, `negatives`, `negatives_by_cell`, `positives_non_tumor`, `positives_non_tumor_by_cell` |
