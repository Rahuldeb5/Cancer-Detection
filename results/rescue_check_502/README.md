# S14: the lesions Dataset502 still misses (2026-10-05)

**Question.** Which lesions does Dataset502 (+ S13's D post-processing) still miss? How do they differ from the
lesions it finds in the same size bin? And does any existing signal flag them more often than chance?
This session is descriptive: there is no gate. PC, CPU, about 7 min for the case pass. Nothing was trained.

**Answer.**
- **Half of the missed 20-40 mm lesions sit inside 502's own predicted pancreas** (61/116, "included"). The other
  half sit **beside** the predicted gland (38 adjacent, mostly GT tier `abutting`) or **away** from it (17, almost
  all GT `separated`). 502 almost never carves a missed lesion out of its gland: **0/116 dents** at 20-40 mm and
  2/144 at 10-20 mm. For lesions it does not include, the median share of the 3 mm shell covered by predicted gland
  is 0.03, so the gland does not wrap around them either.
  → **S15 should look for bulges inside the predicted gland, not for dents.**
- **The misses are enriched** for satellites (35% vs 7% of detected 20-40 mm), lesions outside or abutting the gland,
  and estimated non-contrast scans (31% vs 17%). They are **not** enriched for iso attenuation (42% vs 38%, p 0.46).
- **The faint nnU-Net guess fires above chance on the misses** (20-40 mm: 30.2% vs 9.8% chance, +20.3 points
  [+12.6, +27.7]). Its footprint is a contact, not a segmentation: median Dice 0.05, and 8/35 reach Dice 0.1 when
  the whole p > 0.01 component is counted. It also puts a candidate in 47% of negatives.
- **The duct signal is at or below chance on the misses** (20-40 mm: 12.0% vs 14.6%, −2.5 points [−6.9, +2.0]).
  It is only slightly above chance on detected lesions (+7.6 points).

Every number below was computed in S14 unless marked *carried over*. Every printed number:
`work/rescue_check_502/summary.txt`.

## Definitions (fixed in `src/rescue_check_502/core.py` before any S14 result)

| item | definition |
|---|---|
| model | Dataset502 + **D**: each saved object (component of p > 0.5) regrown to p > 0.1 inside its t_low 0.1 footprint. One fixed rule for all folds. S13's LOFO picked this in 4/5 folds; fold 3 picked the t_low 1e-4 variant, which is not used here. C+ is not used |
| missed | no **raw** 502 component touches the GT lesion. D's extra touches are grazes (S13), so they never count as detection |
| detection rate | reported twice: **touch** (raw 502; this is the missed/detected split) and **Dice ≥ 0.1** (502 + D, per lesion) |
| phase | S12 `est_phase` with S8b's 5 strata (arterial split into early / late by `pv_enh_frac` < 0.5), plus `unmeasurable` |
| thickness | S8 bins <2.5 / 2.5-5 / ≥5 mm |
| satellite | not the largest lesion in its case |
| predicted pancreas | 502 npz channel 0 > 0.5 (S13 sparse cache). `frac_in` = share of lesion voxels inside it. `frac_filled` = share inside its 8 mm closing + hole fill. `shell_frac` = share of the lesion's outer 0-3 mm shell (other lesions removed) inside it. `dist_mm` = min distance from a lesion voxel to the predicted gland |
| category | **included** frac_in ≥ 0.5 → **dent** frac_filled ≥ 0.5 → **adjacent** dist ≤ 3 mm → **away**. First matching rule wins |
| signals | S11's operating points, unchanged. **nnU-Net**: a t = 0.01 component of case rank ≤ 3 (by peak p) touches the lesion. **duct**: (`mpd_dilated` OR head-end caliber ≥ 3.21 mm) AND lesion ≤ 10 mm from the MPD; not evaluable without an MPD mask. **S9 dropped** (S11: its top-10 rate equals chance) |
| chance | uniform placement over the tumorlib search region (GT envelope, 8 mm closing, hole fill, 3 mm dilation; never reads the lesion). **nnU-Net**: each rank ≤ 3 component, with its real shape, is translated so that its peak voxel lands on a random region voxel; chance = 1 − Π(1 − f_i). **duct**: the lesion is translated so that its centroid voxel lands on a random region voxel; g = share of placements with some lesion voxel ≤ 10 mm from the MPD; chance = case flag × g. Exact overlap counts by FFT correlation |
| signal − chance | mean over evaluable lesions of (hit − chance), case-bootstrap 95% CI (S8's `paired_ratio`, B = 1000, seed 0) |
| specificity weights | S8b primary: negatives reweighted to the tumor+ est_phase × thickness mix |

## Regression (asserted before any S14 number was read)

- **Raw 502 matches S8.** Touch, lesion Dice and peak p (lesion + 2 mm) equal S8's `per_lesion_comparison.csv` for
  1235/1235 lesions. Touch is computed two ways (S13 atoms; S11's table built from S8's components) and they agree.
- **Missed counts.** 116/597 missed at 20-40 mm, equal to S13 (*carried over*). 144/275 missed at 10-20 mm.
- **S11 reused unchanged.** `rescue_check.nnunet_table --pred_source` ran on 502 components exported from S8's
  pickles. Its touch / max-t / peak columns equal S8 for every lesion.
- **Identity placement reproduces the real signals.** Placing each component or lesion at its own position gives
  the observed signal: nnU-Net 1235/1235 lesions, duct "≤ 10 mm" 1003/1003 lesions against the master table's
  `dist_to_mpd_mm`.
- **Predicted-pancreas overlap** equals S13's `les_in_panc` for 1235/1235.
- **Specificity.** Raw specificity 0.7217 and FP/negative 0.3303 reproduce S8. S8b-weighted specificity 0.70339
  reproduces S8b (ESS 186.9).
- **D.** Mean lesion Dice is 0.418 (S13's LOFO pool: 0.418, *carried over*). D touches 7/116 of the 20-40 mm misses,
  equal to S13's "rescued 7" (*carried over*).
- **Oracle tests.** 11 tests in `tests/test_rescue_check_502.py`, written before the real run:
  - FFT placement vs a brute-force loop on random 3D shapes
  - hand-counted placement fractions
  - phantoms for included / dent / adjacent / away on an anisotropic grid
  - the 10.0 / 12.5 mm duct boundary
  - uint8 mask input

  A mutation check confirmed the tests can fail: 3 of 4 mutations were caught, and the fourth is equivalent
  (cannot change the output).

## Model performance (502 + D)

| bin | n | missed (raw touch) | touch, raw 502 | Dice ≥ 0.1, 502 + D | touch, 502 + D | mean Dice raw → D |
|---|---|---|---|---|---|---|
| 10-20 | 275 | 144 | 47.6% | 42.5% | 50.5% | 0.196 → 0.216 |
| **20-40** | 597 | **116** | 80.6% | 77.7% | 81.7% | 0.468 → 0.486 |
| ≥40 | 277 | 24 | 91.3% | 87.7% | 91.7% | 0.578 → 0.599 |

- **Misses under D.** None of the 116 20-40 mm misses reaches Dice 0.1 under D (max 0.017). One of the 144
  10-20 mm misses does.
- **Specificity.** Unchanged by D: 0.722 unweighted, 0.703 S8b-weighted. FP components per negative go
  0.330 → 0.315 (weighted 0.375 → 0.341).

## 1. What the misses look like (plain table)

Share of missed lesions with each property, with the detected lesions of the same bin for comparison
(Wilson 95% CI on the missed share):

| property | 20-40 missed (n 116) | 20-40 detected (n 481) | 10-20 missed (n 144) | 10-20 detected (n 131) |
|---|---|---|---|---|
| non-contrast (est. phase) | **31.0%** [23-40] | 17.5% | 33.3% [26-41] | 22.9% |
| iso (v3) | 42.2% [34-51] | 38.5% | 36.8% [29-45] | 35.1% |
| satellite | **35.3%** [27-44] | 6.7% | **54.9%** [47-63] | 15.3% |
| outside gland (GT tier `separated`) | 16.4% [11-24] | 2.1% | 15.3% [10-22] | 3.1% |
| < 50% inside 502's predicted pancreas | **47.4%** [39-56] | 4.2% | **42.4%** [35-51] | 5.3% |
| at least one of the five | 86.2% | 49.7% | 93.1% | 56.5% |

**The rest.** Only 16 of the 20-40 mm misses (and 10 at 10-20 mm) have none of the five properties. All of them are
hypo or hyper lesions, mostly in venous scans.

## 2. Missed vs detected, same size bin

Missed share vs detected share (Fisher p, descriptive; many comparisons). Last column: the detection rate within
that level, as touch / Dice ≥ 0.1 under D.

**20-40 mm (missed 116, detected 481)**

| variable | level | missed | detected | p | detection in level |
|---|---|---|---|---|---|
| phase | non-contrast | 31.0% | 17.5% | 0.002 | 70.0 / 62.5% (n 120) |
| | early arterial | 9.5% | 5.4% | 0.13 | 70.3 / 64.9% (n 37) |
| | late arterial | 8.6% | 6.4% | 0.41 | 75.6 / 73.2% (n 41) |
| | venous | 47.4% | 66.3% | 0.0003 | 85.3 / 84.2% (n 374) |
| | uncertain / unmeasurable | 3.4% | 4.4% | | n 21 / 4, not interpreted |
| attenuation v3 | hypo / iso / hyper | 42 / 42 / 16% | 45 / 38 / 16% | ≥ 0.46 | 81.6 / 79.1 / 81.4% touch |
| satellite | yes | 35.3% | 6.7% | 4e-14 | 43.8 / 39.7% (n 73) |
| tier | inside | 52.6% | 84.2% | 5e-12 | 86.9 / 84.3% |
| | embedded | 9.5% | 6.4% | 0.31 | 73.8 / 66.7% |
| | abutting | 21.6% | 7.3% | 2e-5 | 58.3 / 58.3% |
| | separated | 16.4% | 2.1% | 3e-8 | 34.5 / 27.6% |
| region | head / body / tail | 59 / 21 / 17% | 73 / 15 / 11% | 0.004 (head) | 83.6 / 75.0 / 73.3% touch |
| thickness | <2.5 / 2.5-5 / ≥5 mm | 53 / 25 / 22% | 31 / 52 / 16% | 1e-5 (<2.5) | 70.9 / 89.7 / 75.7% touch |
| predicted-pancreas category | included / adjacent / away | 53 / 33 / 15% | 96 / 4 / 0% | 1e-28 | 88.3 / 33.3 / 0% touch |

| continuous, median [IQR] | missed | detected | Mann-Whitney p |
|---|---|---|---|
| slice thickness (mm) | 1.5 [0.8-3] | 2.5 [1.25-2.5] | 0.007 |
| diameter (mm) | 26.2 [22.6-31.1] | 29.3 [24.6-34.3] | 6e-5 |
| distance to predicted pancreas (mm) | 0 [0-0.75] | 0 [0-0] | 3e-30 |
| share of lesion in predicted pancreas | 0.58 [0-0.99] | 0.99 [0.95-1] | 2e-15 |
| 3 mm shell share in predicted pancreas | 0.39 [0.04-0.71] | 0.69 [0.54-0.85] | 7e-12 |
| 502 peak p in lesion + 2 mm | 0.0020 [4e-5-0.040] | 1.00 [0.999-1] | 1e-62 |

**10-20 mm (missed 144, detected 131).**
- The same pattern holds, mostly weaker:
  - satellite 54.9% vs 15.3% (p 5e-12)
  - separated 15.3% vs 3.1%
  - embedded 16.0% vs 4.6%
  - non-contrast 33.3% vs 22.9% (p 0.06)
  - venous 43.8% vs 61.1% (p 0.005)
  - iso 36.8% vs 35.1% (p 0.8)
- Thickness differs only at 2.5-5 mm (20% vs 40%).
- Predicted-pancreas categories: included 57.6%, adjacent 19.4%, away 21.5%, dent 1.4% (detected: 94.7% included).
- Median peak p in lesion + 2 mm is 4e-4.

**Satellite misses.** Most of them are in cases where raw 502 finds another lesion: 58/79 at 10-20 mm and 28/41 at
20-40 mm.

**Thin slices.** Missed 20-40 mm lesions sit on thinner slices. This is probably the S12 phase ≈ thickness confound
(est non-contrast scans are mostly ≤ 2 mm), not a thickness effect in itself. It was not separated here.

## 3. Included, adjacent or away: the bulge-vs-dent question

| | included | dent | adjacent | away |
|---|---|---|---|---|
| 20-40 missed (116) | **61 (52.6%)** | 0 | 38 (32.8%) | 17 (14.7%) |
| 20-40 detected (481) | 461 (95.8%) | 1 | 19 (4.0%) | 0 |
| 10-20 missed (144) | **83 (57.6%)** | 2 | 28 (19.4%) | 31 (21.5%) |
| 10-20 detected (131) | 124 (94.7%) | 0 | 7 (5.3%) | 0 |
| ≥40 missed (24) | 4 | 0 | 10 | 10 |

**Category × GT tier, 20-40 mm misses.**
- Included: 50 inside, 7 embedded, 3 abutting, 1 separated.
- Adjacent: 22 abutting, 10 inside, 4 embedded, 2 separated.
- Away: 16 separated, 1 inside.

**Profile of the 61 included 20-40 mm misses.**
- iso 37, hyper 13, hypo 11
- non-contrast 23, venous 28
- 13 satellites
- 22 lie 100% inside the predicted gland

The model labels these lesions as pancreas but not as lesion. Most of them look like gland (iso or unenhanced).

**The not-included misses are not dents.** Their 3 mm shell is median 3% predicted gland, at a median distance of
1.0 mm. 502 ends its gland at the lesion and does not wrap around it.

## 4. Signals vs chance (S9 dropped)

| bin | group | signal | hit rate | chance | hit − chance [95% CI] |
|---|---|---|---|---|---|
| 20-40 | missed | nnU-Net OP | 35/116 = 30.2% | 9.8% | **+20.3** [+12.6, +27.7] |
| 20-40 | missed | duct OP | 10/83 = 12.0% (NE 33) | 14.6% | −2.5 [−6.9, +2.0] |
| 20-40 | detected | nnU-Net OP | 478/481 = 99.4% | 26.5% | +72.9 [+71.3, +74.5] |
| 20-40 | detected | duct OP | 201/409 = 49.1% (NE 72) | 41.6% | +7.6 [+5.3, +9.7] |
| 10-20 | missed | nnU-Net OP | 31/144 = 21.5% | 6.1% | **+15.5** [+8.5, +22.3] |
| 10-20 | missed | duct OP | 7/109 = 6.4% (NE 35) | 8.8% | −2.4 [−6.8, +1.7] |
| 10-20 | detected | nnU-Net OP | 131/131 = 100% | 10.1% | +89.9 [+88.4, +91.1] |
| 10-20 | detected | duct OP | 21/110 = 19.1% (NE 21) | 17.7% | +1.4 [−2.6, +5.5] |
| ≥40 | missed | nnU-Net OP | 9/24 = 37.5% | 15.3% | +22.2 [+4.5, +40.5] |
| ≥40 | missed | duct OP | 1/16 (NE 8) | 7.1% | −0.8 |

NE = not evaluable (no MPD mask).

**Duct, split into its two parts (MPD mask present).**
- Case flag: 22.9% of 20-40 mm misses vs 54.8% of detected lesions.
- Within 10 mm of the MPD: **28.9% observed vs 38.6% by placement** for misses (detected: 65.3% vs 58.6%).
- So the missed lesions sit *farther* from the duct than random placement would put them. Neither part helps on the
  misses.

**Costs on the 327 negatives (S8b-weighted in brackets).**
- nnU-Net OP: 0.67 [0.79] FP components per negative; 47.4% [52.5%] of negatives get at least one.
- Duct case flag: fires on 6.4% [9.7%] of negatives.

**Faint candidates are descriptive only (no rescue claim).**
- **This differs from the brief's carried-over count.** S13's 24/116 counts touches by B's *mask* (voxels > 0.5,
  else ≥ half the peak), which is a speck. Re-run here through S13's code: 24/116, max Dice 0.087. S11's signal
  counts the whole p > 0.01 *footprint*, which touches 35/116.
- Footprint Dice on those 35: median 0.050, max 0.382, 8 ≥ 0.1. At 10-20 mm: 31 touched, median 0.129, 16 ≥ 0.1.
- These are post-hoc views of a large faint blob, not segmentations.

**Combined.** Either signal flags 36.2% of 20-40 mm misses and 25.0% of 10-20 mm misses.

## 5. Recommendation for S15 (gland-shape pilot)

**Target: missed 20-40 mm index lesions that 502's own predicted pancreas includes (≥ 50% of voxels), excluding
GT `separated` lesions. n = 48.**

- **Why this subgroup.** It is the largest coherent group of misses. In these lesions the model has already
  segmented the gland around the tumor, so a shape cue (a local bulge or thickening of the predicted gland) is
  available. Intensity cues are weak here: 37/61 of the included misses are iso.
- **Why not dents.** There are none: 0/116 at 20-40 mm.
- **Why not the other misses.** The adjacent and away misses lie outside the predicted gland, so a gland-shape
  method cannot see them.
- **Subsets of the 48:**
  - 30 are in enhanced scans (not est non-contrast)
  - 16 are enhanced and not iso
- **Comparison pool.** The 461 detected 20-40 mm lesions that are also included, for "does a detected tumor bulge
  the same way?"
- **Secondary.** The 10-20 mm equivalent is n = 46 (25 enhanced, 10 enhanced and not iso), with a comparison pool of
  124.
- **Size.** n = 48 is small. Treat S15 as a pilot with a pre-set effect size, not a performance claim.

## Caveats

- **Uniform placement is one null among several.** Lesions are not uniformly placed (the head holds 63-70%,
  *carried over* from CONTEXT). The annotated duct is densest in the head (S10, *carried over*), so a
  head-weighted null would most likely *raise* the duct's chance rate. Its below-chance result is therefore unlikely
  to be an artifact of this choice. This was not tested. The domain is built from GT pancreas masks, which never touch the lesion.
- **Assumptions inside the nnU-Net chance rate:**
  - each rank ≤ 3 component is placed independently
  - placements may leave the domain or overlap each other
  - it is a reference rate, not a calibrated p-value
- **Reading the "detected" rows.** "Detected" means raw touch. The nnU-Net signal's ~100% there is trivial: the
  t = 0.5 object lies inside a t = 0.01 component.
- **Duct signal sources.** The duct signal uses S11's sources unchanged: `mpd_dilated` from the S2-era
  `duct_dilation_flags.csv` and the master table's MPD distance.
- **Unresolved confounds.** Phase and thickness are confounded (S12). Satellites are mostly smaller and in
  multi-lesion cases. No stratum was adjusted for the others.
- **D.** D is the fixed rule, not S13's per-fold LOFO picks. The aggregate Dice equals S13's to 3 decimals.

## Files

| file | content |
|---|---|
| `missed_lesions.csv` | one row per lesion that raw 502 misses (367 rows, all bins). Keys `case_id, lesion_id`; descriptors (phase5, attenuation_v3, satellite, tier, region, thickness); `touch_raw, dice_raw, touch_d, dice_d, det_dice_d`; predicted-pancreas `panc_frac_in, panc_frac_filled, panc_shell_frac, panc_dist_mm, panc_category`; `peak_p_dil2`; S11 signals `sig_nnunet, sig_duct` with `nn_op_chance, duct_near_g, duct_chance`; `nn_op_dice` (descriptive). `work/rescue_check_502/all_lesions.csv` has the same columns for all 1235 lesions |

```
PYTHONPATH=src .venv/bin/python -m pytest tests/test_rescue_check_502.py
PYTHONPATH=src .venv/bin/python -m rescue_check_502.export_s8
PYTHONPATH=src .venv/bin/python -m rescue_check.nnunet_table --pred_source work/rescue_check_502/pred_source --tag ds502
PYTHONPATH=src .venv/bin/python -m rescue_check_502.case_pass --workers 4      # ~7 min, PC
PYTHONPATH=src .venv/bin/python -m rescue_check_502.analyze
```
