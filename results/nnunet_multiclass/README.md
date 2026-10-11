# Dataset502 (pancreas + lesion) vs Dataset501 (binary lesion): lesion channel, identical protocol (session S8)

**Answer: the multi-class model is better on every lesion metric. It also cuts false-positive lesion blobs by a factor of 6.**

All results are 5-fold out-of-fold (OOF), with each case scored by the fold model that never saw it. Both models were scored by the same code on the same 1308 cases.

| lesion set = all | 501 | 502 | delta [95% CI] |
|---|---|---|---|
| mean lesion Dice, 1235 lesions | 0.280 | **0.400** | +0.121 [+0.107, +0.134] |
| detect@any, 1235 lesions | 53.4% | **70.3%** | +16.9% [+14.3, +19.7] |
| 10-20 mm: Dice / detect@any (n = 275) | 0.074 / 25.8% | **0.196 / 47.6%** | +0.122 [+0.091, +0.152] / +21.8% [+15.3, +28.9] |
| 20-40 mm: Dice / detect@any (n = 597) | 0.321 / 59.8% | **0.468 / 80.6%** | +0.147 [+0.127, +0.168] / +20.8% [+16.9, +24.7] |
| tumor+ cases localized (981) | 631 (64.3%) | **823 (83.9%)** | +19.6% [+16.7, +22.3] |
| FP lesion components per negative case (327) | 2.06 | **0.33** | −1.73 [−1.95, −1.51] |
| case specificity (any lesion voxel) | 0.223 | **0.722** | |

The 240 20-40 mm lesions that 501 missed completely:
- **502 detects 140 of them (58.3%)**, with mean Dice 0.441 among those it finds.
- In exchange it loses 16 of the 357 that 501 found.

Every number here was computed in S8 unless it is marked *carried over*.

## How to rerun

```
# server (where the OOF exports are): phantoms, then 1308 cases x 2 models, 6 workers, ~45 min, peak 3.3 GB/worker
PYTHONPATH=src ~/nnunet_setup/.venv/bin/python work/nnunet_multiclass/run_tests.py   # pytest is not installed there
PYTHONPATH=src ~/nnunet_setup/.venv/bin/python -m multiclass_eval.case_pass --workers 6
# PC: rsync work/nnunet_multiclass/cases + the 10 fold summary.json files, then
.venv/bin/python -m pytest tests/test_multiclass_eval.py -q          # 12 phantom / known-answer tests
PYTHONPATH=src .venv/bin/python -m multiclass_eval.summarize         # regression asserts + these files
```

- **Code:** `src/multiclass_eval/` (`core.py`, `case_pass.py`, `stats.py`, `summarize.py`).
- **Full log:** every number in this README is in `work/nnunet_multiclass/summary.txt`.
- **evaluate.py-format aggregates:** `work/nnunet_multiclass/evaluation_{501,502}_{all,excl_separated}.json`.
- **502 ladder:** `work/nnunet_multiclass/ladder_502.csv`.

## Files

| file | rows | content |
|---|---|---|
| `per_lesion_comparison.csv` | 1235 | One row per GT lesion, keyed `(case_id, lesion_id)` (tumorlib, nibabel order). Strata columns: size, slice thickness, metadata phase, S12 estimated phase, v3 attenuation, region, tier, excluded. Then per model (`_baseline` = 501, `_multiclass` = 502): lesion Dice, IoU, detect@any, detect@IoU≥0.1, peak p in lesion + 2 mm, and the highest ladder threshold with a touch hit and with a peak hit. Last columns: `dice_delta` and `change` (both / recovered / lost / neither). |
| `summary_vs_baseline.csv` | 893 | Long table with columns `section, metric, lesion_set, stratum_var, stratum, t, n, n_cases, baseline, multiclass, delta, ci_lo, ci_hi, p_boot, note`. Sections: `case_seg`, `case_detect_any_voxel`, `lesion_by_size_evaluate`, `pancreas`, `localization`, `false_positives`, `ladder`, `ladder_fp_matched`, `paired_lesion`, `per_fold`, `missed_20_40`. |

## Method

**Inputs (both models).**
- Each model's `fold_k/validation/{case}.nii.gz` and `.npz`.
- Lesion GT = Dataset501 `gt_segmentations`. Per case, it was re-checked to be voxel-identical to Dataset502 label 2.

**Lesion mask and lesion probability.**
- 501: mask = `seg == 1`; probability = softmax channel 1.
- 502: mask = `seg == 2`; probability = sigmoid region channel 1.
- In both models `seg == lesion` equals `p > 0.5` for all 1308 cases with 0 mismatching voxels. For 502 the gland region also has 0 mismatches. So the t = 0.5 rung of the ladder *is* the saved segmentation.

**Reused, not re-implemented.**
- `testing/nnunet/evaluate.py`'s own `overlap_metrics`, `boundary_metrics`, `lesion_level` and `detection_block` are imported and run on nibabel-order arrays.
- S2's `analyze()` (threshold ladder, peak and touch rules, search region) is imported unchanged.
- No existing script was edited.

**Keys and definitions.**
- **lesion_id.** evaluate.py's numbering on an (x, y, z) array equals tumorlib's `lesion_id`. This is asserted per case.
- **Lesion Dice.** As in evaluate.py: Dice of the lesion against the union of the predicted components that touch it.
- **Detected.** Any overlap.
- **Localized case.** A predicted component touches GT.
- **FP component.** A 26-connected predicted component that touches no GT lesion. One that touches an excluded lesion is not an FP.

**Without the 70 excluded lesions.**
- Case level uses `src/nnunet/prepare_excluded_eval.py`'s ignore-region rule:
  - excluded GT is removed
  - a predicted component touching only excluded lesions is removed
  - predicted voxels inside an excluded lesion are removed
  - 18 cases with no lesion left are dropped
- Per lesion, the excluded rows are dropped. The phantom test runs the real script and gets the identical masks.

**Bootstrap.**
- Paired and case-level: 1000 resamples of tumor+ cases (seed 0), percentile 95% CI.
- `p_boot` is the two-sided bootstrap p for delta = 0, at a resolution of 1/1000.
- CIs are **not** adjusted for the many strata.

## Regression: 501 reproduces everything on record (asserted)

| check | result |
|---|---|
| localized / wrong place / nothing | **631 / 288 / 62** (*carried over* 631 / 288 / 62) |
| detect@any by bin <5 / 5-10 / 10-20 / 20-40 / ≥40 | **2.0 / 0.0 / 25.8 / 59.8 / 83.0%** (*carried over* same) |
| per-lesion Dice and detect@any vs `master_lesions.csv` | equal for all 1235 (max \|Δ\| 4.9e-7) |
| per-case TP / n_ref / n_pred vs nnU-Net `summary.json` | equal for all 1308 cases. 501 region 1; 502 regions 2 and (1,2). Per-fold mean Dice reproduced (502 lesion 0.4608 / 0.4408 / 0.4040 / 0.4321 / 0.4213) |
| evaluate.py aggregates vs `results/nnunet_fold{0..4}/evaluation.json` and `nnunet_excl_outside_pancreas/fold{0..4}/evaluation.json` | equal (rtol 1e-5) for every field except NSD@3 mm in folds 3 and 4 (see below) |
| threshold ladder vs S2 `nnunet_froc_points.csv` | all 216 rows equal |
| S2 per-lesion peak p, max-t hit (touch, peak) | equal for all 1235 |
| S11's fold-0 502 detections (read independently on the PC) | equal for all 248 fold-0 lesions; missed 20-40 mm recovered 25/36 (*carried over* 25/36) |

**The one difference: NSD@3 mm in two cases.** PanTS_00007130 (0.690 vs 0.713) and PanTS_00001751 (0.293 vs 0.299).
- Both are 1.5 mm isotropic scans. A 2-voxel surface distance is then exactly 3.000 mm, which sits on the `≤ 3 mm` tolerance.
- evaluate.py reads float32 header spacing. tumorlib's affine spacing is 1.5 ± 1e-7. So those distances fall on opposite sides of the cutoff.
- The two S8 models share one spacing, so the comparison is unaffected.

**Phantom tests before real data:** 12 tests, all pass on the PC and the server. They cover:
- label 1 over a lesion counts as missed
- the 501 and 502 code paths give identical output for identical probabilities
- the real `prepare_excluded_eval.py` and `evaluate.py` run as subprocesses: same masks, same metrics
- the bootstrap equals a literal resample loop

**Sabotage check.** Reading 502's lesion as `seg > 0` flips the missed lesion to detected, and the built-in `p > 0.5` vs seg check flags 17,835 voxels.

## Step 2: evaluate.py metrics on the lesion mask

**Case level, lesion set all.** 981 reference-positive cases; detection metrics are over all 1308 cases.

| metric | 501 | 502 |
|---|---|---|
| Dice mean / median | 0.312 / 0.205 | **0.472 / 0.537** |
| IoU mean | 0.234 | 0.367 |
| HD95 mean / median (mm) | 75.4 / 57.8 | **22.2 / 10.7** |
| ASSD mean (mm) | 26.3 | 7.0 |
| NSD@2 mm mean | 0.298 | 0.517 |
| relative volume difference, median | +0.043 | **−0.187** |
| ref-positive cases with an empty prediction | 62 | **83** |
| case Se / Sp (any lesion voxel) | 0.937 / 0.223 | 0.915 / **0.722** |
| PPV / NPV | 0.784 / 0.541 | 0.908 / 0.740 |
| AUROC: predicted volume / max p | 0.747 / 0.683 | **0.906 / 0.887** |

- **Paired case Dice:** +0.159 [+0.145, +0.174].
- **Case sensitivity falls by 2 points.** That is not a loss of localization. Of 501's 62 + 288 cases with nothing or nothing correct, 58 wrong-place cases become empty predictions in 502 (transitions in Step 3).
- **502 under-segments** (median volume −19%). This matches the ≥40 mm picture in CONTEXT.

**Per lesion, by Feret diameter** (evaluate.py bins, lesion set all):

| bin | n | Dice 501 → 502 | Dice among detected | detect@any | detect@IoU≥0.1 |
|---|---|---|---|---|---|
| <5 | 49 | 0.000 → 0.000 | 0.007 → — | 2.0 → 0.0% | 0 → 0% |
| 5-10 | 37 | 0.000 → 0.030 | — → 0.375 | 0.0 → 8.1% | 0 → 8.1% |
| 10-20 | 275 | 0.074 → **0.196** | 0.287 → 0.412 | 25.8 → **47.6%** | 14.2 → 35.3% |
| 20-40 | 597 | 0.321 → **0.468** | 0.536 → 0.581 | 59.8 → **80.6%** | 51.8 → 71.9% |
| ≥40 | 277 | 0.483 → 0.578 | 0.581 → 0.633 | 83.0 → 91.3% | 71.8 → 79.8% |

Without the 70 excluded lesions:
- 10-20 mm: 0.078 → 0.217 Dice, 26.1 → 51.0% detect@any (n = 249).
- 20-40 mm: 0.333 → 0.485, 61.4 → 82.9% (n = 568).

At 20-40 mm most of the Dice gain is detection. Dice among detected lesions rises only 0.536 → 0.581.

## Step 1's separate check: the pancreas channel (502 only; not a gland-segmentation benchmark)

**Setup.** Region (1,2) vs 502 GT label > 0.
- Per-case Dice mean **0.860**, median 0.895, p10 0.758 (1297 cases with a non-empty GT). Pooled voxel Dice 0.869.
- Tumor+ 0.863, tumor− 0.854.
- nnU-Net's per-fold means are reproduced exactly: 0.870 / 0.870 / 0.854 / 0.850 / 0.850.

**Problem cases.**
- 2 of the 11 empty-envelope negatives get a predicted gland.
- 6 negatives score Dice 0. Three are S7's near-empty envelopes (1, 109 and 527 GT voxels). The others (PanTS_00005371, 00007136, 00007508; 23,316 / 3,405 / 1,366 GT voxels) get almost no gland predicted where the GT puts it.

**Caveat (S7):** class 1 absorbs any tumor that bulges outside the gland masks, so this is not a clean gland metric.

**Descriptive only.** In tumor+ cases the gland Dice is 0.872 when 502 finds a lesion vs 0.816 when it finds none.

## Step 3: localization-aware detection

| | 501 | 502 |
|---|---|---|
| localized | 631 (64.3%) | **823 (83.9%)** |
| wrong place only | 288 (29.4%) | **75 (7.6%)** |
| nothing | 62 (6.3%) | 83 (8.5%) |

**Transitions:**
- Of 501's 288 wrong-place cases, 171 become localized, 58 become empty and 59 stay wrong.
- Of 501's 631 localized cases, 604 stay localized; 14 become empty and 13 wrong.

**By largest lesion:**

| largest lesion | n | 501 → 502 localized | delta [95% CI] |
|---|---|---|---|
| <10 mm | 11 | 0 → 1 | |
| 10-20 mm | 176 | 31.8 → **63.1%** | +31.3% [+22.2, +39.3] |
| 20-40 mm | 524 | 65.6 → 87.0% | +21.4% [+17.3, +25.3] |
| ≥40 mm | 270 | 85.6 → 94.4% | +8.9% [+4.7, +13.4] |

Without the excluded lesions (963 cases): 623 / 281 / 59 → 814 / 72 / 77.

**FP lesion components at the saved segmentation:**

| | 501 | 502 |
|---|---|---|
| per negative case | 2.06 | **0.33** [Δ −1.95, −1.51] |
| negatives with any FP component | 77.7% | 27.8% |
| per tumor+ case | 1.60 | 0.33 |

## Step 4: threshold ladder (S2 protocol) on the lesion probability, side by side

Sensitivity is the fraction of all 1235 lesions hit. FP = components per negative case. The full table is in `summary_vs_baseline.csv`, section `ladder`.

| t | touch: all | 10-20 | 20-40 | peak-in-2 mm: all | 10-20 | 20-40 | FP/neg |
|---|---|---|---|---|---|---|---|
| 0.9 | 48.3 → 66.3% | 17.8 → 40.7 | 54.8 → 76.7 | 41.4 → 61.0% | 12.0 → 33.8 | 45.9 → 70.9 | 1.52 → 0.26 |
| 0.5 | 53.4 → 70.3% | 25.8 → 47.6 | 59.8 → 80.6 | 44.5 → 63.5% | 15.6 → 38.2 | 48.9 → 73.4 | 2.06 → 0.33 |
| 0.1 | 57.4 → 74.1% | 30.9 → 54.2 | 64.3 → 83.9 | 47.0 → 65.3% | 17.8 → 41.1 | 52.1 → 75.4 | 2.55 → 0.48 |
| 0.01 | 62.8 → 77.4% | 37.8 → 60.0 | 70.0 → 86.8 | 49.8 → 66.8% | 21.1 → 44.7 | 55.4 → 76.7 | 3.34 → 0.68 |
| 0.001 | 70.3 → 81.4% | 50.2 → 66.2 | 77.1 → 90.1 | 52.6 → 68.4% | 25.5 → 48.0 | 58.8 → 77.9 | 5.09 → 1.22 |

**FP-matched comparison.** The two probabilities are calibrated differently (softmax vs sigmoid), so the comparison that matters is at matched FP.
- 502's *lowest* rung (t = 0.001, 1.22 FP/neg) still has fewer FPs than 501's *highest* rung (t = 0.9, 1.52 FP/neg).
- **So 502 dominates 501 at every operating point on this ladder.** At ≤1.52 FP/neg: 81.4% vs 48.3% overall, 66.2% vs 17.8% at 10-20 mm.
- **The 502 FROC curve is not yet closed at the low-FP end.** The ladder stops at t = 0.001.

## Step 5: paired per-lesion comparison (case bootstrap, B = 1000)

**Lesion set all.** Delta = 502 − 501, with 95% CI.

| stratum | n | Dice 501 → 502 | ΔDice [CI] | detect 501 → 502 | Δdetect [CI] | recovered / lost |
|---|---|---|---|---|---|---|
| all | 1235 | 0.280 → 0.400 | +0.121 [+0.107, +0.134] | 53.4 → 70.3% | +16.9 [+14.3, +19.7] | 252 / 43 |
| 10-20 hypo (v3) | 140 | 0.091 → 0.221 | +0.130 [+0.081, +0.179] | 27.9 → 46.4 | +18.6 [+8.3, +28.6] | 36 / 10 |
| 10-20 iso | 99 | 0.050 → **0.143** | +0.093 [+0.059, +0.130] | 25.3 → 46.5 | +21.2 [+9.8, +33.7] | 29 / 8 |
| 10-20 hyper | 36 | 0.073 → 0.246 | +0.173 [+0.090, +0.259] | 19.4 → 55.6 | +36.1 [+18.2, +53.1] | 14 / 1 |
| 10-20 head / body / tail | 168 / 49 / 57 | 0.095 / 0.014 / 0.065 → 0.200 / 0.205 / 0.182 | all CIs > 0 | 32.1 / 10.2 / 21.1 → 51.8 / 38.8 / 43.9 | | 46/13, 17/3, 16/3 |
| 10-20 est. venous / non-contrast / arterial | 143 / 78 / 42 | 0.098 / 0.034 / 0.086 → 0.234 / 0.166 / 0.175 | all CIs > 0 | 30.1 / 21.8 / 23.8 → 55.9 / 38.5 / 42.9 | | |
| 10-20 slices <2.5 / 2.5-5 / ≥5 mm | 131 / 82 / 62 | 0.069 / 0.108 / 0.039 → 0.174 / 0.264 / 0.154 | all CIs > 0 | 27.5 / 30.5 / 16.1 → 42.7 / 64.6 / 35.5 | | 34/14, 29/1, 16/4 |
| 10-20 tier inside / abutting / embedded / separated | 187 / 33 / 29 / 26 | 0.080 / 0.117 / 0.020 / 0.039 → 0.245 / 0.158 / 0.098 / **0.004** | inside > 0; abutting n.s.; separated −0.035 [−0.096, +0.003] | 27.3 / 27.3 / 17.2 / 23.1 → 57.2 / 42.4 / 20.7 / **15.4** | | |
| 20-40 hypo / iso / hyper | 266 / 234 / 97 | 0.363 / 0.278 / 0.309 → 0.513 / 0.434 / 0.426 | all CIs > 0 | 65.0 / 53.0 / 61.9 → 81.6 / 79.1 / 81.4 | | 55/11, 66/5, 19/0 |
| 20-40 est. non-contrast | 120 | 0.137 → 0.336 | **+0.199** [+0.156, +0.246] | 34.2 → 70.0 | +35.8 [+26.4, +45.4] | 46 / 3 |
| 20-40 tier embedded / abutting / separated | 42 / 60 / 29 | 0.148 / 0.205 / 0.087 → 0.358 / 0.321 / 0.122 | embedded and abutting > 0; separated n.s. | 31.0 / 48.3 / 27.6 → 73.8 / 58.3 / 34.5 | abutting +10.0 [−5.0, +23.1] | |

All other strata are in `summary_vs_baseline.csv` (`section == paired_lesion`). They cover size × v3 attenuation, region, metadata phase, S12 estimated phase, slice thickness and tier, for both lesion sets.

**Per fold.** Each fold is an independently trained pair of models. 502 beats 501 on mean lesion Dice, detect@any, 10-20 mm and 20-40 mm Dice, and FP/neg **in all 5 folds**:
- 10-20 mm Dice: 0.118→0.187, 0.099→0.202, 0.051→0.124, 0.078→0.253, 0.018→0.217.
- FP/neg: 1.92→0.33, 2.00→0.21, 2.29→0.40, 2.17→0.37, 1.94→0.34.

### Did the 20-40 mm lesions the baseline missed completely get recovered? Mostly, yes.

**Lesion set all.**
- 501 missed 240.
- **502 detects 140 (58.3%).** Among them, mean Dice 0.441 (median 0.483) and 67 reach Dice ≥ 0.5.
- Mean Dice over all 240, counting the 100 still missed as 0: 0.258.

**Recovery by stratum:**
- v3 attenuation: iso 66/110, hypo 55/93, hyper 19/37.
- Tier: inside 105/159, embedded 18/29, abutting 12/31, separated 5/21.
- Estimated phase: venous 71/114, non-contrast 46/79, arterial 18/38.

**The 100 still missed.**
- 15 are touched by a 502 component at t ≥ 0.1, 28 at t ≥ 0.01, 44 at t ≥ 0.001.
- Their median peak p in lesion + 2 mm is 8.1e-4 (501: 2.0e-4).
- So most remain flat.

**Losses.** 502 loses 16 of the 357 20-40 mm lesions that 501 detected. Their 501 Dice was low anyway (mean 0.199).

**Without the excluded lesions:** 135/219 recovered (61.6%), 13/349 lost.

## Caveats

- **Not a single-variable experiment.** 501 vs 502 differs in:
  - the label definition
  - the lesion-centred patch rate: 502 sees about **half** as many lesion-centred patches (S7 §6), which works *against* 502
  - CT normalization statistics (gland vs tumor foreground)
  - 11 empty-envelope cases dropped from 502's training lists

  The gain cannot be attributed to "knowing the pancreas" alone.
- **One training run per model and fold.** The bootstrap resamples cases, not training runs. The per-fold table is the only look at run-to-run variation, and the direction holds in all 5 folds.
- **Many strata, unadjusted CIs.** Read the stratum rows as descriptive. Small strata (Delay phase, `unmeasurable`, region `none`) are noise.
- **Separated lesions do not improve.** At 10-20 mm, separated lesions go from 23.1% to 15.4% detected (n = 26, CI spans 0). That is the direction you would expect from a gland prior; they sit fully outside the gland masks (S7).
- **Pancreas Dice is against a label that includes extra-glandular tumor** (S7). The GT gland masks are PanTS auto-segmentation, not expert truth.
- **Negatives are thinner-sliced and, by S12, not phase-matched to positives.** Specificity and FP/neg inherit that. 30 negatives have report-described lesions (S1), so some "FP" may not be false.
- **FP-matched ladder comparisons are at discrete rungs, not interpolated.**

## What surprised me

1. **The size of it.** 10-20 mm detect@any nearly doubles (25.8 → 47.6%) despite the halved lesion-patch rate. The FP-matched gap is larger still: 66% vs 18% at ≤1.5 FP/neg.
2. **501's case sensitivity of 0.937 was propping itself up on wrong-place blobs.** 502's case "sensitivity" is *lower* (0.915) while localization rises from 64% to 84%.
3. **Most of 502's improvement is fewer false blobs.** FP/neg drops from 2.06 to 0.33, and 72% of negatives now get no lesion voxel at all. The gland context seems to suppress off-pancreas detections more than it sharpens on-pancreas ones.
4. **The largest relative gains are where 501 was weakest:**
   - estimated non-contrast 20-40 mm: 34% → 70% detected
   - body 10-20 mm: 10% → 39%
   - embedded 20-40 mm: 31% → 74%
5. **502 under-segments** (median volume −19%) where 501 did not (+4%). The ≥40 mm under-segmentation problem from CONTEXT priority 2 is now general.
