# S13: cheap post-processing of Dataset502's lesion channel (no retraining, leave-one-fold-out)

**Answer.** Only one of the specified families helps: **D, regrowing the saved segmentation's own objects to p > 0.1**. It raises the objective from 0.5342 to 0.5446 (+0.0104, 95% CI +0.008 to +0.013), and it is positive in all 5 held-out folds.

- **A** (out-of-pancreas filter) is nearly a no-op: +0.0023, CI [+0.000, +0.005]. 502's lesion predictions already lie inside its own pancreas prediction.
- **B** (faint-candidate rescue) never pays. The tuning picks the no-op.
- **C = A+B+D as specified** reduces to A.
- **C+** is a post-hoc grid that also allows D's selection. It reaches +0.0183, but it does so by trading 22 S8-detected lesions for 12 cleared negatives.

The objective, fixed before results, is the mean per-case lesion Dice over all 1308 cases. A tumor+ case scores its case Dice (0 if nothing is predicted). A tumor- case scores 1 if nothing is predicted, else 0.

## Scoreboard (pooled held-out folds; each fold uses the setting picked on the other four)

| | S8 502 | A filter | B rescue | D regrow | C = A+B+D | C+ (post hoc) |
|---|---|---|---|---|---|---|
| **objective (1308)** | 0.5342 | 0.5365 | 0.5342 | **0.5446** | 0.5365 | 0.5526 |
| Δ vs S8 [95% CI] | | +0.0023 [+0.000, +0.005] | −0.0000 | **+0.0104 [+0.008, +0.013]** | +0.0023 [−0.000, +0.005] | +0.0183 [+0.013, +0.025] |
| case Dice, tumor+ (981) | 0.472 | 0.472 | 0.472 | **0.486** | 0.472 | 0.484 |
| case specificity, raw | 0.722 | 0.731 | 0.722 | 0.722 | 0.731 | 0.758 |
| case specificity, reweighted* | 0.703 | 0.710 | 0.703 | 0.703 | 0.710 | 0.746 |
| FP components / negative | 0.330 | 0.318 | 0.330 | 0.315 | 0.318 | 0.260 |
| FP components / tumor+ (non-tumor) | 0.334 | 0.332 | 0.334 | 0.286 | 0.332 | 0.255 |
| tumor+ localized / wrong place / nothing | 823 / 75 / 83 | 823 / 75 / 83 | 823 / 75 / 83 | 830 / 68 / 83 | 823 / 75 / 83 | 814 / 65 / 102 |
| lesion Dice (1235) | 0.400 | 0.400 | 0.400 | **0.418** | 0.400 | 0.414 |
| detect@touch 10-20 / 20-40 / ≥40 | 47.6 / 80.6 / 91.3% | same | 47.3 / 80.4 / 91.3% | 50.5 / 81.7 / 91.7%† | 47.3 / 80.4 / 91.3% | 49.8 / 79.2 / 89.2% |
| Dice among detected 10-20 / 20-40 / ≥40 | 0.412 / 0.581 / 0.633 | same | 0.415 / 0.581 / 0.632 | 0.427 / 0.594 / 0.653 | 0.415 / 0.581 / 0.632 | 0.428 / 0.606 / 0.666 |
| median volume ratio among detected, 10-20 / 20-40 / ≥40 | 1.18 / 0.92 / 0.78 | same | 1.17 / 0.92 / 0.78 | 1.43 / 1.08 / 0.89 | 1.17 / 0.92 / 0.78 | 1.42 / 1.07 / 0.90 |
| case median relative volume error | −0.187 | −0.187 | −0.187 | −0.025 | −0.187 | −0.050 |

\* Negatives reweighted to the tumor+ slice-thickness mix (S8's bins <2.5 / 2.5-5 / ≥5 mm; 221 / 62 / 44 negatives vs a tumor+ mix of 0.41 / 0.43 / 0.16; coverage 100%). This is the S4 reweighting. No "S8b" exists in the repo, so this is my reading of the request.

† D's extra detections are contact artifacts, not finds. Of 19 newly touched lesions, 1 has Dice ≥ 0.1. Their median Dice is 0.008, and their median predicted/lesion volume is 3.9×. The regrown mask of a neighbouring object now grazes them.

Supplementary families (attribution only): A+B = A; B+D = B (no-op). All rows are in `lofo_results.csv`.

## What each fold picked

| family | grid | picks (folds 0-4) |
|---|---|---|
| A | 7 | `d 0 mm, ≥50% inside` in **5/5** (margin over the 2nd-best setting 0.001-0.003 on the training folds) |
| B | 65 | no-op in 4/5; fold 4 `t_low 0.5, top-5` (3-way tie; held-out −0.0001) |
| D | 17 | `t_low 0.1, t_seg 0.1` in 4/5; fold 3 `t_low 1e-4, t_seg 0.1` (margin 2e-5). Both mean "grow each core to p > 0.1". |
| C | 1820 | A's pick in 4/5 (12 exact ties; the t_seg variants at t_low 0.5 are identical); fold 4 the same plus top-5 |
| C+ (post hoc) | 1932 | `d 0 mm, ≥50% inside` on the **t_low 0.001** footprint + core-only + `t_seg 0.1` in **5/5** |

Held-out objective Δ per fold:

| family | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| D | +0.008 | +0.011 | +0.013 | +0.011 | +0.009 |
| A | 0 | 0 | +0.004 | 0 | +0.008 |
| C+ | +0.018 | +0.012 | +0.014 | +0.028 | +0.021 |

**D's picks sit on the edge of the grid** (t_seg = 0.1, the lowest value). The best t_seg may be lower. Under-segmentation is size-dependent: after D, 10-20 mm lesions are over-segmented (1.43×) while ≥40 mm ones are still under (0.89×).

## A: the filter, and which lesions it costs, by tier

At the saved segmentation (t = 0.5), **no A setting removes the only touching component of any GT lesion, in any tier**. 868 lesions are detected by S8: 698 inside, 44 embedded, 111 abutting, 15 separated.

The reason is that the model's pancreas channel already wraps its lesion channel:
- the median t = 0.5 component lies 100% inside channel 0 > 0.5;
- only 6 of 1341 components are < 50% inside;
- none is < 50% within 5 mm.

So d = 5 and 10 mm change nothing. d = 0 / ≥ 50% removes 6 components (4 in negatives, 2 in tumor+ cases, none touching a lesion) and clears 3 negatives.

The 15 S8-detected separated lesions are 66% inside the predicted pancreas (median), against 0% for all 70 separated lesions. The model extends its gland around the lesions it does find.

**The filter bites only on faint footprints, and there it costs real lesions.** C+'s filter runs on the t_low 0.001 component. Against the same regrowth without the filter, d 0 mm / ≥ 50% loses **27 detected lesions**:

| | inside | embedded | abutting | separated |
|---|---|---|---|---|
| lost | 9 | 2 | 12 | 4 |

By size, 15 are 20-40 mm and 7 are ≥40 mm. In exchange it clears 12 negatives. Measured against S8, C+ loses 22 and gains 16 lesions that are only touched (2 of the 16 have Dice ≥ 0.1). d 5 mm / ≥ 50% loses 7. The objective accepts this trade because it weights a cleared negative and a tumor+ case equally.

## B: why rescue does not pay

These are descriptive rows at fixed settings on all 1308. They were never used to choose.

| setting | objective | spec | FP / neg | missed 20-40 mm touched (of 116) | of those, Dice ≥ 0.1 | S8-detected lost |
|---|---|---|---|---|---|---|
| top-3 at t 0.01 (S11's idea) | 0.485 | 0.526 | 0.70 | 24 | 0 | 3 |
| top-3 at t 0.001 | 0.446 | 0.373 | 1.11 | 32 | 3 | 3 |
| top-1 at t 0.1 | 0.500 | 0.615 | 0.40 | 8 | 0 | 79 |

The B rule always keeps the first component, so any t_low < 0.5 predicts something in most negatives. And B's mask for a faint component (voxels ≥ half its peak) is a speck, so a rescued lesion adds about 0 Dice. 502 misses 116 of 597 20-40 mm lesions. The "100" in S8's notes and in the brief is the subset that 501 also missed.

## Method

- **Inputs.** Dataset502 out-of-fold `.npz`: channel 1 = lesion sigmoid, channel 0 = pancreas region. For each case, every voxel with p > 1e-4 was extracted on the server (float32, flat index), along with channel 0 > 0.5 and the lesion GT labelled as S8 labelled it. The cache is `~/research/nnunet_probs_502/` (1308 files, 0.4-0.5 GB, on both server and PC).
- **Resolution at 1e-4.** 502's probabilities come from fp16 logits: near 1e-4 the relative step is 0.78% (57 distinct values in [0.8e-4, 1.25e-4]), so float32 resolves p > 1e-4. The float16 *crop* concern applies to S2's crops, not to this cache. 46 cases never exceed 1e-4.
- **Settings.**
  - Components: 26-connected components of p > t_low over the whole volume.
  - A: |component ∩ (Euclidean distance to channel 0 > 0.5 ≤ d mm)| ≥ 1 voxel, or ≥ 50%.
  - B: rank by peak p (ties by size, then raster index; 0 ties occurred at any picked cut). Keep the first, then the following ones while peak ≥ p_min, up to k (k ∈ {1, 2, 3, 5, all}; "all" was added so the no-op is in the grid).
  - D: mask = p > t_seg inside the component, falling back to ≥ peak/2. B's fixed mask rule is D with t_seg = 0.5.
  - D-alone: keep exactly the components containing a voxel > 0.5 (the saved segmentation's objects), regrown inside their t_low footprint.
- **Tuning.** For held-out fold k, take the highest mean objective over the other 4 folds (first maximum wins, and the no-op is listed first). Apply it to fold k, then pool the 5 held-out folds. The CI is a paired case bootstrap of the pooled Δ (B = 1000, seed 0, S8's `stats.py`). It does not include training-run variance; the per-fold table does.
- **Exactness.** Each case reduces to "atoms" (the connected components of each kept component's mask, with sizes and per-lesion overlaps). Every metric is a sum over them, because kept components are never 26-adjacent.
  - On 4 synthetic volumes, the atom arithmetic equals a voxel-level reference scored by evaluate.py's own `lesion_level` / `overlap_metrics` for every one of the 1932 settings.
- **Oracle, before real data** (`tests/test_postprocess.py`, 14 tests). Three hand-computed volumes:
  - vol1: filter survivors for all 7 filters, including the exact 50% and 5.0 mm boundaries on anisotropic spacing; top-k / p_min picks;
  - vol2: B and D masks, Dice for each t_seg, and the fallback;
  - vol3: merging at low t_low, a component-level FP inside a kept component, peak ties, and an empty predicted pancreas.
- **Sabotage.** Each of 8 mutations fails 1-7 tests: ≥ to > in the 50% rule, the size tie order, p_min inclusivity, ≤ d to < d, the fallback fraction, the max(t_seg, t_low) rule, dropping the core-only selection, and skipping the filter in the fast path.
- **S8 regression.**
  - The no-op (no filter, t = 0.5, keep all) reproduces S8 per case and per lesion to 1e-12: case Dice, TP / predicted voxels / #CCs, every lesion's Dice, IoU and detect@any.
  - Components at t ∈ {0.5, 0.1, 0.01, 0.001} match S8's in size, peak and touched lesions, in all 1308 cases.
  - 28 aggregate numbers match `results/nnunet_multiclass/summary_vs_baseline.csv` to its stored 6 digits: 0.4717 / 0.7217 / 0.330 / 0.334 / 823-75-83 / per-bin detect and Dice / per-tier detect / −0.187.

## Caveats

- **C+ is post hoc.** The grid was widened after the first run showed that C as specified cannot express D's core-only selection. Its picks are still leave-one-fold-out, and the same setting won in all 5 folds. Only the grid definition was chosen after seeing results.
- **The objective weights a negative case as much as a tumor+ case**, and the cohort's 25% negatives are not thickness- or phase-matched (S1, S12). That is why C+ can win while losing lesions. Under a lesion-centred metric, D is the better pick.
- **D's touch-detections** (†) inflate detect@touch, most visibly in the <5 mm bin, where 2 lesions are "detected" by a 1371× mask. Read D as a segmentation gain, not a detection gain.
- **The filter loss counts are against the same setting without the filter.** A lesion may lose its touching component and still be touched by another one. Only lesions left untouched count as lost.
- **One training run per fold.** Fold-level variance is in the per-fold table only.

## Files

| file | content |
|---|---|
| `lofo_results.csv` | long table. Columns: `section, family, metric, stratum, fold, value, n, ci_lo, ci_hi, setting, note`. Sections: `picks` (train / held-out objective per fold), `pooled` (every scoreboard metric per family), `lesion_changes`, `filterA_losses` / `filterA_losses_faint_footprint` / `filterA_all_settings` (descriptive), `B_fixed_settings` (descriptive), `A_diagnostics`, `missed_20_40`, `D_new_detections` |
| `work/nnunet_postprocess/summary.txt` | every number above, as printed (git-ignored) |
| code | `src/nnunet-multiclass/postprocess/{extract_sparse.py (server), core.py, atoms_pass.py, lofo.py}`, `tests/test_postprocess.py` |
