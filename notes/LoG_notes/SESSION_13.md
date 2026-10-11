# Session S13: cheap post-processing on Dataset502, tuned leave-one-fold-out (2026-10-05)

**Answer: regrowing 502's own objects is the only clean win. The filter is nearly a no-op, faint-candidate rescue never pays, and C as specified reduces to the filter.**

- **D (regrow).** Each saved-segmentation object is regrown to p > 0.1 inside its footprint. The objective goes 0.5342 → **0.5446** (+0.0104, CI +0.008 to +0.013), positive in all 5 held-out folds. Case Dice goes 0.472 → 0.486, and lesion Dice 0.400 → 0.418.
- **A (filter).** +0.0023 (CI touches 0). It removes 6 components in total, none of which touch a lesion.
- **B (rescue).** The tuning picks the no-op in 4/5 folds.
- **C+ (post hoc).** This widened grid reaches +0.0183, but only by trading 22 S8-detected lesions for 12 cleared negatives.

## Scoreboard (pooled held-out folds)

| | S8 502 | A | B | **D** | C = A+B+D | C+ (post hoc) |
|---|---|---|---|---|---|---|
| objective (1308 cases) | 0.534 | 0.537 | 0.534 | **0.545** | 0.537 | 0.553 |
| case Dice, tumor+ | 0.472 | 0.472 | 0.472 | **0.486** | 0.472 | 0.484 |
| specificity (raw / thickness-reweighted) | 0.722 / 0.703 | 0.731 / 0.710 | 0.722 / 0.703 | 0.722 / 0.703 | 0.731 / 0.710 | 0.758 / 0.746 |
| FP components per negative / per tumor+ | 0.33 / 0.33 | 0.32 / 0.33 | 0.33 / 0.33 | 0.32 / 0.29 | 0.32 / 0.33 | 0.26 / 0.26 |
| lesion Dice (1235) | 0.400 | 0.400 | 0.400 | **0.418** | 0.400 | 0.414 |
| detect@touch 10-20 / 20-40 mm | 47.6 / 80.6% | same | 47.3 / 80.4% | 50.5 / 81.7%* | 47.3 / 80.4% | 49.8 / 79.2% |
| Dice among detected, 20-40 / ≥40 mm | 0.581 / 0.633 | same | same | 0.594 / 0.653 | same | 0.606 / 0.666 |
| median volume ratio among detected, 20-40 / ≥40 mm | 0.92 / 0.78 | same | same | 1.08 / 0.89 | same | 1.07 / 0.90 |
| tumor+ cases with nothing predicted | 83 | 83 | 83 | 83 | 83 | 102 |
| same pick in all 5 folds? | | yes | 4/5 (no-op) | 4/5 (same rule) | 4/5 | yes |

\* Not real detections. Of D's 19 newly touched lesions, 1 has Dice ≥ 0.1 (median 0.008): a regrown neighbour now grazes them.

Full tables, picks per fold, tier losses and caveats: `results/nnunet_postprocess/README.md`. Every printed number: `work/nnunet_postprocess/summary.txt`.

## What was built (`src/nnunet-multiclass/postprocess/`; no existing script edited)

| file | role |
|---|---|
| `extract_sparse.py` | Server, data movement only. For each case it writes every voxel with lesion p > 1e-4 (float32), channel 0 > 0.5, and the S8-labelled lesion GT to `~/research/nnunet_probs_502/`. 1308 cases, 4 workers, 10 min, 0 failures; rsynced to the PC (1308 files) |
| `core.py` | Pure logic: components, filter A, ranking / top-k / p_min (B), masks (D), and scoring. A case is reduced once to "atoms" (each kept component's mask components, with sizes and per-lesion overlaps); all 1932 settings are then sums over atoms |
| `atoms_pass.py` | PC, 4 workers, ~1 min. Builds the atoms and the objective for every setting, and checks each case against S8's 502 pickle |
| `lofo.py` | PC. S8 aggregate regression, leave-one-fold-out picks, pooled report, and descriptive tables → `results/nnunet_postprocess/lofo_results.csv` |
| `tests/test_postprocess.py` | 14 tests, written and passing before any real-data run |

## Oracle results

- **Three hand-computed volumes:**
  - **Filter survivors.** Correct for all 7 filter options, including the exact 50% and 5.0 mm boundaries on 2 mm z spacing.
  - **Top-k / p_min picks.** p_min is inclusive. The first component is kept whatever its peak.
  - **B and D masks.** Correct for each t_seg, including the "≥ half peak" fallback.
  - **Merging and FP counting.** Low t_low merges two cores into one component. An FP is counted per mask component, even inside a kept component.
  - **Edge cases.** Peak ties are broken by size, and an empty predicted pancreas removes everything.
- **Dense reference.** On the three volumes plus a random one, the atom arithmetic equals a voxel-level reimplementation scored by evaluate.py's own `lesion_level` / `overlap_metrics`, for every setting.
- **Sabotage.** 8 one-line mutations, each caught by 1-7 tests.
- **S8 reproduction.** "No filter, t = 0.5, keep all" reproduces S8 exactly.
  - Per case and per lesion, to 1e-12: Dice, IoU, detect, TP, predicted voxels, #CCs.
  - Components at 4 thresholds match S8's in size, peak and touched lesions.
  - 28 aggregates match S8's CSV to its 6 stored digits (0.4717, 0.7217, 0.330, 823 / 75 / 83, per-bin, per-tier, −0.187).
  - Two deliberately corrupted inputs (one voxel moved below 0.5; five GT voxels dropped) were flagged.

## Decisions fixed before results

- **Objective and LOFO protocol** as briefed. The first maximum wins, with the no-op listed first in every family, so ties resolve toward doing nothing.
- **k ∈ {1, 2, 3, 5, all}.** "all" was added so that each family contains the no-op.
- **B's tie-break:** peak, then size, then raster index. Ties at a picked cut: 0.
- **A is measured on the component's t_low footprint.** In C, A runs before the top-k ranking.
- **D-alone** keeps exactly the components containing a voxel > 0.5 (the saved segmentation's objects) and regrows them at t_seg inside their t_low footprint. With t_low = 0.5 the footprint is already all > 0.5, so D needs a lower t_low to do anything.
- **"Reweighted" specificity** = negatives reweighted to the tumor+ slice-thickness mix (S4's method, S8's bins). I found no "S8b" anywhere in the repo, so this is my reading. Coverage 100%.
- **The 1e-4 rung was checked before use.** 502's probabilities come from fp16 logits, with a 0.78% relative step near 1e-4, so float32 resolves it.

## Added after seeing results (reported, labelled, never used to pick a setting)

- **C+.** C as specified cannot express D's core-only selection, so its grid is not a superset of D's. I added "every filter × D's selection" and kept the same LOFO procedure. The same setting won in all 5 folds: d 0 mm / ≥ 50% on the t_low 0.001 footprint, core-only, t_seg 0.1. Only the grid definition is post hoc.
- **A diagnostics** (overlap of components and GT lesions with channel 0), **the filter on a faint footprint**, **B at fixed settings**, and **what D's new detections are**. All are descriptive, on all 1308 cases.

## Process notes

- **`~/research/nnunet_probs_502/` did not exist.** I created it from the server npz files (read-only), as the brief allowed. A, D and the 1e-4 rung all need voxels, which the S8 pickles don't hold.
- **46 cases never exceed p = 1e-4.** That path crashed the first atoms pass on a missing key; it now has its own test.
- **Two of my own asserts were wrong, not the data:**
  - S1's `case_spacing.csv` already stores the 0-based nnU-Net fold.
  - S8's CSV holds 6 significant digits, so the aggregate check compares at that precision (per-case checks stay at 1e-12).

## What surprised me

1. **The location prior is already inside 502.** The median t = 0.5 lesion component is 100% inside the model's own pancreas channel. Only 6 of 1341 are < 50% inside, and none is < 50% within 5 mm. This is the region-based training (lesion ⊂ pancreas region) showing up at inference. The 15 separated lesions 502 detects are 66% inside its pancreas prediction, against 0% for separated lesions overall: the model grows its gland around what it finds.
2. **The same filter becomes costly on faint footprints.** Applied to t_low 0.001 components, d 0 mm / ≥ 50% loses 27 detected lesions:

   | | inside | embedded | abutting | separated |
   |---|---|---|---|---|
   | lost | 9 | 2 | 12 | 4 |

   By size, 15 are 20-40 mm and 7 are ≥40 mm. The faint halo of a large lesion spills out of the gland and drags the whole component under 50%. The objective still prefers it, because a cleared negative counts as much as a tumor+ case.
3. **S11's rescue idea does not survive a Dice objective on 502.** Top-3 at t 0.01 touches 24 of the 116 missed 20-40 mm lesions, but none of the 24 reaches Dice 0.1. Specificity drops to 0.53 and FP/negative rises to 0.70. Faint 502 candidates are found by contact, not segmented.
4. **D's gain is all segmentation.** The volume ratio among detected lesions goes 0.92 → 1.08 (20-40 mm) and 0.78 → 0.89 (≥40 mm), and the case median volume error goes −19% → −2.5%. But 10-20 mm lesions go from 1.18× to 1.43× (over-segmented), so a single t_seg is a compromise. t_seg = 0.1 is also the lowest grid value.
5. **"100 missed 20-40 mm" was the 501∩502 subset.** 502 alone misses 116 of 597.

## Does S6 (leakage audit) still add anything?

**Little, for the current pipeline.** A is the leakage-free version of the old GT-pancreas location re-ranking:
- It uses the model's own pancreas, never the carved-out GT mask.
- On 502 it moves specificity only 0.722 → 0.731, because 502 already encodes "lesions live in the pancreas".

The old figure (carried over: 501 component re-ranking with GT pancreas location, case specificity 0.57 → 0.83 at 90% recoverable) used a different protocol and a different model, so the two can only be compared qualitatively.

S6 would matter only to qualify that parked duct-work number if it is ever cited, or if a later method feeds GT anatomy back in. If S6 is run at all, the cheapest informative version is to swap the GT pancreas feature for 502's predicted pancreas in the parked re-ranker.

## Suggested next steps (not done)

- **Adopt D as the 502 post-processing default** ("grow each saved object to p > 0.1 inside its t_low 0.1 footprint"). It is one line, gains in every fold, and costs no lesion.
- **Extend D's grid below t_seg 0.1, and try a size-dependent t_seg**, still LOFO. Small objects over-grow and large ones still under-grow.
- **Decide whether the objective should weight negatives equally to tumor+ cases.** C+ is "better" only under equal weighting. A lesion-weighted or detect-constrained objective would likely reject it.
- **Update CONTEXT.md** with S8 and S13. Not done: CONTEXT.md is not mine to edit without asking.
