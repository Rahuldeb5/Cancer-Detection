# Session S8: Dataset502 (pancreas + lesion) vs Dataset501, identical protocol, lesion channel only (2026-10-05)

**Answer: 502 is better on every lesion metric. The gain is significant (case bootstrap) and holds in all 5 independently trained folds.**

| | 501 | 502 |
|---|---|---|
| mean lesion Dice (1235) | 0.280 | **0.400**, Δ +0.121 [+0.107, +0.134] |
| 10-20 mm Dice / detect@any (275) | 0.074 / 25.8% | **0.196 / 47.6%** |
| 20-40 mm Dice / detect@any (597) | 0.321 / 59.8% | **0.468 / 80.6%** |
| tumor+ cases localized (981) | 631 (64.3%) | **823 (83.9%)** |
| 10-20 mm index lesion, case localized (176) | 31.8% | **63.1%** |
| FP lesion components per negative | 2.06 | **0.33** |
| case Dice mean (981 ref-positive) | 0.312 | **0.472** |

- **Missed 20-40 mm lesions.** Of the 240 that 501 missed completely, 502 finds 140 (58.3%, mean Dice 0.441 among them). It loses 16 of 501's 357 hits.
- **Operating points.** 502's lowest ladder rung (t = 0.001) still has fewer FP per negative (1.22) than 501's highest rung (t = 0.9, 1.52). So 502 dominates at every operating point on the ladder.

Full tables, method and caveats: `results/nnunet_multiclass/README.md`. Every printed number: `work/nnunet_multiclass/summary.txt`.

## What was built (`src/multiclass_eval/`, importable; no existing script edited)

| file | role |
|---|---|
| `core.py` | Pure per-case evaluation. `lesion_mask` (501 `seg==1`, 502 `seg==2`); `scan_502` (region-npz consistency, whole volume); `exclusion_variant` (`prepare_excluded_eval.py`'s ignore-region rule in nibabel order); `evaluate_case`. It **imports** S2's `analyze()` and evaluate.py's metric functions rather than copying them |
| `case_pass.py` | Server: one pickle per (model, case) in `work/nnunet_multiclass/cases/`, resumable. 1308 × 2 jobs, 6 workers, ~45 min, peak 3.3 GB per worker, 0 failures |
| `stats.py` | Paired case-level bootstrap as a reweighting of per-case sums |
| `summarize.py` | PC: regression asserts, then the two CSVs. `S8_PARTIAL=1` with `python -O` smoke-tests it on an unfinished pass (used once, output never kept) |
| `tests/test_multiclass_eval.py` | 12 tests, passing on PC and server before real data |

**What the tests cover:**
- known-answer phantom: a lesion under label 1 is "missed"
- the 501 and 502 paths give identical output on identical probabilities
- the **real** `prepare_excluded_eval.py` and `evaluate.py` run as subprocesses and match
- the bootstrap equals a literal resample loop

**Sabotage check.** `seg > 0` as 502's lesion mask is caught by the phantom, and by the `p > 0.5` vs seg check (17,835 voxels).

**Server notes.**
- The server venv has no pytest. `work/nnunet_multiclass/run_tests.py` (server only, git-ignored) stubs `pytest.fixture` and `pytest.approx` and calls the tests directly. Nothing was installed.
- Only `src/multiclass_eval/` and the new test file were rsynced to the server.
- The server's `tumorlib`, S2 scripts, evaluate.py, fold files, master table and `prepare_excluded_eval.py` are md5-identical to the PC's.

## Regression (asserted before any 502 number was read)

501 through the S8 code reproduces:
- 631 / 288 / 62
- detect@any 2.0 / 0.0 / 25.8 / 59.8 / 83.0%
- master lesion_dice for all 1235 lesions
- S2's 216 ladder rows
- S2's per-lesion peak / max-t columns
- the 10 `evaluation.json` files (all and excl), every field to rtol 1e-5, with one explained exception below
- nnU-Net's own `summary.json` per-case TP / n_ref / n_pred for both models (and 502's gland region), all 1308 cases

S11's independent fold-0 502 read (on the PC) matches all 248 fold-0 lesions.

**Exception: NSD@3 mm in 2 cases.** 1.5 mm isotropic grids put 2-voxel surface distances exactly on the 3 mm tolerance. evaluate.py's float32 header spacing and tumorlib's affine spacing (1.5 ± 1e-7) round them to opposite sides.
- I did not rerun the pass with header spacing. CONTEXT says to use affine spacing, and both S8 models share it.
- So the assert allows NSD@3 mm only.

## Decisions (made before results)

- **Both models re-scored by one code path**, rather than reusing 501's stored numbers. The 501 reproduction is the proof that the path is evaluate.py / S2.
- **Lesion GT** = 501 `gt_segmentations` for both models, re-checked per case to equal 502 label 2.
- **Without excluded**:
  - case level: prepare_excluded_eval semantics (18 cases dropped)
  - per lesion: drop the 70 rows (kept lesions keep the full-prediction Dice)
  - ladder: S2's `excl_separated` rows
- **Bootstrap.** Unit = case, 1000 replicates, seed 0, percentile CI. Same W for every stratum within a lesion set.
- **Slice-thickness bins** <2.5 / 2.5-5 / ≥5 mm, chosen to match S1's thresholds.
- **Phase reported twice:** metadata and S12 estimate.
- **FP-matched ladder:** for each 501 rung, 502 at the lowest t whose FP/neg ≤ 501's. No interpolation.

## Added after seeing results (reported, not used for the verdict)

- **Per-fold breakdown**, added because the bootstrap cannot see training-run variance. Direction is the same in all 5 folds.
- **The NSD@3 mm diagnosis**, which led to the scoped assert.

## Process notes

- **The first partial run hid a column-naming bug**, because `python -O` also strips the asserts that would have exercised it. The full run caught it.
- **Lesson:** a partial smoke test under `-O` only covers non-assert code paths.

## What surprised me

1. **The size of it.** 10-20 mm detection roughly doubles even though 502 sees about half as many lesion-centred patches (S7 §6).
2. **501's case "sensitivity" (0.937) was propped up by wrong-place blobs.** 502's is lower (0.915) while localization goes from 64% to 84%. Of 501's 288 wrong-place cases, 171 become localized and 58 become empty.
3. **FP suppression is the bigger effect:**
   - 72% of negatives get no lesion voxel (501: 22%)
   - FP/neg falls in every fold, from about 2 to about 0.3
4. **The weakest 501 strata gain most:**
   - estimated non-contrast 20-40 mm: 34% → 70% detected
   - body 10-20 mm: 10% → 39%
   - embedded 20-40 mm: 31% → 74%
5. **Separated lesions don't improve.** At 10-20 mm they slip from 23% to 15% (n = 26, CI spans 0), which is consistent with a gland prior.
6. **502 under-segments** (median volume −19% vs +4% for 501). Dice among detected 20-40 mm lesions rises only 0.536 → 0.581, so the Dice gain is mostly detection.

## Suggested next steps (not done)

- **Make 502 the baseline for later sessions.**
  - Rerun S11 on it via `--pred_source`. Its component tables are in the pickles; `ladder_502.csv` and the per-lesion columns are already here.
  - Recompute S2's sub-threshold readings for 502 misses (100 missed 20-40 mm; median peak p 8e-4).
- **Disentangle the confounds.** A 502 trainer with lesion-forced foreground sampling (S7 §6) would separate "pancreas context" from "sampling rate". The normalization statistics remain a second difference.
- **Extend the 502 ladder below t = 0.001.** The curve is not closed at the low-FP end.
- **Under-segmentation** is now the main Dice limiter among detected lesions (CONTEXT priority 2).
- **Update CONTEXT.md** with the S8 scoreboard. Not done here, because CONTEXT.md has uncommitted edits of yours.
