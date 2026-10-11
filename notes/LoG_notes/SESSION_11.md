# Session S11: rescue check, would an existing signal flag nnU-Net's missed tumors? (2026-10-03)

**Answer: the fixed gate PASSES, but the pass rests on a signal that performs at chance.**

- **Union.** The three operating points catch 114/240 = 47.5% (CI 41.3-53.8%) of the 20-40 mm lesions nnU-Net
  misses. Without the separated lesions: 112/219 = 51.1%.
- **Uniqueness.** S9 top-10 catches 19.6% of the misses on its own, which clears the 10% bar.
- **The catch.** S9 top-10's hit rate on these lesions (33.0%) equals the rate expected if its 10 candidates
  landed uniformly on the gland (32.8%). Without S9, the union of nnU-Net and the duct signal is 27.9%, which
  fails the gate.
- **Every signal fires more often on DETECTED lesions than on missed ones.**
- **The bigger finding is the optional step.** Dataset502 fold 0 recovers 25 of 36 fold-0 20-40 mm lesions that
  501 missed, at 0.39 vs 1.60 FP components per case.

Full tables, definitions and caveats: `results/rescue_check/README.md`. Every printed number:
`work/rescue_check/summary_ds501_s2.txt`.

## What was built (`src/rescue_check/`, importable; no existing script edited)

| file | role |
|---|---|
| `core.py` | every operating point and gate threshold as constants (frozen before results); pure functions: Wilson CI, component ranking with pessimistic ties, per-lesion touch hits, per-case FP counts, t = 0.5 case category, the three signals, overlap / Venn counts, gate, review-list ordering |
| `nnunet_table.py` | the only nnU-Net-dependent step: `--pred_source` → `work/rescue_check/nnunet_<tag>_{lesions,cases}.csv`. The default `ds501_s2` reads S2's component tables plus its per-lesion CSV and asserts equality with S2 |
| `analyze.py` | joins master + nnU-Net table + S9 + v3 + duct outputs with gt_vox asserts → `results/rescue_check/` |
| `ds502_fold0.py` | optional: rsync-pulls 502 fold-0 segmentations (server read only), touch rule on label == 2, FP components |
| `tests/test_rescue_check.py` | 10 tests, written and passing before real data. The synthetic 10-lesion table reproduces hand-counted Venn cells. Mutating any operating point breaks 5 tests. Full suite: 96 passed |

Run time on the PC is about 1 s for the table, about 3 s for the analysis, and about 2 min for the 502 pass.

## Oracle results

- **Synthetic table.** Reproduces the expected overlap counts, unique catches, per-signal n (with not-evaluable
  excluded) and gate pass/fail. Boundary cases (rank 3 vs 4, 10 mm, 3.21 mm, rank 10 vs 11) behave as specified.
- **Real data:**
  - 1235 lesions; 20-40 mm n = 597; detected 59.8% (S2, *carried over*)
  - the missed set equals `detected_any == 0`
  - hit at each t equals S2's `max_t_hit_touch`
  - FP/case at 4 thresholds equals `nnunet_froc_points.csv` to 1e-4
  - gt_vox is equal on every join (nnU-Net, S9, v3, the 248 502 fold-0 lesions)
- **Head-end dilation.** The existing column (`caliber_head_mm` ≥ 3.21 mm, `component_features.py`) reproduces
  CONTEXT's 56.8% / 5.1%, so nothing was recomputed from profiles.
- **502 orientation.** All 264 affines equal the CT affine. Gland Dice is 0.71-0.95 as stored vs 0.12-0.37 with x
  flipped.

## Decisions fixed before results (stated in `core.py`)

- **Not evaluable** (S9 unscorable, or no MPD mask): excluded from that signal's own rate and counted as **not
  caught** in unions and the gate.
- **S9 "could not score"** = `pct_A` NaN, 126 lesions. 94 have no lattice centre in lesion + 2 mm; 32 have a
  footprint too large, and those 32 still have a `cand_rank_A`, which is not used.
- **Gate "unique"** = caught by that signal alone ÷ all missed 20-40 mm lesions. The looser reading (÷ misses the
  other two leave) is reported but not used.
- **Primary gate set** = all lesions. It also passes without the separated lesions.
- **Review-list ordering**: number of signals, then the mean of graded strengths, then diameter.

## Added after seeing results (reported only, never used for the verdict)

- The S9 uniform-placement chance rate, 1 − (1 − f)^10 with f = footprint fraction of the lattice.
- Pairwise unions.
- 502 FP/case. This was added because "recovers more" needed a cost before it could be reported.

## Process notes

- **The duct outputs are not where the brief said.** `results/duct_cutoff/` does not exist; the Session-3 duct
  CSVs are in `results/geometry_results/duct_cutoff/`, even though `main.py`'s default output is
  `results/duct_cutoff/`. SESSION_3.md is the attenuation audit; the duct work has no session note.
- **Two small pandas bugs in the 502 section.** A scalar `True & True` mask, and a read-only array from
  `to_numpy()`. Both were fixed before any number was read.
- **An orientation check was nearly skipped.** I had filtered the affine-warning lines out of the 502 run's
  output. I went back and counted them (0) before trusting the result.

## What surprised me

1. **S9 top-10 at 20-40 mm is pure footprint geometry.** A 20-40 mm lesion covers ~3.5% of the lattice, so 10
   random candidates hit it about a third of the time, which is exactly the observed rate. S9's README reported
   20-40 mm top-10 = 39-44%; on these lesions that is near chance too. Only ≥40 mm missed lesions
   beat chance (63% vs 51%, n = 41).
2. **The signals are enriched in the lesions nnU-Net already finds.**
   - The duct OP fires on 55% of detected 20-40 mm lesions vs 20% of missed ones.
   - The duct case flag fires in 49% of detected-lesion cases vs 21% of missed-lesion cases.
   - Duct dilation and anomaly scores track the same obvious tumors.
3. **Head-end dilation adds essentially nothing over `mpd_dilated`**: +2 missed 20-40 mm lesions, +0 negatives.
4. **Top-3 components at t = 0.01 cost almost the same as the t = 0.5 segmentation**: 1.77 vs 1.72 FP/case. That
   operating point catches 43 of the 240 missed 20-40 mm lesions and drops 13 of the 357 detected ones. It is a
   cheap re-thresholding idea, not tested here (it would need fold-held-out tuning).
5. **Dataset502 fold 0 beats every signal and cuts FPs about 4×.**
   - FP components per case: 0.39 vs 1.60. Negatives with any lesion prediction: 26% vs 80%.
   - This is despite the S7 sampling confound.
   - Fold 0 only. 501's fold-0 20-40 mm detection (67%) is also above its pooled 60%.
6. **Satellites are the hardest case.** Missed lesions in cases where nnU-Net found another lesion are flagged by
   nothing 76% of the time. Non-contrast misses are unflagged 67% of the time.

## Suggested next step (not done)

Wait for all five 502 folds, then rerun this exact pipeline with `--pred_source` on 502 (S8). That requires
writing 502 component tables in S2's schema, i.e. an S2-style case pass over the lesion-channel sigmoid.

A re-ranker built on the current three signals is not recommended despite the literal PASS. Its only unique
contributor is at chance.
