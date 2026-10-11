# Session S8b: phase and negative-matching add-on to S8 (2026-10-05)

Full tables and method: `results/nnunet_multiclass_phase/README.md`. Data: `results/nnunet_multiclass_phase/phase_addon.csv`.
Every printed number: `work/nnunet_multiclass_phase/summary.txt`. PC only, about 2 min. The S8 case pass was **not** rerun and no
probability file was read.

## Scoreboard

| # | question | answer | good / bad |
|---|---|---|---|
| 0 | What did S8 already have? | Est-phase non-contrast / venous / uncertain strata (10-20, 20-40 mm, paired bootstrap) and unweighted Sp / FP per negative. Missing: the early/late arterial split, iso by estimated phase, reweighted negatives, positives' far-from-tumor FP | — |
| 1a | Does 502 beat 501 in every phase stratum? | Yes in every interpretable stratum (≥20 lesions), **except late arterial 10-20 mm**: +0.061 Dice [−0.023, +0.167], n = 29 | good (one weak spot) |
| 1b | Early vs late arterial at 20-40 mm | Late gains more (+31.7 detection points, 13 recovered / 0 lost) than early (+10.8) | good |
| 1c | Early arterial 10-20 mm, uncertain 10-20 mm | n = 13 and 10, **not interpreted** | — |
| 2a | Iso in truly unenhanced scans | **55.1%** (156/283) of lesions on est non-contrast scans; ≥10 mm 61.5% (144/234). These scans hold **23%** of lesions, not a third | bad (worse than thought) |
| 2b | Iso in venous | 33.1% (222/670); ≥10 mm 33.4%. Metadata gave 30.3 / 30.8% | neutral (barely moves) |
| 2c | Iso in pancreatic (late arterial) phase | **19.8%** (20/101), the lowest; CI just reaches the 5-15% literature range (*carried over*) | good (phase explains part of "iso") |
| 3a | Can the negatives be reweighted to the tumor+ phase × thickness mix? | Yes. **No tumor+ cell lacks negatives.** ESS 187 of 327. Thinnest cells: non-contrast ≥5 mm (19 tumor+ vs 3 negatives) and venous 2.5-5 mm (357 vs 36, 3.3× weight) | good (wide CIs) |
| 3b | Specificity after matching | 501 0.223 → **0.178**; 502 0.722 → **0.703**. Gap +0.525 [+0.431, +0.614] | bad for both, small; gap unchanged |
| 3c | FP per negative after matching | 501 2.06 → 2.34; 502 0.33 → 0.38 | small, both worse |
| 3d | FP on positives' non-tumor region (no matching needed) | 501 **1.57**, 502 **0.29** per tumor+ case (peak > 10 mm from every lesion); 69% vs 22% of tumor+ cases have one | good (same ranking) |
| 3e | Are negatives an optimistic FP denominator? | No: within matched cells they usually have *more* FPs than positives' far region (all 6 cells for 501, 4 of 6 for 502) | good (S8's FP numbers are not flattering) |

## Steps
1. **Inventory first.** S8's CSV already held the est_phase strata. Those rows are recomputed only as a regression check: same bootstrap draws, asserted equal to S8. They are labelled `source = S8`.
2. **Item 2's metadata rows** reproduce S3's table exactly (557 / 260 / 413 / 4, and the ≥10 mm figures 543 / 246 / 355).
3. **Item 3's unweighted values** reproduce S8 (Sp 0.2232 / 0.7217, FP per negative 2.0642 / 0.3303). Far + near FP on positives reproduces S8's 1.60 / 0.33 per tumor+ case.
4. **The 10 mm test reads the GT lesion masks** (PC, read-only) for 717 tumor+ cases, because the pickles carry no lesion geometry. Voxel and lesion counts equal the master table in every case.

## Code added (new files only; no existing script changed)
| file | role |
|---|---|
| `src/multiclass_eval/phase_addon.py` | items 1-3. Pure functions: `phase_stratum`, `thickness_bin`, `cell_weights`, `ess`, `weighted_bootstrap`, `peak_distances_mm` |
| `tests/test_phase_addon.py` | 7 known-answer tests (weights, ESS, uncovered cell, bootstrap identity, 9.6 / 10.4 / 12.5 mm distance boundary on an anisotropic grid), written before the real run |

## Process notes
- **Two regression asserts tripped on CSV rounding, not logic.**
  - The thickness CSVs differ by ≤2.4e-4 mm, so the assert now also checks that the bins are identical.
  - S8's `%.6g` CSV moved a near-zero CI bound by 1e-7. That check now uses an absolute tolerance of 1e-5.
- **The first README draft overstated three readings.** These were corrected against the log before saving:
  - the thickness monotonicity
  - the cell count for the negatives-vs-positives comparison
  - the direction of "pessimistic"
- **CONTEXT.md was not edited.** The replacement line for the metadata-based iso claim is in the README, under Item 2.

## What surprised me
1. **No uncovered cells.** Every tumor+ phase × thickness cell has at least one negative, so the whole tumor+ mix could be matched. The price is a 3.3× weight on the 36 venous 2.5-5 mm negatives.
2. **The negatives were *flattering* both models, but only slightly** (≤0.045 Sp). The phase/thickness mismatch S12 found is real but barely moves the nnU-Net numbers.
3. **Within a matched cell, negatives get more FPs than the far region of positives.** This is the reverse of S4's LoG result. One untested idea: nnU-Net spends its probability mass on the real tumor when there is one.
4. **The pancreatic phase is where 502 adds least at 10-20 mm** (n.s.) and where iso is rarest. Both fit the idea that 502's gland context helps most when the contrast phase is poor.
