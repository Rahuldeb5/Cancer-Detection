# Session S14: what Dataset502 still misses, and does any existing signal flag it? (2026-10-05)

Full tables, definitions and caveats: `results/rescue_check_502/README.md`. Data:
`results/rescue_check_502/missed_lesions.csv`. Every printed number: `work/rescue_check_502/summary.txt`. PC only,
CPU, ~7 min. Model = **502 + S13's D** (fixed rule, all folds). Missed = not touched by **raw** 502.

## Scoreboard

| # | question | answer | good / bad |
|---|---|---|---|
| 0 | Do the inputs reproduce S8 / S8b / S13? | Yes. Raw touch, Dice and peak p equal S8 for 1235/1235. 116/597 missed at 20-40 mm. S8b-weighted Sp 0.70339. D mean Dice 0.418 and 7/116 touched match S13. Identity placements reproduce both signals exactly | good |
| 1 | How many misses, and does D rescue any? | 116 at 20-40 mm, 144 at 10-20 mm. Under D: 0/116 reach Dice 0.1 (max 0.017), 1/144 at 10-20 mm | bad (as expected) |
| 2a | Are misses non-contrast? | 20-40: **31%** of misses vs 17% of detected (p 0.002). 10-20: 33% vs 23% | bad (phase matters) |
| 2b | Are misses iso? | 20-40: 42% vs 38% (p 0.46). 10-20: 37% vs 35%. **Not enriched** | neutral |
| 2c | Are misses satellites? | 20-40: **35% vs 7%**. 10-20: **55% vs 15%** | bad |
| 2d | Are misses outside the gland? | GT `separated`: 16% vs 2% (20-40). < 50% inside 502's predicted gland: **47% vs 4%** | bad |
| 2e | Misses with none of these properties | 16/116 (20-40) and 10/144 (10-20). All hypo or hyper, mostly venous | — |
| 2f | 502 peak p in lesion + 2 mm | Missed median 0.002 (20-40), 4e-4 (10-20). Detected 1.00 | — |
| 3 | Does 502 include or exclude the lesions it misses? | 20-40: **included 53%**, adjacent 33%, away 15%, **dent 0%**. Not-included misses: median shell 3% gland, so the gland stops at them rather than wrapping around | **decides S15: bulges, not dents** |
| 4a | Faint nnU-Net guess vs chance | Misses 20-40: 30.2% vs 9.8% chance, **+20.3 pts** [+12.6, +27.7]. 10-20: +15.5 [+8.5, +22.3]. Costs 0.67 FP per negative; 47% of negatives get one | good, but contact only (median Dice 0.05) |
| 4b | Duct signal vs chance | Misses 20-40: 12.0% vs 14.6%, **−2.5 pts** [−6.9, +2.0]. 10-20: −2.4. Detected 20-40: +7.6 | **bad (at chance or below)** |
| 5 | S15 target | **Missed 20-40 mm index lesions that 502 includes in its gland, not GT-separated: n = 48** (30 enhanced; 16 enhanced and not iso). Comparison pool: 461 detected and included. Secondary: 10-20 mm, n = 46 | — |

## Steps
1. **Oracle first.** 11 tests in `tests/test_rescue_check_502.py`, written and passing before any real run.
   - FFT placement equals a brute-force loop.
   - Hand-counted fractions and phantom categories, on an anisotropic grid.
   - 3 of 4 mutations fail the tests; the 4th is an equivalent mutant.
2. **S11 reused unchanged.** `export_s8` wrote S8's 502 components in S2's schema. `rescue_check.nnunet_table
   --pred_source` then ran as-is. Its output equals S8 per lesion.
3. **Case pass** (`case_pass.py`, 1308 cases, 0 failures). For each case it computes:
   - raw and D scores from S13's atoms
   - predicted-pancreas metrics from S13's sparse cache
   - placement chance for both signals, with an identity-placement assert per lesion
4. **Analysis** (`analyze.py`). Imports S11's `build_table`, S8b's `cell_weights` and S8's `paired_ratio`, then runs
   the regression asserts and the tables.

## Code added (new files only; no existing script changed)
| file | role |
|---|---|
| `src/rescue_check_502/__init__.py` | package docstring |
| `src/rescue_check_502/core.py` | frozen constants. Pure functions: `anchor_hits` / `placement_fraction` (exact FFT placement), `chance_any`, `near_region`, `lesion_panc_metrics`, `panc_category`, `phase_level` |
| `src/rescue_check_502/export_s8.py` | S8 502 pickles → S11 `--pred_source` directory (work/) |
| `src/rescue_check_502/case_pass.py` | per-case pass → `work/rescue_check_502/cases/` |
| `src/rescue_check_502/analyze.py` | joins, regression, tables → results + `summary.txt` |
| `tests/test_rescue_check_502.py` | 11 oracle tests |

## Process notes
- **First full run had a bug of mine: `~uint8`.** The case pass inverted the MPD mask with `~` on tumorlib's uint8
  array, which is a bitwise NOT (0 → 255). The duct "near" region came out empty.
  - The identity-placement regression caught it before any result was read: only 504/1003 lesions agreed with the
    master table's MPD distance.
  - Fixed by moving the step into `core.near_region` (casts to bool) and adding a uint8 test. After the rerun,
    1003/1003 agree.
  - This is CONTEXT gotcha 1's failure mode, now in a new place.
- **The brief's "faint candidates touch 24/116" is S13's B mask.** That mask keeps only voxels > 0.5, or ≥ half the
  peak, inside each component. S11's signal uses the whole p > 0.01 footprint, which touches 35/116. I re-ran S13's
  setting to confirm the 24. Both numbers are right; they measure different things.
- **MPD-presence check scope.** The first version of the check compared the case pass against negatives the pass
  never reads. It is now restricted to tumor+ (981/981 agree).
- **Pre-existing test-suite issue (not touched).** The full `pytest` run has 3 failures and 2 errors in
  `test_loader.py` / `test_lesion_ids.py`. They appear with or without the S14 file, and both files pass when run
  alone. Most likely an earlier test module imports S2's `common`, which rewrites tumorlib's env paths.
- **CONTEXT.md was not edited.**

## What surprised me
1. **No dents at all.** I expected that a model trained with lesion ⊂ pancreas might carve missed tumors out of its
   gland. It doesn't. It either swallows them as gland (iso or unenhanced) or stops its gland at their edge.
2. **Iso is not what separates misses from hits at 20-40 mm** (42% vs 38%). Iso is the dominant trait *within* the
   included misses (37/61), but just as common among the detected lesions. Phase, satellite status and location
   separate the two groups instead.
3. **The duct signal is anti-informative on misses.** Missed lesions are within 10 mm of the MPD less often than
   random placement would give (29% vs 39%). The duct flag tracks the obvious, detected tumors, as S11 found for 501.
4. **The faint guess is above chance but expensive.** It reaches 3× chance on misses, but costs about 2× 502's own
   FP rate on negatives. It also only grazes the lesion.
5. **Satellites make up over half of the 10-20 mm misses** (79/144). In 58 of those 79, raw 502 does find another
   lesion in the same case. At 20-40 mm the figure is 28 of 41. So at case level, many of these "misses" are cases
   that are already localized.
