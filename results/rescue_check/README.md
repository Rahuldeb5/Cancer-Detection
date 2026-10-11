# S11 rescue check (2026-10-03)

**Question.** For tumors the baseline nnU-Net (Dataset501, 5-fold OOF) misses, especially 20-40 mm ones, would an
existing signal have flagged them? This is pure analysis of existing outputs. Nothing was trained and nothing was
recomputed from images (except the optional Dataset502 fold-0 comparison at the end). CPU, PC.

**Answer.**
- **The fixed gate PASSES**, with and without the 70 separated lesions. The three operating points together catch
  114/240 = 47.5% of missed 20-40 mm lesions (CI 41.3-53.8%). The normative model catches 19.6% of those misses on
  its own.
- **The signal that carries the pass performs at chance.** S9's top-10 hits 33.0% of missed 20-40 mm lesions.
  10 candidates dropped uniformly on the gland would hit 32.8%.
- **Without S9, the gate would fail.** nnU-Net faint guess ∪ duct = 27.9% (CI 22.6-33.9%), below the 30% bar.
  Neither of those two reaches 10% unique (7.9%, 7.1%).
- **All three signals fire more often on lesions nnU-Net already finds** than on the ones it misses.
- **Dataset502 fold 0 recovers more than any signal.** It finds 25 of the 36 fold-0 20-40 mm lesions that 501
  missed, with about 4× fewer FP components per case.

The gate passing is reported as-is: thresholds were fixed before results and were not changed. The chance
comparison and the pairwise unions were added after the results and are reported alongside the gate, not used
to decide it.

Everything here was computed in S11 unless marked *carried over*. Code: `src/rescue_check/`. Tests:
`tests/test_rescue_check.py`. Readable log of every number: `work/rescue_check/summary_ds501_s2.txt`.

```
PYTHONPATH=src .venv/bin/python -m pytest tests/test_rescue_check.py          # synthetic oracle (10 tests)
PYTHONPATH=src .venv/bin/python -m rescue_check.nnunet_table                  # --pred_source (default: S2 Dataset501)
PYTHONPATH=src .venv/bin/python -m rescue_check.ds502_fold0 --fetch           # optional; reads the server, writes only on the PC
PYTHONPATH=src .venv/bin/python -m rescue_check.analyze                       # -> this folder
```

**Rerunning on another model** (e.g. Dataset502 once all folds exist) takes two steps:
1. Put its component tables in S2's schema under `DIR/components/components_t{t}.csv`.
2. Run `nnunet_table --pred_source DIR --tag ds502` and then `analyze --nnunet_tag ds502`.

The second step writes to `work/rescue_check/results_ds502/`. Only the nnU-Net signal and the MISSED/DETECTED
split depend on the source.

## Definitions (fixed before any result)

| item | definition |
|---|---|
| population | lesion-level, key (case_id, lesion_id) from `master_lesions.csv`; 1235 lesions |
| MISSED | no predicted component at t = 0.5 touches the lesion (`detected_any == 0`). Equal to the component-table touch at t = 0.5 for 1235/1235 |
| 1. faint nnU-Net guess | a component of p > t touches the lesion, t ∈ {0.1, 0.01, 0.001}; components are 26-connected over the whole volume (S2 tables). **Operating point (OP): t = 0.01 and the touching component ranks ≤ 3** among that case's t = 0.01 components by peak p. Ties are ranked pessimistically (only 1 lesion, at t = 0.001, is affected) |
| 2. normative (S9) | score A, matched-radius candidate rank (`cand_rank_A`). **OP: top-10.** Not evaluable = S9 could not score the lesion (`pct_A` NaN). That happens either because there is no lattice centre in lesion + 2 mm (outside the search region, 94 lesions) or because the footprint is too large (32). Not-evaluable lesions are excluded from S9's own hit rate |
| 3. duct | case flag = `mpd_dilated` (S2-era `duct_dilation_flags.csv`) OR **head-end dilation**. Head-end dilation is the existing Session-3 rule from `src/duct-cutoff/component_features.py`: MPD `caliber_head_mm` ≥ 3.21 mm, where `caliber_head_mm` is the median caliber over the first 10 mm of the head→tail oriented MPD centerline in `cutoff_per_duct.csv`. This is on that pipeline's own 1 mm isotropic caliber scale, and it reproduces CONTEXT's 56.8% of evaluable tumor+ / 5.1% of negatives. **OP: case flag AND `dist_to_mpd_mm` ≤ 10.** Not evaluable = no MPD mask in the case (232 lesions) |
| union / overlap / gate | a not-evaluable lesion counts as **not caught** (conservative) |
| gate | missed 20-40 mm: union ≥ 30% with Wilson lower bound ≥ 20%, **and** at least one signal has unique catches (caught by it alone) ≥ 10% of all missed 20-40 mm lesions. Primary lesion set = all; also reported without the separated lesions |

## Oracles

- **Synthetic table.** 10 lesions whose raw columns are chosen so each signal's hit is known, including boundary
  cases: rank 3 vs 4, rank 10 vs 11, distance exactly 10 mm, head caliber exactly 3.21 mm, and not-evaluable
  lesions with an otherwise passing rank. Run through the same functions as the real data, it reproduces the
  hand-counted Venn cells, the 0/1/2/3 counts, unique catches, per-signal n, and pass/fail gates.
- **Mutation check.** Setting the rank cut to 4, the distance to 9.95 mm, or top-N to 9 each breaks 5 of the 10
  tests.
- **Real-data checks against S2:**
  - total 1235 lesions; 20-40 mm n = 597
  - detected at t = 0.5 in 20-40 mm: 357/597 = 59.8% (S2: 59.8%, *carried over*)
  - recomputed hit at t ∈ {0.1, 0.01, 0.001} == (S2 `max_t_hit_touch` ≥ t) for every lesion
  - FP components/case at t = 0.5 / 0.1 / 0.01 / 0.001 equal `nnunet_froc_points.csv` exactly (all cases and
    negatives)
- **Joins.** gt_vox is equal on every join: nnU-Net table, S9 `lesion_ranks`, v3 labels, and the 248 Dataset502
  fold-0 lesions. The v3 class in S9's table equals the v3 file. MPD-mask status equals master `mpd_present`.

## Population

| bin | lesions | detected (t = 0.5) | missed | missed, without separated |
|---|---|---|---|---|
| 10-20 | 275 | 71 (25.8%) | 204 | 184 |
| **20-40** | 597 | 357 (59.8%) | **240** | **219** |
| ≥40 | 277 | 230 (83.0%) | 47 | 40 |

## 1. Hit rate per signal, MISSED vs DETECTED (20-40 mm, all lesions; Wilson 95% CI)

| signal | component | missed | detected |
|---|---|---|---|
| nnU-Net | touch t = 0.1 | 27/240 = 11.2% [7.8-15.9] | 357/357 |
| | touch t = 0.01 | 61/240 = 25.4% [20.3-31.3] | 357/357 |
| | touch t = 0.001 | 103/240 = 42.9% [36.8-49.2] | 357/357 |
| | t = 0.1, rank ≤ 3 | 20/240 = 8.3% | 344/357 |
| | t = 0.001, rank ≤ 3 | 67/240 = 27.9% | 345/357 |
| | **OP: t = 0.01, rank ≤ 3** | **43/240 = 17.9% [13.6-23.3]** | 344/357 = 96.4% |
| normative (S9) | pct_A ≥ 90 | 32/218 = 14.7% [10.6-20.0] | 83/348 = 23.9% [19.7-28.6] |
| | **OP: top-10** | **72/218 = 33.0% [27.1-39.5]** | 161/348 = 46.3% [41.1-51.5] |
| | *uniform-placement chance, top-10* | *32.8%* | *50.3%* |
| duct | (a) case `mpd_dilated` | 52/174 = 29.9% [23.6-37.1] | 188/318 = 59.1% [53.6-64.4] |
| | (a) case `cbd_dilated` | 38/213 = 17.8% | 120/347 = 34.6% |
| | (a) case either | 73/220 = 33.2% | 227/353 = 64.3% |
| | (a) case double | 17/167 = 10.2% | 81/312 = 26.0% |
| | (b) head-end dilated | 24/51 = 47.1% | 119/175 = 68.0% |
| | `mpd_dilated` OR head-end | 54/174 = 31.0% | 189/318 = 59.4% |
| | (c) `dist_to_mpd_mm` ≤ 10 | 61/174 = 35.1% | 230/318 = 72.3% |
| | **OP: flag AND ≤ 10 mm** | **35/174 = 20.1% [14.8-26.7]** | 176/318 = 55.3% [49.9-60.7] |

**How each signal's denominator is formed (20-40 mm):**
- nnU-Net: every lesion is evaluable.
- S9: not evaluable for 22 missed lesions (21 with no lattice centre, 1 footprint too large) and 9 detected
  lesions (8 + 1).
- Duct: 66 missed and 39 detected lesions are in cases without an MPD mask.
- The (a) flags use mask-present denominators per duct (CBD mask for `cbd_dilated`, either mask for "either",
  both masks for "double").
- Head-end caliber exists only where the MPD centerline is status `ok` (418/1308 cases), so (b) is evaluable for
  just 51 missed and 175 detected lesions.

**Other summaries, missed vs detected:**
- Median rank among touched lesions (t = 0.1 / 0.01 / 0.001): missed 2 / 2 / 3, detected 1 / 1 / 1.
- Median peak p in lesion + 2 mm: missed 8.5e-4, detected 0.997.
- Median S9 pct_A: missed 57.2, detected 60.5.

**Without the separated lesions (missed n = 219):**
- nnU-Net OP 41/219 = 18.7% [14.1-24.4]
- S9 OP 72/218 = 33.0% (NE 1)
- duct OP 35/158 = 22.2% [16.4-29.2] (NE 61)

All 21 S9 no-centre misses in this bin are separated lesions.

**Context bins (all lesions, OP):**

| bin | nnU-Net missed / detected | S9 missed / detected (chance, missed) | duct missed / detected |
|---|---|---|---|
| 10-20 | 23/204 = 11.3% / 98.6% | 21/183 = 11.5% / 15.4% (13.3%) | 19/155 = 12.3% / 14.1% |
| ≥40 | 7/47 = 14.9% / 96.5% | 26/41 = 63.4% / 75.1% (51.2%) | 3/34 = 8.8% / 52.7% |

Full grid (every component × bin × lesion set × group) is in `summary.csv`, section `signal_hit`.

## 2. Cost of each signal

| signal | cost |
|---|---|
| nnU-Net, any component at t = 0.1 / 0.01 / 0.001 | 2.19 / 2.94 / 4.68 FP components per case (all 1308); 2.55 / 3.34 / 5.09 on the 327 negatives |
| **nnU-Net OP** (top 3 at t = 0.01) | **1.77 FP/case** (all), **2.09** (negatives). The t = 0.5 segmentation is 1.72 / 2.06 |
| **S9 OP** | **10 candidates per case by design.** S9's `froc_points.csv` reports 7.62 FP/case (r 6) / 7.58 (r 10) on positives' non-tumor region and 9.91 / 9.89 on negatives |
| **duct case flag** | **21/327 negatives = 6.4%** (21/235 = 8.9% of negatives with an MPD mask). It also fires in 312/631 = 49.4% of cases with a detected lesion, and in 89/433 = 20.6% of cases with a missed lesion |

**Head-end dilation adds almost nothing on top of `mpd_dilated`.**
- In negatives, both head-end-flagged cases are already `mpd_dilated`.
- In detected-lesion cases it adds 1 (311 → 312).
- Among missed 20-40 mm lesions it adds 2 (52 → 54).

**On the "detected" columns:** the union of signals reaches 98.3% of detected 20-40 mm lesions. This is
trivial: nnU-Net's own signal is near 100% on lesions it already touches at t = 0.5. It is not a cost.

## 3. Overlap among missed 20-40 mm lesions

Not evaluable counts as not caught.

| set | n | caught by 0 / 1 / 2 / 3 | union | nnU-Net only | S9 only | duct only | nn+S9 | nn+duct | S9+duct | all 3 |
|---|---|---|---|---|---|---|---|---|---|---|
| **all** | 240 | 126 / 83 / 26 / 5 | **114 = 47.5% [41.3-53.8]** | 19 | 47 | 17 | 13 | 6 | 7 | 5 |
| without separated | 219 | 107 / 81 / 26 / 5 | 112 = 51.1% [44.6-57.7] | 17 | 47 | 17 | 13 | 6 | 7 | 5 |
| evaluable by all 3 | 158 | 69 / 61 / 23 / 5 | 89 = 56.3% [48.5-63.8] | 12 | 32 | 17 | 10 | 6 | 7 | 5 |

Signal totals among the 240: nnU-Net 43, S9 72, duct 35. Each lesion falls in exactly one cell of the row.

**Pairwise unions** (reported only, not the gate), all lesions:
- nnU-Net ∪ duct: 67/240 = 27.9% [22.6-33.9]
- nnU-Net ∪ S9: 97/240 = 40.4%
- S9 ∪ duct: 95/240 = 39.6%

Without the separated lesions, the same three are 65/219 = 29.7% [24.0-36.0], 43.4% and 43.4%.

**Context bins, missed, all lesions:**
- 10-20 mm: union 57/204 = 27.9% [22.2-34.5]; caught by 0 / 1 / 2 / 3 = 147 / 53 / 2 / 2; unique nn 21, S9 17,
  duct 15.
- ≥40 mm: union 30/47 = 63.8% [49.5-76.0]; 17 / 25 / 4 / 1; unique 3 / 21 / 1.

Without the separated lesions: 30.4% (10-20) and 75.0% (≥40).

## 4. GATE: PASS

| lesion set | union (≥ 30%) | CI lower bound (≥ 20%) | unique / all missed (≥ 10% for one signal) | verdict |
|---|---|---|---|---|
| all (n = 240) | 47.5% ✓ | 41.3% ✓ | nnU-Net 7.9%, **S9 19.6% ✓**, duct 7.1% | **PASS** |
| without separated (n = 219) | 51.1% ✓ | 44.6% ✓ | nnU-Net 7.8%, **S9 21.5% ✓**, duct 7.8% | **PASS** |

A secondary reading of "unique" was reported but not used: unique catches over the misses the other two signals
leave. It gives nnU-Net 19/145 = 13.1%, S9 47/173 = 27.2%, duct 17/143 = 11.9%. That reading would pass on every
signal, so the primary (stricter) reading is the binding one.

### What the pass rests on

1. **S9 top-10 is indistinguishable from chance on these lesions.**
   - The chance model: if 10 candidates were lattice centres drawn uniformly from the case's search region, a
     lesion covering a fraction f of the centres would be hit with probability 1 − (1 − f)^10. f comes from S9's
     `n_centres_dil2 / n_centres`.
   - The median 20-40 mm missed lesion covers 3.5% of the gland's lattice, so chance ≈ 33%. The observed rate is
     33.0%.
   - It is also at or below chance for detected 20-40 mm (46.3% vs 50.3%) and for 10-20 mm (11.5% vs 13.3%).
   - Only ≥40 mm missed lesions beat this chance rate (63.4% vs 51.2%, n = 41).
   - Local-maximum suppression spreads real candidates apart, so the true chance rate is a little higher than this
     estimate, which strengthens the point.
2. **Without S9, the remaining union fails the gate.** nnU-Net ∪ duct = 27.9%, and neither reaches 10% unique.
3. **Every signal is enriched in DETECTED lesions, not MISSED ones**: S9 46% vs 33%, duct OP 55% vs 20%,
   duct-flagged cases 49% vs 21%.
   - This is what you would expect if duct dilation and normative anomaly track the same "obvious" tumors
     nnU-Net already segments.
   - A re-ranker would mostly re-find detected lesions.

So the brief's literal criterion says "build the re-ranker". But the only signal that clears the uniqueness bar
behaves as random placement at 7.6 FP/case.

## 5. Missed 20-40 mm lesions caught by NOTHING

126/240 (all lesions); 107/219 without the separated ones. Each cell shows caught-by-nothing / missed in that
stratum.

| stratum | all lesions | without separated |
|---|---|---|
| v3 hypo / iso / hyper | 54/93 (58%) / 59/110 (54%) / 13/37 (35%) | 42/80 / 54/105 / 11/34 |
| phase non-contrast / arterial / venous / delay | 64/95 (67%) / 19/38 (50%) / 42/106 (40%) / 1/1 | 58/89 / 17/35 / 31/94 / 1/1 |
| region head / body / tail / none | 78/151 (52%) / 28/47 (60%) / 17/38 (45%) / 3/4 | 69/141 / 24/43 / 14/35 / — |
| slice spacing < 2.5 / 2.5-5 / ≥ 5 mm | 71/113 (63%) / 25/77 (32%) / 30/50 (60%) | 62/102 / 23/75 / 22/42 |
| case at t = 0.5: no prediction / wrong place only / another lesion of the case found | 16/38 (42%) / 81/164 (49%) / 29/38 (76%) | 11/32 / 76/158 / 20/29 |

Patterns:
- **Contrast matters.** Two thirds of missed non-contrast lesions have no signal, versus 40% of venous ones.
- **Satellites in multi-lesion cases.** When nnU-Net found another lesion in the case, the missed one is the
  hardest to flag (76%).
- **Phase and thickness are not separated here.** Thin-slice (< 2.5 mm) misses are also mostly unflagged (63%).
  This analysis did not separate thickness from phase.

## 6. Visual-review list (missed 20-40 mm)

**How the lists are ordered** (fixed before looking; `core.evidence_order`):
- Sort first by number of signals caught (descending).
- Then by `graded`, the mean of the available graded strengths:
  - percentile of peak p in lesion + 2 mm among these lesions
  - pct_A / 100
  - for duct-flagged cases, 1 − min(dist to MPD, 30)/30; for unflagged cases, 0
- Then by diameter.
- "None" = caught by nothing, weakest graded evidence first.
- The `review_list` / `review_rank` columns in `rescue_by_lesion.csv` mark both lists.

**Strongest combined evidence**

| # | case_id | lesion | diam mm | v3 | region | phase | signals (nn/S9/duct) | peak p | pct_A | S9 rank | MPD mm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PanTS_00008601 | 1 | 38.2 | hypo | tail | venous | ✓✓✓ | 0.368 | 100 | 1 | 3.4 |
| 2 | PanTS_00003969 | 1 | 32.8 | hypo | head | venous | ✓✓✓ | 0.231 | 83.5 | 10 | 0 |
| 3 | PanTS_00009887 | 1 | 39.1 | hypo | head | venous | ✓✓✓ | 0.249 | 62.5 | 9 | 0 |
| 4 | PanTS_00003318 | 1 | 35.1 | hypo | head | venous | ✓✓✓ | 0.031 | 71 | 7 | 0 |
| 5 | PanTS_00001601 | 1 | 38.9 | hyper | head | venous | ✓✓✓ | 0.309 | 47 | 8 | 2.5 |
| 6 | PanTS_00005180 | 1 | 26.7 | hypo | tail | non-contrast | ✓✓ NE | 0.675 | 87.5 | 4 | — |
| 7 | PanTS_00008001 | 1 | 32.2 | hyper | head | venous | ·✓✓ | 0.0024 | 78 | 4 | 0.8 |
| 8 | PanTS_00006987 | 2 | 21.5 | hyper | tail | venous | ✓·✓ | 0.014 | 64.5 | 11 | 1.3 |
| 9 | PanTS_00001277 | 1 | 20.5 | hypo | head | arterial | ✓·✓ | 0.012 | 54 | 11 | 1.5 |
| 10 | PanTS_00003812 | 1 | 29.0 | hyper | head | venous | ✓✓· | 0.646 | 62.5 | 5 | 12.9 |
| 11 | PanTS_00008726 | 1 | 22.5 | iso | head | venous | ✓✓· | 0.049 | 84.5 | 3 | 15 |
| 12 | PanTS_00000911 | 1 | 36.2 | iso | head | venous | ✓✓ NE | 0.043 | 60.5 | 5 | — |
| 13 | PanTS_00003287 | 1 | 25.0 | iso | head | venous | ✓·✓ | 0.028 | 28 | 14 | 0.8 |
| 14 | PanTS_00004542 | 1 | 35.5 | hypo | body | venous | ·✓✓ | 5.4e-4 | 87.5 | 4 | 9.0 |
| 15 | PanTS_00001086 | 1 | 21.4 | iso | head | venous | ✓·✓ | 0.461 | 3 | 21 | 0 |

**No evidence**

| # | case_id | lesion | diam mm | v3 | region | phase | excluded | peak p | pct_A | MPD mm |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PanTS_00003900 | 3 | 22.3 | hypo | head | arterial | yes | 1.2e-5 | NE | — |
| 2 | PanTS_00005162 | 2 | 23.0 | hypo | — | venous | yes | 1.3e-5 | NE | — |
| 3 | PanTS_00002451 | 1 | 29.0 | iso | head | non-contrast | | 2.7e-5 | 0 | — |
| 4 | PanTS_00001678 | 1 | 20.4 | hypo | tail | non-contrast | | 3.8e-6 | 11.5 | 73.8 |
| 5 | PanTS_00007578 | 2 | 25.4 | iso | tail | venous | yes | 5.1e-5 | NE | 69.8 |
| 6 | PanTS_00007661 | 2 | 31.5 | hypo | head | arterial | yes | 3.6e-5 | NE | — |
| 7 | PanTS_00002081 | 1 | 36.9 | hyper | head | non-contrast | | 9.7e-5 | 0 | 1.3 |
| 8 | PanTS_00003968 | 2 | 26.2 | iso | head | non-contrast | | 1.1e-4 | 0 | 9.3 |
| 9 | PanTS_00006135 | 1 | 20.9 | hypo | head | venous | yes | 7.9e-5 | NE | 25.1 |
| 10 | PanTS_00007868 | 1 | 20.0 | iso | head | non-contrast | | 1.8e-4 | 3 | 12.2 |
| 11 | PanTS_00001751 | 1 | 24.8 | iso | tail | non-contrast | | 1.9e-4 | 4.5 | 54.9 |
| 12 | PanTS_00001351 | 1 | 28.8 | hypo | head | arterial | | 2.6e-4 | 2 | 41.3 |
| 13 | PanTS_00008242 | 1 | 25.7 | hypo | — | non-contrast | yes | 1.5e-4 | NE | 27.0 |
| 14 | PanTS_00008854 | 1 | 27.7 | hypo | head | venous | | 1.1e-4 | 17.5 | 7.3 |
| 15 | PanTS_00002177 | 1 | 29.4 | iso | head | non-contrast | | 7.2e-5 | 26.5 | 20.4 |

**Why some "no evidence" lesions sit far from the MPD.** Several have MPD distances of 40-74 mm. GT MPD masks
are short fragments (median centerline 28 mm, *carried over*), so a large distance may only mean the fragment is
elsewhere in the gland. These lesions, and the 6 separated ones in this list, are still worth a check for label
or anatomy problems before they are read as true misses.

## 7. Optional: Dataset502 vs Dataset501, nnU-Net fold 0 only

**Method.**
- Read `fold_0/validation/{case}.nii.gz` from the server, which only reads there (rsync to
  `work/rescue_check/ds502_fold0_seg/`, 9 MB). Nothing was written on the server and no GPU was used.
- Lesion prediction = label == 2. Touch rule at the saved segmentation for both models.
- 264 cases, 248 lesions, gt_vox equal to master for all 248.
- **Orientation check.** All 264 affines equal the CT affine. On 8 spot cases, 502's label > 0 against the GT
  gland envelope gives Dice 0.71-0.95, versus 0.12-0.37 when x is flipped.

**n is small, this is one fold, and 501's fold-0 20-40 mm detection (67.3%) is above its pooled 59.8%.**

| bin (all lesions) | n | detected 501 | detected 502 | missed by both | recovered by 502 | newly missed by 502 |
|---|---|---|---|---|---|---|
| all | 248 | 143 (57.7%) | 180 (72.6%) | 58 | 47 | 10 |
| 10-20 | 57 | 18 (31.6%) | 29 (50.9%) | 25 | 14 | 3 |
| **20-40** | 110 | 74 (67.3%) | 95 (86.4%) | **11** | **25** | **4** |
| ≥40 | 60 | 50 (83.3%) | 55 (91.7%) | 3 | 7 | 2 |

- **Without the separated lesions,** 20-40 mm is n = 105: 72 → 92 detected; missed by both 9, recovered 24,
  newly missed 4.
- **Signals on the recovered lesions.** Of the 25 fold-0 20-40 mm lesions 502 recovers, 10 had been caught by
  ≥1 S11 signal. Of the 11 missed by both models, 5 had.

**Cost, fold 0.** FP components per case (touch rule): 501 at t = 0.5 vs 502.

| cases | 501 FP/case | 502 FP/case | cases with any lesion prediction, 501 | same, 502 |
|---|---|---|---|---|
| all 264 | 1.60 | 0.39 | 91.3% | 77.7% |
| 66 negatives | 1.92 | 0.33 | 80.3% | 25.8% |

**502's gain is not bought with more false positives; it has far fewer.** That is despite the
lesion-patch-sampling confound, which halves lesion-centred patches in 502 (S7, *carried over*).

## Caveats

- **GT anatomy.** All duct inputs and S9's search region use GT anatomy masks: upper bounds for an automatic
  pipeline.
- **Head-end dilation coverage.** It is defined only where the MPD centerline is status `ok` and oriented (418
  cases). Elsewhere the case flag falls back to `mpd_dilated` alone. The 3.21 mm threshold came from Session 2's
  native-grid percentiles and was applied to Session 3's 1 mm isotropic calibers without re-derivation. That is
  inherited from `component_features.py`, not changed here.
- **Hit rule at low t.** The nnU-Net signal uses the touch rule on whole-volume components. At t = 0.01 components
  grow and merge, so a "touch" can come from a large low-confidence blob anchored elsewhere. The ranks are case
  ranks by peak p, not a calibrated score.
- **S9 rank radius.** S9 top-10 uses the matched-radius rank (r 6 or r 10 per lesion), as in S9's README. Its
  cost row is per radius.
- **Not-evaluable handling.** Not evaluable = not caught in unions. Restricting to lesions evaluable by all three
  raises the 20-40 mm union to 56.3% (n = 158). The gate verdict is the same either way.
- **Phase labels** may be noisy ("arterial" parenchyma is darker than venous, S3 *carried over*).
- `slice_thickness_mm` is voxel spacing, not reconstruction thickness (S1).

## Files

| file | rows | content |
|---|---|---|
| `rescue_by_lesion.csv` | 1235 | master keys + size / tier / v3 / phase / region / spacing; `detected_t05`, `missed`, `case_pred_t05`; `peak_p_dil2`; per t: `hit_t*`, `rank_t*`, `n_comps_case_t*`; `sig_nnunet`; `pct_A`, `cand_rank_A`, `s9_not_evaluable_reason`, `sig_normative`, `s9_chance_topN`; duct flags, `caliber_head_mm`, `head_end_dilated`, `duct_case_flag`, `dist_to_mpd_mm`, `sig_duct` (empty = not evaluable); `n_signals`, `evidence_graded`, `review_list`, `review_rank`; `detected_502_fold0` (fold 0 only) |
| `summary.csv` | 716 | long format: `section` (population, signal_hit, not_evaluable, cost, overlap, gate, nothing_caught, ds502_fold0) × `lesion_set` × `size_bin` × `group` × `signal` × `metric` with n, k, value, Wilson CI |
| `README.md` | | this file |
