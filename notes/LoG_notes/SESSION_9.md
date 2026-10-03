# Session S9: normative ("what does normal pancreas look like") anomaly model, pilot (2026-10-02)

**Answer: all gates FAIL.** The positive control (≥40 mm median within-pancreas percentile ≥ 90) fails:
71.2 for score A, 53.0 for B. Per the brief, that means the features are treated as broken and no target
claim is made.

- **Target group** (10-20 mm, hypo/hyper v3, enhanced, n = 106): median percentile 71.8 (A) vs S4's 23.5
  (*carried over*).
- **Top-10 hit rate:** 16.0% (CI 10.3-24.2%) vs S4's 2.8%. That half of the target gate would clear S4's CI
  on its own. The median half does not.

Full tables, caveats and the reproduce recipe: `results/normative_pilot/README.md`.
Every printed number: `work/normative/summary.txt`.

## What was built (`src/normative/`, importable; no existing script edited)

| file | role |
|---|---|
| `common.py` | paths and every a-priori constant (design freeze), thickness/phase groups |
| `pool.py` | Phase 0: mask checks over 8497 non-cohort negatives (~20 min), filters, stratified sample, LFS include lists |
| `extract_bank_cts.sh`, `pull_bank_masks.sh` | Phase 0 data: 300 CTs (9.3 GB) into a separate `ct_normal_bank/`; 1500 masks (207 MB), one structure per pull |
| `features.py` | pure functions: patch lattice, 15/17 rotation-invariant features, u frame, native-grid sphere insertion, lattice local maxima, same-footprint random sets |
| `case_pass.py` | per-case pass (bank / cohort / insert); cached gzip pickles in `work/normative/features/` (~0.25 MB/case). The lesion mask is read only after the features exist. The insert mode also runs S4's own `analyze()` (imported, not modified) on the same planted CT |
| `model.py` | standardize + PCA (95%), strata and merge rule, k-means/Mahalanobis (A), k-NN (B), union fallback for unknown phase/u, leave-one-out within-case z (C) |
| `evaluate.py` | fit / score / calibrate / insertion / report → deliverables |
| `tests/test_normative.py` | 21 phantom tests, written and passing before any real case ran. Full suite: 74 passed |

Run times on the PC (4 workers, CPU only): cohort pass 24 min; bank 6 min; insertion pass 15 min;
fit 30 s; scoring 10 s; report about 1 min. The server was not touched.

## Oracle results

1. **Phantom.** All 21 tests pass, including an end-to-end chain in which a 15 mm −30 HU sphere ranks
   ≥ 90th against a phantom bank. They caught one real bug: a stratum lookup for a phase with no bank
   patches.
2. **Regressions against earlier sessions.**
   - u reproduces the master table to 4.9e-6 over 1205 lesions.
   - `gt_vox` equals the master table for all 1235 lesions.
   - The 11 empty-envelope negatives are exactly S2/S4's list.
   - S4's target baseline was recomputed from its CSV: median beaten-by 0.765, n = 106.
3. **3a calibration: PASS.** Pseudo-lesions in the 316 held-out negatives give near-uniform percentiles
   (KS p ≥ 0.14). There is a mild excess ≥ 90 (12-15%) for footprints of 20-60 centres.
4. **3b insertion: nearly flat.** Only the 20 mm −30 HU sphere separates from its null (median 63.5 vs
   49.5; 27% ≥ 90). 10-15 mm spheres of any contrast are indistinguishable from the dHU = 0 controls, and
   S4's LoG is equally flat.
5. **3c scan level.** Score A shows no positive-vs-negative difference within any thickness bin. Score B
   differs in the 1.5-2.5 mm bin only (p = 1.8e-7, 48 negatives there).

## Gate outcomes

| gate | verdict |
|---|---|
| Positive control (≥40 mm median ≥ 90, any score) | **FAIL**: A 71.2, B 53.0 |
| Target, A | **FAIL**: median 71.8; top-10 16.0% (CI 10.3-24.2%), which alone would clear S4's 1.0-8.0% |
| Target, B | **FAIL**: median 48.0; top-10 11.3% (CI 6.6-18.8%) |
| Target, C_A / C_B | **FAIL**: identical to A / B by construction |
| nnU-Net-missed ∩ target (n = 74) | A: median 73.0, top-10 13.5% (CI 7.5-23.1%) |

As instructed, nothing was scaled to more negatives and no attention model was built.

## Surprises

1. **"Normal" is broader than a tumor's contrast.**
   - Within one normal pancreas, the patch median HU has an SD of ~44 HU, and 23% of normal patches have
     a median below 20 HU. A centred patch in a thin gland, inside a search region of envelope + 3 mm,
     mixes in fat.
   - So a −30 HU lesion just looks like another normal patch type. The insertion test showed this before
     any lesion was scored.
   - The features see the sphere: the centre patch's mean drops 20-26 HU. The density model doesn't flag
     it, and in some cases the score *falls*.
2. **The normative score still beats S4 by 3-6× at every size.** Top-10 hit rates for A vs S4 on the same
   lesions: 16% vs 3% (target), 43% vs 11% (20-40 mm), 74% vs 18% (≥40 mm). Its false alarms mostly
   have no vein, duodenum, SMA or aorta within 3 mm (81% for A). S4's top false peaks were labelled
   veins/duodenum 55% of the time (*carried over*), but that was a nearest-structure-within-30-mm label,
   so the two are not directly comparable. The signal is real; the lesion just isn't the *single* most anomalous region of its
   pancreas often enough (top-1 = 2.8% target, 27% ≥40 mm).
3. **The positive control is partly a protocol effect at extreme sizes.** The 16 lesions with > 1000
   footprint centres score a median of 0: a same-size random set covers almost the whole remaining gland
   and always contains some clutter peak. Another 30 ≥40 mm lesions are unscorable because the far region
   is smaller than the footprint. The rest still have medians of only 69-77, so the verdict stands, but a
   max-over-footprint percentile is a poor positive control for very large lesions. Worth redefining
   (e.g. a fixed-size core patch) *before* any future session, not now.
4. **Score C is a no-op for within-scan metrics.** A per-case monotone rescaling cannot change a
   within-case rank or a top-N list (max |C − A| = 0.0 over all lesions). It only matters for global
   thresholds, where C_A gets 13% of the target group at ~5 FP/case vs 6% for raw A at 3.7.
5. **More bank data doesn't help.** The target median is 74 / 71 / 72 at 100 / 200 / 300 bank cases.
6. **Score A (k-means, Rahul's) beats score B (k-NN) almost everywhere** (target 71.8 vs 48.0). B's false
   alarms sit on veins more (22.8% vs 8.4%). The likely reason is that k-NN in a 5-D PCA space is dominated
   by the dense fat/edge patches.
7. **PCA at 95% variance keeps only 5-6 components** from 15-17 features: the HU quantiles, mean and ring
   terms are nearly collinear. The model is effectively five-dimensional.

## Process notes

- Phase-0 thickness came from max(metadata spacing), validated on the cohort beforehand (1306/1308
  bins). 6 bank cases moved bin once their true affine thickness was known; the achieved mix is reported.
- 17 bank cases have no head or tail mask (u undefined). Their patches were left out of the fit rather
  than given a fake u.
- An early `pkill -f` matched its own shell and killed my wait loop. That was harmless, but avoid
  `pkill -f` with patterns that appear in the invoking command.
- Test failures during development were three wrong expectations plus one real bug:
  - the soft-edge profile value at d = R + 0.2 mm
  - the merge-rule outcome for 7000-patch phases
  - head/tail selection on translated points in the u test
  - the real bug: an unfitted stratum lookup
- CONTEXT.md was not edited. Suggested addition: an "S9 result" block mirroring the S4 one.
