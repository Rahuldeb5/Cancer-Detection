# S9 normative anomaly model, pilot (2026-10-02)

**Question.** If we learn what normal pancreas patches look like from unused PanTS negatives, does a
10-20 mm tumor stand out from the rest of its own pancreas better than S4's blob map did?

**Answer: better than S4, but nowhere near the gates. All three gates FAIL.**

- **Positive control fails.** ≥40 mm lesions reach a median within-pancreas percentile of 71.2 (score A).
  The gate needs ≥ 90. Under the rule fixed beforehand, the features count as broken and no target claim is made.
- **The target group fails too.** 10-20 mm, hypo/hyper on v3, contrast-enhanced, n = 106:
  - median percentile 71.8 (A) / 48.0 (B); the gate needs ≥ 90
  - S4's blob map on the same lesions: 23.5 (carried over)
- **The insertion oracle shows why.** A smooth, homogeneous 10-20 mm sphere of ±15-30 HU planted in a
  normal pancreas is barely more anomalous than the gland's own patches.

Every number below was computed in S9 unless it is marked *carried over*. Code: `src/normative/`.
Intermediates: `work/normative/` (git-ignored). Session notes: `notes/LoG_notes/SESSION_9.md`.

## Gates (fixed before any cohort lesion was scored)

| gate | requirement | result | verdict |
|---|---|---|---|
| Positive control | ≥40 mm median percentile ≥ 90 for at least one score | A 71.2, B 53.0, C_A 71.2, C_B 53.0 (n = 238 scored) | **FAIL** |
| Target, score A | median ≥ 90 **and** top-10 CI clear of S4's 1.0-8.0% | median 71.8; top-10 17/106 = 16.0% (CI 10.3-24.2%) | **FAIL** (median) |
| Target, score B | same | median 48.0; top-10 12/106 = 11.3% (CI 6.6-18.8%) | **FAIL** (both) |
| Target, C_A / C_B | same | identical to A / B (see "Score C") | **FAIL** |
| nnU-Net-missed ∩ target (n = 74) | reported | A: median 73.0, top-10 10/74 = 13.5% (CI 7.5-23.1%); B: 56.0, 10.8% | — |

For score A, the top-10 half of the target gate would pass on its own: its CI is clear of S4's.
The median half fails, and so does the positive control.

## What was built

### Phase 0: normal bank (300 non-cohort negatives)

**Pool filters**, applied in order:

| filter | cases passing |
|---|---|
| PanTS cases | 9901 |
| 1. not in the 1308 cohort | 8593 |
| 2. metadata `tumor?` == 0 | 8497 |
| 3. no "Pancreas lesions" report section (S1's regex) | 7828 |
| 4. `pancreatic_lesion` readable via `tumorlib.io.load_mask`, on-grid, **empty** | 7828 |
| 5. non-empty envelope | **7349** |

Notes on the filters:
- Filter 4 removed nobody: all 8497 tumor-free non-cohort cases have an empty lesion mask.
- Filter 5 needs `pancreas.nii.gz`, which is an un-pulled LFS stub for every non-cohort case, while
  head/body/tail are real files. head|body|tail being non-empty proves the envelope is non-empty.
- **479 cases had empty head/body/tail.** Deciding their envelope would have meant pulling 479 extra
  masks, so they were excluded instead. This makes filter 5 conservative, never permissive.

**Patient overlap with the cohort is unverifiable.** The metadata has no patient, study or accession
identifier: its 15 columns are ID, shape, spacing, phase, sex, age, scanner, study type, site, year,
`tumor?` and report. No bank case can be checked for sharing a patient with a cohort case.

**Sample.**
- 300 cases, seed 42, stratified to the 981 cohort positives' raw phase × slice-thickness-bin mix
  (largest-remainder quotas).
- Thickness came from max(metadata spacing), which reproduces S1's affine-derived bin in 1306/1308
  cohort cases.
- After extraction, 6 bank cases changed bin on their true affine thickness.

Achieved mix (affine thickness):

| phase | ≤1.5 | 1.5-2.5 | 2.5-5 | >5 |
|---|---|---|---|---|
| Arterial | 40 | 17 | 8 | 0 |
| Delay | 0 | 1 | 0 | 0 |
| Non-contrast | 51 | 14 | 21 | 0 |
| Venous | 34 | 81 | 32 | 1 |

**Data fetched.**
- 300 CTs, **9.3 GB**, extracted into a separate `~/research/datasets/pants/ct_normal_bank/` by
  `src/normative/extract_bank_cts.sh`. The cohort's `ct_staging` is untouched. The script refuses any
  cohort ID.
- 1500 masks (pancreas, veins, duodenum, SMA, aorta), **207 MB**, git-lfs pulled one structure per pull.
  The longest include string was 16.5 kB. Nothing was reset or committed in the mask repo.

**No cohort case is in the bank.** This is asserted in `pool.py`, `extract_bank_cts.sh`, `case_pass.py`
(both directions) and `evaluate.py`.

### Phase 1: patch features (fixed a priori; no lesion mask is ever read in the feature path)

**Image and patch grid.**
- Per case: `tumorlib` envelope, then `fill_search_region` (the same region as `tumorlib.search_region`),
  then a bbox crop padded by 30 mm, then 1 mm isotropic, then HU clipped to [-100, 300].
- Spherical patches of r = 6 and 10 mm, centred on a 4 mm lattice inside the search region.
- Median 1954 patches per cohort case and 1989 per bank case.

**Features (15 for r 6 mm, 17 for r 10 mm):**
- HU mean, SD, and the 5/25/50/75/95th percentiles
- mean HU in 2 mm shells minus the patch median
- patch mean minus the mean of a 5 mm surrounding ring
- Gaussian (σ 1 mm) gradient-magnitude mean and SD
- histogram entropy (25 HU bins)
- raw σ²·LoG at the centre, σ = r/√3 (not MAD-normalized)

**Context covariates.**
- u: the S1 gland frame (PCA of `pancreas.nii.gz` in world mm, head→tail sign fix). It reproduces the
  master table's `u` to **4.9e-6** over 1205 lesions.
- Distance to veins / duodenum / SMA / aorta, from GT masks.
- Phase and slice thickness.

**All anatomy is GT** (the gland envelope, u and the structure distances), so every number here is an
upper bound for an automatic pipeline.

### Phase 2: normal model (bank only)

**Fit.**
- Standardize, then PCA at 95% variance. This keeps only **5 (r 6) / 6 (r 10) components**, so the
  hand-made features are highly redundant.
- Strata = phase group × thickness (≤2.5 / >2.5 mm) × u tercile:
  - phase groups: non-contrast, arterial, venous; Delay is folded into venous
  - u tercile edges: bank quantiles 0.241 / 0.633
- **All 18 strata had ≥ 2000 bank patches at n = 300, so no merging happened.** The smallest is
  arterial|thick|u3 with 4111; the largest is venous|thin|u3 with 93,926.
- 17 bank cases have an empty head or tail mask, so u is undefined. Their 20,510 patches were left out of
  the fit, leaving 630,863 patches.
- At n = 100 the merge rule collapsed the strata to 15.

**Scores.**
- **A** (Rahul's method): k-means with K = 16 per stratum. A cluster's diagonal variance is shrunk toward
  the stratum variance with weight 10 patches. The score is the minimum diagonal-Mahalanobis distance over
  the 16 centroids. "Nearest" is read in the same metric: an interpretation, fixed beforehand.
- **B**: mean Euclidean distance to the 5 nearest bank patches in PCA space.
- **C**: A or B, z-scored leave-one-out against every other patch of the same pancreas.
- A patch with an unknown phase or u is scored against the union of every compatible stratum. One cohort
  case has no phase; 27 lesions lack u.

## Oracles (run before any lesion was evaluated)

### Phantom (`tests/test_normative.py`, 21 tests, all pass)

The tests cover:
- feature values on constant and ball phantoms
- invariance to lattice rotations
- the LoG scale law
- the insertion profile: integral preserved across 1 / 2.5 / 5 mm slice grids, centre = dHU, analytic
  edge values
- the u frame, which matches `lesion-sectioning/pca_hitrate.build_frame`
- the merge rule
- leave-one-out z
- footprint-protocol calibration on a null map
- an end-to-end phantom: a 15 mm −30 HU sphere in a textured gland ranks ≥ 90th for both scores against a
  12-case phantom bank

They caught one real bug: a stratum lookup for a phase that has no bank patches.

### 3a. Calibration: PASS

Pseudo-lesions (random compact footprints of 1/5/20/60 lattice centres) in the 316 held-out cohort
negatives were run through the full protocol.

- Percentiles are close to uniform for every footprint size, radius and score: mean 48-53, KS D ≤ 0.064,
  KS p ≥ 0.14.
- **Mild skew:** at 20-60 centres the share ≥ 90 is 12-15% instead of 10%, so the protocol is slightly
  generous to large footprints. That works *against* the gate failures, not for them.
- The literal check ("one random patch against all others in its pancreas") is uniform by
  exchangeability. It is a bookkeeping check, also reported in `work/normative/summary.txt`.

### 3b. Insertion test: nearly flat (`insertion_test.csv`)

Setup:
- 50 random cohort negatives, one random gland location each (≥ 2 mm inside the envelope), 9 sphere
  conditions plus 3 dHU = 0 null controls.
- Each sphere was planted on the **native** grid with partial-volume averaging and a soft 1 mm edge, then
  resampled.
- Each method's matched-radius percentile is reported. S4 is its own LoG pass on the same planted CT, run
  through S4's own `analyze()` with matching polarity.

Median percentile [share ≥ 90]:

| diam | dHU | A | B | S4 LoG |
|---|---|---|---|---|
| 10 | −30 | 51.0 [20%] | 46.5 [6%] | 33.2 [6%] |
| 10 | −15 | 46.0 [10%] | 40.8 [6%] | 30.0 [6%] |
| 10 | +15 | 35.0 [12%] | 43.8 [4%] | 47.8 [14%] |
| 10 | 0 (null) | 44.5 [8%] | 36.8 [2%] | 30.5 [8%] |
| 15 | −30 | 56.0 [18%] | 48.0 [16%] | 45.8 [12%] |
| 15 | −15 | 45.2 [12%] | 50.8 [16%] | 41.5 [10%] |
| 15 | +15 | 41.8 [10%] | 41.5 [14%] | 51.5 [10%] |
| 15 | 0 (null) | 36.0 [10%] | 39.2 [14%] | 39.0 [10%] |
| 20 | −30 | **63.5 [27%]** | **65.5 [22%]** | 51.8 [8%] |
| 20 | −15 | 57.0 [16%] | 55.5 [20%] | 51.5 [10%] |
| 20 | +15 | 55.0 [14%] | 56.0 [14%] | 64.2 [20%] |
| 20 | 0 (null) | 49.5 [14%] | 47.0 [14%] | 59.0 [10%] |

- Only the 20 mm −30 HU sphere separates from its null.
- **This is not a planting bug.** At the sphere centre the r 10 patch's mean HU drops 20-26 HU and its
  ring contrast 15-25 HU (12 cases checked). The features see the sphere; the *anomaly score* often does
  not move.
- **Why it doesn't move.** In the bank, the patch median HU has a **within-case SD of 44 HU** (r 6,
  venous, thin), and **23% of normal patches have a median below 20 HU**. A thin gland means a centred
  patch mixes in peripancreatic fat. A −30 HU lesion in 80 HU parenchyma reads ~50 HU, which is an
  ordinary "normal" patch.
- Real tumors rank higher than planted spheres (target median 71.8 vs 56.0 for a 15 mm −30 HU sphere). They
  carry texture, shape and duct effects beyond a homogeneous HU offset.

### 3c. Scan-level confound (report only)

Per-case mean score, cohort positives (far-from-lesion patches) vs negatives:
- **Score A:** no difference in any thickness bin (Mann-Whitney U, p ≥ 0.1).
- **Score B** is higher in positives in the 1.5-2.5 mm bin only: medians 0.517 vs 0.414 (r 6), p = 1.8e-7,
  with just 48 negatives there.

This is why within-scan ranking is the primary metric.

## Cohort results (`lesion_ranks.csv`, `froc_points.csv`)

### Within-pancreas percentile at the matched radius, vs S4 on the same lesions

The S4 column is 100 × (1 − `patch_frac_ge_dark`), *carried over* from `results/log_candidates/`.

| stratum | n | scored | A | B | S4 | A ≥ 90 | A top-10 |
|---|---|---|---|---|---|---|---|
| **TARGET 10-20 mm hypo/hyper enhanced** | 106 | 92 | **71.8** | 48.0 | 23.5 | 20.7% | 16.0% |
| nnU-Net missed ∩ TARGET | 74 | 64 | 73.0 | 56.0 | 23.5 | 18.8% | 13.5% |
| 10-20 mm all | 275 | 248 | 58.5 | 36.8 | 22.5 | 13.7% | 11.3% |
| 10-20 mm iso (v3) | 99 | 95 | 52.0 | 34.5 | 16.5 | 10.5% | 12.1% |
| **20-40 mm hypo/hyper enhanced** | 295 | 279 | 65.0 | 53.0 | 17.5 | 21.5% | 44.1% |
| 20-40 mm all | 597 | 566 | 59.2 | 46.5 | 22.0 | 20.3% | 39.4% |
| **≥40 mm hypo/hyper enhanced** | 140 | 121 | 69.5 | 44.5 | 8.7 | 36.4% | 74.3% |
| ≥40 mm all | 277 | 238 | 71.2 | 53.0 | 13.5 | 34.5% | 72.9% |
| <10 mm all | 86 | 57 | 56.0 | 44.0 | 35.5 | 17.5% | 4.7% |

- "Matched radius" = the patch radius closest to the lesion's equivalent-sphere radius. It is r 6 for
  102/106 target lesions; at ≥ 20 mm it is mostly r 10.
- The A top-10 column comes from the matched-radius map.
- Per-stratum breakdowns (v3 class, phase, region, thickness, tier, nnU-Net detected/missed) are in
  `work/normative/summary.txt`.
- There is no stratum in which the target-size median reaches 90. The best 10-20 mm subsets are about
  65-73 (A): v3 hypo, venous, tail, missed ∩ target.

**Unscorable lesions: 126.**
- 94 have no lattice centre in lesion + 2 mm, because they are outside the search region; 65 of these
  are `separated`.
- 32 (30 of them ≥ 40 mm) have a far region smaller than their own footprint.

**The ≥ 40 mm percentile is also pulled down by the protocol itself.** The 16 lesions with > 1000
footprint centres have a median of **0**: a same-size random set covers nearly the whole remaining gland
and always contains some clutter peak. Excluding them would not pass the gate: the other buckets have
medians 69-77.

### Candidate lists (top-N per case, FP source (b) = positives' search region > 10 mm from any lesion)

Hit rate at top-10 (top-1 in parentheses), headline maps:

| | normative A | normative B | S4 LoG, same lesions (from S4's CSV) |
|---|---|---|---|
| TARGET, r 6 map (7.6 FP/case) | **16.0%** (2.8%) | 11.3% (2.8%) | 2.8% (0%) |
| 20-40 mm hypo/hyper enh., r 10 | **43.4%** (6.1%) | 25.4% (4.4%) | 10.8% (1.0%) |
| ≥40 mm hypo/hyper enh., r 10 | **74.3%** (27.1%) | 51.4% (20.0%) | 17.9% (2.1%) |

- Across all 1235 lesions: 36.4% (A, r 6) at top-10 and 69.5% at top-50 (38 FP/case).
- At ≤ 3 FP/case (top-3), TARGET hit rate is 10.4% (A, r 6). S4's was 0% *carried over*.
- FP per case on negatives (source a) is slightly *higher* than source (b) at a fixed top-N. That is
  expected: with no lesion present, every candidate is false.
- Global-threshold operating points and the full grid (score × radius × top-N / threshold × FP source ×
  stratum) are in `froc_points.csv`.

### Where top false candidates sit

Rank ≤ 5, nearest GT structure within 3 mm:
- Score A: veins 8.4%, duodenum 9.4%, none 81%.
- Score B: veins 22.8%, duodenum 10.7%.

Most of the normative score's false alarms have no vein, duodenum, SMA or aorta within 3 mm.
S4's top false peaks had veins (30.4%) or duodenum (24.9%) as their *nearest* structure (*carried over*).
S4's figure is a nearest-structure label within 30 mm, with a median of 3.5 mm away, so it is not the
same measure. Read the comparison qualitatively.

### Bank-size curve (nested seed-42 subsets; everything refit)

| bank cases | TARGET A | TARGET B | ≥40 A | ≥40 B |
|---|---|---|---|---|
| 100 | 74.2 | 55.5 | 58.0 | 52.5 |
| 200 | 71.0 | 46.2 | 69.8 | 54.0 |
| 300 | 71.8 | 48.0 | 71.2 | 53.0 |

The target median is flat from 100 to 300 bank cases; ≥ 40 mm improves from 100 to 200 and then
plateaus. **More normal data is not the bottleneck.**

### Per fold

Fold 0 was designated the dev fold. No design change was made after cohort results, so all folds are
evaluation. TARGET median A by fold: 68.0 / 73.0 / 76.0 / 62.0 / 57.0.

## Score C

C is a per-case z-transform of A or B. Leave-one-out makes it not strictly monotone, but with ~2000
patches per case it is monotone in practice.

So C cannot change:
- any within-scan percentile (max |C − A| over all 1235 lesions = 0.0)
- any top-N-per-case list (the FROC rows are identical)

It only moves operating points defined by a **global** threshold across cases. There:
- at ~5 FP/case, C_A r 6 gets 13% of the target group (z ≥ 3)
- at ~3.7 FP/case, raw A gets 6%

The "attention-lite" ablation therefore helps calibrate *across* scans, not rank *within* them.

## Caveats

- **GT anatomy everywhere** (envelope, search region, u, structure distances): upper bound.
- The 479 cases with an undetermined envelope were excluded from the pool, not checked. Patient overlap
  is unverifiable.
- One cohort positive (PanTS_00003188) has no phase and is scored against all phase strata.
- The search region is the envelope + 3 mm, hole-filled. Patches near its edge are mostly fat, which is a
  large part of why "normal" is so broad (see 3b).
- `slice_thickness` is voxel spacing, not reconstruction thickness (as in S1).
- Following the brief, nothing was scaled up and no attention model was built.

## Files

| file | rows | content |
|---|---|---|
| `lesion_ranks.csv` | 1235 | master-table columns + v3 / S2 / S4 joins + per-lesion percentiles. `pct_{A,B,CA,CB}` at the matched radius, `pct_*_r6` / `_r10` both radii, `fge_*` = share of random sets ≥ lesion (S4's "beaten by"), `cand_rank_*` = best in-lesion candidate rank, `n_centres_dil2`, `n_far_centres`, flags `target`, `target_20_40`, `target_ge40` |
| `froc_points.csv` | 12896 | score × radius × operating point × FP source × stratum, with Wilson CIs |
| `insertion_test.csv` | 5400 | case × diameter × dHU × method (A/B/C at both radii, S4 LoG): percentile, frac_ge, footprint size |
| `README.md` | | this file |

Reproduce from the repo root on the PC (CPU, 4 workers):

```
PYTHONPATH=src .venv/bin/python -m normative.pool checks && PYTHONPATH=src .venv/bin/python -m normative.pool sample
bash src/normative/extract_bank_cts.sh work/normative/pool/bank_ids.txt && bash src/normative/pull_bank_masks.sh
PYTHONPATH=src .venv/bin/python -m normative.case_pass cohort      # ~24 min
PYTHONPATH=src .venv/bin/python -m normative.case_pass bank        # ~6 min
PYTHONPATH=src .venv/bin/python -m normative.evaluate fit
PYTHONPATH=src .venv/bin/python -m normative.evaluate score
PYTHONPATH=src .venv/bin/python -m normative.evaluate calibrate
PYTHONPATH=src .venv/bin/python -m normative.case_pass insert      # ~15 min
PYTHONPATH=src .venv/bin/python -m normative.evaluate insertion
PYTHONPATH=src .venv/bin/python -m normative.evaluate report
```
