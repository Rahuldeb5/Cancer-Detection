# Scale-space LoG blob candidates (session S4)

**Question.** Does a scale-normalized Laplacian-of-Gaussian blob map, restricted to the pancreas,
separate small tumors from normal-pancreas clutter well enough to be a candidate generator?

**Answer: no, at every threshold and in every stratum.** For the realistic target group — 10-20 mm
lesions that are hypo- or hyper-attenuating on the v3 labels, in contrast-enhanced scans (n=106) —
the lesion's own strongest blob response is beaten by a median **76.5%** of same-size random patches
of the *same* pancreas. A budget of 10 candidates per case (8.3 false peaks per case) finds
**2.8%** of them (95% CI 1.0-8.0%); the whole 50-candidate list finds 14.2% (8.8-22.0%) at 41.6
false peaks per case. Iso lesions within ±5 HU behave no differently, as expected.

A second, methodological result is arguably more useful than the negative one: **the a-priori design
of pooling scales by noise-normalized score is strictly dominated.** Ranking by z = response/MAD
sends 80.6% of the candidate list to the finest scale (σ=1.5 mm), because on real scans MAD rises
with σ. A single fixed σ=5 mm produces only 6.5 candidates per case and still gets 20.8% of the
target group into the top 10 — about 7x the pooled list's 2.8% at an eighth of the candidate count.
Even that is too weak to be useful, but it means the headline number below measures a flawed
ranking as much as it measures the pancreas.

**Data.** All 1308 cohort cases, CTs and masks on the PC. Lesions keyed `(case_id, lesion_id)` from
`tumorlib.label_lesions`, joined onto `results/master_lesion_table/master_lesions.csv` (S1),
`results/attenuation_v3/attenuation_labels_v3.csv` (S3) and
`results/nnunet_subthreshold/per_lesion_subthreshold.csv` (S2).

Every number in this file was computed in S4 unless marked *carried over*.

```
# PC, repo root
.venv/bin/python src/log-candidates/phantom.py      # step 0, ~1 min, 4 known-answer checks
.venv/bin/python src/log-candidates/case_pass.py --workers 4   # 1308 cases, 33 min, ~1.5 GB RSS/worker
.venv/bin/python src/log-candidates/summarize.py    # asserts + the three CSVs
.venv/bin/python -m pytest tests/test_logblob.py    # 16 unit tests on the detector functions
```

## Files

| file | rows | content |
|---|---|---|
| `candidates.csv.gz` | 129034 | one row per candidate peak: the pooled scale-space list, top 50 per case per polarity. Score, scale, Hessian shape terms, distance to the nearest GT lesion, nearest anatomical structure |
| `separability_by_lesion.csv` | 1235 | one row per GT lesion: search-region coverage, in-lesion peak rank per scale and pooled, continuous max response, the random-patch comparison, and every stratifier |
| `log_froc_points.csv` | 6696 | 36 strata x 3 FP sources x [4 rankings x 12 z thresholds (5184 rows, `z_thresh` set) + 3 rankings x 7 budget sizes (1512 rows, `top_n` set)]: sensitivity with Wilson CI, false peaks per case and per 100 cm³, and the censored fraction |

Intermediates (per-case pickles, `summary.txt` with every number here, per-scale peak counts,
`fp_by_thickness.csv`, phantom tables) are in the git-ignored `work/log_candidates/`.

## Detector, fixed a priori

Nothing below was chosen by looking at an evaluation lesion.

- **Search region.** `tumorlib.search_region`: the pancreas ∪ head ∪ body ∪ tail envelope, closed
  8 mm, 3D hole-filled, dilated 3 mm. It never reads the lesion mask. It is empty for the same **11
  tumor-negative cases** S2 found (all four pancreas masks empty), so those are **excluded** and all
  region-based numbers use **1297 cases: 981 tumor+, 316 tumor-**.
- **CT.** Crop to the region's bbox + 30 mm, resample to 1 mm isotropic (`tumorlib.resample`), clip
  HU to [-100, 300]. **No z-score** — per-case intensity normalization would erase the HU contrast
  the detector exists to see. 30 mm exceeds the widest kernel radius (4σ = 24 mm); for 410 of 1297
  cases the CT volume ends before that, and those edges use `mode="nearest"`.
- **Response.** R_σ = σ²·∇²(G_σ * f) at σ ∈ {1.5, 2, 2.5, 3, 4, 5, 6} mm. Dark-blob polarity is +R,
  bright is −R, scored separately.
- **Score.** z = R_σ / (1.4826·MAD of R_σ over the search region), per case **and per scale**.
- **Peaks.** 26-neighbourhood local maxima inside the region, greedily thinned to a 5 mm minimum
  separation, top **K = 50** per scale; the seven scale lists are then pooled and thinned again to
  give the case's candidate list (top 50 per polarity). A fixed budget of 50 per polarity means a
  single-polarity detector, the realistic one, gets exactly the specified budget.
- **Shape.** Hessian eigenvalues of σ²·(G_σ * f) at the peak, sign-flipped by polarity so blob-like
  is always positive, ordered |λ1| ≤ |λ2| ≤ |λ3|. `blobness` = |λ1|/|λ3| is ~1 for a ball, ~0 for a
  tube. Also Frangi's Ra, Rb, the Frobenius norm and how many eigenvalues have the blob sign.
- **Hit rule.** A lesion is hit when a candidate peak's voxel lies inside that lesion dilated 2 mm
  (Euclidean, in mm) — the same rule as S2, so the two sessions' sensitivities are comparable.
- **False peak.** A peak more than **10 mm** from *every* GT lesion in its case, so response on a
  tumor's edge is never counted against the detector.

## Step 0: phantoms (all four passed before any real data)

| check | result |
|---|---|
| **A. scale law.** Uniform ball of radius r in correlated noise | The argmax over σ of the **raw** σ²·LoG at the ball centre is within one ladder step of r/√3 in **66/66** runs (noiseless error 0-1 steps). But ranking by **z** prefers one step *coarser*, because in pure noise MAD falls with σ. On real scans MAD *rises* with σ, so z there prefers *finer* — see the headline. |
| **B. anisotropy.** Ball built at (0.8, 0.8, 5) mm with PSF + partial volume, resampled to 1 mm, centre offset randomized over a full native voxel each trial | In-plane error ≤ 1 mm in **20/20** trials (max 0.97 mm). **z limitation, measured:** median 0.38 mm, max 1.04 mm; a 5 mm slice grid cannot do better than ~2.5 mm and resampling to 1 mm interpolates rather than adding z information. |
| **C. noise-only null**, 200 cm³ mask (the order of a real search region) | ~5-7 false peaks per volume at z ≥ 4, ~0 at z ≥ 5, 0 at z ≥ 6. The 50th peak sits at z ≈ 3.4, so **counts at z ≤ 3.4 are censored by K = 50, not rates.** This is the floor the real FROC is read against, and it fixes `NULL_Z_FLOOR = 5` as "a response exists". |
| **D. blob vs tube.** r = 2 mm dark ball vs r = 2 mm long dark tube, same contrast | `blobness` separates them completely: ball ≥ 0.503, tube ≤ 0.153. The midpoint **0.33** is frozen as the shape gate used below. |

The phantoms caught a real bug: the pooled thinning used `all()` over an empty kept-list, which is
`True`, so it rejected every candidate and the pooled list came back empty.

## 1. Search-region coverage — the ceiling for any restricted method

Fraction of each lesion's voxels inside the region. **Reported, not tuned**: no radius was changed
to raise it.

| tier | n | mean | median | entirely outside | ≥50% in | fully in |
|---|---|---|---|---|---|---|
| inside | 880 | 0.983 | 1.000 | 0.0% | 100.0% | 73.3% |
| embedded | 111 | 0.837 | 0.976 | 0.9% | 92.8% | 40.5% |
| abutting | 174 | 0.358 | 0.281 | 3.4% | 31.6% | 4.6% |
| separated | 70 | 0.015 | 0.000 | 94.3% | 1.4% | 1.4% |

| size | n | mean | entirely outside |
|---|---|---|---|
| <5 | 49 | 0.956 | 4.1% |
| 5-10 | 37 | 0.945 | 5.4% |
| **10-20** | **275** | **0.814** | **10.5%** |
| 20-40 | 597 | 0.855 | 5.0% |
| ≥40 | 277 | 0.741 | 3.6% |

Cohort mean 0.827; **73 of 1235 lesions are entirely outside** the region and cannot be found by any
detector built on it. Target group: mean 0.761, **15 of 106 entirely outside**. The `abutting` tier
is the real cost — a lesion mostly outside the envelope is invisible to a restricted search.

Regression: this coverage reproduces S2's independently computed `in_sr_frac` (different machine,
different mask root) to max |diff| 1.6e-3, mean 1.4e-6.

## 2. Where the lesion's own peak ranks (dark polarity, pooled list)

`hit` = an in-lesion peak anywhere in the case's 50 candidates. CNR is *carried over* from S3.

| stratum | n | hit | top1 | top3 | top10 | median rank | median CNR |
|---|---|---|---|---|---|---|---|
| **TARGET: 10-20 mm hypo/hyper, enhanced** | **106** | **14.2%** | **0.0%** | **1.9%** | **2.8%** | **36** | −2.33 |
| 10-20 mm, all | 275 | 10.9% | 0.0% | 1.1% | 2.5% | 27 | −0.53 |
| 10-20 mm, enhanced | 157 | 12.7% | 0.0% | 1.9% | 3.8% | 32 | −1.29 |
| 10-20 mm, iso (v3) | 99 | 10.1% | 0.0% | 1.0% | 4.0% | 16 | −0.03 |
| 10-20 mm, iso within ±5 HU | 66 | 13.6% | 0.0% | 0.0% | 4.5% | 19 | 0.00 |
| <10 mm | 86 | 2.3% | 1.2% | 1.2% | 1.2% | 18 | −0.58 |
| ≥20 mm | 874 | 43.0% | 2.7% | 6.9% | 14.6% | 16 | −0.53 |
| all lesions | 1235 | 33.0% | 2.0% | 5.2% | 11.0% | 16 | −0.54 |
| nnU-Net missed | 576 | 19.6% | 1.0% | 1.7% | 5.2% | 26 | −0.44 |
| nnU-Net missed ∩ TARGET | 74 | 12.2% | 0.0% | 1.4% | 2.7% | 34 | −2.17 |

Size dominates everything. Contrast class barely moves the result and **iso within ±5 HU is not
worse than hypo/hyper** — which says the detector is not limited by contrast in this range; it is
limited by clutter. Full stratification by phase, region, thickness and tier is in `summary.txt`
and `separability_by_lesion.csv`.

### 2b. The pooled ranking is dominated by any single coarse scale

`cand/case` is how many peaks that ranking actually produces — the cap is 50, but a coarse response
is smooth and yields fewer local maxima, so a coarse scale is also *cheaper*.

TARGET group (n = 106):

| ranking | cand/case | hit | top1 | top3 | top10 | median rank |
|---|---|---|---|---|---|---|
| σ = 1.5 mm | 49.7 | 13.2% | 0.0% | 1.9% | 2.8% | 36 |
| σ = 2 mm | 49.5 | 21.7% | 0.0% | 0.9% | 1.9% | 30 |
| σ = 2.5 mm | 48.2 | 34.9% | 0.0% | 0.0% | 3.8% | 27 |
| σ = 3 mm | 41.4 | **46.2%** | 0.0% | 0.9% | 9.4% | 20 |
| σ = 4 mm | 16.4 | 34.0% | 0.0% | 2.8% | **22.6%** | 8 |
| σ = 5 mm | 6.5 | 20.8% | 0.9% | 8.5% | 20.8% | 4 |
| σ = 6 mm | 2.7 | 9.4% | 3.8% | 7.5% | 9.4% | 2 |
| **POOLED (z)** | **50.0** | **14.2%** | **0.0%** | **1.9%** | **2.8%** | **36** |

Mechanism: the response MAD, which is the z denominator, **rises** with σ on real scans — median
16.9, 18.2, 19.3, 20.2, 21.3, 22.3, 22.8 HU across the ladder (sigma^2 in mm^2 times a Laplacian in HU/mm^2 is HU) — because inside a real pancreas
it measures vessels, ducts and bowel wall, not photon noise. So z penalizes exactly the scales that
match lesion size, and **80.6% of the pooled list, and 82.3% of its rank-1 peaks, are σ = 1.5 mm.**

Two cautions. First, *picking* σ = 4 or 5 from this table would be tuning on evaluation lesions
(CONTEXT rule 3); the whole ladder is shown so that nothing is selected, and a corrected ranking
would need out-of-fold selection to claim a number. Second, even the best row is 22.6% top-10, and
its median rank of 8 out of 16 candidates means **~7 non-lesion peaks in the same pancreas still
outscore the lesion**. The conclusion does not change; only the size of the gap does.

## 3. False peaks, and the slice-thickness problem

Negatives are thinner-sliced than positives (*carried over*, S1: median 1.5 vs 2.5 mm), and blob
false-alarm rates depend on thickness, so false peaks are counted from **two** sources and the
negatives are never headlined alone:

- **(a) negatives** — the 316 with a search region; every peak is false.
- **(b) non-tumor parts of positive scans** — peaks >10 mm from every GT lesion, over the 981
  positives. Denominator is the tumor-excluded region volume (median 94.8 cm³ vs 124.2 cm³ for the
  full region), which is why the per-100 cm³ column matters.

Slice-thickness mix: positives {≤1.5: 39.7%, 1.5-2.5: 39.3%, 2.5-5: 20.7%, >5: 0.3%}; negatives
{63.3%, 15.2%, 20.9%, 0.6%}. Reweighting (a) to the positives' mix covers 100% of that mass.

**Headline FROC — slice-matched, source (b), dark polarity:**

| z | FP/case | FP/100 cm³ | censored | TARGET | 10-20 all | 10-20 iso | <10 | ≥20 | all |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 41.34 | 40.83 | 5.1% | 14.2% | 10.5% | 9.1% | 2.3% | 42.9% | 32.9% |
| 3 | 34.44 | 34.01 | 4.3% | 10.4% | 7.6% | 9.1% | 2.3% | 36.3% | 27.5% |
| 4 | 18.05 | 17.83 | 1.6% | 6.6% | 4.7% | 6.1% | 0.0% | 20.0% | 15.2% |
| 5 | 7.66 | 7.57 | 0.5% | 2.8% | 1.5% | 1.0% | 0.0% | 9.3% | 6.9% |
| 6 | 2.94 | 2.90 | 0.0% | **0.0%** | 0.4% | 1.0% | 0.0% | 4.2% | 3.1% |
| 8 | 0.49 | 0.48 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 1.0% | 0.7% |
| ≥10 | ≤0.13 | ≤0.13 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | ≤0.2% | ≤0.2% |

There is no operating point with useful sensitivity at a tolerable false-peak rate. By the time the
false-peak rate is clinically plausible (≤3 per case, z ≥ 6) the target-group sensitivity is **zero**.

**Fixed-budget FROC** (top N candidates per case — no censoring, and the natural curve for a
candidate generator), source (b), dark:

| top N | FP/case | FP/100 cm³ | TARGET (95% CI) | 10-20 all | <10 | ≥20 | all |
|---|---|---|---|---|---|---|---|
| 1 | 0.80 | 0.79 | 0.0% (0.0-3.5) | 0.0% | 1.2% | 2.7% | 2.0% |
| 3 | 2.43 | 2.40 | 1.9% (0.5-6.6) | 1.1% | 1.2% | 6.9% | 5.2% |
| 10 | 8.27 | 8.16 | 2.8% (1.0-8.0) | 2.5% | 1.2% | 14.6% | 11.0% |
| 20 | 16.64 | 16.44 | 2.8% (1.0-8.0) | 4.0% | 1.2% | 24.5% | 18.3% |
| 50 | 41.61 | 41.10 | 14.2% (8.8-22.0) | 10.9% | 2.3% | 43.0% | 33.0% |

**(a) vs (b) disagree, and (b) — the slice-matched estimate — is the larger one.** Per 100 cm³,
thickness-bin mean: z ≥ 4, 14.26 (a) vs 19.89 (b); z ≥ 6, 2.43 vs 3.63; z ≥ 8, 0.44 vs 0.51. So the
negatives **understate** the false-peak rate by roughly a third in the z range that matters.
Reweighting (a) to the positives' thickness mix moves it only a little (z ≥ 6: 3.60 → 3.82 per
case), so most of the (a)/(b) gap is not thickness — it is that a pancreas containing a tumour is a
more cluttered organ than a normal one.

False peaks per case by thickness, source (b): z ≥ 6 gives 2.90 (≤1.5 mm), 2.15 (1.5-2.5), 4.44
(2.5-5), 6.00 (>5, n=3). **Thicker slices produce more false peaks, not fewer**, so a detector
tuned on thin scans degrades on the thick ones that dominate this cohort.

## 4. Against random same-size patches in the same pancreas

Comparing a small lesion against a whole-region maximum is a stacked comparison, so each lesion's
max z inside its 2 mm dilation is also compared against **200 random patches** of the same
equivalent radius, placed elsewhere in the same case's region (≥10 mm from every lesion).
`patch_frac_ge` is the fraction of those patches that match or beat the lesion; 0.5 means the lesion
is indistinguishable from a random piece of its own pancreas.

| stratum | n | median | ≤0.05 | ≤0.10 | ≤0.25 | best in case |
|---|---|---|---|---|---|---|
| **TARGET** | **93** | **0.765** | **3.2%** | **3.2%** | **6.5%** | **1.1%** |
| 10-20 mm all | 252 | 0.775 | 2.0% | 2.8% | 8.3% | 0.4% |
| 10-20 mm iso ±5 HU | 64 | 0.812 | 1.6% | 3.1% | 10.9% | 0.0% |
| <10 mm | 76 | 0.645 | 1.3% | 2.6% | 7.9% | 1.3% |
| ≥20 mm | 840 | 0.820 | 7.3% | 10.1% | 16.8% | 2.6% |
| all | 1168 | 0.785 | 5.7% | 8.0% | 14.4% | 2.1% |

The median target lesion is beaten by **76.5%** of same-size patches of its own pancreas. Only 1.1%
are the strongest patch in their own case. The comparison is available for 1168 of 1235 lesions; the
other 67 have no region voxel inside their 2 mm dilation.

Caveat: both sides are read off the max-over-scales z map, so this inherits the anti-coarse-scale
bias section 2b exposes. The per-scale clutter measure that does not is `median rank` in 2b, and it
agrees (rank 8 of 16 at σ = 4 mm).

## 5. Verdict per stratum

Two independent axes: `resp` = fraction of the stratum reaching z ≥ 5 anywhere in the lesion + 2 mm
(the phantom null's ceiling for a pancreas-sized region), and median `patch_frac_ge`.

| stratum | n | resp ≥5 | median frac | verdict |
|---|---|---|---|---|
| TARGET 10-20 mm hypo/hyper enhanced | 93 | 5.4% | 0.765 | **featureless** |
| 10-20 mm all | 252 | 3.6% | 0.775 | **featureless** |
| 10-20 mm iso (v3) | 95 | 2.1% | 0.835 | **featureless** |
| 10-20 mm iso within ±5 HU | 64 | 3.1% | 0.812 | **featureless** |
| <10 mm | 76 | 0.0% | 0.645 | **featureless** |
| 20-40 mm | 570 | 9.5% | 0.780 | **featureless** |
| ≥40 mm | 270 | 27.4% | 0.865 | **clutter-dominated** |
| nnU-Net missed ∩ TARGET | 65 | 4.6% | 0.765 | **featureless** |

Only lesions ≥40 mm produce a response above the pure-noise floor often enough to call the failure
"clutter"; everything at or below 20 mm is **featureless** on this map — there is no response to
rank, let alone to separate. No stratum reaches `signal-exists`. Iso within ±5 HU is undetectable as
expected, and is reported rather than chased.

## 6. What the false peaks actually are

Nearest structure to the top false peaks (dark, z ≥ 6, ranks 1-5; n = 1622), median distance 3.5 mm:

veins **30.4%**, duodenum **24.9%**, stomach 17.0%, celiac artery 8.2%, superior mesenteric artery
6.2%, aorta 3.5%, IVC 3.3%, common bile duct 2.8%, pancreatic duct 2.1%, gall bladder 0.7%, none 1.0%.

So the clutter is peripancreatic **vessels and bowel**, as expected — which is what the Hessian shape
term was supposed to suppress.

**It does not work.** Gating on the phantom-frozen cut `blobness ≥ 0.33` cuts false peaks ~11x
(41.34 → 3.85 per case at z ≥ 2) but cuts target sensitivity nearly in half (14.2% → 7.5%), and the
reason is decisive: **in-lesion peaks are *less* blob-like than the false ones** — median blobness
0.086 for in-lesion dark peaks with z ≥ 6 versus 0.123 for the top false peaks. The gate keeps only
6.9% of true in-lesion peaks. Real pancreatic lesions at these sizes are not isotropic blobs in the
LoG sense; they are irregular, partly volume-averaged across thick slices, and often contiguous with
the vessels and duct they distort.

## Caveats

1. **Ground truth anatomy.** The search region is built from GT pancreas masks, so every number is
   an upper bound relative to a pipeline that must segment the pancreas first.
2. **The pooled ranking is flawed** (section 2b). The headline FROC measures the a-priori design; a
   per-scale ranking is better and still insufficient. Any corrected ranking must be selected
   out-of-fold before a number is claimed.
3. **The patch comparison inherits that bias** — see section 4's caveat.
4. **Censoring.** The list is capped at 50 per case per polarity, so false-peak counts at low z are
   floors. The `censored` column gives the affected fraction; at z ≥ 5 it is under 4%.
5. **z localization.** On 5 mm slices the peak's z position carries an error up to ~1 mm (phantom B)
   and the grid itself cannot do better than ~2.5 mm. The 2 mm dilation hit rule absorbs this.
6. **27 (case, polarity) lists hold fewer than 50 peaks**, from tiny search regions (smallest 0.056
   cm³). Coarse scales are legitimately below the cap everywhere — see `cand/case` in 2b.
7. **Structure labels** come from PanTS organ masks, which are themselves model output, not
   radiologist annotation. `celiac_artery` is missing for a small number of cases.
8. **`hu` in `candidates.csv.gz` is post-clip**, so −100 means "≤ −100 HU" (gas or fat).
9. Lesion `tier`, `region`, `attenuation_v3`, `cnr`, `detected_any` and slice thickness are all
   *carried over* from S1/S2/S3, not recomputed here.

## What surprised me

1. **The phantom's noise model had the wrong sign.** In pure noise MAD falls with σ; inside a real
   pancreas it rises. The z-normalization that the phantom validated is therefore actively harmful on
   real data, and it sent 80.6% of the candidate list to the finest scale. A phantom can validate a
   formula and still mislead about which regime you are in.
2. **The shape term points the wrong way.** True lesion peaks are *less* blob-like than vessel and
   bowel peaks. The single most obvious way to suppress clutter suppresses tumors harder.
3. **The negatives understate false peaks by ~1/3**, and reweighting for slice thickness barely
   changes it. The confound is real but it is mostly *not* thickness — a pancreas with a tumor is
   simply more cluttered. This is a reason to prefer source (b) generally, beyond this session.
4. **Thicker slices give more false peaks, not fewer.**
5. **Contrast class hardly matters below 20 mm.** Iso within ±5 HU is no worse than hypo/hyper with
   median CNR −2.33. At these sizes the limit is not contrast; there is simply no blob-shaped
   response above the noise floor.
6. **73 lesions are entirely outside the search region**, concentrated in the `abutting` tier —
   a 5.9% ceiling loss before any detector runs.
