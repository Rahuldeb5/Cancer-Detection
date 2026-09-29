# nnU-Net sub-threshold probe (session S2)

**Question.** Where the vanilla nnU-Net baseline misses a tumor, is its tumor probability raised inside the lesion
without crossing 0.5 (a "whisper"), or is it flat at background?

**Data.** Out-of-fold validation predictions only (`fold_{0..4}/validation/{case}.npz`, `--npz` softmax) for
all 1308 cohort cases. GT comes from nnU-Net `gt_segmentations`. The pancreas masks come from the server's
`~/research/datasets/pants/duct_masks/`. Lesions are keyed `(case_id, lesion_id)` from `tumorlib.label_lesions`
and joined onto `results/master_lesion_table/master_lesions.csv`.

Every number below was computed in S2 unless it is marked *carried over*.

```
# server, repo root (~/nnunet_setup/.venv/bin/python)
src/nnunet-subthreshold/oracle.py        # step 0: 3-case oracle
src/nnunet-subthreshold/case_pass.py     # phantom oracle, then 1308 cases (4 workers, one case per worker, ~32 min, peak RSS 3.0 GB/worker)
src/nnunet-subthreshold/summarize.py     # asserts + the two CSVs; intermediates in work/nnunet_subthreshold/
```

## Files

| file | rows | content |
|---|---|---|
| `per_lesion_subthreshold.csv` | 1235 | one row per GT lesion: peak tumor probability in the lesion and in the lesion dilated 2 mm, compared against the same case's non-lesion search region and against the 316 negatives that have a search region |
| `nnunet_froc_points.csv` | 216 | 9 thresholds × 2 hit rules × 2 lesion sets × 6 size bins: lesion sensitivity, FP components per case |

The following live outside `results/`:
- **PC export** (`~/research/nnunet_probs/`, 5.6 GB):
  - `probs/{case}.npz` for the 1297 cases that have a search region. Arrays: `prob` float16, `logit` float16 (clipped logit), `crop_start`, `crop_stop`, `full_shape`, `affine`, `spacing`, `fold`, `case_id`. The crop is the search-region bbox + 20 mm, in nibabel (x, y, z) voxel indices on the CT grid.
  - `components/components_t{t}.csv`, one table per threshold.
- **Work dir** (`work/nnunet_subthreshold/`): `summary.txt` (full log of every number here), `saturation_hist.png`, `case_pass.log`.

## Definitions

- **p** = channel 1 of the saved softmax: `npz['probabilities'][1].transpose(2,1,0)`.
- **Logit scale.** `logit = log(p/(1-p))` after clipping p to [1e-7, 1-1e-7], so its range is ±16.12. `peak_raw_logit_*` = `log p1 - log p0` without clipping.
- **Search region (SR).** `tumorlib.search_region`: the pancreas | head | body | tail envelope, closed 8 mm, hole-filled, then dilated 3 mm. It never reads the lesion mask. It is empty for 11 cases, all tumor-negative, whose four pancreas masks are empty on both the server and the PC. So the SR-based negative comparisons use **316 negatives**.
- **Lesion + 2 mm** (`dil2`): voxels within 2.0 mm (Euclidean, in mm) of the lesion.
- **Components.** 26-connected components of `p > t`, over the **whole volume** (not restricted to the SR).
- **Two hit rules:**
  - `peak_in_dil2mm` (strict): the component's max-p voxel lies inside the lesion + 2 mm.
  - `touch` (lenient): the component overlaps at least one lesion voxel.

  A component is an FP when it hits no lesion under that rule. A component that hits an excluded (separated) lesion is not an FP.
- **Missed** = the saved segmentation does not touch the lesion. This equals `detected_any == False` in the master table for all 1235 lesions.

## Step 0: oracle (3 cases + phantom) and the same checks cohort-wide

**Oracle cases:**

| case | fold | description |
|---|---|---|
| PanTS_00000771 | 0 | 40.7 mm, Dice 0.921 |
| PanTS_00003513 | 0 | missed 9.4 mm index lesion, 1.0 mm slices |
| PanTS_00000020 | 0 | negative |

**Results on the three cases:**
- **Layout.** The npz is float32 `(2, z, y, x)`. Channel 1 `.transpose(2,1,0)` has the `.nii` shape. The seg, GT and pancreas-mask affines agree.
- **Orientation.** For case 771, mean p inside GT is 0.876 with the transpose. Flipping x, y or z instead gives 0.121, 7e-7 and 2e-6.
- **Dice.** Case 771's Dice against the union of touching predicted components is 0.9211, matching the master table.
- **Negative case.** PanTS_00000020 has max p 0.9994 (*carried over* value 0.9994).

**Cohort-wide (all 1308 cases):**
- argmax ≠ saved segmentation: **0 voxels**. `p > 0.5` ≠ saved segmentation: **0 voxels**. Max |p0 + p1 − 1| = 8.8e-8.
- Spacing read from the `gt_segmentations` header (no CTs on the server) differs from S1's CT-affine spacing by at most 5.4e-8 mm.

**Phantom** (`case_pass.py --phantom`, 0.8 × 0.8 × 2.5 mm grid). It covers:
- a blob 1.6 mm outside a lesion: a peak hit that is not a touch
- a shell that touches a lesion but peaks 5.8 mm outside it: a touch that is a peak-rule FP
- sub-threshold lesions that appear only at t ≤ 0.003
- clip-floor behaviour

It passes. Changing the dilation to 1 mm, or the clip to 1e-12, makes it fail.

## Step 1: regression (saved segmentation, "a predicted component touches GT")

| quantity | S2 | carried over |
|---|---|---|
| tumor+ cases localized / only wrong place / nothing | **631 / 288 / 62** of 981 | 631 / 288 / 62 |
| localized, largest lesion <10 mm | 0/11 = 0.0% | 0% |
| 10-20 mm | 56/176 = 31.8% | 31% |
| 20-40 mm | 344/524 = 65.6% | 66% |
| ≥40 mm | 231/270 = 85.6% | 86% |
| lesion detect@any by bin <5 / 5-10 / 10-20 / 20-40 / ≥40 | 2.0 / 0.0 / 25.8 / 59.8 / 83.0% | 2 / 0 / 26 / 60 / 83% |

The touch hit at t = 0.5 computed from the probabilities equals the saved-segmentation touch for all 1235 lesions.

## Step 2: peak probability in missed lesions (lesion + 2 mm)

Negative-case SR peak p (n = 316): median **0.317**, p10 6.5e-5, p90 0.998. The whole-volume peak median is 0.993 (n = 327).

| missed, lesion set = all | n | median peak p [IQR] | median logit | ≥0.1 | ≥0.01 | <1e-4 | median frac of negative SR peaks ≥ it (min) | above own-case non-lesion SR peak | ≤ own-case SR p99 | median own-case SR voxel frac ≥ it |
|---|---|---|---|---|---|---|---|---|---|---|
| <10 mm | 85 | 1.3e-4 [4.0e-5 – 4.1e-4] | −8.96 | 5 | 7 | 33 | 0.82 (0.484) | 0 | 76 | 0.103 |
| <10 mm, index lesion | 11 | — | — | 1 | 2 | 7 | 0.91 (0.513) | 0 | — | 0.056 |
| 10-20 mm | 204 | 2.8e-4 [7.3e-5 – 3.8e-3] | −8.18 | 18 | 43 | 68 | 0.73 (0.402) | 13 | 155 | 0.041 |
| 10-20 mm, index lesion | 121 | — | — | 11 | 28 | 43 | 0.76 (0.411) | — | — | 0.031 |
| 20-40 mm | 240 | 8.5e-4 [1.5e-4 – 1.5e-2] | −7.07 | — | — | — | 0.66 | 42 | 137 | 0.016 |
| ≥40 mm | 47 | 1.8e-3 [2.9e-4 – 3.9e-2] | −6.30 | — | — | — | 0.64 | 3 | 26 | 0.014 |

Rows without the 70 separated lesions are in `work/nnunet_subthreshold/summary.txt`. The <10 mm row becomes n = 80, median 1.3e-4; the 10-20 mm row becomes n = 184, median 3.0e-4.

For the 11 missed index lesions under 10 mm, the peak p in lesion + 2 mm is:
- ≈0.18 in PanTS_00003513
- 0.026 in PanTS_00004856
- ≤3e-4 in the other nine

**Clipping.** No lesion peak sits at either clip edge. The minimum unclipped peak logit is −13.7. Where 1e-7 < p < 1-1e-7, the clipped and unclipped logits differ by at most 3.3e-3.

### The two readings, stated as numbers

**(a) Elevated sub-threshold probability that ranks above negative-case peaks.**
- 0 of 85 missed <10 mm lesions exceed every negative's SR peak, and 1 of 85 exceeds the median negative SR peak.
- For the median missed <10 mm lesion, 82% of negatives have an SR peak at or above it.
- The lesion with the highest rank against negatives still has 48.4% of negative SR peaks at or above it.
- In the 10-20 mm bin: 0/204 exceed every negative SR peak, 12/204 exceed the median, and the median fraction is 0.73.

**(b) Flat at background.**
- 1 of 85 missed <10 mm peaks is ≤ its own case's median non-lesion SR p.
- 76 of 85 are ≤ its own case's non-lesion SR 99th percentile.
- The median missed <10 mm peak (1.3e-4) has a median of 10.3% of its own case's non-lesion SR voxels at or above it, and 0/85 exceed their own case's non-lesion SR peak.

Both comparisons have a region-size asymmetry: a lesion + 2 mm region is compared with the max over a whole SR, which is a much larger region.

## Step 3: nnU-Net-only FROC points (lesion set = all, whole-volume components)

Sensitivity by size bin. FP/case columns: `all` over 1308 cases, `neg` over 327 negatives.

| t | peak rule: all | <5 | 5-10 | 10-20 | 20-40 | ≥40 | FP/all | FP/neg | touch rule: all | <5 | 5-10 | 10-20 | 20-40 | ≥40 | FP/all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.9 | 41.4% | 0.0 | 0.0 | 12.0 | 45.9 | 73.6 | 1.35 | 1.52 | 48.3% | 2.0 | 0.0 | 17.8 | 54.8 | 79.1 | 1.29 |
| 0.7 | 43.3% | 0.0 | 0.0 | 14.2 | 47.9 | 75.8 | 1.63 | 1.89 | 51.6% | 2.0 | 0.0 | 23.3 | 58.0 | 81.6 | 1.56 |
| 0.5 | 44.5% | 0.0 | 0.0 | 15.6 | 48.9 | 77.3 | 1.80 | 2.06 | 53.4% | 2.0 | 0.0 | 25.8 | 59.8 | 83.0 | 1.72 |
| 0.3 | 45.7% | 0.0 | 2.7 | 16.7 | 50.1 | 79.1 | 2.00 | 2.26 | 55.1% | 2.0 | 2.7 | 27.6 | 61.6 | 84.8 | 1.91 |
| 0.1 | 47.0% | 0.0 | 2.7 | 17.8 | 52.1 | 79.1 | 2.29 | 2.55 | 57.4% | 4.1 | 5.4 | 30.9 | 64.3 | 85.2 | 2.19 |
| 0.03 | 48.3% | 0.0 | 2.7 | 18.9 | 53.8 | 80.1 | 2.66 | 2.92 | 59.6% | 4.1 | 8.1 | 33.5 | 66.7 | 87.0 | 2.55 |
| 0.01 | 49.8% | 0.0 | 5.4 | 21.1 | 55.4 | 80.9 | 3.05 | 3.34 | 62.8% | 4.1 | 13.5 | 37.8 | 70.0 | 89.2 | 2.94 |
| 0.003 | 51.2% | 0.0 | 5.4 | 23.6 | 57.1 | 80.9 | 3.70 | 4.08 | 66.2% | 8.2 | 16.2 | 43.6 | 73.2 | 90.3 | 3.58 |
| 0.001 | 52.6% | 0.0 | 2.7 | 25.5 | 58.8 | 82.3 | 4.82 | 5.09 | 70.3% | 12.2 | 18.9 | 50.2 | 77.1 | 92.8 | 4.68 |

Bin sizes are n = 1235 / 49 / 37 / 275 / 597 / 277. Touch-rule FP/neg equals peak-rule FP/neg, since negatives have no lesions.

Of the 85 missed <10 mm lesions, 12 are touched at some t ≤ 0.3. Only 2 are ever hit under the peak rule.

Peak-rule sensitivity is not monotone in t: for 5-10 mm it is 5.4% at 0.01 and 2.7% at 0.001. When components merge, the merged component's peak can move outside the lesion's 2 mm ring.

## Step 4: saturation (SR voxels; plot in `work/nnunet_subthreshold/saturation_hist.png`)

| voxels in the SR | n | p in [0.01, 0.99] | p ≥ 0.99 | p < 0.01 |
|---|---|---|---|---|
| lesion (tumor+) | 7,081,853 | 34.2% | 29.5% | 36.3% |
| non-lesion (all cases) | 160,301,212 | 2.67% | 0.46% | 96.87% |
| all SR voxels: tumor+ cases / tumor− cases | 122.5 M / 44.9 M | 5.13% / 0.94% | | |

**Components with peak p in [0.01, 0.99]** (whole volume): 31.1% at t = 0.9, 46.1% at 0.5, 66.3% at 0.01, 40.1% at 0.001. Among components whose peak is inside the SR: 19.0 / 29.2 / 44.8 / 30.7%.

**What the float ceiling hides: nothing at the top in this data.**
- The npz is float32, which can hold p up to 1 − 6e-8 (logit ≈ 16.6).
- The highest p1 anywhere in 1308 volumes is **0.9999875** (logit 11.29). No voxel has p1 = 1.0 or p0 = 0.
- So the upper clip at 1 − 1e-7 (logit 16.12) never binds.

**The clip floor does bind:**
- 1.23% of SR voxels have p ≤ 1e-7 and collapse onto logit −16.12. Their order among themselves is lost; no lesion peak is among them.

**Soft ceiling near logit 6.** Both logit histograms have a step near logit 6 (p ≈ 0.9975):

| lesion voxels, logit bin | [5.75, 6) | [6, 6.25) |
|---|---|---|
| count | 383,272 | 89,313 |

Only 2.7% of lesion voxels and 0.024% of non-lesion SR voxels have logit ≥ 6.25. This is a property of the saved softmax, not of the float format.

**float16 export.** p ≥ 0.99976 rounds to 1.0 (18,640 voxels) and p below about 3e-8 rounds to 0 (108 M voxels). The `logit` array in each file keeps both ends; use it for ranking.

## Caveats

- **Search region.** The SR and GT pancreas masks are ground truth (PanTS auto-segmentation), not predicted anatomy.
- **Excluded lesions.** 70 lesions (tier `separated`) lie outside the envelope. 55 of the 576 missed lesions have no voxel in the SR; 49 of those 55 are separated lesions. `lesion_set = excl_separated` rows are in the FROC CSV.
- **Anisotropic slices.** 2 mm dilation on 5 mm slices adds almost nothing along the slice axis. Hit rates on thick-slice cases are therefore closer to the lesion-only peak.
- **Negative comparisons.** Negatives are thinner-sliced than positives (median 1.5 vs 2.5 mm, S1).
- **"FP" definition.** An FP is a component not hitting any GT lesion. 30 negatives have report-described pancreatic lesions (S1), so some "FP" components in negatives may not be false.
- **Spacing source.** The server has no CTs, so tumorlib's grid was read from `gt_segmentations` headers through a symlink farm (`TUMORLIB_CT_ROOT=work/nnunet_subthreshold/ctgrid`). Its spacing equals the S1 CT-affine spacing within 5.4e-8 mm in every case.

## What surprised me

1. The "saturated" probabilities are not at the float limit. The top is a soft cap near p ≈ 0.9975, and the maximum anywhere is 0.9999875.
2. 11 negatives have entirely empty pancreas masks (all four), not only the 3 positives mentioned earlier.
3. The strict peak rule loses lesions as t drops, because merged components peak elsewhere. Touch-rule and peak-rule sensitivities diverge by 7 to 18 points.
