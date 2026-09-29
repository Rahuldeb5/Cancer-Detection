# Master lesion table (session S1)

`master_lesions.csv` has **one row per GT lesion component (1235 rows, 981 tumor+ cases)** and the key
**(case_id, lesion_id)**. It is the shared join table: later sessions join onto it and do not re-derive
these columns. Every number below was computed in S1 by `src/master-lesion-table/` unless it is marked
*carried over*.

```
.venv/bin/python3 src/master-lesion-table/case_pass.py     # phantom oracle, then 1308-case mask pass (~2 min, 4 workers)
.venv/bin/python3 src/master-lesion-table/build_table.py   # joins + oracles + audits -> this CSV
```
Intermediates (per-case spacing, lesion recount, audit tables, full build log) go to `work/master_lesion_table/`,
which is git-ignored. The `work/` line was added to `.gitignore` this session.

**Lesion identity.** A lesion is a 26-connected component of `pancreatic_lesion.nii.gz`, labeled in nibabel
(x,y,z) order. That is the same numbering as every upstream CSV. Tiny lesions are **flagged, not dropped**.

## Column dictionary

| column | unit / values | meaning and source |
|---|---|---|
| `case_id`, `lesion_id` | | key. `lesion_id` is the `scipy.ndimage.label` index (26-conn, nibabel order) |
| `fold` | 0-4 | nnU-Net fold. `fold_k_ids.txt` gives fold k-1 |
| `official_split` | train / test | PanTS's own ID ranges (1-9000 train, 9001-9901 test). Informational only: evaluation is our 5-fold CV |
| `ct_phase` | Venous / Non-contrast / Arterial / Delay | metadata `ct phase`. NaN ×1: PanTS_00003188 has no phase in the metadata |
| `spacing_x_mm`, `spacing_y_mm`, `spacing_z_mm` | mm | `nib.affines.voxel_sizes(CT affine)` in **voxel-axis order** (nibabel x,y,z array axes, not world axes). Equal to header zooms in 1308/1308 cases |
| `slice_axis` | 0/1/2 | voxel axis most aligned with world superior-inferior. It is 0 in 39 cases, where `spacing_z_mm` is in-plane |
| `slice_thickness_mm` | mm | spacing along `slice_axis` ("z thickness"). This is slice *spacing*: NIfTI does not store the reconstruction thickness |
| `gt_vox` | voxels | component voxel count, recounted from the mask and equal in every source (see oracles) |
| `vol_mm3` | mm³ | `gt_vox` × voxel volume from the spacing above |
| `diam_mm` | mm | Feret diameter: max distance between voxel centres (same function as upstream; matches `lesion_dice_attenuation.csv` to 1e-6 mm) |
| `diam_bin` | `<5`, `5-10`, `10-20`, `20-40`, `>=40` | project bins, left-closed |
| `tiny_flag` | bool | `vol_mm3 < 8` |
| `n_lesions_in_case` | int | number of components in the case |
| `largest_in_case` | bool | the case's component with the most voxels. Exactly one per case: the only tie, PanTS_00009056 (732 = 732 voxels), goes to the larger Feret diameter (lesion 2) |
| `dHU` | HU | v2 median tumor HU minus median parenchyma HU; the parenchyma pool excludes lesions + a 2.5 mm shell (`attenuation_labels.csv delta_hu`). NaN ×3 |
| `attenuation_v2` | hypo/iso/hyperattenuating/unknown | v2 class, ±10 HU (`attenuation_labels.csv`). Under audit in S3 |
| `lesion_dice` | 0-1 | nnU-Net OOF Dice of this lesion against the union of predicted components touching it (`lesion_dice_attenuation.csv`) |
| `detected_any` | bool | any predicted voxel overlaps the lesion (= `lesion_dice > 0`). Equals `per_lesion_metrics.detected_any` for every case |
| `region` | head / body / tail / NaN | section holding most voxels of (lesion + 3-voxel ring) over the head/body/tail masks. If the ring touches none of them, falls back to `u` against the pooled median head/body and body/tail boundaries. This reproduces `lesion_duct_position.csv section` exactly on its 1142 rows and extends it to all 1235 |
| `region_source` | mask_ring / pc1_boundary / none | 1134 / 93 / 8. 55 of the 93 fallbacks and 7 of the 8 `none` rows are excluded lesions: treat `region` of excluded lesions as nominal |
| `u` | 0 = head end … 1 = tail end | PC1 position of the lesion centroid (`lesion_pca_position.csv pc1_norm`). Identical to `u_head_to_tail` / `pc1_norm` in the published CSVs (up to 3-dp rounding). NaN ×30: cases without a head or tail mask |
| `tier` | inside / embedded / abutting / separated | contact with the pancreas envelope (`src/lesion-sectioning/work/lesion_inclusion.csv`): 880 / 111 / 174 / 70 |
| `excluded` | bool | `tier == separated`. Equal to the 70 rows of `excluded_lesions.csv` |
| `mpd_present` | bool | case has any `pancreatic_duct` voxels (`attenuation_labels.csv`) |
| `dist_to_mpd_mm` | mm | min distance from lesion to the MPD mask. NaN exactly when `mpd_present` is False (232) |

## Oracles (all pass)
- 1308 cases, each in exactly one `fold_k_ids.txt`. The fold mapping agrees with `lesion_dice_attenuation`,
  `lesion_inclusion`, `excluded_lesions` and `per_case_metrics`.
- 1235 rows, (case_id, lesion_id) unique.
- `gt_vox` from the fresh mask recount equals, with 0 mismatches:
  - `attenuation_labels.n_vox` (1235 rows)
  - `lesion_dice_attenuation.gt_vox` (1235 rows; this comes from the server's nnU-Net `gt_segmentations`, so the
    preprocessed GT equals the raw masks)
  - `lesion_inclusion.n_vox` (1235 rows)
  - `excluded_lesions.n_vox` (70 rows)
  - work `lesion_duct_position.n_vox` (1142 rows)
  - `per_lesion_metrics.gt_vox` as a per-case multiset. That file has no lesion_id, and (case, gt_vox) is not
    unique there (6 duplicate keys in 4 cases), so it cannot be joined row-wise.
- `tumor?` == (lesion mask non-empty) for 1308/1308.
- The phantom test (`case_pass.py --phantom`) checks voxel counts, Feret diameter, ring-vote region (including a
  lesion carved out of its section mask) and slice-axis detection on axis-permuted and oblique affines.
- Lesion affine ≠ CT affine in 6 cases. These are exactly CONTEXT's 6 tumor+ corrupted-affine cases; the CT grid is used.

## Audits

### a. Cohort counts: which file gives which
| source | cases | tumor+ | tumor− |
|---|---|---|---|
| union of `fold_{1..5}_ids.txt` | 1308 | | |
| metadata `tumor?` over the 1308 | | **981** | **327** |
| `per_case_metrics.csv gt_tumor` (5 folds) | | 981 | 327 |
| `train_pos_ids.txt` / `train_neg_ids.txt`, parsed | | 981 | 327 |
| same files through `wc -l` (no trailing newline) | | 980 | **326** |
| metadata `tumor?` over IDs 1-9000 (official train) | 9000 | **926** | 8074 |
| metadata `tumor?` over IDs 9001-9901 (official test) | 901 | 151 | 750 |
| `testing/PanTS/old/pants_utils.py TRAIN_NEG_COUNT` | | | **309** |

The authoritative numbers are **981 / 327**. 926 is the official-train positive count, 326 is a `wc -l` artefact,
and 309 is an old planned negative count that was never the realised cohort.

184 of the 1308 cases (138 tumor+, 46 tumor−; 146 of 1235 lesions) have IDs 9001-9901, which PanTS labels
as its test range (`PanTSMini_ImageTe_00009001_00009901`). The project evaluates only with its own fixed 5-fold CV,
so these are ordinary CV cases. `official_split` is kept for information only.

### b. Slice thickness (spacing along the slice axis) over the 1308
| p0 | p5 | p10 | p25 | **p50** | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| 0.363 | 0.8 | 0.8 | 1.0 | **2.5** | 2.5 | 5.0 | 5.0 | 7.5 | 2.26 |

- **≥2.5 mm: 683/1308 (52.2%). ≥5 mm: 199/1308 (15.2%).** Per lesion (n=1235): median 2.5; 55.4% ≥2.5; 17.5% ≥5.
- The "disagreeing medians" come from different populations or a wrong axis:
  - 1.25 mm = all 9901 PanTS cases
  - 1.5 mm = the 327 cohort negatives
  - 2.5 mm = the cohort positives and the cohort overall
  - 2.21 mm = metadata `spacing` 3rd entry over the cohort. That entry is the last *voxel* axis, which is in-plane
    for the 39 slice-axis-first cases (23 of them are 5 mm scans).
- **Confound:** negatives are thinner-sliced than positives. Median 1.5 vs 2.5 mm; ≥2.5 mm 32.4% vs 58.8%.
  Negatives were phase-matched, not thickness-matched.
- The values below 0.6 mm (6 cases) are genuine thin-slice reconstructions with clean diagonal affines.

### c. Metadata columns and a diagnosis field
The 15 columns are: `PanTS ID, shape, spacing, ct phase, sex, age, manufacturer, manufacturer model, study type,
site, site detail, site nationality, study year, tumor?, structured report`.

**There is no diagnosis, pathology or lesion-type field, and none can be parsed from the reports.** The structured
report is template-generated from segmentations:
- organ volume / mean HU, "enlarged", and lesion location / size / volume / enhancement
- 204 distinct pancreas-related line templates once numbers are masked
- 0 of 9901 reports contain any of adenocarcinoma, PDAC, carcinoma, cyst, IPMN, neuroendocrine, PNET, metastasis,
  pancreatitis, malignant, benign, mucinous, serous, pathology, biopsy, diagnosis or lymphoma

PanTS_00009901 (a tumor+ cohort case) has no report at all. No column was added. You were right that the reports only restate size, which the masks already give.

### d. Negatives with a pancreatic lesion in the report
**30 of the 327 negatives** have a "Pancreas lesions" section describing 37 lesions:
- median long axis 2.0 cm; 34 are ≥1 cm and 20 are ≥2 cm
- the largest is an 8.6 × 5.1 cm hypoattenuating body/tail mass (PanTS_00006771)
- only 6 of the 30 mention it in the IMPRESSION

Full list: `work/master_lesion_table/audit_d_negatives_report_lesions.csv`.

**The report section does not track the masks in either direction.** 677/981 cohort positives have *no* lesion
section; across all PanTS, 699 negatives have one and 763 positives do not. So the report is weak evidence that
these negatives are false negatives of the label. It is also consistent with the reports coming from a different
(model?) segmentation than the released masks, which may be why report attenuation was only 75.6% self-consistent
(*carried over*). The larger ones (the ≥3 cm cases) are still worth eyeballing before treating those scans as
tumor-free in specificity numbers.

### e. Size distribution with and without the tiny flag
| set | n | p5 diam | p10 | median | <5 | 5-10 | 10-20 | 20-40 | ≥40 |
|---|---|---|---|---|---|---|---|---|---|
| all | 1235 | 7.21 | 12.02 | 27.33 | 49 | 37 | 275 | 597 | 277 |
| tiny_flag = False | 1196 | **11.11** | 13.27 | 27.74 | 11 | 36 | 275 | 597 | 277 |
| tiny only | 39 | 1.60 | 1.74 | 2.40 | 38 | 1 | 0 | 0 | 0 |

The 39 tiny components:
- have 3-18 voxels, ≤7.8 mm³ and diameter ≤5.05 mm
- sit in 21 cases; **none is the largest lesion in its case**
- 0 are detected and 2 are excluded
- tiers: embedded 18, inside 13, abutting 6, separated 2

The 5th-percentile diameter moves from **7.2 → 11.1 mm**.

### f. Lesions < 10 mm: isolated vs secondary
| bin | largest in case | n | tiny | detected | excluded |
|---|---|---|---|---|---|
| <5 | no | 49 | 38 | 1 | 3 |
| 5-10 | no | 26 | 1 | 0 | 2 |
| 5-10 | yes | 11 | 0 | 0 | 0 |

- **75 of the 86 sub-10 mm components are secondary components** of a case that has a larger lesion. All 49 of
  the <5 mm components are secondary.
- Only **11 cases** have a largest lesion <10 mm; 10 of those are single-lesion cases. nnU-Net detects none of the 11.
- By comparison, 176 of the 275 lesions in the 10-20 mm bin are the largest in their case.
- So the "<10 mm" population is mostly satellite fragments. This supports making 10-20 mm the primary target.

## Caveats
- `region` / `u` rely on GT head/body/tail masks. `region` for the 70 excluded lesions is nominal: they lie outside
  the gland, and 55 of them are placed only by the PC1 fallback.
- `slice_thickness_mm` is voxel spacing, not reconstruction thickness.
- `detected_any` is overlap-based and not a localization-aware case hit (see S5). `lesion_dice` and `detected_any`
  are out-of-fold nnU-Net results for every row.
- The published `lesion_duct_position.csv` has a `section_source` column that the committed `duct_position.py`
  does not write. It is added later by `plot_duct_position.py` (PC1 fallback), which this table reproduces.

## What surprised me
1. 184 cohort cases come from PanTS's test ID range (irrelevant here: the project uses its own 5-fold CV).
2. Negatives are systematically thinner-sliced than positives (1.5 vs 2.5 mm median).
3. Report lesion sections disagree with the masks in both directions (30 cohort negatives have one, 677 positives
   don't).
4. Almost no sub-10 mm lesion is a case's index lesion (11 of 86).
