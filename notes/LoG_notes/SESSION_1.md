# Session S1: master lesion table (2026-09-28)

Built the shared per-lesion join table and ran the requested cohort/metadata audits. No detector work this session.

## What was built

| File | Purpose |
|---|---|
| `src/master-lesion-table/case_pass.py` | Phantom-tested pass over all 1308 cases straight from the masks: `nib.affines.voxel_sizes` spacing, slice-axis detection (the axis whose direction cosine is most aligned with world superior-inferior), an independent 26-connected lesion recount, Feret diameter, and head/body/tail region by 3-voxel ring vote. Writes `work/master_lesion_table/{case_spacing,lesion_recount}.csv` (git-ignored). |
| `src/master-lesion-table/build_table.py` | Joins that recount onto every existing per-lesion CSV on `(case_id, lesion_id)`, asserts `gt_vox` equality after each join, then runs oracles and audits a-f and writes `results/master_lesion_table/master_lesions.csv`. |

`.gitignore` gained a `work/` line so all repo-root intermediates from this session onward are excluded by default.

## Table

`results/master_lesion_table/master_lesions.csv`: **1235 rows**, key `(case_id, lesion_id)`, one row per GT lesion connected component. Columns: fold, `official_split` (informational only — see below), ct_phase, spacing_{x,y,z}_mm, slice_axis, slice_thickness_mm, gt_vox, vol_mm3, diam_mm, diam_bin, tiny_flag, n_lesions_in_case, largest_in_case, dHU, attenuation_v2, lesion_dice, detected_any, region, region_source, u, tier, excluded, mpd_present, dist_to_mpd_mm. Full dictionary in `results/master_lesion_table/README.md`.

## Oracles (all passed)

- 1308 cases, each in exactly one `fold_k_ids.txt`.
- 1235 rows, `(case_id, lesion_id)` unique.
- `gt_vox` from the fresh recount equals, with 0 mismatches: `attenuation_labels.n_vox` (1235), `lesion_dice_attenuation.gt_vox` (1235, computed server-side from nnU-Net's own `gt_segmentations`), `lesion_inclusion.n_vox` (1235), `excluded_lesions.n_vox` (70), work `lesion_duct_position.n_vox` (1142), and `per_lesion_metrics.gt_vox` as a per-case multiset (that file has no `lesion_id` and 6 duplicate `(case, gt_vox)` keys, so it can't be joined row-wise).
- `tumor?` == (lesion mask non-empty) for 1308/1308.
- `detected_any` == `lesion_dice > 0` for every case, checked against `per_lesion_metrics.detected_any`.
- Phantom test (`case_pass.py --phantom`) covers voxel counts, Feret diameter, ring-vote region (including a lesion carved out of its own section mask), and slice-axis detection on axis-permuted and oblique affines.
- 6 cases have lesion affine ≠ CT affine — exactly CONTEXT's known corrupted-affine list; CT grid used throughout, as the existing scripts already do.
- One exact tie for largest-in-case (PanTS_00009056, 732 = 732 voxels), broken by Feret diameter.

## Audit results (all computed this session)

**a. Cohort counts.** 981 tumor+ / 327 tumor− is correct, and agrees across every source that stores it. 926 = PanTS tumor+ over IDs 1-9000 only; 326/980 = `wc -l` on ID files with no trailing newline; 309 = an old planned negative count that was never realized. Separately: 184 of the 1308 cases (138 tumor+, 46 tumor−) carry IDs 9001-9901, which is PanTS's own test-set ID range. **User decision: this is irrelevant** — the project evaluates only with its fixed 5-fold CV, not PanTS's train/test split, so these are ordinary CV cases. `official_split` is kept on the table for information only.

**b. Slice thickness.** Median (along the true slice axis, not always the last voxel axis) is 2.5 mm over the 1308; 52.2% ≥ 2.5 mm, 15.2% ≥ 5 mm. The disagreeing "1.25 / 1.5 / 2.5 mm" medians from old notes are different populations: 1.25 = all of PanTS, 1.5 = cohort negatives only, 2.5 = cohort positives/overall. Negatives are phase-matched but not thickness-matched (median 1.5 mm tumor− vs 2.5 mm tumor+) — a possible confound for any negative-side (specificity) comparison.

**c. Diagnosis field.** None exists and none is parsable. The 15 metadata columns are listed in the README. The structured report is template-generated (organ volume/HU, "enlarged", and for lesions: location/size/volume/enhancement only) — 204 distinct line templates once numbers are masked, and 0/9901 reports contain any of a 15-term diagnosis-keyword list (adenocarcinoma, PDAC, carcinoma, cyst, IPMN, neuroendocrine, PNET, metastasis, pancreatitis, malignant, benign, mucinous, serous, pathology, biopsy, diagnosis, lymphoma). No column was added — confirms the report text is size/enhancement only, nothing more.

**d. Negatives with a reported lesion.** 30 of the 327 negatives have a "Pancreas lesions" report section describing 37 lesions; median long axis 2.0 cm, 34 ≥ 1 cm, 20 ≥ 2 cm, largest an 8.6×5.1 cm hypoattenuating body/tail mass (PanTS_00006771). Only 6 of the 30 are named in the IMPRESSION line. But the report doesn't track the masks in either direction: 677/981 positives have no lesion section at all, and across all of PanTS 699 negatives have one vs 763 positives without. Full list in `work/master_lesion_table/audit_d_negatives_report_lesions.csv`.

**e. Size distribution with/without tiny flag.** 39 lesions are `tiny_flag` (< 8 mm³; all ≤ 5.05 mm diameter, 3-18 voxels). None is the largest lesion in its case; 0 detected; 2 excluded. 5th-percentile diameter moves 7.2 → 11.1 mm when tiny lesions are excluded from that one statistic (they are never dropped from the table).

**f. Lesions < 10 mm: isolated vs secondary.** 75 of 86 sub-10mm components are secondary (not the largest lesion in their case) — including all 49 under 5 mm. Only 11 cases have an index (largest) lesion under 10 mm, 10 of those single-lesion cases, and nnU-Net detects none of them. By contrast 176/275 in the 10-20 mm bin are the case's largest lesion. Supports treating "<10 mm" as mostly satellite fragments rather than a primary detection target, consistent with CONTEXT's existing 10-20 mm focus.

## Caveats

- `region`/`u` depend on GT head/body/tail masks; for the 70 excluded (separated) lesions `region` is nominal — most are placed only by the PC1 fallback, not the mask ring.
- `slice_thickness_mm` is voxel spacing, not reconstruction slice thickness (NIfTI doesn't store the latter).
- `detected_any` is plain overlap, not localization-aware (see S5 for that distinction).
- `lesion_dice`/`detected_any` are out-of-fold nnU-Net results for every row, joined from the server-computed CSVs, not recomputed this session.

## Surprises

1. 184 cohort cases sit in PanTS's own test ID range — surfaced as a question, resolved by the user as not meaningful here (own 5-fold CV, case numbers don't matter).
2. Negatives are systematically thinner-sliced than positives, a confound not previously flagged.
3. Report lesion sections disagree with the masks in both directions, not just "positives sometimes lack a report lesion" — 30 masked-negative cases have one.
4. Almost no sub-10mm lesion is a case's index lesion (11/86); the small-lesion problem is overwhelmingly about satellite fragments of already-detected masses.

## CONTEXT.md changes (user-approved)

- Removed "Train split only (IDs 1-9000)"; added that evaluation is the project's own 5-fold CV and PanTS ID ranges don't matter.
- Added the negative-cohort thickness confound.
- Added a pointer to `results/master_lesion_table/master_lesions.csv` as the shared join table.
- Replaced "No diagnosis label located yet" with: there isn't one, and why (template reports).
- Added that report lesion sections don't track the masks.
- Replaced "counts disagree, don't trust them" with the resolved 981/327 numbers and where each old number came from.
- Rule 3 changed from "report the official test set only at the very end" to "all reported results are 5-fold CV (out-of-fold)" — the official-test-set framing no longer applies.
