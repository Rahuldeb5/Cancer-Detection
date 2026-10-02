# CONTEXT.md - small pancreatic tumor detection on PanTS
MSKCC collaboration. Author: Rahul (intermediate Python dev; wants to understand
each step - explain design choices in 1-2 sentences, run phantom/oracle checks BEFORE real data,
ask before changing existing scripts).

## Goal (updated after S4)
Original aim: raise recall for small (<20 mm) tumors with a LoG blob candidate generator + nnU-Net. S4 result: NEGATIVE
(see "S4 result"). Current priorities:
1. Stronger baseline: multi-class nnU-Net (pancreas + lesion), same folds (S7 build/launch, S8 evaluation).
   Pancreas is a TRAINING label only, never an input (the carved-out hole marks the tumor; an input mask would leak).
2. Decompose >=20 mm failures. Approx. (derived from baseline tables, recompute before citing): 20-40 mm has ~40% of
   lesions fully missed and Dice among detected ~0.54, so Dice gain there is mostly detection; >=40 mm is
   under-segmentation (carried over: recall 0.41, precision 0.66).
3. 10-20 mm remains the open frontier. Idea, NOT started: normative/anomaly model fit on unlabeled normal pancreas
   patches (extra negatives from the ~8,500 unused PanTS cases; cohort negatives held out for evaluation), scored by
   within-scan rank vs random same-size patches; requires a same-data nnU-Net control and a patient-overlap check.
Segmentation quality (Dice/IoU), not only detection, is the end goal. Priority lesions: 10-20 mm (<10 mm is too sparse).
Session status: S0-S4 done; S5 (union FROC) NOT run (S4 negative); S6 (leakage audit) pending; S7/S8 = multi-class baseline. S9 = normative-model pilot (PC): normal-patch bank from 300 non-cohort PanTS negatives (phase/thickness-matched to
cohort positives); k-means (Rahul's method) and k-NN scores in PCA space, stratified by phase x thickness x gland
position; within-scan percentile rank as the primary metric; fixed gates (>=40 mm control >= 90th percentile; 10-20 mm
target >= 90th percentile and beating S4). No cohort case ever enters the bank. If it passes, the next step is the
fairness control (nnU-Net retrained with the same extra negatives) before any claim.

## Some notes to keep in mind

Note that there is a paucity of lesions < 10 mm in diameter so it's more important we focus on the 10-20mm diameter lesions. Also note that some of the lesions landed outside the pancreas map and you can find results/lesion_data/lesion_location_results to tell you the excluded lesions and nnunet_excl_outside_pancreas to tell you the updated baseline without the excluded lesions (although that could make the sizes a bit different for each fold that's a little sketchy). Go through all the code files at the start of each session to get a good understanding.

## Cohort
- 1308 cases = union of src/data/fold_{1..5}_ids.txt: 981 tumor+ (1235 lesion components), 327 tumor-
  (phase-matched random negatives). ~75% positive vs ~10.9% in PanTS overall:
  sens/spec transfer, PPV/NPV/accuracy do not.
- Evaluation = our fixed 5-fold CV only. PanTS's own train/test ID ranges (1-9000 / 9001-9901) are irrelevant
  here: 184 cohort cases (138 tumor+, 46 tumor-) have IDs >9000 and are ordinary CV cases.
- Negatives are phase-matched but NOT slice-thickness-matched: median slice spacing 1.5 mm (tumor-) vs 2.5 mm
  (tumor+) (S1).
- Shared per-lesion join table: results/master_lesion_table/master_lesions.csv (1235 rows, key
  (case_id, lesion_id); fold, spacing, size, tiny_flag, attenuation, Dice, region, u, tier, MPD). Join onto it
  instead of re-deriving; column dictionary in its README.
- Tumor label: results/PanTS_metadata_new.csv column `tumor?` (matches empty/non-empty lesion mask 1308/1308).
- pancreatic_lesion is NOT only PDAC (other lesion types likely). No diagnosis label exists: the metadata has no
  such field and the structured reports are template text (location/size/volume/HU only) (S1).
- Structured reports do not track the masks: 30/327 negatives have a "Pancreas lesions" section, 677/981
  positives don't (S1). Don't use report text as ground truth.
- fold_k_ids.txt <-> nnU-Net fold k-1 (fold_1 -> fold 0 ... fold_5 -> fold 4).
- Old-note count disagreements resolved (S1): 981/327 is correct. 926 = PanTS tumor+ over IDs 1-9000,
  326/980 = `wc -l` on ID files without a trailing newline, 309 = an old planned negative count. Median slice
  spacing (along the slice axis) is 2.5 mm over the 1308 (1.25 = all PanTS, 1.5 = cohort negatives);
  52.2% >= 2.5 mm, 15.2% >= 5 mm.
- Lesions <10 mm are mostly satellite fragments: 75/86 are not the largest lesion in their case; only 11 cases
  have an index lesion <10 mm. 39 components are < 8 mm^3 (tiny_flag; flagged, never dropped) (S1).

## nnU-Net baseline (vanilla binary tumor, 3d_fullres, 5-fold CV, 1000 epochs)
- Lesion Dice / detect@any-overlap by lesion Feret diameter, pooled: <5 mm 0.000/2% (n=49); 5-10 0.000/0% (37);
  10-20 0.074/26% (275); 20-40 0.321/60% (597); >=40 0.483/83% (277).
- evaluate.py case sensitivity (0.937) only checks "any predicted voxel anywhere". NOT localization-aware.
  Localization-aware: 631/981 tumor+ cases (64%) have a predicted component touching GT; 288 predict only in the
  wrong place; 62 predict nothing. By largest lesion: <10 mm 0%, 10-20 31%, 20-40 66%, >=40 86%.
- Case specificity 0.223: 78% of negatives get a predicted blob.
- OOF predictions on server: .../nnUNet_results/Dataset501_PanTSTumor/nnUNetTrainer__nnUNetPlans__3d_fullres/
  fold_{0..4}/validation/{case}.nii.gz + .npz. npz['probabilities'] is (C,z,y,x); channel 1 .transpose(2,1,0)
  matches the nibabel array.
- Probabilities are float32 and never hit the float ceiling (max 0.9999875 over all 1308 OOF volumes, S2); the
  histogram has a soft cap near logit 6 (p~0.9975). PanTS_00000020: GT negative, max prob 0.9994.
- S2 (results/nnunet_subthreshold/): missed <10 mm lesions peak at median p 1.3e-4 in lesion+2 mm; for the
  median one, 82% of negatives' search-region peaks are >= it. nnU-Net FROC ladder (t 0.9..0.001, peak-in-2mm
  and touch rules) in nnunet_froc_points.csv; float16 prob/logit crops + component tables on the PC at
  ~/research/nnunet_probs/. 11 negatives have empty pancreas masks -> no search region.
- Failure is NOT explained by z-undersampling (an old "49%" claim was wrong; 4-26% depending on definition).

## Attenuation (v2 audited in S3 -> v3; use results/attenuation_v3/attenuation_labels_v3.csv)
v2: dHU = median tumor HU - median parenchyma HU (pool excludes lesion + 2.5 mm shell); hyper >+10, iso within +-10,
hypo <-10. Counts: hypo 561 / iso 503 / hyper 168 / unknown 3 (reproduced exactly in S3). Below 10 mm every class
has Dice 0.000 (size, not contrast); iso is worse only at 10-40 mm. Report-text attenuation is only 75.6%
self-consistent - not truth.
v3 (S3): PV-aware eroded-core median vs envelope pool minus lesions+5 mm minus ducts/CBD. Phantom: whole-mask median
loses 16-27% of contrast at r=4 mm, 8-14% at r=8; core fixes r>=4, nothing fixes r=2. Iso 40.7% -> 37.3% (134/1235
change class, both directions). Iso is mostly real: venous >=10 mm 30.8% (CI 27-35%), flat with size >=10 mm;
non-contrast scans are 1/3 of lesions and ~50% iso. Residual: >4 mm slices have the highest iso in every phase.
"Arterial" parenchyma median 61 HU < venous 81 HU: phase labels may be early-arterial/noisy.

## Location
Head holds ~50% of gland tissue but 63-70% of lesions. Small (<20 mm) tumors are found more often in the head
(zero-Dice rate 68% head vs 95% mid-gland). Only the PC1 head->tail axis is reliable; PC2/PC3 are not anatomical.
Lesion tiers vs pancreas envelope: inside 880, embedded 111, abutting 174, separated 70
(results/lesion_location_results/excluded_lesions.csv lists the 70 separated). Report results with and without them.

## S4 result (pancreas-restricted scale-normalized LoG blob map): NEGATIVE
- Target group (10-20 mm, hypo/hyper v3, contrast-enhanced, n=106): the lesion's strongest blob response is beaten by a
  median 76.5% of same-size random patches from the same pancreas. 10 candidates/case (8.3 FP/case) finds 2.8%
  (CI 1.0-8.0); 50 candidates finds 14.2% (8.8-22.0) at 41.6 FP/case; at <=3 FP/case target sensitivity is 0%.
  On nnU-Net-missed ∩ target (n=74): 12.2% hit, 0% at rank 1.
- Featureless at <=20 mm in every stratum (3.6% of 10-20 mm lesions exceed the phantom noise floor); clutter-dominated
  only at >=40 mm. Iso within +-5 HU undetectable.
- z = response/MAD sends ~81% of candidates to sigma=1.5 mm (MAD rises with sigma on real scans). Fixed sigma=5 mm gets
  20.8% into the top 10 but was NOT adopted (tuning on evaluation lesions); even then ~7 clutter peaks outscore the lesion.
- Blobness points the wrong way (in-lesion 0.086 vs false peaks 0.123).
- Negatives understate false peaks by ~1/3; slice-thickness reweighting barely changes it; thicker slices give MORE false
  peaks. Default FP denominator in later work = source (b): positives, search region >10 mm from every GT lesion.
- 73/1235 lesions lie entirely outside the search region (mostly `abutting` tier).
- Top false peaks sit on vessels (30.4%) and duodenum (24.9%).
- Files: results/log_candidates/, notes/LoG_notes/SESSION_4.md, src/log-candidates/. Vessel/bowel masks for the 1308
  cohort cases were git-lfs pulled on the PC (~2 GB).

## Duct work (PARKED - ablation only)
Step-cutoff detector fires in 7.1% of tumor+ (0/327 neg): GT MPD masks are short fragments (median centerline
28 mm; 255 empty). Head-end dilation: 56.8% of evaluable tumor+ MPDs vs 5.1% of negatives, concentrated in
>=20 mm tumors. Component re-ranking (HistGB, leave-one-fold-out) at 90% recoverable: case spec raw prob 0.34,
+nnU-Net feats 0.57, +GT pancreas location 0.83, +GT duct 0.79 (duct adds ~0). All GT anatomy = upper bound.
Possible leakage in the pancreas-location gain (pancreas mask is carved at the tumor) - S6 tests this.
pdac_classification.csv (2 cm periampullary rule) is circular - do not use it to scope PDAC.

## Data gotchas (each one has already cost time)
1. PanTS masks are int8 but split across two on-disk encodings (verified S0, 1308-case cohort, 5
   pancreas/lesion masks each: 3130 files raw {0,1} slope 1; 3410 files raw {-128,127} slope 1/255
   inter 0.502). img.header shows scl_slope=NaN for every file, but that's only because nibabel moves
   the scaling onto img.dataobj once loaded, not because the on-disk header is NaN. Read with nibabel
   img.dataobj.get_unscaled() (int8, no float64 upcast) and threshold at raw>0.5 in scaled units, i.e.
   raw > (0.5-inter)/slope -- `raw != 0` or `.astype(bool)` lights up the whole volume for the second
   encoding. NEVER SimpleITK BinaryThreshold (same failure mode). Canonical loader:
   src/tumorlib/io.py load_mask() (tested against nibabel's own scaled reading, S0).
2. Work in nibabel (x,y,z) order everywhere. Lesion IDs = scipy.ndimage.label with generate_binary_structure(3,3).
   Mixing SimpleITK (z,y,x) permuted IDs for ~21% of lesions. Join key = (case_id, lesion_id); assert gt_vox
   equality after every join.
3. Spacing: nib.affines.voxel_sizes(affine). Never header zooms or np.diag. Slice axis is not always last;
   z-spacing goes up to 7.5 mm.
4. 9 cases have corrupted mask affines (tumor+: PanTS_00000259, 00005731, 00006447, 00006466, 00006927, 00007151;
   neg: 00005915, 00007132, 00009737). Voxel grids are still aligned: use the CT grid/affine, warn, don't reject.
5. Pancreas masks: lesion is carved out of pancreas.nii.gz in many cases; head/body/tail disagree with it.
   Use envelope = union(pancreas, head, body, tail) and hole-fill for any candidate search. NEVER let a search
   region read the lesion mask.
6. Memory: crop to bbox BEFORE EDT/label/resample. Worst case PanTS_00008854 (510x431x918). PC has 15 GB RAM:
   <=2-4 workers. Never cache full-volume 1 mm isotropic arrays (this exhausted disk before).
7. Un-pulled git-LFS masks are ~131-byte pointer stubs ("not a gzip file"). Scoped `git lfs pull -I` include
   string must stay under 131072 bytes.
8. Do NOT run src/nnunet/cleanup.sh (deletes ct_staging, needed for CT-based work).
9. Multi-class labels (Dataset502_PanTSPancLesion): 0 background; 1 pancreas = union(pancreas, head, body, tail, lesion)
   (carved-out lesion voxels kept, no morphological ops); 2 lesion (overrides; identical to the Dataset501 binary label).
   nnU-Net regions: labels {background:0, pancreas:[1,2], lesion:2}, regions_class_order [1,2]. 11 negatives have all four
   pancreas masks empty: kept in validation, excluded from every fold's TRAIN list. Server has no raw CTs: build and
   preprocess on the PC, rsync nnUNet_preprocessed to the server, train there (GPUs 0,1 only).

## Machines
- Laptop (WSL, /home/rahuldeb5/Cancer-Detection, ssh rahuldeb5@100.81.217.58, but there is nothing extra there and it could be offline): code + result CSVs, NO dataset. IDE has twice overwritten
  edited files with a stale buffer: re-read from disk before trusting contents.
- PC -- this is what we are currently in (ssh rahuldeb5@100.103.52.84, RTX 5070, 15 GB RAM, 16 cores, py3.14): CTs in
  ~/research/datasets/pants/ct_staging; masks in .../masks/mask_only/{case}/segmentations/*.nii.gz.
  Has unrelated uncommitted work: only add files, never reset/commit others.
- Server (ssh rahul@deep-server.tail8e65db.ts.net, 12 cores, 31 GB, py at ~/nnunet_setup/.venv/bin/python;
  source ~/research/nnunet_env/env.sh): nnU-Net predictions/checkpoints, preprocessed data, gt_segmentations,
  6 masks/case in ~/research/datasets/pants/duct_masks/. Its repo was force-synced to origin/main in S2
  (old uncommitted MedFormer edits backed up in ~/repo_backup_pre_S2_2026-09-28/; untracked exp/ log/ kept). Untracked-on-server scripts (evaluate.py etc.) must be brought into git.
- For CPU intensive tasks use the PC, for GPU intensive tasks use the server (only use the 2 2080 TIs)

## Rules
1. Oracle first: synthetic phantom or known-answer regression before any real-data run.
2. Never report a number you did not compute this session. Label carried-over numbers as carried over.
3. Nothing tuned on evaluation lesions. Detector settings are fixed a priori; anything learned uses
   leave-one-nnU-Net-fold-out. All reported results are 5-fold CV (out-of-fold).
4. Units explicit (mm vs voxels). Bin edges: <5, 5-10, 10-20, 20-40, >=40 mm (Feret diameter).
5. results/ holds ONLY the named deliverables (one CSV + README per session). Intermediates go in a
   git-ignored work/ dir.
6. Hit definitions must be localization-aware (see S5).
7. Each session ends with README: what was computed, how, caveats, and what surprised you. You can name it as SESSION_{x}.md and put it in notes/LoG_notes/