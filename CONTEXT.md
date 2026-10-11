# CONTEXT.md - pancreatic tumor segmentation on PanTS
MSKCC collaboration (Prof. Yingli Tian; Dr. Oguz Akin). Author: Rahul (intermediate Python dev; wants to understand
each step - explain design choices in 1-2 sentences, run phantom/oracle checks BEFORE real data,
ask before changing existing scripts).
Original framing was multi-organ detection with LLM-read radiology reports. The report arm is PARKED: PanTS reports
are template text with no diagnosis; Merlin needs a Stanford AIMI request.

## Goal (updated after S14)
Segmentation quality (Dice/IoU) of pancreatic lesions, not only detection. Priority sizes: 10-20 mm and 20-40 mm
(<10 mm is too sparse and mostly satellites).
1. BASELINE = Dataset502 (multi-class nnU-Net: pancreas + lesion) with S13's post-processing D. It beats the binary
   Dataset501 on every lesion metric in all 5 folds (S8). Pancreas is a TRAINING label only, never an input (the
   carved-out hole marks the tumor; an input mask would leak).
2. Current work: S15 gland shape pilot (secondary signs: contour bulge, upstream atrophy) on the predicted gland,
   targeting 502's missed 20-40 mm lesions that 502 swallows into its own gland (n=48, S14).
3. Only if S15 passes: small case-level classifier on 502 confidence + shape features (duct dropped: S14).
4. Optional GPU job: 502 retrain with lesion-forced foreground sampling (separates "gland context" from "sampling
   rate"; S7 sec. 6 confound). Not started.
Likely paper framing: anatomical supervision (+ secondary signs if S15 holds) for pancreatic tumor segmentation,
with S2/S4/S9 as the analysis of why small tumors stay invisible.

Session status: S0-S4, S7, S8, S8b, S9, S10, S11, S12, S13, S14 done. S5 (union FROC) NOT run (S4 negative).
S6 (leakage audit) superseded by S13 - only needed if the parked duct re-ranking number is ever cited. S15 next.

## Some notes to keep in mind

Note that there is a paucity of lesions < 10 mm in diameter so it's more important we focus on the 10-20mm diameter lesions. Also note that some of the lesions landed outside the pancreas map and you can find results/lesion_data/lesion_location_results to tell you the excluded lesions and nnunet_excl_outside_pancreas to tell you the updated baseline without the excluded lesions (although that could make the sizes a bit different for each fold that's a little sketchy). Go through all the code files at the start of each session to get a good understanding.

## Cohort
- 1308 cases = union of src/data/fold_{1..5}_ids.txt: 981 tumor+ (1235 lesion components), 327 tumor-
  (random negatives matched on the METADATA phase label, which is noisy - see S12). ~75% positive vs ~10.9% in PanTS
  overall: sens/spec transfer, PPV/NPV/accuracy do not. Read IDs with tumorlib.io.case_ids() (gotcha 11).
- Each CT is an independent patient (per PanTS docs; metadata has no patient ID - cite PanTS docs in a paper).
- Evaluation = our fixed 5-fold CV only. PanTS's own train/test ID ranges (1-9000 / 9001-9901) are irrelevant
  here: 184 cohort cases (138 tumor+, 46 tumor-) have IDs >9000 and are ordinary CV cases.
- Negatives are NOT slice-thickness-matched: median slice spacing 1.5 mm (tumor-) vs 2.5 mm (tumor+) (S1).
  They are matched on metadata phase only; by scan-derived phase they are NOT matched (S12: venous 58.6% tumor+
  vs 41.6% tumor-, p 8e-6). Within estimated venous the thickness gap persists (1.5 vs 2.5 mm).
- S8b: reweighting negatives to the tumor+ est_phase x thickness mix is possible (no empty cells; ESS 187/327) and
  barely moves specificity (501 0.223 -> 0.178; 502 0.722 -> 0.703). Within matched cells negatives get MORE FPs than
  positives' far region, so unmatched negatives are not an optimistic FP denominator for nnU-Net.
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

## Dataset502 = current baseline (S7 build, S8 eval, S8b phase add-on, S13 post-processing)
- Labels / regions / training: gotcha 9. Same folds and plan as 501; only CT normalization stats differ. Lesion-centred
  patch sampling ~25% vs ~50% in 501 (confound; it works AGAINST 502). "Pancreas Dice" is not a clean gland metric
  (44.8% of lesion voxels lie outside the 4-mask envelope). Suspect gland masks: PanTS_00002205, 00002102.
- Predictions on server: .../nnUNet_results/Dataset502_PanTSPancLesion/.../fold_{0..4}/validation/ (~135 GB).
  npz channel 0 = pancreas region, channel 1 = lesion, (C,z,y,x) -> .transpose(2,1,0). Lesion mask = seg == 2
  (NOT seg > 0). Sparse cache on PC: ~/research/nnunet_probs_502/ (every voxel with lesion p > 1e-4 as float32,
  channel 0 > 0.5, lesion GT). 46 cases never exceed p 1e-4. S8 per-case pickles: work/nnunet_multiclass/cases/.
- S8 (501 -> 502, all 5 folds, case bootstrap; direction identical in every fold):
  mean lesion Dice 0.280 -> 0.400 (delta +0.121, CI +0.107 to +0.134); 10-20 mm Dice/detect 0.074/25.8% ->
  0.196/47.6%; 20-40 mm 0.321/59.8% -> 0.468/80.6%; tumor+ cases localized 631 (64.3%) -> 823 (83.9%);
  10-20 mm index lesion localized 31.8% -> 63.1%; FP comps per negative 2.06 -> 0.33 (72% of negatives clean vs 22%);
  case Dice (981 tumor+) 0.312 -> 0.472. Of 501's 240 missed 20-40 mm, 502 finds 140 and loses 16 of 501's 357 hits.
  502 at t=0.001 (1.22 FP/neg) is below 501 at t=0.9 (1.52): 502 dominates the whole ladder.
  502 case "sensitivity" 0.915 < 501's 0.937 only because 501's was propped up by wrong-place blobs.
  Biggest gains in 501's weakest strata (est non-contrast 20-40 mm 34% -> 70%; body 10-20 mm 10% -> 39%).
  Separated lesions do not improve (10-20 mm 23% -> 15%, n=26, CI spans 0).
- S8b: 502 beats 501 in every phase stratum with >=20 lesions except late arterial 10-20 mm (+0.061 Dice,
  CI -0.023 to +0.167, n=29; a hint, not a finding).
- S13 post-processing (leave-one-fold-out, objective = mean per-case lesion Dice over all 1308 cases):
  * D ADOPTED as default: grow each saved-segmentation object to p > 0.1 inside its t_low 0.1 footprint.
    Objective 0.5342 -> 0.5446 (+0.0104, CI +0.008 to +0.013, all 5 folds); case Dice 0.472 -> 0.486; lesion Dice
    0.400 -> 0.418; median volume ratio among detected 0.92 -> 1.08 (20-40 mm), 0.78 -> 0.89 (>=40 mm). 10-20 mm
    over-grow (1.18x -> 1.43x). t_seg 0.1 is the grid edge. D's 19 "new touches" are grazes (median Dice 0.008).
  * A (out-of-pancreas filter): no-op (+0.0023, CI touches 0). 502 already keeps lesions inside its own gland, and
    grows its gland around lesions it detects.
  * B (faint-candidate rescue): dead. Faint candidates touch missed lesions but never segment them.
  * C+ (filter x D): post hoc; trades 22 detected lesions for 12 cleared negatives. NOT adopted.
- S14 (what 502 still misses; model = 502 + D; missed = not touched by RAW 502): 116/597 at 20-40 mm, 144 at 10-20 mm.
  D rescues none (0/116 reach Dice 0.1). Misses vs detected (20-40 mm): satellite 35% vs 7%; est non-contrast 31% vs
  17%; <50% inside predicted gland 47% vs 4%; GT separated 16% vs 2%; iso 42% vs 38% (NOT enriched).
  Missed lesions are inside the predicted gland 53%, adjacent 33%, away 15%, dents 0% (gland stops at them; no
  carving). 10-20 mm: 79/144 misses are satellites, 58 of those in cases 502 already localizes.
  Faint 502 guess (p>0.01): +20.3 pts over chance on misses but median Dice 0.05 and 0.67 FP/negative.
  S15 target: missed 20-40 mm INDEX lesions >=50% inside 502's predicted gland, not GT-separated: n=48 (30 enhanced,
  16 enhanced and not iso); same-size reference: 461 detected-and-included; secondary 10-20 mm n=46.
  Files: results/rescue_check_502/missed_lesions.csv.

## Dataset501 (old binary baseline: vanilla binary tumor, 3d_fullres, 5-fold CV, 1000 epochs)
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
change class, both directions). Residual: >4 mm slices have the highest iso in every phase.
Iso by ESTIMATED phase (S8b; replaces the old metadata-based "non-contrast = 1/3 of lesions, ~50% iso"):
est non-contrast holds 23% of lesions and is 55.1% iso (156/283; >=10 mm 61.5%); venous 33.1% (>=10 mm 33.4%);
late arterial (pancreatic phase) 19.8% (20/101), the lowest.
"Arterial" parenchyma median 61 HU < venous 81 HU (S3) is explained in S12: metadata "Arterial" is a mixture.

## Contrast phase (S12; use results/phase_estimate/phase_estimate.csv, NOT metadata `ct phase`)
- est_phase per case from a frozen 3-component GMM on (aorta HU, portal-vein HU): eroded (1 voxel) median over the
  middle 80% of vessel slices; masks aorta.nii.gz + veins.nii.gz (PanTS `veins` = portal/splenic system, inferred
  from geometry). Max posterior < 0.8 -> uncertain (53); 17 unmeasurable (mostly empty veins masks, 13 negatives).
  Code src/phase/. Centroids (aorta, PV): non-contrast (41, 39), venous (148, 163), arterial (243, 125).
- Standard strata for all reporting (S8b): non-contrast, early arterial, late arterial, venous, uncertain.
  `early_arterial` = PV enhancement fraction < 0.5 (frozen in src/phase/core.py). Don't interpret strata with
  <20 lesions.
- Metadata phase agrees only 69.0% (CI 66-72). 143/380 metadata "Non-contrast" are enhanced; metadata "Arterial"
  (282) = 34.8% venous-looking, 32.3% early arterial, 16.0% late arterial, 12.8% non-contrast. Gland HU (not in the
  rule) follows the estimate: est non-contrast 33 HU, early arterial 64, venous 81, late arterial 93.
- Phase ~ thickness: est non-contrast median 0.8 mm (84% <= 2 mm), arterial 1.25, venous 2.5.
- The wide arterial component claims 10 scans with PV >= aorta; filter `aorta_hu > pv_hu` for clean arterial.
- Oracle caveat: 1-voxel-shifted r=5 mm vein reads up to -7.2 HU low (fails 5 HU tolerance; aorta passes).
  Not "fixed" by changing the erosion (that would be tuning to the gate).

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
- Negatives understate false peaks by ~1/3 (consistent with negatives being thinner-sliced, S12); slice-thickness
  reweighting barely changes it; thicker slices give MORE false peaks. Default FP denominator in later work =
  positives' search region >10 mm from every GT lesion.
- 73/1235 lesions lie entirely outside the search region (mostly `abutting` tier).
- Top false peaks sit on vessels (30.4%) and duodenum (24.9%).
- Files: results/log_candidates/, notes/LoG_notes/SESSION_4.md, src/log-candidates/. Vessel/bowel masks for the 1308
  cohort cases were git-lfs pulled on the PC (~2 GB).

## S9 result (normative normal-patch anomaly model): FAILED all gates
- Bank: 300 non-cohort PanTS negatives (~/research/datasets/pants/ct_normal_bank/, 9.3 GB), matched to cohort positives
  on METADATA phase and thickness (metadata phase is noisy, S12 - re-estimate before reusing the bank).
- 10-20 mm target median within-scan percentile 71.8 (S4: 23.5); top-10 16%. >=40 mm control 71.2 (gate 90).
  Insertion test nearly flat. Cause: normal patches vary ~44 HU SD within one pancreas; 23% of normal patches <20 HU
  (fat in spherical patches). Bank size 100/200/300 is flat. Within-scan z-scoring cannot change within-scan ranks.
- Not a standalone detector. One revision "S9b" only if it proves complementary to 502.

## S10 result (duct atlas, 6061 glands, PC1 head->tail frame)
One continuous curved ridge ~3.5-5.5 mm anterior to the gland centreline; densest head/neck, fades to tail (56% of
cases have duct at u=0.2, 8% at u=0.8); axial head bimodal (posterior lobe near u 0.1); tumor+ glands carry ~4x more
annotated duct, reaching body/tail. Duct label is disjoint from head/body/tail. Annotation fragmentary (median extent
27.6 mm) -> atlas darkness = coverage x anatomy.

## S11 rescue check on 501 (results/rescue_check/) - superseded by S14 on 502
- Missed 20-40 mm (n=240): union of nnU-Net faint guess (t=0.01, case rank<=3; 17.9%), S9 top-10 (33.0%) and duct
  (mpd_dilated|head-end AND <=10 mm from MPD; 20.1% of evaluable) = 47.5% (CI 41-54); S9 unique 19.6% -> fixed gate PASS.
- But S9 top-10 = uniform-placement chance (32.8%; footprint ~3.5% of lattice), and nnU-Net|duct alone = 27.9% (fails).
  All signals fire MORE on detected than missed lesions. Head-end dilation adds ~nothing over mpd_dilated.
- Lesson: report the chance rate next to every hit rate.
- Duct CSVs live in results/geometry_results/duct_cutoff/ (not results/duct_cutoff/).

## Duct work (PARKED - not used as a signal)
Step-cutoff detector fires in 7.1% of tumor+ (0/327 neg): GT MPD masks are short fragments (median centerline
28 mm; 255 empty). Head-end dilation: 56.8% of evaluable tumor+ MPDs vs 5.1% of negatives, concentrated in
>=20 mm tumors. Component re-ranking on 501 (HistGB, leave-one-fold-out) at 90% recoverable: case spec raw prob 0.34,
+nnU-Net feats 0.57, +GT pancreas location 0.83, +GT duct 0.79 (duct adds ~0). All GT anatomy = upper bound.
S14: on 502's misses the duct signal is at or below chance (-2.5 pts, CI -6.9 to +2.0); missed lesions sit FARTHER
from the MPD than random placement. Duct dilation tracks tumors 502 already finds. Dropped from the classifier plan.
pdac_classification.csv (2 cm periampullary rule) is circular - do not use it to scope PDAC.

## S15 plan (gland shape profile pilot) - gates must be confirmed by Rahul BEFORE Part C runs
Outline = RAW 502 predicted pancreas (channel 0 > 0.5); never GT masks. Centerline every 2 mm, perpendicular
cross-sections, 24 rays (first exit, max 40 mm) -> radius map r(position, direction) + area curve A(position).
Primary score = positive (outward) bulge_local (S14: no dents). Normal reference = S9 bank glands run through 502
(5-fold ensemble), 250 reference / 50 held out for insertion tests; re-estimate bank phase with src/phase/ first.
Gates: insertion 20 mm bulge median percentile >= 95; control detected >=40 mm median >= 90; target (S14's
n=48) median >= 80 AND top-3 hit rate >= 2x chance AND >= 25%. Scope: S15 can address at most 48 of the 116 missed
20-40 mm lesions.

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
4. Mask affines are unreliable; voxel grids align with the CT, so always use the CT grid/affine, warn, don't reject.
   - 9 cases have corrupted mask affines (tumor+: PanTS_00000259, 00005731, 00006447, 00006466, 00006927, 00007151;
     neg: 00005915, 00007132, 00009737).
   - aorta.nii.gz is much worse: its affine differs from the CT in 367/1308 cohort cases (300 LPS vs CT RAS,
     55 translation only, 12 other flips); world-remapping puts it on the veins (S12 src/phase/affine_check.py).
   - Duct affines: 21 across PanTS differ from the head affine by a y (A<->P) flip. This is a mask-affine problem,
     not evidence that the gland is mirrored on the CT grid.
   - Mirrored glands (aorta-posterior test, S15): S10 flagged 13/1608 glands, but most do not hold up. They were
     mask-affine corruptions that do not affect the CT grid, or artifacts of using the whole aorta. S15 reran the test
     on the CT voxel grid with the aorta placed relative to the gland (qualifying band: offset within +-10 mm =
     inconclusive). Result on the cohort: 5 flagged, 34 inconclusive, 1256 normal, 1 missing.
       * Solid (4): PanTS_00001361, 00002916, 00005731 (all agree with S10), and 00006447 (its CT affine itself is
         mirrored).
       * Uncertain (1): PanTS_00008480 (+18 mm offset, but its predicted-gland outline is fragmented, so the
         largest component may not be the pancreas).
       * Inconclusive: 34, of which 32 are small negative offsets (-10 to -0.5 mm); only PanTS_00001826 (+0.8 mm)
         and 00002205 (+1.0 mm) are positive. Do not treat these as mirrored.
   - Rules: any per-case anterior/posterior analysis needs the aorta test, run on the CT voxel grid and never
     through the aorta's own affine. Report flipped and inconclusive cases separately. Scores that rank positions
     within one scan (S15 bulge_local, width) do not depend on which direction is called "anterior"; only
     bulge_global and any atlas-frame analysis do. Head->tail sign: take it from the CT affine (tail toward patient
     left), not from GT head/tail masks (S15; agrees with S10's GT sign in 1005/1006; the exception, PanTS_00007757,
     has a GT head->tail pointing to patient right and is worth a look by Dr. Akin).
   load_mask does the right thing; never resample masks by their own affine.
5. Pancreas masks: lesion is carved out of pancreas.nii.gz in many cases; head/body/tail disagree with it.
   Use envelope = union(pancreas, head, body, tail) and hole-fill for any candidate search. NEVER let a search
   region read the lesion mask. Duct is carved out of head/body/tail.
6. Memory: crop to bbox BEFORE EDT/label/resample. Worst case PanTS_00008854 (510x431x918). PC has 15 GB RAM:
   <=2-4 workers. Never cache full-volume 1 mm isotropic arrays (this exhausted disk before).
7. Un-pulled git-LFS masks are ~131-byte pointer stubs ("not a gzip file"). Scoped `git lfs pull -I` include
   string must stay under 131072 bytes; pull one structure at a time. pancreas.nii.gz is a stub for 8293 non-cohort
   cases. tumorlib's stub guard also trips on tiny synthetic phantoms (<1 KB gzipped): use a realistic FOV (S12).
8. Do NOT run src/nnunet/cleanup.sh (deletes ct_staging, needed for CT-based work).
9. Multi-class labels (Dataset502_PanTSPancLesion): 0 background; 1 pancreas = union(pancreas, head, body, tail, lesion)
   (carved-out lesion voxels kept, no morphological ops); 2 lesion (overrides; identical to the Dataset501 binary label).
   nnU-Net regions: labels {background:0, pancreas:[1,2], lesion:2}, regions_class_order [1,2]. 11 negatives have all four
   pancreas masks empty: kept in validation, excluded from every fold's TRAIN list. Server has no raw CTs: build and
   preprocess on the PC, rsync nnUNet_preprocessed to the server, train there (GPUs 0,1 only).
10. tumorlib masks are uint8: `~mask` is a bitwise NOT (0 -> 255), not a logical NOT. Use np.logical_not or cast to
    bool first (S14 bug: emptied the near-duct region; caught by a regression check).
11. `cat fold_*_ids.txt` merges IDs (files have no trailing newline). Use tumorlib.io.case_ids().
12. Test suite: full `pytest` shows 3 failures + 2 errors in test_loader.py / test_lesion_ids.py that pass when run
    alone (likely S2's `common` rewriting tumorlib env paths at import). Run those files individually until isolated.
    Server venv has no pytest.
13. `python -O` strips asserts: a smoke test under -O does not exercise assert-guarded code (S8).

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
  Dataset502 training is finished; GPUs 0,1 are free.
- For CPU intensive tasks use the PC, for GPU intensive tasks use the server (only use the 2 2080 TIs)

## Rules
1. Oracle first: synthetic phantom or known-answer regression before any real-data run.
2. Never report a number you did not compute this session. Label carried-over numbers as carried over.
3. Nothing tuned on evaluation lesions. Detector settings are fixed a priori; anything learned uses
   leave-one-nnU-Net-fold-out. All reported results are 5-fold CV (out-of-fold). Gates are fixed before results
   and never relaxed after.
4. Units explicit (mm vs voxels). Bin edges: <5, 5-10, 10-20, 20-40, >=40 mm (Feret diameter).
5. results/ holds ONLY the named deliverables (one CSV + README per session). Intermediates go in a
   git-ignored work/ dir.
6. Hit definitions must be localization-aware: a predicted component touching the GT lesion. Report detection
   twice: touch AND per-lesion Dice >= 0.1 (S13: post-processing can create touches that are only grazes).
   Report the chance rate next to every hit rate (S11).
7. Standard reporting: estimated phase in S8b's 5 strata; specificity raw AND reweighted to the tumor+
   est_phase x thickness mix (S8b weights); FP per negative AND per tumor+ case in the region >10 mm from every
   GT lesion; with and without the 70 separated lesions.
8. Each session ends with README: what was computed, how, caveats, and what surprised you. You can name it as SESSION_{x}.md and put it in notes/LoG_notes/