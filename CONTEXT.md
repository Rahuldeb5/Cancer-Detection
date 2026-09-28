# CONTEXT.md - small pancreatic tumor detection on PanTS
MSKCC collaboration. Author: Rahul (intermediate Python dev; wants to understand
each step - explain design choices in 1-2 sentences, run phantom/oracle checks BEFORE real data,
ask before changing existing scripts).

## Goal
Raise recall (and Dice/Iou) for small (<20 mm) (and possibly low-contrast although not as important) pancreatic tumors that nnU-Net misses, using a simple
interpretable candidate generator (scale-normalized LoG / Hessian blob map, pancreas-restricted) combined
with nnU-Net, judged by lesion-level FROC. The paper only exists if the blob map finds small tumors that
nnU-Net misses at EVERY nnU-Net threshold. A negative result is acceptable and reportable. 

## Some notes to keep in mind

Note that there is a paucity of lesions < 10 mm in diameter so it's more important we focus on the 10-20mm diameter lesions. Also note that some of the lesions landed outside the pancreas map and you can find results/lesion_data/lesion_location_results to tell you the excluded lesions and nnunet_excl_outside_pancreas to tell you the updated baseline without the excluded lesions (although that could make the sizes a bit different for each fold that's a little sketchy). Go through all the code files at the start of each session to get a good understanding.

## Cohort
- 1308 cases = union of src/data/fold_{1..5}_ids.txt: 981 tumor+ (1235 lesion components), 327 tumor-
  (phase-matched random negatives). Train split only (IDs 1-9000). ~75% positive vs ~10.9% in PanTS overall:
  sens/spec transfer, PPV/NPV/accuracy do not.
- Tumor label: results/PanTS_metadata_new.csv column `tumor?` (matches empty/non-empty lesion mask 1308/1308).
- pancreatic_lesion is NOT only PDAC (other lesion types likely). No diagnosis label located yet.
- fold_k_ids.txt <-> nnU-Net fold k-1 (fold_1 -> fold 0 ... fold_5 -> fold 4).
- Counts in older notes disagree (926 vs 981 positives; 309/326/327 negatives; median z-spacing 1.25/1.5/2.5 mm).
  Recompute from files; do not trust them.

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
- Probabilities look saturated (PanTS_00000020: GT negative, max prob 0.9994).
- Failure is NOT explained by z-undersampling (an old "49%" claim was wrong; 4-26% depending on definition).

## Attenuation (existing v2 labels, being audited in S3)
dHU = median tumor HU - median parenchyma HU (pool excludes lesion + 2.5 mm shell); hyper >+10, iso within +-10,
hypo <-10. Counts: hypo 561 / iso 503 / hyper 168 / unknown 3. Below 10 mm every class has Dice 0.000 (size, not
contrast); iso is worse only at 10-40 mm. Report-text attenuation is only 75.6% self-consistent - not truth.
Iso 40.7% looks high vs my recollection of the PDAC literature (~5-15%, unverified) - hence the audit.

## Location
Head holds ~50% of gland tissue but 63-70% of lesions. Small (<20 mm) tumors are found more often in the head
(zero-Dice rate 68% head vs 95% mid-gland). Only the PC1 head->tail axis is reliable; PC2/PC3 are not anatomical.
Lesion tiers vs pancreas envelope: inside 880, embedded 111, abutting 174, separated 70
(results/lesion_location_results/excluded_lesions.csv lists the 70 separated). Report results with and without them.

## Duct work (PARKED - ablation only)
Step-cutoff detector fires in 7.1% of tumor+ (0/327 neg): GT MPD masks are short fragments (median centerline
28 mm; 255 empty). Head-end dilation: 56.8% of evaluable tumor+ MPDs vs 5.1% of negatives, concentrated in
>=20 mm tumors. Component re-ranking (HistGB, leave-one-fold-out) at 90% recoverable: case spec raw prob 0.34,
+nnU-Net feats 0.57, +GT pancreas location 0.83, +GT duct 0.79 (duct adds ~0). All GT anatomy = upper bound.
Possible leakage in the pancreas-location gain (pancreas mask is carved at the tumor) - S6 tests this.
pdac_classification.csv (2 cm periampullary rule) is circular - do not use it to scope PDAC.

## Data gotchas (each one has already cost time)
1. PanTS masks have scl_slope=NaN. Read with nibabel img.dataobj.get_unscaled() (int8, no float64 upcast).
   NEVER SimpleITK BinaryThreshold (lights up the whole volume). Just check what we've implemented throughout.
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

## Machines
- Laptop (WSL, /home/rahuldeb5/Cancer-Detection, ssh rahuldeb5@100.81.217.58, but there is nothing extra there and it could be offline): code + result CSVs, NO dataset. IDE has twice overwritten
  edited files with a stale buffer: re-read from disk before trusting contents.
- PC -- this is what we are currently in (ssh rahuldeb5@100.103.52.84, RTX 5070, 15 GB RAM, 16 cores, py3.14): CTs in
  ~/research/datasets/pants/ct_staging; masks in .../masks/mask_only/{case}/segmentations/*.nii.gz.
  Has unrelated uncommitted work: only add files, never reset/commit others.
- Server (ssh rahul@deep-server.tail8e65db.ts.net, 12 cores, 31 GB, py at ~/nnunet_setup/.venv/bin/python;
  source ~/research/nnunet_env/env.sh): nnU-Net predictions/checkpoints, preprocessed data, gt_segmentations,
  6 masks/case in ~/research/datasets/pants/duct_masks/. Its repo has uncommitted work: move files with scp,
  never git pull/reset there. Untracked-on-server scripts (evaluate.py etc.) must be brought into git.
- For CPU intensive tasks use the PC, for GPU intensive tasks use the server (only use the 2 2080 TIs)

## Rules
1. Oracle first: synthetic phantom or known-answer regression before any real-data run.
2. Never report a number you did not compute this session. Label carried-over numbers as carried over.
3. Nothing tuned on evaluation lesions. Detector settings are fixed a priori; anything learned uses
   leave-one-nnU-Net-fold-out. Report the official test set only at the very end.
4. Units explicit (mm vs voxels). Bin edges: <5, 5-10, 10-20, 20-40, >=40 mm (Feret diameter).
5. results/ holds ONLY the named deliverables (one CSV + README per session). Intermediates go in a
   git-ignored work/ dir.
6. Hit definitions must be localization-aware (see S5).
7. Each session ends with README: what was computed, how, caveats, and what surprised you. You can name it as SESSION_{x}.md and put it in notes/LoG_notes/