# SESSION 7 - multi-class nnU-Net baseline (pancreas + lesion), Dataset502_PanTSPancLesion

Purpose: a stronger segmentation baseline for the >=20 mm work (CONTEXT.md priority 1).
Same 1308 cases, same CT grids and the **same 5 folds** as `Dataset501_PanTSTumor`, so
502 vs 501 is a controlled comparison of the label definition and nothing else.

The pancreas is a **training label only**. `channel_names` stays `{0: CT}`; there is no
pancreas input channel, because PanTS carves the lesion out of `pancreas.nii.gz` and an
input mask would hand the net the tumor location (CONTEXT.md gotcha 5).

Nothing in `src/nnunet/`, `testing/nnunet/`, Dataset501, its checkpoints or the S2 exports
was modified. S7 only adds files; `testing/nnunet/build_dataset.py` is *imported* for its
direction-cosine repair rather than copied or edited.

---

## 1. Label definition (fixed; implemented in `src/nnunet-multiclass/labels502.py`)

```
0  background
1  pancreas = union(pancreas, pancreas_head, pancreas_body, pancreas_tail, lesion)
2  lesion   (overrides 1; voxel-identical to the Dataset501 binary label)
```

No morphological operations: no closing, no hole filling, no dilation. The lesion is
unioned into class 1 on purpose, so the carved-out cavity in `pancreas.nii.gz` is filled
by the lesion itself and class 1 means "gland tissue", not "gland minus tumor".

`dataset.json` (region-based):

```json
{"channel_names": {"0": "CT"},
 "labels": {"background": 0, "pancreas": [1, 2], "lesion": 2},
 "regions_class_order": [1, 2],
 "numTraining": 1308, "file_ending": ".nii.gz",
 "dataset_name": "Dataset502_PanTSPancLesion"}
```

Masks are read only through `tumorlib.io.load_mask`: the two on-disk int8 encodings mean
`raw != 0` or `.astype(bool)` lights up whole volumes, and SimpleITK `BinaryThreshold`
has the same failure mode (CONTEXT.md gotcha 1). No SimpleITK thresholding anywhere in S7.

Phantom tests ran before any real scan was touched (`tests/test_ds502_labels.py`, 6 tests,
all pass): carved hole refilled, lesion override, separated lesion, empty envelope, the
{0,1} vs {0,255} encodings agreeing, and the region block in `dataset.json`.

### The nesting consequence (reported, not fixed)

Because "pancreas" is the region `{1, 2}`, every lesion voxel is also a pancreas-region
voxel. Measured on the real data: **all 70 S1 `separated` lesions lie entirely outside the
4-mask gland envelope (100% of their voxels), so all 70 are pancreas-region purely by
nesting.** They are not alone - see §3.

---

## 2. Disk (step 1)

| | free | Dataset501 reference | Dataset502 actual / expected | 1.5x need | verdict |
|---|---|---|---|---|---|
| PC `/dev/sdd` | 270 G | raw 37 G, preprocessed 46 G | raw: 47 M labels + 0 bytes images (hard-linked from 501); preprocessed ~46 G | 70 G | OK |
| server `/dev/sda2` | 544 G | preprocessed 46 G, results 67 G | preprocessed 50 G + results **~135 G** (revised, see below) | 278 G | OK |

Measured afterwards: preprocessed is **50 G** (501: 46 G), and the deleted 1-epoch
rehearsal showed one fold's `validation/` costs **27 G** (~102 MB/case) against 501's
~13 G/fold, because 502's npz carries a structured pancreas channel where 501's tumor
channel was almost all zeros and compressed away. So expect **~135 G** of results, not
67 G. Server free space after the rehearsal was deleted and the real run started: 494 G.

`imagesTr` costs nothing: 1302 of 1308 files are hard links to Dataset501's `imagesTr`
(which are themselves hard links to `ct_staging`), and the 6 direction-cosine-repaired CTs
are hard-linked from Dataset501's already-repaired copies. `du` reports 11 G for the
directory but `df` did not move.

---

## 3. Oracles on all 1308 cases, before preprocessing (step 3)

`src/nnunet-multiclass/oracle502.py` -> `work/nnunet_ds502/oracle_summary.json`,
`oracle_per_case.csv`, `oracle_per_lesion.csv`.

**Hard checks - all pass, 0 cases with problems:**

| check | result |
|---|---|
| `(label == 2)` vs the Dataset501 binary label, voxel for voxel | **0 mismatching voxels, 0 cases** |
| label values subset of {0,1,2}; dtype uint8; scl_slope/inter clean | pass, 1308/1308 |
| `(label > 0)` vs a fresh recomposition from the 5 source masks | **0 mismatching voxels, 0 cases** |
| label on the image grid (shape + affine, atol 1e-6) | pass, 1308/1308 |
| pancreas region fraction of scan < 10% | max **2.17%** |
| lesion fraction of scan < 30% | max **1.71%** |

Label 2 non-empty in 981 cases, empty in 327 - exactly the cohort's 981/327.

**Pancreas volume (class 1 + class 2, i.e. the whole gland region), cm^3:**

| min | p5 | median | mean | p95 | max |
|---|---|---|---|---|---|
| 0.00 | 28.01 | **78.03** | 84.23 | 156.53 | 822.69 |

Outliers: **40 cases < 20 cm^3** (11 of them are the empty-envelope cases at 0.00; the next
four are near-empty: `PanTS_00006379` with **1 voxel**, `PanTS_00006166` 109 vox,
`PanTS_00003123` 527 vox, `PanTS_00009147` 1724 vox - all four are negatives). **7 cases
> 250 cm^3**, and 6 of those 7 are tumor-dominated, not gland-segmentation errors:

| case | region cm^3 | lesion cm^3 | lesion share |
|---|---|---|---|
| PanTS_00002720 | 822.7 | 732.4 | 0.89 |
| PanTS_00002205 | 512.5 | 2.2 | **0.00** |
| PanTS_00005420 | 401.0 | 300.5 | 0.75 |
| PanTS_00009367 | 356.7 | 304.0 | 0.85 |
| PanTS_00000246 | 336.3 | 261.9 | 0.78 |
| PanTS_00002102 | 298.6 | 2.7 | **0.01** |
| PanTS_00009726 | 259.2 | 147.9 | 0.57 |

`PanTS_00002205` (512 cm^3) and `PanTS_00002102` (299 cm^3) are the two genuinely suspect
gland masks. Flagged, not touched.

**Empty-envelope cases: 11, all negatives, no positives among them** - exactly the 11 known
from S2:
`PanTS_00001333, 00001560, 00003199, 00004859, 00005016, 00006797, 00007273, 00007760,
00008326, 00008619, 00008926`.

**Lesion voxels outside the gland masks** (join to `master_lesions.csv` on
`(case_id, lesion_id)`: 1235 of 1235 rows matched, 0 `gt_vox` disagreements, so the
lesion_id key is sound):

Pooled over all 10,663,814 lesion voxels: **44.8% lie outside the 4-mask envelope** and
**60.3% outside the original `pancreas.nii.gz`**. By S1 tier:

| tier | n lesions | with any voxel outside envelope | % of voxels outside envelope | % outside `pancreas.nii.gz` | entirely outside envelope |
|---|---|---|---|---|---|
| inside | 880 | 520 | 14.8% | 37.1% | 0 |
| embedded | 111 | 111 | 78.5% | 91.1% | 44 |
| abutting | 174 | 174 | 85.1% | 91.1% | 47 |
| separated | 70 | 70 | **100.0%** | 100.0% | **70** |

Case level: 683 of 981 positives have at least one lesion voxel outside the envelope; 44
cases have *all* their lesion voxels outside it.

**This was the surprise of the session.** The S1 tiers are not containment statements: 520
of the 880 `inside` lesions still have voxels outside the union of all four gland masks,
and by size the leakage is worst at both ends (fraction of voxels outside the envelope:
<5 mm 76.4%, 5-10 mm 50.5%, 10-20 mm 25.7%, 20-40 mm 25.9%, >=40 mm 51.1%). The practical
consequence for 502 is that class 1 ("pancreas") picks up a substantial volume of tissue
that no gland mask covers, wherever a tumor bulges out of the gland. That is the fixed
definition doing what it was told; it is a caveat for any S8 "pancreas Dice" number, which
is therefore not a clean gland-segmentation metric.

---

## 4. Plan diff vs Dataset501, 3d_fullres (step 4)

`nnUNetv2_plan_and_preprocess -d 502 -c 3d_fullres -np 3 --verify_dataset_integrity`,
default planner, no overrides. `np 3` because `np 4` OOM-killed this 15 GB WSL box once.
Integrity verification passed. Full diff in `work/nnunet_ds502/plans_diff.txt`.

**No material difference. Nothing to STOP for.**

| field | Dataset501 | Dataset502 |
|---|---|---|
| target spacing | [2.2100000381469727, 0.7910159826278687, 0.7949219942092896] | **identical** |
| patch size | [56, 160, 224] | **identical** |
| batch size | 2 | **identical** |
| architecture | PlainConvUNet, 6 stages, (32,64,128,256,320,320), InstanceNorm3d, LeakyReLU, kernels [[1,3,3],[3,3,3]x5], strides [[1,1,1],[1,2,2],[2,2,2],[2,2,2],[2,2,2],[1,2,2]] | **identical** |
| normalization scheme | CTNormalization, `use_mask_for_norm` False | **identical** |
| `batch_dice` | True | **identical** |
| median image size | [107, 351, 476] | **identical** |

The only planning difference is the CT normalization statistics, which **must** differ:
they are foreground statistics, and the foreground is now the whole gland rather than only
tumor.

| | mean | std | p0.5 (clip low) | p99.5 (clip high) | median |
|---|---|---|---|---|---|
| Dataset501 | 57.7793 | 48.0313 | -86 | 184 | 57 |
| Dataset502 | 63.6232 | 57.8832 | -111 | 206 | 68 |

CTNormalization clips to [p0.5, p99.5] then z-scores with that mean/std, so **the two
baselines see slightly differently normalized CT intensities**. This is unavoidable under
default planning and is the one non-label difference between 501 and 502; note it when
comparing lesion Dice in S8.

---

## 5. Splits (step 5)

`src/nnunet-multiclass/make_splits502.py` copies Dataset501's `splits_final.json`
(fold_k_ids.txt <-> nnU-Net fold k-1) and removes the 11 empty-envelope cases from every
fold's **TRAIN** list only. They stay in validation, so the 502 validation sets are
byte-identical to 501's and the two baselines are scored on exactly the same cases.

Asserts that passed: folds disjoint; every val list byte-identical to Dataset501's; no
excluded case in any train list; no train/val overlap; each excluded case appears in
exactly one validation list; 1308 unique cases over the val lists.

| nnU-Net fold | train 501 | train 502 | dropped | val |
|---|---|---|---|---|
| 0 | 1044 | **1035** | 9 | 264 |
| 1 | 1045 | **1035** | 10 | 263 |
| 2 | 1047 | **1039** | 8 | 261 |
| 3 | 1048 | **1039** | 9 | 260 |
| 4 | 1048 | **1040** | 8 | 260 |

Rationale: a case whose four gland masks are all empty has no pancreas label at all;
training on it teaches "this abdomen contains no pancreas", which is a labelling gap, not a
fact. Validating on it is still meaningful (all 11 are tumor-negative, so a correct
prediction is "no lesion").

Judgment call left open: the four *near*-empty envelopes (1, 109, 527 and 1724 voxels;
`PanTS_00006379`, `00006166`, `00003123`, `00009147`, all negatives) were **kept** in
training, because the brief says to drop empty envelopes and these are not empty. Worth
revisiting if class-1 training looks odd.

---

## 6. Foreground patch sampling with two classes - a known confound (observe only)

Read from the installed nnunetv2 2.8.1 on the server; **nothing was modified**.

`nnUNetDataLoader.get_bbox` (`training/dataloading/data_loader.py`) on a forced-foreground
patch builds `eligible_classes_or_regions` from the keys of `properties['class_locations']`
(dropping the `annotated_classes_key`) and picks one **uniformly at random**, then centres
the patch on a random voxel of that class. `class_locations` keys come from
`DefaultPreprocessor._sample_foreground_locations`, which uses
`label_manager.foreground_regions if has_regions else foreground_labels`.

| | eligible keys on a forced-fg patch | P(patch is centred on a lesion voxel \| forced fg) |
|---|---|---|
| Dataset501 (binary, no regions) | `{1}` -> 1 key | **1.0** |
| Dataset502 (region-based) | `{(1,2), 2}` -> 2 keys | **0.5** |

With `oversample_foreground_percent = 0.33` and `batch_size = 2`,
`_oversample_last_XX_percent` forces foreground on `round(2 * (1 - 0.33)) = 1` of the 2
samples, i.e. exactly one patch per batch. So:

- Dataset501: ~1 of 2 patches per batch is lesion-centred (~50%).
- Dataset502: ~1 of 4 patches per batch is lesion-centred (~25%); the other forced-fg
  patch is centred on a random voxel of the `(1,2)` region, which is overwhelmingly
  non-tumor gland (median gland 78 cm^3 vs a median lesion far smaller).

**Lesion-centred patch exposure is therefore roughly halved relative to the binary
baseline.** For <20 mm lesions, which already get Dice 0.000-0.074 in the 501 baseline,
this cuts the wrong way, and it is a confound for any "multi-class is better/worse" reading
of S8: part of any lesion-Dice change is a sampling-rate change, not a representation
change. `class_locations` is also subsampled (`min_num_samples=10000`,
`min_percent_coverage=0.01`, seed 1234) per class independently, so small lesions are not
additionally penalised within their own key.

(If S8 wants to remove this confound, the knob is a trainer with
`probabilistic_oversampling` or an `overwrite_class` forced to the lesion region - NOT
attempted in S7.)

---

## 7. Region-based `.npz` layout, verified (step 9)

For a region-based dataset nnU-Net applies a **sigmoid**, not a softmax, and builds the
saved segmentation with
`for i, c in enumerate(regions_class_order): seg[prob[i] > 0.5] = c`
(`nnunetv2/utilities/label_handling/label_handling.py`). `regions` follow the insertion
order of `dataset.json["labels"]` minus background.

**Confirmed against nnU-Net's own `LabelManager`, built from Dataset502's real
`dataset.json`** (`src/nnunet-multiclass/oracle_regions.py`, passes):

```
LabelManager.has_regions        : True
LabelManager.foreground_regions : [(1, 2), 2]
num_segmentation_heads          : 2
inference_nonlin                : torch.sigmoid        <- not softmax
```

- `npz['probabilities']` channel **0** = P(pancreas region = gland **including** tumor)
- `npz['probabilities']` channel **1** = P(lesion)
- array layout is **`(C, z, y, x)`**, float32; channel c in nibabel (x,y,z) order is
  `prob[c].transpose(2, 1, 0)` - same convention as S2's Dataset501 exports
- channels are independent sigmoids and **do not sum to 1** (`sigmoid([2,2])` sums to 1.76;
  measured channel sums on real exports ranged 0.00-0.26 for the untrained rehearsal net)

Known-answer table driven through the real `convert_probabilities_to_segmentation`:

| p(pancreas) | p(lesion) | seg | why |
|---|---|---|---|
| 0.10 | 0.10 | 0 | neither region fires |
| 0.90 | 0.10 | 1 | pancreas only |
| 0.90 | 0.90 | **2** | lesion assigned last, overrides |
| 0.10 | 0.90 | **2** | lesion fires while pancreas does not - channel 0 need not agree |

and on that non-empty segmentation both equalities hold exactly:
`(channel 1 > 0.5) == (seg == 2)` and `(channel0>0.5 | channel1>0.5) == (seg > 0)`.

### Short-trainer rehearsal on fold 0 (run, verified, then deleted)

`nnUNetTrainer_1epoch` exists in the installed nnunetv2 2.8.1
(`variants/training_length/nnUNetTrainer_Xepochs.py`), so the END of the run was exercised
first: `DATASET=502 TRAINER=nnUNetTrainer_1epoch MAX_TRIES=1 bash train_ddp_ds.sh 0`.

Result: **exit 0, "training + final validation complete"**, all **264** `.nii.gz` + **264**
`.npz` exported plus `summary.json`. `summary.json` reports the two regions separately -
`region (1,2)` with `n_ref` 83,730 voxels/case and `region 2` with 8,829 - which
independently confirms channel 0 is the gland and channel 1 the lesion. Both Dices were
0.0, as expected after one epoch. The 27 G results folder was then deleted.

`verify_npz_regions.py --n 3` on those exports passed every structural check (keys, 4D
shape, `(C,z,y,x)` order, 2 channels, channel sums, both equalities). **Caveat, stated
plainly: at 1 epoch the net's maximum probability was 0.16, so the saved segmentation was
empty and the two equality checks were satisfied trivially (0 == 0).** That is exactly why
`oracle_regions.py` above was added - it proves the mapping without a trained model. Re-run
`verify_npz_regions.py` on the real fold 0 `validation/` when it appears (~14 h after
launch) for the non-vacuous confirmation on real predictions.

### torch.compile: checked, not a new risk

The rehearsal logged *"WARNING! batch size is 1 during training and torch.compile is
enabled ... If you encounter crashes in validation ... rerun with --val"*. Investigated
rather than acted on:

- nnunetv2 2.8.1 **defaults `nnUNet_compile` to True** when the env var is unset
  (`nnUNetTrainer.py:303-306`), and it is set nowhere in `env.sh` or the shell profile.
- Dataset501's own training logs contain the same *"nnUNet_compile is enabled"* line, so
  **the 501 baseline was trained with compile on too**, under the same batch 2 / 2 GPUs
  (per-GPU batch 1) condition, and its validation completed on all 5 folds.
- The 502 rehearsal's validation then also completed, exit 0, 264/264 exported.

So the warning is advisory here and 501 vs 502 are comparable on this axis as well. Nothing
was changed. If a fold *does* crash in final validation, the documented fix is to re-run
that fold with `--val`; `train_ddp_ds.sh` currently retries with `--c`, which is the right
move for a mid-training crash. Noted, not pre-emptively changed.

## 8. Commands

Everything below assumes `source ~/research/nnunet_env/env.sh`.

**PC (build / preprocess / splits / sync):**

```bash
cd ~/Cancer-Detection
.venv/bin/python -m pytest tests/test_ds502_labels.py -q            # phantoms first
.venv/bin/python src/nnunet-multiclass/build_dataset502.py --nproc 3
.venv/bin/python src/nnunet-multiclass/oracle502.py --nproc 3
bash src/nnunet-multiclass/preprocess_502.sh                        # -d 502 -c 3d_fullres -np 3
.venv/bin/python src/nnunet-multiclass/diff_plans.py -c 3d_fullres
.venv/bin/python src/nnunet-multiclass/make_splits502.py
bash src/nnunet-multiclass/sync_preprocessed_502.sh                 # --dry-run first if unsure
```

**Server (train):**

```bash
ssh rahul@deep-server.tail8e65db.ts.net
source ~/research/nnunet_env/env.sh && cd ~/Cancer-Detection
nvidia-smi                                                          # GPUs 0,1 must be idle
python src/nnunet-multiclass/oracle_regions.py                      # channel-order oracle

# short-trainer end-to-end rehearsal on fold 0 (deleted afterwards)
DATASET=502 TRAINER=nnUNetTrainer_1epoch MAX_TRIES=1 \
  bash src/nnunet-multiclass/train_ddp_ds.sh 0
python src/nnunet-multiclass/verify_npz_regions.py --n 3 --pred_dir \
  $nnUNet_results/Dataset502_PanTSPancLesion/nnUNetTrainer_1epoch__nnUNetPlans__3d_fullres/fold_0/validation

# the real run: folds 0 -> 4 sequentially, detached, resumable
nohup bash src/nnunet-multiclass/run_folds502.sh </dev/null \
      > ~/research/nnunet_env/logs/run_folds502.log 2>&1 &
```

**Monitor / resume:**

```bash
bash src/nnunet-multiclass/status.sh      # fold, epoch, losses, pseudo-dice, ETA, retries, GPU, disk
tail -f ~/research/nnunet_env/logs/run_folds502.log
tail -f ~/research/nnunet_env/logs/train_502_3d_fullres_nnUNetTrainer_f0.log

# non-vacuous re-check of the .npz equalities once fold 0's real validation exists (~14 h)
python src/nnunet-multiclass/verify_npz_regions.py --n 3 --pred_dir \
  $nnUNet_results/Dataset502_PanTSPancLesion/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation
```

Resume after any interruption: re-run the same `nohup bash run_folds502.sh` line. The
driver skips any fold whose `validation/summary.json` exists, `flock` refuses a second
copy, and each fold retries up to 30 times with `--c` and a 30 s backoff.

## 9. Expected output paths

```
PC:
  $nnUNet_raw/Dataset502_PanTSPancLesion/{imagesTr,labelsTr,dataset.json}
  $nnUNet_preprocessed/Dataset502_PanTSPancLesion/
      nnUNetPlans.json  dataset.json  dataset_fingerprint.json  splits_final.json
      nnUNetPlans_3d_fullres/{case}.b2nd + {case}_seg.b2nd + {case}.pkl   (blosc2, not npy)
      gt_segmentations/{case}.nii.gz
  work/nnunet_ds502/   (git-ignored: build.log, oracle*.csv/json, plans_diff.txt, splits.txt,
                        sync.log, sync_spotcheck.txt)

Server:
  $nnUNet_preprocessed/Dataset502_PanTSPancLesion/   (rsynced from the PC)
  $nnUNet_results/Dataset502_PanTSPancLesion/nnUNetTrainer__nnUNetPlans__3d_fullres/
      fold_{0..4}/checkpoint_{best,final,latest}.pth
      fold_{0..4}/training_log_*.txt  progress.png  debug.json
      fold_{0..4}/validation/{case}.nii.gz + {case}.npz + summary.json
  ~/research/nnunet_env/logs/run_folds502.log, train_502_3d_fullres_nnUNetTrainer_f{0..4}.log
```

S8 (evaluation) reads `fold_{0..4}/validation/` against
`$nnUNet_preprocessed/Dataset502_PanTSPancLesion/gt_segmentations/`.

---

## 10. rsync to the server, verified (step 6)

`bash src/nnunet-multiclass/sync_preprocessed_502.sh`, ~11 min at ~78 MB/s.

| | PC | server |
|---|---|---|
| files | 5236 | **5236** |
| total file bytes | 52,756,079,446 | **52,756,079,446** (exact match) |
| `nnUNetPlans_3d_fullres` entries | 3924 (1308 x `.b2nd` + `_seg.b2nd` + `.pkl`) | **3924** |
| `gt_segmentations` | 1308 | **1308** |
| `dataset.json` md5 | 2e0fa65687ac4172d43541f639dea0d2 | **same** |
| `nnUNetPlans.json` md5 | eff4def431d828f213bd566d69caab89 | **same** |
| `dataset_fingerprint.json` md5 | bdbd5e0de325a7e75a1bd165c8817dea | **same** |
| `splits_final.json` md5 | 315bafa5d409de6f52b5134b3c4cc8db | **same** |

Spot-check of 5 cases (`PanTS_00000003, 00000012, 00000020, 00000026, 00000029`): every
preprocessed file and every `gt_segmentations` file md5-identical.

Note: nnunetv2 2.8.1 writes preprocessed arrays as **blosc2 `.b2nd`**, not `.npy`/`.npz`.
`du -sb` differs by 229 KB between the two machines - that is directory-inode overhead, not
content; the summed file bytes match exactly.

Two things to know about this script: it had a `set -e` + `ls | head` SIGPIPE bug that
killed it silently right after the transfer and before the md5 spot-check; fixed (now
`find -printf | sort | head`, and `-e` dropped). The transfer itself had already completed
and was verified by hand.

---

## 11. Real run: launched and verified (step 10)

Launched 2026-10-02 13:28:17 EDT, detached, `CUDA_VISIBLE_DEVICES=0,1`:

```
nohup bash src/nnunet-multiclass/run_folds502.sh </dev/null \
      > ~/research/nnunet_env/logs/run_folds502.log 2>&1 &
```

Pre-launch: `nvidia-smi` showed **no compute processes at all** on any GPU (the 133 MiB
resident on GPU 0 is not a compute app). Post-launch: exactly **one** `run_folds502.sh`
(pid 1621813) and exactly **one** `nnUNetv2_train 502 3d_fullres 0 -tr nnUNetTrainer
-num_gpus 2 --npz` - no duplicate process. A `flock` on
`logs/run_folds_502_3d_fullres.lock` refuses a second driver.

**Fold 0 health, first 5 epochs - loss finite and falling, per-region pseudo-Dice logged
and rising, both GPUs busy:**

| epoch | train_loss | val_loss | pseudo-Dice [pancreas, lesion] | epoch time |
|---|---|---|---|---|
| 0 | 0.0886 | 0.0010 | [0.0000, 0.0000] | 74.7 s (compile warm-up) |
| 1 | -0.0424 | -0.0538 | [0.0013, 0.0000] | 52.55 s |
| 2 | -0.0812 | -0.1420 | [0.3688, 0.0000] | 47.01 s |
| 3 | -0.1277 | -0.1486 | [0.3579, 0.1142] | 46.54 s |
| 4 | -0.1496 | -0.2152 | [0.4492, 0.2024] | 46.46 s |

The pseudo-Dice list has **two** entries, in region order: `[region (1,2) = pancreas,
region 2 = lesion]`. Pancreas learns first (0.45 by epoch 4), lesion follows (0.20) - the
expected ordering, and a useful live signal that the multi-class target is wired correctly.

GPU 0: 4164 MiB / 98%; GPU 1: 4036 MiB / 97%. GPUs 2 and 3 (the 1080 Ti) at 6 MiB / 0% -
untouched, as required. Disk 494 G free.

**ETA.** Steady-state epoch time **46.5 s** (Dataset501's median was 48.11 s over its 1000
epochs, so 502 is marginally faster despite the extra output channel):
1000 x 46.5 s = **12.9 h** training + ~0.9 h final validation = **~13.8 h per fold**,
**~2.9 days** for all 5 folds. Slightly under the brief's 14 h 45 min / 3.5 days estimate.

Session ended here, as instructed; the run was not waited on.


---

## 12. New files (all uncommitted)

```
src/nnunet-multiclass/labels502.py              label definition + dataset.json (pure, tested)
src/nnunet-multiclass/build_dataset502.py       builds imagesTr/labelsTr/dataset.json
src/nnunet-multiclass/oracle502.py              the all-1308 pre-preprocessing oracles
src/nnunet-multiclass/preprocess_502.sh         plan_and_preprocess -d 502 -c 3d_fullres -np 3
src/nnunet-multiclass/diff_plans.py             502 vs 501 plan diff, exits 2 on a material diff
src/nnunet-multiclass/make_splits502.py         splits_final.json (drop empty envelopes from TRAIN)
src/nnunet-multiclass/sync_preprocessed_502.sh  rsync to the server + verification
src/nnunet-multiclass/train_ddp_ds.sh           train_ddp.sh with DATASET/TRAINER/CFG overridable
src/nnunet-multiclass/run_folds502.sh           folds 0->4 driver: resumable, flock, detached
src/nnunet-multiclass/status.sh                 one-screen run status
src/nnunet-multiclass/verify_npz_regions.py     region .npz layout check on real exports
src/nnunet-multiclass/oracle_regions.py         known-answer channel-order oracle via LabelManager
tests/test_ds502_labels.py                      6 phantom tests for the label definition
notes/LoG_notes/SESSION_7.md                    this file
```

No files under `results/` were created or changed (rule 5). Intermediates are in the
git-ignored `work/nnunet_ds502/`: `build.log`, `oracle.log`, `oracle_summary.json`,
`oracle_per_case.csv`, `oracle_per_lesion.csv`, `plans_diff.txt`, `splits.txt`, `sync.log`,
`sync_spotcheck.txt`, `preprocess_launch.log`.

Nothing existing was edited. `testing/nnunet/build_dataset.py` is imported (for
`is_orthonormal` / `orthonormalized`), not modified; Dataset501, its checkpoints and the S2
exports were only read. The S7 scripts were `rsync`-ed to the server's
`~/Cancer-Detection/src/nnunet-multiclass/` so the run could be launched there; they are
uncommitted on both machines.

---

## 13. What surprised me

1. **The S1 tiers are not containment statements.** 520 of the 880 `inside` lesions still
   have voxels outside the union of all four gland masks, and 44.8% of all lesion voxels
   lie outside it. I expected the envelope to contain `inside` lesions by definition. The
   consequence is that class 1 in Dataset502 absorbs a lot of tissue no gland mask covers,
   so an S8 "pancreas Dice" is not a clean gland-segmentation number.
2. **Foreground patch sampling halves lesion exposure** (§6). This was predictable from the
   source but is easy to miss, and it works directly against the 10-20 mm priority.
3. **Dataset501 was already trained with `torch.compile` on** - it defaults to True in
   2.8.1. I expected it off, because `train_ddp.sh` carefully comments the export out.
4. **Results are ~2x larger per fold than 501's** (27 G vs ~13 G) purely because the
   pancreas probability channel does not compress to nothing the way the tumor channel did.
5. The plan came out **bit-identical** on spacing, patch, batch and architecture. Only the
   CT normalization statistics moved, and they had to.
