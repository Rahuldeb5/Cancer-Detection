# nnU-Net v2 baseline — PanTS-Mini binary tumor segmentation

Baseline for comparison against the in-house 3D model. Trained on the **same
1308-case subset and the same stratified 5 folds** (`src/data/fold_[1-5]_ids.txt`),
binary target (`background=0`, `tumor=1`) built from `pancreatic_lesion.nii.gz`.

## Layout

| Path | What |
|------|------|
| `~/research/datasets/pants/images/*.tar.gz` | 10 source image shards (ids 1–9901), `PanTS_XXXXXXXX/ct.nii.gz` |
| `~/research/datasets/pants/masks/` | git-LFS checkout of `BodyMaps/iPanTSMini` |
| `~/research/datasets/pants/ct_staging/` | CTs extracted for the 1308 CV cases (intermediate, disposable) |
| `~/research/nnunet_env/{nnUNet_raw,nnUNet_preprocessed,nnUNet_results}` | nnU-Net dirs (from `env.sh`, sourced in `~/.bashrc`) |
| `~/research/nnunet_env/logs/` | run logs |

## One-time setup (done)

1. `pip install nnunetv2` into `.venv` (v2.8.1; installs fine on Python 3.14).
2. `git-lfs` 3.8.0 binary in `~/.local/bin` (`git lfs install`).
3. Env vars: `source ~/research/nnunet_env/env.sh` (also appended to `~/.bashrc`).

## Build pipeline (run on the PC)

```bash
source ~/research/nnunet_env/env.sh
cd ~/Cancer-Detection

# 1. CT images -> ct_staging/PanTS_XXXXXXXX/ct.nii.gz   (one sequential pass per shard)
bash src/nnunet/extract_cts.sh <(.venv/bin/python src/data/build_fold_id_list.py)

# 2. Lesion masks -> real files in the masks/ working tree (git-LFS batch pull; the
#    per-file HTTP route gets rate-limited unauthenticated, git-lfs's batch API doesn't)
cd ~/research/datasets/pants/masks && git lfs pull --include="**/pancreatic_lesion.nii.gz" && cd -

# 3. Assemble $nnUNet_raw/Dataset501_PanTSTumor (imagesTr, labelsTr, dataset.json).
#    - nibabel only for masks: the PanTS lesion files have scl_slope=NaN, which makes
#      SimpleITK's BinaryThreshold light up the WHOLE volume (silent all-ones labels).
#    - MAX_FG_FRACTION=0.30 guard rejects any implausible label.
#    - auto-orthonormalises 6 CTs whose direction cosines carry drift ITK rejects.
#    Wipe labelsTr first so nothing is skipped:  rm -rf $nnUNet_raw/Dataset501_PanTSTumor/labelsTr
.venv/bin/python src/nnunet/build_dataset.py
# verify: 981 positive / 327 empty labels, 0 mismatches vs metadata 'tumor?' column

# 4. Fingerprint + plan + preprocess, 3d_fullres only.
#    -np 3: image resampling is RAM-hungry; 15 GB WSL OOMs at the default -np 4.
nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres --verify_dataset_integrity -np 3 -npfp 3
#    (if interrupted after planning: nnUNetv2_preprocess -d 501 -c 3d_fullres -np 3 --no_pbar)

# 5. Overwrite nnU-Net's random split with our stratified folds
.venv/bin/python src/nnunet/make_splits.py
```

### Fold mapping

`fold_1_ids.txt` → nnU-Net fold **0** (its validation set), … `fold_5_ids.txt` → nnU-Net fold **4**.

### Generated 3d_fullres plan

| | |
|---|---|
| target spacing | `2.21 × 0.79 × 0.79` mm |
| patch size | `56 × 160 × 224` |
| batch size | `2` |
| network | `PlainConvUNet`, 6 stages, features `[32,64,128,256,320,320]`, **InstanceNorm** |
| CT normalization | clip `[-1000, 400]`, then z-score (µ≈−123, σ≈376) |
| median volume | `107 × 351 × 476` voxels |

## Server (`deep-server`, via Tailscale)

- 2× RTX 2080 Ti (GPU 0,1) + 2× GTX 1080 Ti (GPU 2,3), driver 535, 31 GB RAM, 667 GB free.
- venv: `~/nnunet_setup/.venv` (uv, Python 3.12, torch 2.5.1+cu121, nnunetv2 2.8.1, surface-distance).
- env: `source ~/research/nnunet_env/env.sh` (dirs + venv on PATH; also in `~/.bashrc`).

### Transfer preprocessed data PC → server

```bash
rsync -a --info=progress2 -e ssh \
  ~/research/nnunet_env/nnUNet_preprocessed/Dataset501_PanTSTumor \
  rahul@deep-server.tail8e65db.ts.net:~/research/nnunet_env/nnUNet_preprocessed/
```

`splits_final.json` and `dataset.json` travel inside that folder — `nnUNet_raw` is **not**
needed on the server (only for later `nnUNetv2_predict` on brand-new CTs).

### Training — DDP on the two 2080 Ti

```bash
bash src/nnunet/train_ddp.sh 0        # fold 0; crash-resilient wrapper, resumes with --c
```

The wrapper sets `CUDA_VISIBLE_DEVICES=0,1`, `OMP_NUM_THREADS=1`, `nnUNet_n_proc_DA=6`
and runs `nnUNetv2_train 501 3d_fullres 0 -num_gpus 2 --npz`. nnU-Net handles the DDP
spawn itself (no torchrun). `batch_size=2` total → 1/GPU (the DDP floor); InstanceNorm
so no BN-sync. `--npz` saves validation softmax for probability-based detection AUC.
Log: `~/research/nnunet_env/logs/train_501_3d_fullres_f0.log`.

Optional speedups (enable once a run is stable): uncomment `nnUNet_compile=1` in the
wrapper (~15–25% faster, Turing-supported).

If an 11 GB card OOMs (batch is already at the floor): re-plan smaller and train against it —
```bash
nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres -gpu_memory_target 9 \
  -overwrite_plans_name nnUNetPlans_9gb --no_pp
# then edit train_ddp.sh: add  -p nnUNetPlans_9gb
```

### Evaluation — full metric suite

nnU-Net's own end-of-training validation writes `fold_0/validation/*.nii.gz` (+ `.npz`)
and a `summary.json` with Dice/IoU only. For the complete suite:

```bash
source ~/research/nnunet_env/env.sh
R=$nnUNet_results/Dataset501_PanTSTumor/nnUNetTrainer__nnUNetPlans__3d_fullres
python src/nnunet/evaluate.py \
  --pred_dir $R/fold_0/validation \
  --gt_dir   $nnUNet_preprocessed/Dataset501_PanTSTumor/gt_segmentations \
  --out_dir  $R/fold_0/evaluation
```

Produces `evaluation.json` + `per_case_metrics.csv` + `per_lesion_metrics.csv`:
- **tumor segmentation** (over ref-positive cases): Dice, IoU, HD95, ASSD, NSD@{1,2,3} mm,
  relative volume difference, count missed entirely
- **case-level tumor detection** (all val cases): Sensitivity, Specificity, PPV, NPV, F1,
  Accuracy, Balanced Acc, AUROC + AUPRC — scored both by predicted tumor volume and by
  max tumor probability, at volume thresholds 0/50/100 mm³
- **size-stratified** (per GT lesion connected component, binned by 3D max/Feret diameter,
  edges 5/10/20/40 mm): lesion-wise Dice (all + detected-only), detection rate at
  any-overlap and IoU≥0.1, plus per-scan Dice binned by the scan's largest lesion.
  `src/nnunet/lesion_sizes.py --fold N` prints just the GT size distribution.

### Inference on new CTs

```bash
nnUNetv2_predict -i <ct_dir> -o <out_dir> -d 501 -c 3d_fullres -f 0   # needs nnUNet_raw dataset.json
```

## Cleanup

`bash src/nnunet/cleanup.sh` — see the script header. Run only after preprocessing
succeeds. `old/cache` (111 G, MedFormer-era) is printed but not auto-deleted.
