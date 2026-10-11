#!/usr/bin/env bash
# S15 Part B step 7 (SERVER): Dataset502 5-fold ensemble inference on the 300 S9 normal-bank CTs, GPUs 0 and 1 only.
# The bank CTs are not in the cohort, so every fold model may be used. Default nnU-Net settings (mirroring TTA on,
# step 0.5), like the cohort's out-of-fold validation predictions. Probabilities are saved so that bank_extract.py
# can keep the pancreas-region channel only.
#   inputs : ~/research/gland_shape_bank/ct/{case}/ct.nii.gz   (rsynced from the PC's ct_normal_bank/)
#   outputs: ~/research/gland_shape_bank/pred/{case}.npz/.nii.gz/.pkl, logs in ~/research/gland_shape_bank/logs/
# Usage (server): bash bank_predict.sh            (launches both halves in the background)
set -euo pipefail
source ~/research/nnunet_env/env.sh
ROOT=~/research/gland_shape_bank
mkdir -p "$ROOT/imagesTs" "$ROOT/pred" "$ROOT/logs"
for d in "$ROOT"/ct/PanTS_*; do
  c=$(basename "$d")
  ln -sf "$d/ct.nii.gz" "$ROOT/imagesTs/${c}_0000.nii.gz"
done
echo "inputs: $(ls "$ROOT/imagesTs" | wc -l)"
for k in 0 1; do
  CUDA_VISIBLE_DEVICES=$k nohup nnUNetv2_predict -i "$ROOT/imagesTs" -o "$ROOT/pred" -d 502 -c 3d_fullres \
    -f 0 1 2 3 4 --save_probabilities -npp 2 -nps 2 -num_parts 2 -part_id $k -device cuda \
    > "$ROOT/logs/predict_part$k.log" 2>&1 &
  echo "part $k on GPU $k: pid $!"
done
