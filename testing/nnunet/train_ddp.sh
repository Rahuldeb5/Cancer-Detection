#!/usr/bin/env bash
# Crash-resilient nnU-Net DDP training on the 2x RTX 2080 Ti (GPUs 0,1).
# Retries with --c (resume from checkpoint_latest) on any non-zero exit, so a
# transient CUDA/driver hiccup (these Turing cards do it) doesn't lose progress.
#
#   bash train_ddp.sh [FOLD]        # default FOLD=0
set -u

source "$HOME/research/nnunet_env/env.sh"

DATASET=501
CFG=3d_fullres
FOLD="${1:-0}"

export CUDA_VISIBLE_DEVICES=0,1          # the two 2080 Ti only
export OMP_NUM_THREADS=1                  # nnU-Net DDP recommendation
export nnUNet_n_proc_DA=6                 # 2 ranks x 6 = 12 DA workers = core count
# export nnUNet_compile=1                 # torch.compile: ~15-25% faster, enable once stable

LOG="$HOME/research/nnunet_env/logs/train_${DATASET}_${CFG}_f${FOLD}.log"
mkdir -p "$(dirname "$LOG")"

max_tries=30
cont=""
for ((try=1; try<=max_tries; try++)); do
  echo "=================== attempt $try  $(date -Is) ===================" | tee -a "$LOG"
  nnUNetv2_train "$DATASET" "$CFG" "$FOLD" -num_gpus 2 --npz $cont >>"$LOG" 2>&1
  rc=$?
  echo "=================== exit $rc  $(date -Is) ===================" | tee -a "$LOG"
  if [ $rc -eq 0 ]; then
    echo "training + final validation complete" | tee -a "$LOG"
    exit 0
  fi
  cont="--c"                              # resume on every subsequent attempt
  sleep 30
done
echo "gave up after $max_tries attempts" | tee -a "$LOG"
exit 1
