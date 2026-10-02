#!/usr/bin/env bash
# Crash-resilient nnU-Net DDP training on the 2x RTX 2080 Ti (GPUs 0,1).
#
# Same behaviour as testing/nnunet/train_ddp.sh (-num_gpus 2, --npz, 30 retries with
# --c, 30 s backoff, same OMP/DA settings); the dataset, trainer, configuration and
# retry count are overridable through the environment, with DATASET defaulting to 501
# so a bare call is byte-equivalent to the original script.
#
#   bash train_ddp_ds.sh [FOLD]                       # Dataset501, nnUNetTrainer
#   DATASET=502 bash train_ddp_ds.sh 0                # the S7 multi-class baseline
#   DATASET=502 TRAINER=nnUNetTrainer_1epoch MAX_TRIES=1 bash train_ddp_ds.sh 0
set -u

source "$HOME/research/nnunet_env/env.sh"

DATASET="${DATASET:-501}"
CFG="${CFG:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer}"
EPOCHS_NOTE="${TRAINER}"          # 1000 epochs is nnUNetTrainer's own default
MAX_TRIES="${MAX_TRIES:-30}"
BACKOFF="${BACKOFF:-30}"
FOLD="${1:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"   # the two 2080 Ti only
export OMP_NUM_THREADS=1                  # nnU-Net DDP recommendation
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-6}"             # 2 ranks x 6 = 12 = core count

LOG="${LOG:-$HOME/research/nnunet_env/logs/train_${DATASET}_${CFG}_${TRAINER}_f${FOLD}.log}"
mkdir -p "$(dirname "$LOG")"

echo "### dataset $DATASET  cfg $CFG  trainer $EPOCHS_NOTE  fold $FOLD  gpus $CUDA_VISIBLE_DEVICES" | tee -a "$LOG"

cont=""
for ((try=1; try<=MAX_TRIES; try++)); do
  echo "=================== attempt $try  $(date -Is) ===================" | tee -a "$LOG"
  nnUNetv2_train "$DATASET" "$CFG" "$FOLD" -tr "$TRAINER" -num_gpus 2 --npz $cont >>"$LOG" 2>&1
  rc=$?
  echo "=================== exit $rc  $(date -Is) ===================" | tee -a "$LOG"
  if [ $rc -eq 0 ]; then
    echo "training + final validation complete" | tee -a "$LOG"
    exit 0
  fi
  cont="--c"                              # resume on every subsequent attempt
  sleep "$BACKOFF"
done
echo "gave up after $MAX_TRIES attempts" | tee -a "$LOG"
exit 1
