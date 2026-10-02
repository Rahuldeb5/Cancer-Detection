#!/usr/bin/env bash
# S7 driver: Dataset502 folds 0 -> 4, sequentially, resumable, fully detached.
#
# A fold is skipped when its validation/summary.json already exists (nnU-Net writes that
# only after training AND final validation have finished), so re-running this script
# after a reboot continues where it stopped. Each fold itself retries up to 30 times
# with --c inside train_ddp_ds.sh.
#
# Launch detached (the only supported way -- an ssh session must not own it):
#   source ~/research/nnunet_env/env.sh
#   cd ~/Cancer-Detection
#   nohup bash src/nnunet-multiclass/run_folds502.sh </dev/null \
#         > ~/research/nnunet_env/logs/run_folds502.log 2>&1 &
set -u

source "$HOME/research/nnunet_env/env.sh"

DATASET="${DATASET:-502}"
DS_NAME="${DS_NAME:-Dataset502_PanTSPancLesion}"
CFG="${CFG:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer}"
FOLDS="${FOLDS:-0 1 2 3 4}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESDIR="$nnUNet_results/$DS_NAME/${TRAINER}__nnUNetPlans__${CFG}"

# refuse to start a second copy
LOCK="$HOME/research/nnunet_env/logs/run_folds_${DATASET}_${CFG}.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another run_folds502.sh already holds $LOCK -- refusing to start a duplicate"
  exit 1
fi
echo $$ >&9

echo "=== S7 driver  $(date -Is)  dataset $DATASET  cfg $CFG  trainer $TRAINER  pid $$ ==="
for FOLD in $FOLDS; do
  done_marker="$RESDIR/fold_${FOLD}/validation/summary.json"
  if [ -f "$done_marker" ]; then
    echo "--- fold $FOLD: SKIP, $done_marker exists  $(date -Is)"
    continue
  fi
  echo "--- fold $FOLD: START  $(date -Is)"
  DATASET="$DATASET" CFG="$CFG" TRAINER="$TRAINER" bash "$HERE/train_ddp_ds.sh" "$FOLD"
  rc=$?
  echo "--- fold $FOLD: END rc=$rc  $(date -Is)"
  if [ $rc -ne 0 ]; then
    echo "fold $FOLD gave up; stopping the driver so the failure is not hidden by later folds"
    exit $rc
  fi
done
echo "=== all folds done  $(date -Is) ==="
