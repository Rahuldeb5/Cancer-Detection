#!/usr/bin/env bash
# One-screen status of the S7 Dataset502 run. Read-only.
#   bash src/nnunet-multiclass/status.sh
set -u
source "$HOME/research/nnunet_env/env.sh"

DS_NAME="${DS_NAME:-Dataset502_PanTSPancLesion}"
CFG="${CFG:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer}"
DATASET="${DATASET:-502}"
RESDIR="$nnUNet_results/$DS_NAME/${TRAINER}__nnUNetPlans__${CFG}"
LOGDIR="$HOME/research/nnunet_env/logs"

echo "=== $DS_NAME / $TRAINER / $CFG   $(date -Is) ==="
echo
echo "-- driver --"
if pgrep -af "run_folds502.sh" | grep -v grep >/dev/null; then
  pgrep -af "run_folds502.sh" | grep -v grep
else
  echo "  no run_folds502.sh process"
fi
n=$(pgrep -cf "nnUNetv2_train $DATASET" || true)
echo "  nnUNetv2_train processes for dataset $DATASET: ${n:-0}"
echo
printf "%-7s %-6s %-9s %-11s %-11s %-9s %s\n" fold epoch last_loss val_loss pseudo_dice retries eta
for f in 0 1 2 3 4; do
  log=$(ls -t "$RESDIR/fold_${f}/training_log_"*.txt 2>/dev/null | head -1)
  if [ -z "${log:-}" ]; then
    state="-"
    [ -f "$RESDIR/fold_${f}/validation/summary.json" ] && state="DONE"
    printf "%-7s %-6s %-9s %-11s %-11s %-9s %s\n" "$f" "$state" - - - - -
    continue
  fi
  ep=$(grep -oP '^.*Epoch \K[0-9]+' "$log" | tail -1)
  tl=$(grep -oP 'train_loss \K[-0-9.]+' "$log" | tail -1)
  vl=$(grep -oP 'val_loss \K[-0-9.]+' "$log" | tail -1)
  pd=$(grep -oP 'Pseudo dice \K.*' "$log" | tail -1)
  et=$(grep -oP 'Epoch time: \K[0-9.]+' "$log" | tail -1)
  rt=$(grep -c '^### dataset' "$LOGDIR/train_${DATASET}_${CFG}_${TRAINER}_f${f}.log" 2>/dev/null || echo 0)
  at=$(grep -c '=== attempt' "$LOGDIR/train_${DATASET}_${CFG}_${TRAINER}_f${f}.log" 2>/dev/null || echo 0)
  eta="-"
  if [ -n "${ep:-}" ] && [ -n "${et:-}" ]; then
    eta=$(awk -v e="$ep" -v t="$et" 'BEGIN{r=(999-e)*t; printf "%.1f h left", r/3600}')
  fi
  [ -f "$RESDIR/fold_${f}/validation/summary.json" ] && eta="DONE (validation written)"
  printf "%-7s %-6s %-9s %-11s %-11s %-9s %s\n" "$f" "${ep:--}" "${tl:--}" "${vl:--}" "${pd:--}" "${at:-0}" "$eta"
done
echo
echo "-- GPU --"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
echo
echo "-- disk --"
df -h "$nnUNet_results" | tail -1
echo
echo "-- last 3 log lines of the newest fold log --"
newest=$(ls -t "$RESDIR"/fold_*/training_log_*.txt 2>/dev/null | head -1)
[ -n "${newest:-}" ] && { echo "  $newest"; tail -3 "$newest" | sed 's/^/  /'; } || echo "  none yet"
