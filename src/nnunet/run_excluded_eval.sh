#!/usr/bin/env bash
# Re-score all 5 folds' existing validation predictions with a set of GT lesions excluded.
# No retraining and no new inference: the predictions in fold_k/validation were written by
# that fold's trained checkpoint, so re-scoring them is equivalent to reloading and re-testing.
#
# Usage (on the server, from ~/Cancer-Detection):
#   bash src/nnunet/run_excluded_eval.sh
# Excluded lesions: results/lesion_location_results/excluded_lesions.csv
# -> results/nnunet_excl_outside_pancreas/fold{0..4}/{evaluation.json,per_case_metrics.csv,per_lesion_metrics.csv,manifest.csv}
set -uo pipefail
source "$HOME/research/nnunet_env/env.sh"

NAME=outside_pancreas
REPO="$HOME/Cancer-Detection"
RES="$nnUNet_results/Dataset501_PanTSTumor/nnUNetTrainer__nnUNetPlans__3d_fullres"
GT="$nnUNet_preprocessed/Dataset501_PanTSTumor/gt_segmentations"
CSV="$REPO/results/lesion_location_results/excluded_lesions.csv"
WORK="$HOME/research/nnunet_env/eval_excl/$NAME"
OUT="$REPO/results/nnunet_excl_$NAME"
mkdir -p "$OUT"
rm -f "$OUT/DONE"

pids=()
for f in 0 1 2 3 4; do
  (
    set -e
    python "$REPO/src/nnunet/prepare_excluded_eval.py" \
      --pred_dir "$RES/fold_$f/validation" --gt_dir "$GT" \
      --work_dir "$WORK/fold_$f" --excluded_csv "$CSV"
    python "$REPO/src/nnunet/evaluate.py" \
      --pred_dir "$WORK/fold_$f/pred" --gt_dir "$WORK/fold_$f/gt" --out_dir "$OUT/fold$f"
    cp "$WORK/fold_$f/manifest.csv" "$OUT/fold$f/manifest.csv"
  ) > "$OUT/fold$f.log" 2>&1 &
  pids+=($!)
done

fails=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "fold $i FAILED (see $OUT/fold$i.log)"; fails=$((fails+1)); }
done
echo "finished, failed folds: $fails" | tee "$OUT/DONE"
