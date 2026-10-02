#!/usr/bin/env bash
# rsync the preprocessed Dataset502 from the PC to the server, then verify.
#
# The server has no raw CTs, so everything training needs must travel: the
# nnUNetPlans_3d_fullres .npz/.npy/.pkl blobs, gt_segmentations, nnUNetPlans.json,
# dataset.json, dataset_fingerprint.json and splits_final.json. nnUNet_raw is NOT sent
# (nnUNetv2_train only reads nnUNet_preprocessed).
#
#   bash src/nnunet-multiclass/sync_preprocessed_502.sh [--dry-run]
set -uo pipefail   # no -e: the md5 spot-check below tolerates per-file failures
source "$HOME/research/nnunet_env/env.sh"

DS="${DS_NAME:-Dataset502_PanTSPancLesion}"
REMOTE="${REMOTE:-rahul@deep-server.tail8e65db.ts.net}"
RPRE="${RPRE:-/home/rahul/research/nnunet_env/nnUNet_preprocessed}"
SRC="$nnUNet_preprocessed/$DS/"
LOG="$HOME/research/nnunet_env/logs/rsync_502.log"
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="--dry-run"

echo "=== rsync $SRC -> $REMOTE:$RPRE/$DS/  $(date -Is) ===" | tee -a "$LOG"
ssh "$REMOTE" "mkdir -p '$RPRE/$DS'"
rsync -a --info=progress2 --partial $DRY \
      "$SRC" "$REMOTE:$RPRE/$DS/" 2>&1 | tee -a "$LOG"
[ -n "$DRY" ] && { echo "--dry-run: stopping before verification"; exit 0; }

echo "=== verify ===" | tee -a "$LOG"
for side in local remote; do
  if [ "$side" = local ]; then
    C="bash -c"
    BASE="$nnUNet_preprocessed/$DS"
  else
    C="ssh $REMOTE"
    BASE="$RPRE/$DS"
  fi
  echo "-- $side ($BASE)"
  $C "find '$BASE' -type f | wc -l; du -sb '$BASE' | cut -f1; \
      ls '$BASE'/nnUNetPlans_3d_fullres | wc -l; \
      for f in dataset.json nnUNetPlans.json dataset_fingerprint.json splits_final.json; do \
        printf '%s %s\n' \"\$f\" \"\$(md5sum '$BASE'/\$f 2>/dev/null | cut -d' ' -f1)\"; done; \
      printf 'gt_segmentations %s files\n' \"\$(ls '$BASE'/gt_segmentations | wc -l)\""
done

echo "=== spot-check 5 preprocessed cases (md5 of the .npz/.npy/.pkl) ===" | tee -a "$LOG"
CASES=$(find "$nnUNet_preprocessed/$DS/nnUNetPlans_3d_fullres" -name '*.pkl' -printf '%f\n' | sort | head -5 | sed 's/\.pkl$//')
for c in $CASES; do
  echo "-- $c"
  l=$(cd "$nnUNet_preprocessed/$DS/nnUNetPlans_3d_fullres" && md5sum ${c}.* 2>/dev/null | sort)
  r=$(ssh "$REMOTE" "cd '$RPRE/$DS/nnUNetPlans_3d_fullres' && md5sum ${c}.* 2>/dev/null | sort")
  if [ "$l" = "$r" ]; then echo "   OK  $(echo "$l" | wc -l) files identical"; else
    echo "   MISMATCH"; diff <(echo "$l") <(echo "$r") || true; fi
  lg=$(md5sum "$nnUNet_preprocessed/$DS/gt_segmentations/${c}.nii.gz" | cut -d' ' -f1)
  rg=$(ssh "$REMOTE" "md5sum '$RPRE/$DS/gt_segmentations/${c}.nii.gz' | cut -d' ' -f1")
  [ "$lg" = "$rg" ] && echo "   OK  gt_segmentations identical" || echo "   MISMATCH gt_segmentations"
done
echo "=== done $(date -Is) ===" | tee -a "$LOG"
