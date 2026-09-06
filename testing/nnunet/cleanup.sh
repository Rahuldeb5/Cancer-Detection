#!/usr/bin/env bash
# Reclaim disk after the Dataset501 build is verified. Run only once
# `nnUNetv2_plan_and_preprocess` has finished successfully -- everything training
# needs then lives in $nnUNet_preprocessed and does not depend on the paths below.
#
#   bash src/nnunet/cleanup.sh          # safe intermediates only
#   bash src/nnunet/cleanup.sh --raw    # also drop the 8593 non-subset lesion masks
#   bash src/nnunet/cleanup.sh --all    # also delete the 322 GB image tarballs (NOT recoverable without re-download)
set -euo pipefail

PANTS="/home/rahuldeb5/research/datasets/pants"
FOLD_IDS="$(mktemp)"
python /home/rahuldeb5/Cancer-Detection/src/data/build_fold_id_list.py > "$FOLD_IDS"
trap 'rm -f "$FOLD_IDS"' EXIT

echo "### safe intermediates"
# ct_staging: imagesTr hard-links these inodes, so removing the tree frees only the
# handful of sanitised (copied) CTs + directory entries. Harmless and tidy.
if [ -d "$PANTS/ct_staging" ]; then
  du -sh "$PANTS/ct_staging"
  rm -rf "$PANTS/ct_staging"
  echo "  removed ct_staging"
fi
# MedFormer-era preprocessed cache (superseded by nnU-Net)
for p in "$PANTS/old/cache" "$PANTS/old/cache.tar.gz"; do
  [ -e "$p" ] && { du -sh "$p"; echo "  -> rm -rf '$p'   (uncomment to delete)"; }
done

if [ "${1:-}" = "--raw" ] || [ "${1:-}" = "--all" ]; then
  echo "### non-subset lesion masks"
  keep=$(wc -l < "$FOLD_IDS")
  deleted=0
  while IFS= read -r d; do
    cid=$(basename "$d")
    grep -qxF "$cid" "$FOLD_IDS" || { rm -rf "$d"; deleted=$((deleted+1)); }
  done < <(find "$PANTS/masks/mask_only" -mindepth 1 -maxdepth 1 -type d)
  echo "  kept $keep, deleted $deleted mask folders"
  ( cd "$PANTS/masks" && git lfs prune -f )
fi

if [ "${1:-}" = "--all" ]; then
  echo "### image tarballs (322 GB, source of truth for the CTs)"
  du -sh "$PANTS/images"
  read -r -p "  delete $PANTS/images/*.tar.gz ? [type YES] " ans
  [ "$ans" = "YES" ] && rm -f "$PANTS"/images/*.tar.gz && echo "  deleted tarballs" || echo "  skipped"
fi

echo "### done"
df -h / | tail -1
