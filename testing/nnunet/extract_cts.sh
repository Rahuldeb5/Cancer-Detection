#!/usr/bin/env bash
# Extract ct.nii.gz for the 1308 cross-validation cases out of the 10 PanTS-Mini
# image shards, one sequential pass per shard, into a staging directory.
#   staging/PanTS_XXXXXXXX/ct.nii.gz
set -euo pipefail

IMAGES_DIR="/home/rahuldeb5/research/datasets/pants/images"
STAGING_DIR="/home/rahuldeb5/research/datasets/pants/ct_staging"
ID_LIST="${1:?usage: extract_cts.sh <fold_ids.txt>}"

mkdir -p "$STAGING_DIR"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# shard lower-bound -> filename
declare -A SHARD=(
  [1]="PanTSMini_ImageTr_00000001_00001000.tar.gz"
  [1001]="PanTSMini_ImageTr_00001001_00002000.tar.gz"
  [2001]="PanTSMini_ImageTr_00002001_00003000.tar.gz"
  [3001]="PanTSMini_ImageTr_00003001_00004000.tar.gz"
  [4001]="PanTSMini_ImageTr_00004001_00005000.tar.gz"
  [5001]="PanTSMini_ImageTr_00005001_00006000.tar.gz"
  [6001]="PanTSMini_ImageTr_00006001_00007000.tar.gz"
  [7001]="PanTSMini_ImageTr_00007001_00008000.tar.gz"
  [8001]="PanTSMini_ImageTr_00008001_00009000.tar.gz"
  [9001]="PanTSMini_ImageTe_00009001_00009901.tar.gz"
)

# bucket ids by shard
while read -r cid; do
  [ -z "$cid" ] && continue
  num=$((10#${cid#PanTS_}))
  lb=$(( (num - 1) / 1000 * 1000 + 1 ))
  echo "${cid}/ct.nii.gz" >> "$tmp/members_${lb}.txt"
done < "$ID_LIST"

total=0
for lb in "${!SHARD[@]}"; do
  mf="$tmp/members_${lb}.txt"
  [ -f "$mf" ] || continue
  n=$(wc -l < "$mf")
  total=$((total + n))
  echo ">>> shard $lb (${SHARD[$lb]}): $n cases"
  tar -xzf "$IMAGES_DIR/${SHARD[$lb]}" -C "$STAGING_DIR" -T "$mf"
done

got=$(find "$STAGING_DIR" -name ct.nii.gz | wc -l)
echo "extracted $got ct.nii.gz (expected $total)"
