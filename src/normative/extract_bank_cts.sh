#!/usr/bin/env bash
# S9 PHASE 0: extract ct.nii.gz for the normal-bank cases only, following
# testing/nnunet/extract_cts.sh (one sequential pass per shard), into a SEPARATE
# directory so the cohort's ct_staging never contains a non-cohort case:
#   ~/research/datasets/pants/ct_normal_bank/PanTS_XXXXXXXX/ct.nii.gz
# Refuses to run if any requested id is in the 1308-case cohort.
# --occurrence=1 lets tar stop reading a shard once every requested member was found.
#
# usage: bash src/normative/extract_bank_cts.sh work/normative/pool/bank_ids.txt [parallel_shards]
set -euo pipefail

IMAGES_DIR="/home/rahuldeb5/research/datasets/pants/images"
STAGING_DIR="/home/rahuldeb5/research/datasets/pants/ct_normal_bank"
ID_LIST="${1:?usage: extract_bank_cts.sh <bank_ids.txt> [parallel_shards]}"
PAR="${2:-3}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

cat "$REPO"/src/data/fold_{1,2,3,4,5}_ids.txt | tr -s ' \r' '\n' | grep -o 'PanTS_[0-9]\{8\}' | sort -u > /tmp/s9_cohort_ids.$$
overlap=$(grep -o 'PanTS_[0-9]\{8\}' "$ID_LIST" | sort -u | comm -12 - /tmp/s9_cohort_ids.$$ | wc -l)
rm -f /tmp/s9_cohort_ids.$$
if [ "$overlap" -ne 0 ]; then echo "ABORT: $overlap bank ids are cohort cases"; exit 1; fi

mkdir -p "$STAGING_DIR"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

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

total=0
while read -r cid; do
  [ -z "$cid" ] && continue
  [ -f "$STAGING_DIR/$cid/ct.nii.gz" ] && continue          # resumable
  num=$((10#${cid#PanTS_}))
  lb=$(( (num - 1) / 1000 * 1000 + 1 ))
  echo "${cid}/ct.nii.gz" >> "$tmp/members_${lb}.txt"
  total=$((total + 1))
done < "$ID_LIST"

extract_shard() {
  local lb="$1" mf="$2" shard="$3"
  echo ">>> shard $lb ($shard): $(wc -l < "$mf") cases"
  tar -xzf "$IMAGES_DIR/$shard" -C "$STAGING_DIR" --occurrence=1 -T "$mf"
  echo "<<< shard $lb done"
}

running=0
for lb in "${!SHARD[@]}"; do
  mf="$tmp/members_${lb}.txt"
  [ -f "$mf" ] || continue
  extract_shard "$lb" "$mf" "${SHARD[$lb]}" &
  running=$((running + 1))
  if [ "$running" -ge "$PAR" ]; then wait -n; running=$((running - 1)); fi
done
wait

got=$(grep -c . "$ID_LIST")
have=$(find "$STAGING_DIR" -name ct.nii.gz | wc -l)
echo "requested $total new; bank dir now has $have ct.nii.gz (list has $got ids)"
du -sh "$STAGING_DIR"
