#!/usr/bin/env bash
# S9 PHASE 0: git-lfs pull ONLY the masks the normal bank needs (pancreas, veins, duodenum, SMA,
# aorta; head/body/tail/lesion are already real files for every PanTS case), one structure per
# pull so each include string stays < 131072 bytes (CONTEXT gotcha 7). Lists are written by
# `python -m normative.pool sample` to work/normative/pool/lfs_include_<structure>.txt.
# Never resets/commits anything in the mask repo.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MASKS="/home/rahuldeb5/research/datasets/pants/masks"
export PATH="$HOME/.local/bin:$PATH"
for st in pancreas veins duodenum superior_mesenteric_artery aorta; do
  f="$REPO/work/normative/pool/lfs_include_${st}.txt"
  n=$(wc -c < "$f")
  [ "$n" -lt 131072 ] || { echo "include list for $st is $n bytes"; exit 1; }
  echo ">>> $st ($n bytes of include)"
  git -C "$MASKS" lfs pull -I "$(cat "$f")"
done
echo "done"
