#!/usr/bin/env bash
# Default nnU-Net planning + 3d_fullres preprocessing for Dataset502, on the PC.
#
# -np 3: np 4 OOM-killed this 15 GB WSL box once (CONTEXT.md / S7 brief). No planner
# overrides of any kind -- the plan must be nnU-Net's own default so the only difference
# from Dataset501 is the label definition.
#
#   bash src/nnunet-multiclass/preprocess_502.sh
set -euo pipefail
source "$HOME/research/nnunet_env/env.sh"

NP="${NP:-3}"
LOG="$HOME/research/nnunet_env/logs/preprocess_502.log"
mkdir -p "$(dirname "$LOG")"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "=== nnUNetv2_plan_and_preprocess -d 502 -c 3d_fullres -np $NP  $(date -Is) ===" | tee -a "$LOG"
.venv/bin/nnUNetv2_plan_and_preprocess -d 502 -c 3d_fullres -np "$NP" --verify_dataset_integrity 2>&1 | tee -a "$LOG"
echo "=== exit $?  $(date -Is) ===" | tee -a "$LOG"
