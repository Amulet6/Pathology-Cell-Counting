#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ "$#" -lt 1 ]; then
  echo "Usage: bash tools/augmentation/run_bcdata_unified_aug.sh /path/to/original/BCData [output_dir] [patch_size] [num_augments]"
  exit 1
fi

BCDATA_RAW="$1"
OUTPUT_DIR="${2:-${REPO_ROOT}/outputs/BCData_unified_aug}"
PATCH_SIZE="${3:-256}"
NUM_AUGMENTS="${4:-1}"
SEED="${SEED:-3035}"

RAW_AUG_DIR="${OUTPUT_DIR}/raw"
STEERER_DIR="${OUTPUT_DIR}/steerer"

mkdir -p "${OUTPUT_DIR}"

echo "[1/2] Generate unified-augmentation BCData in original format"
python "${REPO_ROOT}/tools/augmentation/augment_bcdata.py" \
  --input "${BCDATA_RAW}" \
  --output "${RAW_AUG_DIR}" \
  --split train \
  --num-augments "${NUM_AUGMENTS}" \
  --patch-size "${PATCH_SIZE}" \
  --seed "${SEED}" \
  --scale-min 0.8 \
  --scale-max 1.2 \
  --hflip-prob 0.5 \
  --vflip-prob 0.5 \
  --affine \
  --affine-prob 1.0 \
  --rotate-deg 179 \
  --translate-frac 0.01 \
  --shear-deg 5 \
  --affine-scale-min 0.8 \
  --affine-scale-max 1.2 \
  --pixel-aug \
  --blur-noise-prob 1.0 \
  --color-aug-prob 1.0 \
  --min-points 0

echo "[2/2] Convert augmented BCData to STEERER processed format"
python "${REPO_ROOT}/tools/augmentation/convert_bcdata_unified_aug_to_steerer.py" \
  --input "${RAW_AUG_DIR}" \
  --output "${STEERER_DIR}" \
  --count-mode all \
  --box-size 16 \
  --category 0

echo "Done."
echo "Original-format augmented BCData: ${RAW_AUG_DIR}"
echo "STEERER-format augmented BCData: ${STEERER_DIR}"
