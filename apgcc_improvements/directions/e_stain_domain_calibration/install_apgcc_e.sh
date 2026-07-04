#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash install_apgcc_e.sh /path/to/APGCC/apgcc" >&2
  exit 1
fi

TARGET_APGCC="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${SCRIPT_DIR}/apgcc_patch"

if [ ! -d "${TARGET_APGCC}" ]; then
  echo "Target APGCC directory does not exist: ${TARGET_APGCC}" >&2
  exit 1
fi

if [ ! -f "${TARGET_APGCC}/main.py" ] || [ ! -d "${TARGET_APGCC}/models" ]; then
  echo "Target does not look like the APGCC/apgcc directory: ${TARGET_APGCC}" >&2
  exit 1
fi

cp "${PATCH_DIR}/config.py" "${TARGET_APGCC}/config.py"
cp "${PATCH_DIR}/engine.py" "${TARGET_APGCC}/engine.py"
cp "${PATCH_DIR}/eval_domain_threshold.py" "${TARGET_APGCC}/eval_domain_threshold.py"
cp "${PATCH_DIR}/eval_domain_density_threshold.py" "${TARGET_APGCC}/eval_domain_density_threshold.py"
cp "${PATCH_DIR}/datasets/dataset.py" "${TARGET_APGCC}/datasets/dataset.py"
cp "${PATCH_DIR}/models/APGCC.py" "${TARGET_APGCC}/models/APGCC.py"
cp "${PATCH_DIR}/models/__init__.py" "${TARGET_APGCC}/models/__init__.py"

mkdir -p "${TARGET_APGCC}/configs"
cp "${PATCH_DIR}"/configs/*.yml "${TARGET_APGCC}/configs/"

echo "APGCC_E files installed into ${TARGET_APGCC}"
echo "Main recommended config: configs/CoNIC_finetune_stain.yml"
echo "Main recommended post-process: eval_domain_threshold.py --select-metric mae"
