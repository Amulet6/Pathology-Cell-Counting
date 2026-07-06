#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$ROOT_DIR/repos/CellVTA"
ENV_DIR="$ROOT_DIR/envs/cellvta-official-py397"
PYTHON_BIN="$ENV_DIR/bin/python"
ENV_BIN_DIR="$ENV_DIR/bin"
TORCH_LIB_DIR="$ENV_DIR/lib/python3.9/site-packages/torch/lib"
OPS_BUILD_DIR="$REPO_DIR/models/ops/build/lib.linux-x86_64-cpython-39"
TRAIN_CONFIG="$REPO_DIR/configs/train/CellVTA_Conic_training.yaml"
INFER_CONFIG="$REPO_DIR/configs/inference/CellVTA_Conic_upscale_inference.yaml"
TRAIN_DATA_DIR="$REPO_DIR/datasets/conic_cellvit_patient_x40_linear_withOverlap"
TEST_DATA_DIR="$REPO_DIR/datasets/conic_cellvit_patient"
UNI_WEIGHT="$REPO_DIR/pretrained_models/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin"
CONIC_CKPT="$REPO_DIR/pretrained_models/cellvta/CellVTA_UNI_conic.pth"
CUDA_HOME_DEFAULT="${CUDA_HOME:-/usr/local/cuda}"
OUTPUT_DIR_DEFAULT="$REPO_DIR/logs/inference_conic_official"

usage() {
  cat <<'EOF'
Usage:
  run_cellvta_conic_official.sh check
  run_cellvta_conic_official.sh compile_ops
  run_cellvta_conic_official.sh train [config_path]
  run_cellvta_conic_official.sh infer [output_dir]

Subcommands:
  check        Check dataset, weights, CUDA, and extension status
  compile_ops  Compile MultiScaleDeformableAttention with the local env
  train        Run CoNIC training config; default is the official config unless an explicit config_path is provided
  infer        Run official CoNIC upscale inference config
EOF
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing directory: $path" >&2
    exit 1
  fi
}

setup_runtime_env() {
  export CUDA_HOME="$CUDA_HOME_DEFAULT"
  export PATH="$ENV_BIN_DIR:$CUDA_HOME_DEFAULT/bin:$PATH"
  export LD_LIBRARY_PATH="$ENV_DIR/lib:$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"
  # Do not inherit stale PYTHONPATH entries that point at a previously built
  # ops/build tree from another machine.
  export PYTHONPATH="$REPO_DIR"
  # Force the host toolchain so the extension links against the current
  # machine's glibc/libstdc++, not the conda cross-compiler sysroot.
  if [[ -x /usr/bin/gcc ]]; then
    export CC=/usr/bin/gcc
  else
    unset CC || true
  fi
  if [[ -x /usr/bin/g++ ]]; then
    export CXX=/usr/bin/g++
  else
    unset CXX || true
  fi
}

show_check() {
  setup_runtime_env
  require_dir "$TRAIN_DATA_DIR"
  require_dir "$TEST_DATA_DIR"

  echo "Repo: $REPO_DIR"
  echo "Python: $PYTHON_BIN"
  echo "Env: $ENV_DIR"
  echo "CUDA_HOME: $CUDA_HOME"
  echo "Train data: $TRAIN_DATA_DIR"
  echo "Test data: $TEST_DATA_DIR"
  echo "UNI weight: $UNI_WEIGHT"
  echo "CoNIC checkpoint: $CONIC_CKPT"
  echo "Torch lib dir: $TORCH_LIB_DIR"
  echo "Ops build dir: $OPS_BUILD_DIR"

  if [[ -f "$UNI_WEIGHT" ]]; then
    echo "UNI weight: OK"
  else
    echo "UNI weight: MISSING"
  fi

  if [[ -f "$CONIC_CKPT" ]]; then
    echo "CoNIC checkpoint: OK"
  else
    echo "CoNIC checkpoint: MISSING"
  fi

  MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-cellvta-check" \
  "$PYTHON_BIN" - <<'PY'
import importlib.util
import os
import torch
from torch.utils.cpp_extension import CUDA_HOME

mods = ["MultiScaleDeformableAttention", "albumentations", "wandb", "timm", "einops", "natsort"]
print("torch_version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
print("cuda_home:", CUDA_HOME)
print("env_cuda_home:", os.environ.get("CUDA_HOME"))
print("cc:", os.environ.get("CC"))
print("cxx:", os.environ.get("CXX"))
print("ld_library_path:", os.environ.get("LD_LIBRARY_PATH"))
print("pythonpath:", os.environ.get("PYTHONPATH"))
print("modules:", {m: bool(importlib.util.find_spec(m)) for m in mods})
try:
    import MultiScaleDeformableAttention as msda
    print("msda_import: OK", msda)
except Exception as e:
    print("msda_import: ERR", type(e).__name__, e)
PY
}

compile_ops() {
  setup_runtime_env
  require_dir "$REPO_DIR/models/ops"
  echo "Compiling MultiScaleDeformableAttention..."
  (
    cd "$REPO_DIR/models/ops"
    rm -rf build dist MultiScaleDeformableAttention.egg-info
    "$PYTHON_BIN" setup.py build install
  )
}

run_train() {
  setup_runtime_env
  local train_config="${1:-$TRAIN_CONFIG}"
  require_file "$train_config"
  require_file "$UNI_WEIGHT"
  require_dir "$TRAIN_DATA_DIR"
  (
    cd "$REPO_DIR"
    MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-cellvta-train" \
    "$PYTHON_BIN" cell_segmentation/run_cellvit.py --config "$train_config"
  )
}

run_infer() {
  setup_runtime_env
  local output_dir="${1:-$OUTPUT_DIR_DEFAULT}"
  require_file "$INFER_CONFIG"
  require_file "$CONIC_CKPT"
  require_dir "$TEST_DATA_DIR"
  mkdir -p "$output_dir"
  (
    cd "$REPO_DIR"
    MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-cellvta-infer" \
    "$PYTHON_BIN" cell_segmentation/inference/inference_cellvit_upscale.py \
      --config "$INFER_CONFIG" \
      --output_dir "$output_dir" \
      --gpu 0
  )
}

cmd="${1:-}"
case "$cmd" in
  check)
    show_check
    ;;
  compile_ops)
    compile_ops
    ;;
  train)
    run_train "${2:-}"
    ;;
  infer)
    run_infer "${2:-}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
