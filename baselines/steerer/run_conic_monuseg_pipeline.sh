#!/usr/bin/env bash
set +e

STEERER=/root/autodl-tmp/steerer_project/STEERER
PD=/root/autodl-tmp/steerer_project/ProcessedData
cd "$STEERER" || exit 1

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$STEERER/tools:$STEERER:$PYTHONPATH"
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR="outputs/conic_monuseg_pipeline_${RUN_ID}"
mkdir -p "$RUN_DIR/configs"

AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-1}

finish() {
  echo "[EXIT] Pipeline finished or interrupted at $(date)" | tee -a "$RUN_DIR/pipeline.log"
  if [ "$AUTO_SHUTDOWN" = "1" ]; then
    echo "[EXIT] Auto shutdown enabled." | tee -a "$RUN_DIR/pipeline.log"
    shutdown -h now || poweroff || halt
  fi
}
trap finish EXIT

log_step() {
  echo "" | tee -a "$RUN_DIR/pipeline.log"
  echo "========== $1 ==========" | tee -a "$RUN_DIR/pipeline.log"
  echo "$(date)" | tee -a "$RUN_DIR/pipeline.log"
}

cat > /tmp/steerer_early_stop.py <<'PY'
import os, re, signal, sys, time

log_path = sys.argv[1]
pid = int(sys.argv[2])
patience = int(sys.argv[3])
min_delta = float(sys.argv[4])

pat = re.compile(
    r"Loss:.*?MAE:\s*([0-9.]+),\s*Best_MAE:\s*([0-9.]+).*?"
    r"MSE:\s*([0-9.]+),Best_MSE:\s*([0-9.]+)"
)

best = None
stale = 0
val_count = 0

def signal_train():
    try:
        pgid = os.getpgid(pid)
        if pgid == pid:
            os.killpg(pgid, signal.SIGINT)
        else:
            os.kill(pid, signal.SIGINT)
    except Exception as exc:
        print(f"[early-stop] failed to stop training: {exc}", flush=True)

def consume(line):
    global best, stale, val_count
    m = pat.search(line)
    if not m:
        return False
    mae = float(m.group(1))
    best_mae = float(m.group(2))
    val_count += 1
    if best is None or best_mae < best - min_delta:
        best = best_mae
        stale = 0
        print(f"[early-stop] val {val_count}: improved/current best_mae={best:.6f}, mae={mae:.6f}", flush=True)
    else:
        stale += 1
        print(f"[early-stop] val {val_count}: no improvement {stale}/{patience}, best_mae={best:.6f}, mae={mae:.6f}", flush=True)
    return True

while not os.path.exists(log_path):
    print(f"[early-stop] waiting for log: {log_path}", flush=True)
    time.sleep(10)

with open(log_path, "r", errors="ignore") as f:
    for line in f:
        consume(line)

print(f"[early-stop] initial best={best}, stale={stale}/{patience}", flush=True)

if best is not None and stale >= patience:
    print("[early-stop] already converged from existing log, stopping now.", flush=True)
    signal_train()
    sys.exit(0)

with open(log_path, "r", errors="ignore") as f:
    f.seek(0, os.SEEK_END)
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            print("[early-stop] train process exited", flush=True)
            break
        line = f.readline()
        if not line:
            time.sleep(20)
            continue
        if consume(line) and stale >= patience:
            print("[early-stop] convergence reached, stopping training.", flush=True)
            signal_train()
            break
PY

best_ckpt() {
  python - "$1" <<'PY'
import glob, os, re, sys
exp = sys.argv[1]
items = []
for p in glob.glob(os.path.join(exp, "Ep_*_mae_*_mse_*.pth")):
    m = re.search(r"_mae_([0-9.]+)_mse_([0-9.]+)\.pth$", os.path.basename(p))
    if m:
        items.append((float(m.group(1)), float(m.group(2)), p))
if not items:
    sys.exit(1)
items.sort()
print(items[0][2])
PY
}

train_with_early_stop() {
  name="$1"
  cfg="$2"
  exp_dir="$3"
  patience="$4"
  min_delta="$5"
  shift 5

  mkdir -p "$exp_dir"
  train_log="$exp_dir/$(basename "$exp_dir")_train.log"

  log_step "TRAIN EARLY STOP: $name"

  setsid python tools/train_cc.py \
    --cfg "$cfg" \
    --cfg-options "$@" train.resume_path="$exp_dir" \
    >> "$exp_dir/stdout_train.log" 2>&1 &

  pid=$!
  echo "$pid" > "$exp_dir/train.pid"
  echo "[train] $name pid=$pid exp=$exp_dir" | tee -a "$RUN_DIR/pipeline.log"

  python /tmp/steerer_early_stop.py "$train_log" "$pid" "$patience" "$min_delta" \
    2>&1 | tee "$exp_dir/early_stop.log"

  wait "$pid"
  rc=$?
  echo "[train] $name wait rc=$rc" | tee -a "$RUN_DIR/pipeline.log"

  ckpt=$(best_ckpt "$exp_dir")
  if [ -n "$ckpt" ] && [ -f "$ckpt" ]; then
    echo "$ckpt" | tee "$exp_dir/best_checkpoint.txt"
    return 0
  fi
  return 1
}

train_fixed_50() {
  name="$1"
  cfg="$2"
  exp_dir="$3"
  pretrained="$4"
  data_root="$5"

  mkdir -p "$exp_dir"
  log_step "TRAIN 50 EPOCHS: $name"

  python tools/train_cc.py \
    --cfg "$cfg" \
    --cfg-options \
      dataset.root="${data_root}/" \
      train.resume_path="$exp_dir" \
      train.pretrained_counter="$pretrained" \
      train.end_epoch=50 \
    2>&1 | tee "$exp_dir/stdout_train.log"

  ckpt=$(best_ckpt "$exp_dir")
  if [ -n "$ckpt" ] && [ -f "$ckpt" ]; then
    echo "$ckpt" | tee "$exp_dir/best_checkpoint.txt"
    return 0
  fi
  return 1
}

test_eval_eff() {
  dataset="$1"
  cfg_template="$2"
  data_root="$3"
  ckpt="$4"
  tag="$5"
  out_dir="$6"

  mkdir -p "$out_dir"
  test_cfg="$RUN_DIR/configs/${tag}_test.py"
  cp "$cfg_template" "$test_cfg"

  log_step "TEST + EVAL + EFFICIENCY: $tag"

  python tools/test_loc.py \
    --cfg "$test_cfg" \
    --checkpoint "$ckpt" \
    --cfg-options \
      dataset.root="${data_root}/" \
      dataset.test_set=test.txt \
      dataset.loc_gt=test_gt_loc.txt \
      test.loc_base_size=640 \
      test.base_size=640 \
    2>&1 | tee "$out_dir/test_loc_stdout.txt"

  test_exp="exp/${dataset}/test/$(basename "$test_cfg" .py)"
  pred_txt="$test_exp/pred_points.txt"

  if [ ! -f "$pred_txt" ]; then
    echo "[ERROR] pred_points.txt not found: $pred_txt" | tee -a "$out_dir/error.log"
    return 1
  fi

  cp "$pred_txt" "$out_dir/pred_points.txt"
  cp "${data_root}/test_gt_loc.txt" "$out_dir/test_gt_loc.txt"

  python tools/augmentation/export_steerer_points_json.py \
    --input "$out_dir/test_gt_loc.txt" \
    --output "$out_dir/gt.json" \
    --dataset "$dataset" \
    --method ground_truth \
    --role gt \
    --extraction-method gt_centroids_from_processed_txt \
    2>&1 | tee "$out_dir/export_gt_json.txt"

  python tools/augmentation/export_steerer_points_json.py \
    --input "$out_dir/pred_points.txt" \
    --output "$out_dir/pred.json" \
    --dataset "$dataset" \
    --method STEERER \
    --role pred \
    --extraction-method steerer_density_peak_postprocess \
    2>&1 | tee "$out_dir/export_pred_json.txt"

  python tools/augmentation/centroid_eval.py \
    --gt "$out_dir/gt.json" \
    --pred "$out_dir/pred.json" \
    --thresholds 6 12 24 \
    2>&1 | tee "$out_dir/centroid_eval_6_12_24.txt"

  python tools/augmentation/profile_steerer_efficiency.py \
    --cfg "$test_cfg" \
    --checkpoint "$ckpt" \
    --height 256 \
    --width 256 \
    --warmup 10 \
    --iters 100 \
    --output "$out_dir/efficiency_standard_256.json" \
    2>&1 | tee "$out_dir/efficiency_standard_256.txt"

  python tools/augmentation/profile_steerer_efficiency.py \
    --cfg "$test_cfg" \
    --checkpoint "$ckpt" \
    --height 640 \
    --width 640 \
    --warmup 10 \
    --iters 100 \
    --output "$out_dir/efficiency_native_640.json" \
    2>&1 | tee "$out_dir/efficiency_native_640.txt"

  echo "$ckpt" > "$out_dir/checkpoint_used.txt"
  return 0
}

choose_better_conic() {
  python - "$1" "$2" "$3" "$4" <<'PY'
import json, sys
native_json, native_ckpt, unified_json, unified_ckpt = sys.argv[1:5]

def load(path):
    with open(path) as f:
        data = json.load(f)
    mae = data["counting"]["mae"]
    f1 = data["localization"]["12px"]["f1"]
    return mae, f1

n_mae, n_f1 = load(native_json)
u_mae, u_f1 = load(unified_json)

if (u_mae < n_mae) or (abs(u_mae - n_mae) < 1e-8 and u_f1 > n_f1):
    print(unified_ckpt)
else:
    print(native_ckpt)
PY
}

# 1. CoNIC native 继续训
CONIC_NATIVE_ROOT="$PD/CoNIC_cellvit_seed19_native"
CONIC_NATIVE_EXP="exp/CoNIC/MocHRBackbone_hrnet48/CoNIC_cellvit_seed19_native_train_2026-06-13-12-40"

existing_pid=$(ps -ef | grep "tools/train_cc.py" | grep -v grep | awk '{print $2}' | head -n 1)
if [ -n "$existing_pid" ]; then
  log_step "MONITOR EXISTING TRAIN PROCESS: $existing_pid"
  python /tmp/steerer_early_stop.py \
    "$CONIC_NATIVE_EXP/$(basename "$CONIC_NATIVE_EXP")_train.log" \
    "$existing_pid" 4 0.001 \
    2>&1 | tee "$CONIC_NATIVE_EXP/early_stop_existing_native.log"
  wait "$existing_pid"
else
  train_with_early_stop \
    "CoNIC native resume" \
    configs/CoNIC_train.py \
    "$CONIC_NATIVE_EXP" \
    4 0.001 \
    dataset.root="${CONIC_NATIVE_ROOT}/" \
    train.end_epoch=100
fi

CONIC_NATIVE_CKPT=$(best_ckpt "$CONIC_NATIVE_EXP")
echo "$CONIC_NATIVE_CKPT" | tee "$RUN_DIR/conic_native_best_checkpoint.txt"

# 2. CoNIC native 测试
test_eval_eff \
  CoNIC \
  configs/CoNIC_test.py \
  "$CONIC_NATIVE_ROOT" \
  "$CONIC_NATIVE_CKPT" \
  "conic_native_${RUN_ID}" \
  "$RUN_DIR/conic_native"

# 3. CoNIC unified 数据准备 + 训练
CONIC_UNIFIED_ROOT="$PD/CoNIC_cellvit_seed19_unified_aug_640"
if [ ! -f "$CONIC_UNIFIED_ROOT/train.txt" ]; then
  log_step "BUILD CoNIC unified processed data"
  CONIC_RELEASE=$(python - "$PD" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
matches = sorted(root.rglob("docs/conic_split_seed19.json"))
if not matches:
    sys.exit(1)
print(matches[0].parents[1])
PY
)
  python tools/augmentation/convert_conic_cellvit_seed19_to_steerer.py \
    --release-root "$CONIC_RELEASE" \
    --output "$CONIC_UNIFIED_ROOT" \
    --augment unified \
    --num-augments 1 \
    --include-original \
    --patch-size 640 \
    --seed 3035 \
    2>&1 | tee "$RUN_DIR/build_conic_unified.txt"
fi

CONIC_UNIFIED_EXP="exp/CoNIC/MocHRBackbone_hrnet48/CoNIC_cellvit_seed19_unified_aug_640_train_${RUN_ID}"
train_with_early_stop \
  "CoNIC unified aug 640" \
  configs/CoNIC_train.py \
  "$CONIC_UNIFIED_EXP" \
  4 0.001 \
  dataset.root="${CONIC_UNIFIED_ROOT}/" \
  train.end_epoch=100

CONIC_UNIFIED_CKPT=$(best_ckpt "$CONIC_UNIFIED_EXP")
echo "$CONIC_UNIFIED_CKPT" | tee "$RUN_DIR/conic_unified_best_checkpoint.txt"

# 4. CoNIC unified 测试
test_eval_eff \
  CoNIC \
  configs/CoNIC_test.py \
  "$CONIC_UNIFIED_ROOT" \
  "$CONIC_UNIFIED_CKPT" \
  "conic_unified_${RUN_ID}" \
  "$RUN_DIR/conic_unified"

# 5. 严格按 JSON 重建 MoNuSeg split
log_step "PREPARE MoNuSeg strict split"

MON_TRAIN_ZIP="$PD/MoNuSeg 2018 Training Data.zip"
MON_TEST_ZIP="$PD/MoNuSegTestData.zip"
MON_RAW_ALL="$PD/MoNuSeg_raw_extracted_${RUN_ID}"
MON_STRICT_ORIG="$PD/MoNuSeg_split_json_original_${RUN_ID}"
MON_NATIVE_ROOT="$PD/MoNuSeg_split_json_native_${RUN_ID}"
MON_UNIFIED_ORIG="$PD/MoNuSeg_split_json_unified_original_${RUN_ID}"
MON_UNIFIED_ROOT="$PD/MoNuSeg_split_json_unified_aug_640_${RUN_ID}"

mkdir -p "$MON_RAW_ALL"

python - "$MON_TRAIN_ZIP" "$MON_TEST_ZIP" "$MON_RAW_ALL" <<'PY'
import sys, zipfile
from pathlib import Path
for idx, z in enumerate(sys.argv[1:3]):
    z = Path(z)
    out = Path(sys.argv[3]) / f"zip{idx+1}_{z.stem}"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(out)
    print(f"extracted {z} -> {out}")
PY

python tools/augmentation/prepare_monuseg_split_from_json.py \
  --input "$MON_RAW_ALL" \
  --split-json tools/augmentation/monuseg_split.json \
  --output "$MON_STRICT_ORIG" \
  2>&1 | tee "$RUN_DIR/prepare_monuseg_split.txt"

python tools/convert_monuseg_to_steerer.py \
  --input "$MON_STRICT_ORIG" \
  --output "$MON_NATIVE_ROOT" \
  --seed 3035 \
  2>&1 | tee "$RUN_DIR/convert_monuseg_native.txt"

python tools/augmentation/augment_monuseg.py \
  --input "$MON_STRICT_ORIG" \
  --output "$MON_UNIFIED_ORIG" \
  --split train \
  --patch-size 640 \
  --num-augments 1 \
  --include-original \
  --copy-eval-splits \
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
  --seed 3035 \
  2>&1 | tee "$RUN_DIR/augment_monuseg_unified.txt"

python tools/convert_monuseg_to_steerer.py \
  --input "$MON_UNIFIED_ORIG" \
  --output "$MON_UNIFIED_ROOT" \
  --seed 3035 \
  2>&1 | tee "$RUN_DIR/convert_monuseg_unified.txt"

# 6. 选择 CoNIC 更好权重，然后 MoNuSeg native 微调 50 轮
BEST_CONIC_CKPT=$(choose_better_conic \
  "$RUN_DIR/conic_native/pred_centroid_eval.json" "$CONIC_NATIVE_CKPT" \
  "$RUN_DIR/conic_unified/pred_centroid_eval.json" "$CONIC_UNIFIED_CKPT")

echo "$BEST_CONIC_CKPT" | tee "$RUN_DIR/best_conic_checkpoint_for_monuseg.txt"

MON_NATIVE_EXP="exp/MoNuSeg/MocHRBackbone_hrnet48/MoNuSeg_from_best_conic_native_${RUN_ID}"
train_fixed_50 \
  "MoNuSeg native from best CoNIC" \
  configs/MoNuSeg_finetune_conic.py \
  "$MON_NATIVE_EXP" \
  "$BEST_CONIC_CKPT" \
  "$MON_NATIVE_ROOT"

MON_NATIVE_CKPT=$(best_ckpt "$MON_NATIVE_EXP")
test_eval_eff \
  MoNuSeg \
  configs/MoNuSeg_test.py \
  "$MON_NATIVE_ROOT" \
  "$MON_NATIVE_CKPT" \
  "monuseg_native_from_conic_${RUN_ID}" \
  "$RUN_DIR/monuseg_native"

# 7. MoNuSeg unified 微调 50 轮
MON_UNIFIED_EXP="exp/MoNuSeg/MocHRBackbone_hrnet48/MoNuSeg_from_best_conic_unified_aug_640_${RUN_ID}"
train_fixed_50 \
  "MoNuSeg unified from best CoNIC" \
  configs/MoNuSeg_finetune_conic.py \
  "$MON_UNIFIED_EXP" \
  "$BEST_CONIC_CKPT" \
  "$MON_UNIFIED_ROOT"

MON_UNIFIED_CKPT=$(best_ckpt "$MON_UNIFIED_EXP")
test_eval_eff \
  MoNuSeg \
  configs/MoNuSeg_test.py \
  "$MON_UNIFIED_ROOT" \
  "$MON_UNIFIED_CKPT" \
  "monuseg_unified_from_conic_${RUN_ID}" \
  "$RUN_DIR/monuseg_unified"

log_step "DONE"
echo "All outputs saved in: $RUN_DIR" | tee -a "$RUN_DIR/pipeline.log"
