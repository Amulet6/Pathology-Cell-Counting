#!/usr/bin/env bash
# Wait for the A+D+E CoNIC training to finish, then run the full eval/threshold comparison.
# Each step is independent (set +e) so one failure does not abort the rest.
set -u
cd "$(dirname "$0")"

PY=/home/lixinli/anaconda3/envs/apgcc/bin/python
CFG=./configs/CoNIC_finetune_dcnv2_edge_stain.yml
DR=/data1/llx/CoNICdata
GPU=3
OUT=output/CoNIC_ADE_dcnv2_edge_stain
W=$OUT/best.pth
TRAIN_PAT="main.py -c ./configs/CoNIC_finetune_dcnv2_edge_stain.yml"
REPORT=$OUT/EVAL_PROGRESS.log

echo "[$(date)] waiting for training to finish..." > "$REPORT"
# 1) wait until the training python process is gone
while pgrep -f "$TRAIN_PAT" >/dev/null 2>&1; do sleep 120; done
echo "[$(date)] training process gone." >> "$REPORT"
sleep 10

if [ ! -f "$W" ]; then
  echo "[$(date)] ERROR: best.pth not found at $W -- training may have crashed. Aborting eval." >> "$REPORT"
  echo "TRAIN_FAILED" > "$OUT/EVAL_DONE"
  exit 1
fi
echo "[$(date)] best.pth found. last train.log lines:" >> "$REPORT"
tail -6 "$OUT/train.log" >> "$REPORT"

run() {  # $1 = label, rest = command
  local label="$1"; shift
  echo "[$(date)] >>> START $label" >> "$REPORT"
  ( set +e; "$@" ) >> "$OUT/eval_${label}.log" 2>&1
  local rc=$?
  echo "[$(date)] <<< END   $label (exit=$rc)" >> "$REPORT"
}

# A) fixed score-threshold 0.5 on test  (counting + localization baseline)
run "fixed0.5" $PY eval_centroid.py --config $CFG --weight $W --data-root $DR --gpu $GPU \
    --score-threshold 0.5 --eval-list test.list --out-dir $OUT/eval_test_t0.50

# B) Direction-A: pick a single global threshold on val (counting sweep)
run "scan_val" $PY scan_threshold.py --config $CFG --weight $W --data-root $DR --gpu $GPU \
    --eval-list val.list --thresholds 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 \
    --out $OUT/val_scan/scan.json

# C) Direction-A: density-adaptive per-image threshold on test (CV-honest)
run "density_A" $PY density_threshold.py --config $CFG --weight $W --data-root $DR --gpu $GPU \
    --eval-list test.list --ref 0.15 --out $OUT/density_thr.json

# D) Direction-E: per-source-domain MAE-optimal threshold (val selects, test reports)
run "domainTh_E" $PY eval_domain_threshold.py --config $CFG --weight $W --data-root $DR --gpu $GPU \
    --val-list val.list --test-list test.list --select-metric mae \
    --score-candidates 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 --thresholds 6 12 24 \
    --out-dir $OUT/domain_threshold_eval_mae

echo "[$(date)] ALL EVAL STEPS DONE" >> "$REPORT"
echo "OK" > "$OUT/EVAL_DONE"
