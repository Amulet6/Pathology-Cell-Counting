#!/usr/bin/env bash
# Wait (by PID, not pgrep) for a training run to finish, then run the eval/threshold comparison.
# Usage: watch_train_then_eval.sh <TRAIN_PID>
set -u
cd "$(dirname "$0")"
PID="${1:?need training PID}"

PY=/home/lixinli/anaconda3/envs/apgcc/bin/python
CFG=./configs/CoNIC_finetune_dcnv2_edge_stain_eos0.25.yml
DR=/data1/llx/CoNICdata
GPU=8
OUT=output/CoNIC_ADE_dcnv2_edge_stain_eos0.25
W=$OUT/best.pth
REPORT=$OUT/EVAL_PROGRESS.log
mkdir -p "$OUT"

echo "[$(date)] waiting for training PID=$PID" > "$REPORT"
while kill -0 "$PID" 2>/dev/null; do sleep 120; done   # PID-based, no pgrep self-match
echo "[$(date)] training PID gone." >> "$REPORT"
sleep 15

if [ ! -f "$W" ]; then
  echo "[$(date)] ERROR: best.pth missing -> training likely failed." >> "$REPORT"
  echo "TRAIN_FAILED" > "$OUT/EVAL_DONE"; exit 1
fi
echo "[$(date)] best.pth found; train.log tail:" >> "$REPORT"; tail -4 "$OUT/train.log" >> "$REPORT"

run() { local label="$1"; shift
  echo "[$(date)] >>> START $label" >> "$REPORT"
  ( set +e; "$@" ) >> "$OUT/eval_${label}.log" 2>&1
  echo "[$(date)] <<< END   $label (exit=$?)" >> "$REPORT"
}

run "fixed0.5" $PY eval_centroid.py --config $CFG --weight $W --data-root $DR --gpu $GPU \
    --score-threshold 0.5 --eval-list test.list --out-dir $OUT/eval_test_t0.50
run "scan_val" $PY scan_threshold.py --config $CFG --weight $W --data-root $DR --gpu $GPU \
    --eval-list val.list --thresholds 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 \
    --out $OUT/val_scan/scan.json
run "density_A" $PY density_threshold.py --config $CFG --weight $W --data-root $DR --gpu $GPU \
    --eval-list test.list --ref 0.15 --out $OUT/density_thr.json
run "domainTh_E" $PY eval_domain_threshold.py --config $CFG --weight $W --data-root $DR --gpu $GPU \
    --val-list val.list --test-list test.list --select-metric mae \
    --score-candidates 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 --thresholds 6 12 24 \
    --out-dir $OUT/domain_threshold_eval_mae

echo "[$(date)] ALL EVAL STEPS DONE" >> "$REPORT"
echo "OK" > "$OUT/EVAL_DONE"
