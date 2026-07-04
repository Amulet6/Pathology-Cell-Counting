#!/usr/bin/env bash
# Wait (by PID) for a training run to finish, then run the eval/threshold comparison.
# Usage: watch_generic.sh <TRAIN_PID> <CONFIG> <OUTDIR> [GPU]
set -u
cd "$(dirname "$0")"
PID="${1:?need training PID}"
CFG="${2:?need config}"
OUT="${3:?need outdir}"
GPU="${4:-8}"

PY=/home/lixinli/anaconda3/envs/apgcc/bin/python
DR=/data1/llx/CoNICdata
W=$OUT/best.pth
REPORT=$OUT/EVAL_PROGRESS.log
mkdir -p "$OUT"

echo "[$(date)] waiting for training PID=$PID (cfg=$CFG)" > "$REPORT"
while kill -0 "$PID" 2>/dev/null; do sleep 120; done
echo "[$(date)] training PID gone." >> "$REPORT"; sleep 15

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
