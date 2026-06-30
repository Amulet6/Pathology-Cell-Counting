#!/usr/bin/env bash
# Direct eval on the trained best.pth (training already finished). No waiting logic.
set -u
cd "$(dirname "$0")"

PY=/home/lixinli/anaconda3/envs/apgcc/bin/python
CFG=./configs/CoNIC_finetune_dcnv2_edge_stain.yml
DR=/data1/llx/CoNICdata
GPU=8
OUT=output/CoNIC_ADE_dcnv2_edge_stain
W=$OUT/best.pth
REPORT=$OUT/EVAL_PROGRESS2.log

echo "[$(date)] START eval on $W (gpu $GPU)" > "$REPORT"
[ -f "$W" ] || { echo "[$(date)] ERROR: $W missing" >> "$REPORT"; echo "NO_WEIGHT" > "$OUT/EVAL_DONE"; exit 1; }

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
