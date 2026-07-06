# Unified evaluation

`centroid_eval.py` evaluates predictions from every baseline using the same
center-point protocol.

It reports image-level counting MAE, MSE, and RMSE, plus Hungarian-matched
Precision, Recall, and F1-score at one or more pixel thresholds.

```bash
python evaluation/centroid_eval.py \
  --gt path/to/gt.json \
  --pred path/to/pred.json \
  --thresholds 6 12 24
```

The accepted JSON schema is documented in `formats/predictions_json.md`.
