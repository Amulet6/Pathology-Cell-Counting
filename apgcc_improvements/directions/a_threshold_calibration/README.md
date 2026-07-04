# Direction A: threshold calibration

Direction A calibrates APGCC confidence thresholds on validation data and supports
density-adaptive thresholds. The runnable implementation is integrated with D and
E under `../../integrations/a_d_e/apgcc/`:

- `scan_threshold.py`: validation-set global threshold sweep;
- `density_threshold.py`: density-adaptive test-time thresholding;
- `density_threshold_val.py`: validation support for density calibration.

Thresholds must be selected on `val.list`; `test.list` is evaluated once with the
selected rule.
