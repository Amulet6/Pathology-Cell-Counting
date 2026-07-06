# Direction C: Adaptive NMS

Direction C applies inference-time spatial suppression and score-threshold
calibration to reduce duplicate predictions. Its runnable APGCC integration is in
`../../integrations/b_d_c/`; see `CODE_CHANGES_DIRECTION_C.md` and
`RUN_DIRECTION_B_D_C_MONUSEG.md` there.

Adaptive NMS controls the spatial suppression radius. The confidence threshold is
a separate parameter selected using validation data.
