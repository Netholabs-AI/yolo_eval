# Known issues & methodology caveats

Read this before trusting an eval report or using the suite for a go/no-go model decision.
Context: reviewed during the yolo_0626→yolo_1001 fleet deployment (2026-10-10).

## 1. Model output-format divergence (raw anchors vs end-to-end NMS)  — the big one
`ensure_engine()` builds a TRT engine from each `.pt`. Depending on the **ultralytics / TensorRT
version**, the SAME `.pt` exports to DIFFERENT output tensor layouts:
- **raw anchors**: `output0` shape `(B, 5, 1344)` (older/newer ultralytics, e.g. 8.4.152) — pre-NMS,
  many overlapping boxes per object.
- **end-to-end NMS**: `output0` shape `(B, 300, 6)` = `[x1,y1,x2,y2,conf,cls]` (e.g. ultralytics
  8.4.9 + TRT 10.16) — already deduplicated.

Symptoms if not handled:
- One run parsed yolo_1001 as the wrong layout → **confidence median 26.25**, det@0.3 = 100%,
  jitter 0.1px (all garbage). (See `run_1001.log` vs the fixed `run_1001b.log`.)
- Even when conf parses correctly, comparing **`n_detections` / "detections per frame"** across a
  raw-format model and an end2end model is **apples-to-oranges**: the raw one counts un-deduplicated
  anchors (inflated), the end2end one counts post-NMS boxes. A "fewer false positives (1.0 vs 1.9)"
  headline can be an artifact of this, not a real model difference.

**Fix (now in `run_yolo_eval.py`):** auto-detect the layout (`pred.shape[1] < pred.shape[2]`),
transpose + conf-threshold + apply `torchvision.ops.nms` on the raw path, so BOTH models are compared
**post-NMS**. After this, `conf` and `detections/frame` are comparable.

**Still TODO / recommended:** pin the ultralytics+TRT version so all models export the same way, and
ideally build/test the EXACT engine that production runs (the fleet uses the raw `(B,5,1344)` format
via `pose_pipeline/scripts_yolo_lp/yolo_nvdec_inference_bbox.py`).

## 2. Inference is CACHED — clear it after changing inference code
Per-session detections are cached at `<out>/cache/<model>/<session>.parquet` and reused if present
(see `main()`), so **editing `run_yolo_on_video` does NOT change an existing report** — it reloads the
stale cache. After any inference/format fix: delete `<out>/cache/` (and rebuild the engine) then re-run.
The published reports (`output/`, `output_headstage_clean/`) were generated BEFORE the NMS fix — regen
with cleared cache for a clean detections/frame comparison. (Confidence/det-rate are unaffected.)

## 3. Reference-free — proxy metrics only
There are no ground-truth boxes, so the suite cannot report true precision/recall. It uses proxies:
top-box confidence, detection rate vs threshold, temporal jitter/size-stability, cross-camera
agreement, and cross-model IoU. These are **relative** signals. Treat "detections/frame" as a proxy
for over-detection tendency, not a literal animal count.

## What IS trustworthy
The **confidence** and **detection-rate** comparisons are robust (they use the top-confidence box,
computed the same way regardless of NMS). On the headstage set these correctly showed yolo_1001 more
confident than yolo_0626 (≈0.75 vs 0.38) — independently confirmed by frame-level analysis during the
deployment. That finding drove the (correct) decision to deploy yolo_1001.

See also: `pose_pipeline/fleet/YOLO_1001_DEPLOYMENT_NOTES.md`.
