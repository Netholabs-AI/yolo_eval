# YOLO Model Evaluation Suite

Compare YOLO detection models on out-of-sample test videos without ground truth.
Measures detection consistency, temporal stability, cross-camera agreement, and
downstream impact on LP pose estimation.

## Usage

### 1. Prepare test videos

Place test session directories in a folder. Each session should have 5 camera mp4s:

```
test_videos/
  6.01.016_2026_09_17_06_59_00_000_right/
    2026_09_17_06_59_00_000_cam1.mp4
    ...cam5.mp4
  6.01.001_2026_09_19_06_00_00_000_right/
    ...
```

### 2. Run evaluation

```bash
python3 run_yolo_eval.py \
    --test-dir /path/to/test_videos \
    --models yolo_0519:/path/to/yolo_0519/weights/best.pt \
             yolo_0626:/path/to/yolo_0626/weights/best.pt \
    --out /path/to/output \
    --gpu 0
```

### 3. View report

Open `output/report.html` in a browser.

## Metrics

### Detection
- Detection rate (% frames with conf > threshold)
- Confidence distribution (mean, median, per-camera)
- Multi-detection rate (2+ boxes per frame)
- Bbox area and aspect ratio distribution

### Temporal stability
- Frame-to-frame bbox center jitter (px)
- Frame-to-frame bbox size change
- Confidence stability (rolling std)

### Cross-camera agreement
- Per-session: how many cameras detect simultaneously
- Disagreement rate between cameras

### Cross-model comparison
- Detection agreement between models
- Bbox IoU between models on same frames
- Confidence correlation
- Frames where models disagree (one detects, other doesn't)

## Files

```
yolo_eval/
  README.md
  run_yolo_eval.py      # Main evaluation script
  report_generator.py   # HTML report generation
```
