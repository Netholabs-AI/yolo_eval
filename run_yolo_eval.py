#!/usr/bin/env python3
"""
YOLO Model Evaluation Suite — compare detection models without ground truth.

Uses production-style inference: TRT engine + NVDEC GPU decode + batched inference.
Runs multiple YOLO models on the same test videos, computes per-model and
cross-model metrics, generates an interactive HTML report.

Usage:
    python3 run_yolo_eval.py \
        --test-dir /path/to/test_videos \
        --models yolo_0519:/path/to/best.pt yolo_0626:/path/to/best.pt \
        --out /path/to/output
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from PyNvVideoCodec import OutputColorType, SimpleDecoder
from ultralytics import YOLO

sys.path.insert(0, os.path.expanduser("~/simon/pose_pipeline/scripts_yolo_lp"))
from yolo_nvdec_inference_bbox import read_yolo_metadata, _xyxy_to_corners

CONF_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def find_sessions(test_dir):
    test_dir = Path(test_dir)
    sessions = []
    for d in sorted(test_dir.iterdir()):
        if not d.is_dir():
            continue
        mp4s = sorted(d.glob("*cam*.mp4"))
        if len(mp4s) >= 3:
            sessions.append(d)
    return sessions


def ensure_engine(pt_path, imgsz=256, batch=192):
    """Build TRT engine if needed, return engine path."""
    engine_path = pt_path.replace(".pt", ".engine")
    if os.path.exists(engine_path):
        meta = read_yolo_metadata(engine_path)
        if meta.get("batch") == batch:
            return engine_path
    print(f"  Building TRT engine (imgsz={imgsz}, batch={batch})...")
    model = YOLO(pt_path)
    model.export(format='engine', imgsz=imgsz, half=True, batch=batch,
                 int8=False, workspace=None, task='detect', dynamic=False)
    return engine_path


def load_trt_model(engine_path, gpu_id=0):
    """Load TRT model and prepare batch buffer."""
    model = YOLO(engine_path, task='detect')
    meta = read_yolo_metadata(engine_path)
    imsz = meta.get('imgsz', [256])[0] if isinstance(meta.get('imgsz'), list) else int(meta.get('imgsz', 256))
    batch_size = int(meta.get('batch', 192))

    device = torch.device(f'cuda:{gpu_id}')
    dummy = torch.zeros(batch_size, 3, imsz, imsz, dtype=torch.float16, device=device)
    model.predict(source=dummy, device=gpu_id, imgsz=imsz, verbose=False)
    trt_model = model.predictor.model
    fp16 = getattr(trt_model, 'fp16', False)
    buf_dtype = torch.float16 if fp16 else torch.float32
    batch_buf = torch.empty(batch_size, 3, imsz, imsz, dtype=buf_dtype, device=device)
    del dummy

    names = meta.get('names', {0: 'mouse'})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    return trt_model, batch_buf, batch_size, imsz, names, device


def run_yolo_on_video(video_path, trt_model, batch_buf, batch_size, imsz,
                      gpu_id, conf_threshold=0.05):
    """Run batched TRT YOLO on a single video via NVDEC. Returns DataFrame."""
    from torchvision.ops import nms as _nms

    device = batch_buf.device
    decoder = SimpleDecoder(str(video_path), gpu_id=gpu_id,
                            use_device_memory=True, output_color_type=OutputColorType.RGBP)
    n_total = len(decoder)

    first_frames = decoder.get_batch_frames_by_index(np.array([0], dtype=int))
    first_tensor = torch.from_dlpack(first_frames[0])
    _, orig_h, orig_w = first_tensor.shape
    del first_frames, first_tensor

    decoder = SimpleDecoder(str(video_path), gpu_id=gpu_id,
                            use_device_memory=True, output_color_type=OutputColorType.RGBP)

    # Detect raw vs post-NMS output format once
    with torch.no_grad():
        _test = trt_model(batch_buf)
    _test_pred = _test[0] if isinstance(_test, (list, tuple)) else _test
    raw_format = _test_pred.shape[1] < _test_pred.shape[2]  # [B, 5+nc, anchors] vs [B, max_det, 6+]

    lb_scale = min(imsz / orig_h, imsz / orig_w)
    lb_new_h = int(round(orig_h * lb_scale))
    lb_new_w = int(round(orig_w * lb_scale))
    lb_pad_top = (imsz - lb_new_h) // 2
    lb_pad_left = (imsz - lb_new_w) // 2

    rows = []
    batch_starts = list(range(0, n_total, batch_size))

    for batch_start in batch_starts:
        n_frames = min(batch_size, n_total - batch_start)
        indices = np.arange(batch_start, batch_start + n_frames, dtype=int)

        decoded_frames = decoder.get_batch_frames_by_index(indices)
        decoded_tensors = [torch.from_dlpack(f) for f in decoded_frames]
        torch.cuda.synchronize(gpu_id)

        batch_buf[:n_frames] = 114.0 / 255.0
        for i, t_src in enumerate(decoded_tensors):
            t = t_src.unsqueeze(0).to(dtype=batch_buf.dtype) * (1.0 / 255.0)
            if t.shape[-2:] == (imsz, imsz):
                batch_buf[i:i+1] = t
            else:
                batch_buf[i:i+1, :, lb_pad_top:lb_pad_top+lb_new_h,
                          lb_pad_left:lb_pad_left+lb_new_w] = F.interpolate(
                    t, size=(lb_new_h, lb_new_w), mode="bilinear", align_corners=False)
        if n_frames < batch_size:
            batch_buf[n_frames:] = batch_buf[n_frames-1:n_frames]
        torch.cuda.synchronize(gpu_id)

        with torch.no_grad():
            raw_pred = trt_model(batch_buf)
        torch.cuda.synchronize(gpu_id)

        pred_tensor = raw_pred[0] if isinstance(raw_pred, (list, tuple)) else raw_pred
        pred = pred_tensor[:n_frames].cpu()

        for i in range(n_frames):
            frm_idx = batch_start + i
            det = pred[i]

            if raw_format:
                # Raw anchors: [5+nc, num_anchors] → transpose to [anchors, 5+nc]
                det = det.permute(1, 0).contiguous()
                scores = det[:, 4]
                mask = scores > conf_threshold
                det = det[mask] if mask.any() else det[:0]
                if len(det) > 0:
                    # Convert cxcywh → xyxy for NMS
                    cx_r, cy_r, w_r, h_r = det[:, 0], det[:, 1], det[:, 2], det[:, 3]
                    x1_r = cx_r - w_r / 2
                    y1_r = cy_r - h_r / 2
                    x2_r = cx_r + w_r / 2
                    y2_r = cy_r + h_r / 2
                    boxes = torch.stack([x1_r, y1_r, x2_r, y2_r], dim=1)
                    keep = _nms(boxes, det[:, 4], 0.5)
                    det = det[keep]
                    # Rewrite det as [n, 5]: x1,y1,x2,y2,conf
                    cx_r, cy_r, w_r, h_r = det[:, 0], det[:, 1], det[:, 2], det[:, 3]
                    det = torch.stack([cx_r - w_r/2, cy_r - h_r/2,
                                       cx_r + w_r/2, cy_r + h_r/2, det[:, 4]], dim=1)
                v_det = det
            else:
                # Post-NMS: [max_det, 6+], already deduplicated
                scores = det[:, 4]
                mask = scores > conf_threshold
                v_det = det[mask] if mask.any() else det[:0]

            n_detections = len(v_det)
            if n_detections > 0:
                best_idx = v_det[:, 4].argmax()
                vals = v_det[best_idx]
                x1_lb, y1_lb, x2_lb, y2_lb, conf = vals[:5].tolist()
                x1 = (x1_lb - lb_pad_left) / lb_scale
                y1 = (y1_lb - lb_pad_top) / lb_scale
                x2 = (x2_lb - lb_pad_left) / lb_scale
                y2 = (y2_lb - lb_pad_top) / lb_scale
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                bw = x2 - x1
                bh = y2 - y1
                area = bw * bh
                aspect = bh / max(bw, 1)
            else:
                conf = 0.0
                x1 = y1 = x2 = y2 = cx = cy = bw = bh = area = aspect = 0.0

            rows.append({
                "frame": frm_idx,
                "confidence": conf,
                "n_detections": n_detections,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx, "cy": cy,
                "bw": bw, "bh": bh,
                "area": area, "aspect": aspect,
                "frame_w": orig_w, "frame_h": orig_h,
            })

    return pd.DataFrame(rows)


def run_yolo_on_session(trt_model, batch_buf, batch_size, imsz, session_dir, gpu_id):
    """Run YOLO on all cameras in a session."""
    results = {}
    mp4s = sorted(session_dir.glob("*cam*.mp4"))
    for mp4 in mp4s:
        cam = mp4.stem.split("cam")[-1].split("_")[0]
        cam_key = f"cam{cam}"
        results[cam_key] = run_yolo_on_video(mp4, trt_model, batch_buf, batch_size, imsz, gpu_id)
    return results


def compute_detection_metrics(cam_data):
    all_confs, all_areas, all_aspects, all_n_det = [], [], [], []
    per_cam_det_rates = {}

    for cam, df in cam_data.items():
        confs = df["confidence"].values
        all_confs.extend(confs)
        all_areas.extend(df["area"][df["area"] > 0].values)
        all_aspects.extend(df["aspect"][df["aspect"] > 0].values)
        all_n_det.extend(df["n_detections"].values)
        per_cam_det_rates[cam] = {t: float((confs > t).mean()) for t in CONF_THRESHOLDS}

    all_confs = np.array(all_confs)
    all_areas = np.array(all_areas) if all_areas else np.array([0])
    all_aspects = np.array(all_aspects) if all_aspects else np.array([0])
    all_n_det = np.array(all_n_det)

    return {
        "det_rate": {t: float((all_confs > t).mean()) for t in CONF_THRESHOLDS},
        "per_cam_det_rate": per_cam_det_rates,
        "conf_mean": float(np.mean(all_confs)),
        "conf_median": float(np.median(all_confs)),
        "conf_std": float(np.std(all_confs)),
        "conf_values": all_confs,
        "area_median": float(np.median(all_areas)),
        "area_std": float(np.std(all_areas)),
        "aspect_median": float(np.median(all_aspects)),
        "aspect_std": float(np.std(all_aspects)),
        "aspect_values": all_aspects,
        "multi_det_rate": float((all_n_det > 1).mean()),
        "zero_det_rate": float((all_n_det == 0).mean()),
    }


def compute_temporal_metrics(cam_data):
    jitters, size_changes, conf_stds = [], [], []
    for cam, df in cam_data.items():
        det = df[df["confidence"] > 0.3]
        if len(det) < 10:
            continue
        dcx = np.diff(det["cx"].values)
        dcy = np.diff(det["cy"].values)
        jitters.extend(np.sqrt(dcx**2 + dcy**2))
        sc = np.abs(np.diff(det["bw"].values)) + np.abs(np.diff(det["bh"].values))
        size_changes.extend(sc)
        confs = det["confidence"].values
        if len(confs) >= 10:
            for i in range(len(confs) - 10):
                conf_stds.append(np.std(confs[i:i+10]))

    jitters = np.array(jitters) if jitters else np.array([])
    size_changes = np.array(size_changes) if size_changes else np.array([])
    return {
        "jitter_median": float(np.median(jitters)) if len(jitters) else 0,
        "jitter_p95": float(np.percentile(jitters, 95)) if len(jitters) else 0,
        "jitter_values": jitters,
        "size_change_values": size_changes,
        "size_change_median": float(np.median(size_changes)) if len(size_changes) else 0,
        "size_change_p95": float(np.percentile(size_changes, 95)) if len(size_changes) else 0,
        "conf_stability": float(np.median(conf_stds)) if conf_stds else 0,
    }


def compute_cross_camera_metrics(cam_data):
    cams = sorted(cam_data.keys())
    if len(cams) < 2:
        return {"cams_agree_mean": 0, "cams_agree_dist": {}, "cams_agree_values": np.array([])}
    n_frames = min(len(cam_data[c]) for c in cams)
    agree = np.zeros(n_frames)
    for c in cams:
        agree += (cam_data[c]["confidence"].values[:n_frames] > 0.3).astype(float)
    dist = {i: float((agree == i).mean()) for i in range(len(cams) + 1)}
    return {"cams_agree_mean": float(np.mean(agree)), "cams_agree_dist": dist, "cams_agree_values": agree}


def compute_cross_model_metrics(data_a, data_b):
    cams = set(data_a.keys()) & set(data_b.keys())
    if not cams:
        return {}
    ious, conf_pairs = [], []
    agree_both = agree_a = agree_b = agree_none = total = 0
    for cam in sorted(cams):
        da, db = data_a[cam], data_b[cam]
        n = min(len(da), len(db))
        for fi in range(n):
            ra, rb = da.iloc[fi], db.iloc[fi]
            ca, cb = ra["confidence"] > 0.3, rb["confidence"] > 0.3
            if ca and cb:
                agree_both += 1
                inter_x1 = max(ra["x1"], rb["x1"]); inter_y1 = max(ra["y1"], rb["y1"])
                inter_x2 = min(ra["x2"], rb["x2"]); inter_y2 = min(ra["y2"], rb["y2"])
                inter = max(0, inter_x2-inter_x1) * max(0, inter_y2-inter_y1)
                union = ra["area"] + rb["area"] - inter
                ious.append(inter / max(union, 1e-6))
                conf_pairs.append((ra["confidence"], rb["confidence"]))
            elif ca: agree_a += 1
            elif cb: agree_b += 1
            else: agree_none += 1
            total += 1
    m = {}
    if total:
        m = {"both_detect": agree_both/total, "a_only": agree_a/total,
             "b_only": agree_b/total, "neither": agree_none/total}
    if ious:
        ious = np.array(ious)
        m["iou_median"] = float(np.median(ious))
        m["iou_mean"] = float(np.mean(ious))
        m["iou_values"] = ious
    if conf_pairs:
        cp = np.array(conf_pairs)
        m["conf_correlation"] = float(np.corrcoef(cp[:,0], cp[:,1])[0,1])
        m["conf_a_values"] = cp[:, 0]
        m["conf_b_values"] = cp[:, 1]
    return m


def main():
    parser = argparse.ArgumentParser(description="YOLO Model Evaluation Suite")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--models", nargs="+", required=True,
                        help="name:path pairs (e.g. yolo_0626:best.pt)")
    parser.add_argument("--out", default="./yolo_eval_output")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--max-sessions", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    for spec in args.models:
        name, path = spec.split(":", 1)
        models[name] = path

    sessions = find_sessions(args.test_dir)
    print(f"Found {len(sessions)} test sessions")
    if args.max_sessions > 0:
        sessions = sessions[:args.max_sessions]

    # Run inference per model
    all_results = {name: {} for name in models}
    t0 = time.time()

    for model_name, model_path in models.items():
        print(f"\n=== {model_name} ===")

        # Check cache completeness
        cached = 0
        for s in sessions:
            cache_path = out_dir / "cache" / model_name / f"{s.name}.parquet"
            if cache_path.exists():
                cached += 1
        if cached == len(sessions):
            print(f"  All {cached} sessions cached, loading...")
            for s in sessions:
                cache_path = out_dir / "cache" / model_name / f"{s.name}.parquet"
                df = pd.read_parquet(cache_path)
                cam_data = {c: g.drop(columns=["cam"]).reset_index(drop=True)
                            for c, g in df.groupby("cam")}
                all_results[model_name][s.name] = cam_data
            continue

        # Build/load TRT engine
        engine_path = ensure_engine(model_path, imgsz=args.imgsz, batch=args.batch)
        trt_model, batch_buf, batch_size, imsz, names, device = load_trt_model(engine_path, args.gpu)
        print(f"  Engine: batch={batch_size}, imsz={imsz}")

        for si, session in enumerate(sessions):
            sess_name = session.name
            cache_path = out_dir / "cache" / model_name / f"{sess_name}.parquet"

            if cache_path.exists():
                df = pd.read_parquet(cache_path)
                cam_data = {c: g.drop(columns=["cam"]).reset_index(drop=True)
                            for c, g in df.groupby("cam")}
                all_results[model_name][sess_name] = cam_data
                continue

            t_sess = time.time()
            cam_data = run_yolo_on_session(trt_model, batch_buf, batch_size, imsz, session, args.gpu)
            elapsed_sess = time.time() - t_sess
            n_frames = sum(len(df) for df in cam_data.values())
            fps = n_frames / max(elapsed_sess, 0.001)

            all_results[model_name][sess_name] = cam_data

            # Cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            dfs = []
            for cam, df in cam_data.items():
                dc = df.copy()
                dc["cam"] = cam
                dfs.append(dc)
            if dfs:
                pd.concat(dfs).to_parquet(cache_path, index=False)

            print(f"  [{si+1}/{len(sessions)}] {sess_name}: {n_frames} frames, {fps:.0f} fps")

    elapsed = time.time() - t0
    print(f"\nInference done in {elapsed:.0f}s")

    # Compute metrics
    print("\nComputing metrics...")
    model_metrics = {}
    model_names = list(models.keys())

    for model_name in model_names:
        all_cam_data = {}
        for sess_name, cam_data in all_results[model_name].items():
            for cam, df in cam_data.items():
                all_cam_data[f"{sess_name}_{cam}"] = df

        det = compute_detection_metrics(all_cam_data)
        temp = compute_temporal_metrics(all_cam_data)

        per_sess_xcam = {}
        for sess_name, cam_data in all_results[model_name].items():
            per_sess_xcam[sess_name] = compute_cross_camera_metrics(cam_data)

        # Aggregate cross-camera across sessions
        all_agree = np.concatenate([v["cams_agree_values"] for v in per_sess_xcam.values()
                                    if len(v["cams_agree_values"]) > 0])
        n_cams = max(len(next(iter(all_results[model_name].values()))), 5)
        agg_dist = {i: float((all_agree == i).mean()) for i in range(n_cams + 1)}

        model_metrics[model_name] = {
            "detection": det,
            "temporal": temp,
            "cross_camera": {
                "cams_agree_mean": float(np.mean(all_agree)),
                "cams_agree_dist": agg_dist,
                "cams_agree_values": all_agree,
            },
            "per_session_cross_camera": per_sess_xcam,
        }

        print(f"  {model_name}: det@0.3={det['det_rate'][0.3]:.1%}, "
              f"conf_med={det['conf_median']:.3f}, jitter={temp['jitter_median']:.1f}px")

    # Cross-model
    cross_model = {}
    if len(model_names) >= 2:
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                a, b = model_names[i], model_names[j]
                all_cross = []
                for sess in all_results[a]:
                    if sess in all_results[b]:
                        cm = compute_cross_model_metrics(all_results[a][sess], all_results[b][sess])
                        if cm:
                            all_cross.append(cm)
                if all_cross:
                    cross_model[f"{a}_vs_{b}"] = {
                        "both_detect": np.mean([c["both_detect"] for c in all_cross]),
                        "a_only": np.mean([c["a_only"] for c in all_cross]),
                        "b_only": np.mean([c["b_only"] for c in all_cross]),
                        "neither": np.mean([c["neither"] for c in all_cross]),
                        "iou_median": np.mean([c.get("iou_median", 0) for c in all_cross]),
                        "conf_correlation": np.mean([c.get("conf_correlation", 0) for c in all_cross]),
                        "iou_values": np.concatenate([c.get("iou_values", []) for c in all_cross]),
                    }
                    print(f"  {a} vs {b}: IoU={cross_model[f'{a}_vs_{b}']['iou_median']:.3f}")

    from report_generator import generate_report
    report_path = out_dir / "report.html"
    generate_report(model_metrics, cross_model, model_names, sessions, report_path)


if __name__ == "__main__":
    main()
