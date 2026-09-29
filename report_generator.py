#!/usr/bin/env python3
"""Generate interactive HTML report from YOLO evaluation metrics."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
          '#1abc9c', '#e67e22', '#34495e', '#e91e63', '#00bcd4',
          '#ff5722', '#607d8b', '#8bc34a', '#673ab7', '#009688',
          '#ff9800']  # 16 colors for up to 16 models


def _fig_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _winner(model_metrics, model_names, key, higher_is_better=True):
    vals = {}
    for name in model_names:
        det = model_metrics[name]["detection"]
        temp = model_metrics[name]["temporal"]
        xcam = model_metrics[name]["cross_camera"]
        lookup = {
            "det_rate_03": det["det_rate"][0.3],
            "det_rate_05": det["det_rate"][0.5],
            "conf_median": det["conf_median"],
            "jitter": temp["jitter_median"],
            "cams_agree": xcam["cams_agree_mean"],
        }
        if key in lookup:
            vals[name] = lookup[key]
    if not vals:
        return ""
    winner = max(vals, key=vals.get) if higher_is_better else min(vals, key=vals.get)
    return f'<span style="background:#2ecc71;color:white;padding:3px 10px;border-radius:12px;font-size:12px;">Best: {winner}</span>'


def make_detection_rate_chart(model_metrics, model_names):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["All Thresholds", "Zoomed: 0.5 – 0.95"],
                        horizontal_spacing=0.08)
    for mi, name in enumerate(model_names):
        det = model_metrics[name]["detection"]
        thresholds = sorted(det["det_rate"].keys())
        rates = [det["det_rate"][t] for t in thresholds]
        fig.add_trace(go.Scatter(
            x=thresholds, y=rates, name=name,
            mode="lines+markers", line=dict(color=COLORS[mi % len(COLORS)], width=3),
            marker=dict(size=8), legendgroup=name, showlegend=True,
            hovertemplate="%{y:.1%} at conf>%{x}<extra>%{fullData.name}</extra>",
        ), row=1, col=1)
        # Zoomed
        hi_thresh = [t for t in thresholds if t >= 0.5]
        hi_rates = [det["det_rate"][t] for t in hi_thresh]
        fig.add_trace(go.Scatter(
            x=hi_thresh, y=hi_rates, name=name,
            mode="lines+markers", line=dict(color=COLORS[mi % len(COLORS)], width=3),
            marker=dict(size=8), legendgroup=name, showlegend=False,
        ), row=1, col=2)
    fig.update_layout(
        title="Detection Rate vs Confidence Threshold",
        template="plotly_white", height=450,
        legend=dict(font=dict(size=14), orientation='h', yanchor='bottom', y=1.02),
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    fig.update_xaxes(title_text="Confidence Threshold", row=1, col=1)
    fig.update_xaxes(title_text="Confidence Threshold", row=1, col=2)
    return _fig_html(fig)


def make_confidence_chart(model_metrics, model_names):
    """Three panels: all, detected>0.3, detected>0.5."""
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["All Frames", "Detected (conf > 0.3)", "Detected (conf > 0.5)"],
                        horizontal_spacing=0.05)
    for mi, name in enumerate(model_names):
        confs = model_metrics[name]["detection"]["conf_values"]
        fig.add_trace(go.Violin(
            y=confs, name=name, legendgroup=name, showlegend=True,
            marker_color=COLORS[mi % len(COLORS)],
            box_visible=True, meanline_visible=True, opacity=0.7,
            scalemode="width", width=0.8, points=False,
        ), row=1, col=1)
        det03 = confs[confs > 0.3]
        fig.add_trace(go.Violin(
            y=det03, name=name, legendgroup=name, showlegend=False,
            marker_color=COLORS[mi % len(COLORS)],
            box_visible=True, meanline_visible=True, opacity=0.7,
            scalemode="width", width=0.8, points=False,
        ), row=1, col=2)
        det05 = confs[confs > 0.5]
        fig.add_trace(go.Violin(
            y=det05, name=name, legendgroup=name, showlegend=False,
            marker_color=COLORS[mi % len(COLORS)],
            box_visible=True, meanline_visible=True, opacity=0.7,
            scalemode="width", width=0.8, points=False,
        ), row=1, col=3)
    fig.update_layout(
        title="YOLO Confidence Distribution",
        template="plotly_white", height=450,
        legend=dict(font=dict(size=13), orientation='h', yanchor='bottom', y=1.05),
    )
    for c in [1, 2, 3]:
        fig.update_yaxes(title_text="Confidence", row=1, col=c)
    return _fig_html(fig)


def make_per_cam_chart(model_metrics, model_names):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Detection Rate (conf > 0.3)", "Detection Rate (conf > 0.5)"],
                        horizontal_spacing=0.08)
    for mi, name in enumerate(model_names):
        per_cam = model_metrics[name]["detection"]["per_cam_det_rate"]
        from collections import defaultdict
        cam_rates_03 = defaultdict(list)
        cam_rates_05 = defaultdict(list)
        for key, rates in per_cam.items():
            cam = key if key.startswith("cam") else key.split("_cam")[-1]
            if not cam.startswith("cam"):
                cam = f"cam{cam}"
            cam_rates_03[cam].append(rates.get(0.3, 0))
            cam_rates_05[cam].append(rates.get(0.5, 0))
        cams = sorted(cam_rates_03.keys())
        fig.add_trace(go.Bar(
            x=cams, y=[np.mean(cam_rates_03[c]) for c in cams], name=name,
            marker_color=COLORS[mi % len(COLORS)], opacity=0.85,
            legendgroup=name, showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=cams, y=[np.mean(cam_rates_05[c]) for c in cams], name=name,
            marker_color=COLORS[mi % len(COLORS)], opacity=0.85,
            legendgroup=name, showlegend=False,
        ), row=1, col=2)
    fig.update_layout(barmode="group", template="plotly_white", height=400,
                      legend=dict(font=dict(size=13), orientation='h', yanchor='bottom', y=1.05))
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    return _fig_html(fig)


def make_jitter_charts(model_metrics, model_names):
    """4 panels: jitter full, jitter zoomed, size change full, size change zoomed."""
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Center Jitter (full range, clipped p95)",
                                        "Center Jitter (zoomed: 0–5px)",
                                        "Size Change (full range, clipped p95)",
                                        "Size Change (zoomed: 0–10px)"],
                        vertical_spacing=0.12, horizontal_spacing=0.08)
    for mi, name in enumerate(model_names):
        temp = model_metrics[name]["temporal"]
        jitter = temp.get("jitter_values", np.array([]))
        size_ch = temp.get("size_change_values", np.array([]))

        if len(jitter) > 0:
            clip95 = np.percentile(jitter, 95)
            fig.add_trace(go.Violin(
                y=jitter[jitter <= clip95], name=name, legendgroup=name, showlegend=True,
                marker_color=COLORS[mi % len(COLORS)],
                box_visible=True, meanline_visible=True, opacity=0.7,
                scalemode="width", width=0.8, points=False,
            ), row=1, col=1)
            # Zoomed
            zoomed = jitter[jitter <= 5]
            fig.add_trace(go.Violin(
                y=zoomed, name=name, legendgroup=name, showlegend=False,
                marker_color=COLORS[mi % len(COLORS)],
                box_visible=True, meanline_visible=True, opacity=0.7,
                scalemode="width", width=0.8, points=False,
            ), row=1, col=2)

        if len(size_ch) > 0:
            clip95 = np.percentile(size_ch, 95)
            fig.add_trace(go.Violin(
                y=size_ch[size_ch <= clip95], name=name, legendgroup=name, showlegend=False,
                marker_color=COLORS[mi % len(COLORS)],
                box_visible=True, meanline_visible=True, opacity=0.7,
                scalemode="width", width=0.8, points=False,
            ), row=2, col=1)
            zoomed = size_ch[size_ch <= 10]
            fig.add_trace(go.Violin(
                y=zoomed, name=name, legendgroup=name, showlegend=False,
                marker_color=COLORS[mi % len(COLORS)],
                box_visible=True, meanline_visible=True, opacity=0.7,
                scalemode="width", width=0.8, points=False,
            ), row=2, col=2)

    fig.update_layout(
        title="Temporal Stability — Lower is Better",
        template="plotly_white", height=700,
        legend=dict(font=dict(size=13), orientation='h', yanchor='bottom', y=1.03),
    )
    fig.update_yaxes(title_text="Pixels", row=1, col=1)
    fig.update_yaxes(title_text="Pixels", range=[0, 5], row=1, col=2)
    fig.update_yaxes(title_text="Pixels", row=2, col=1)
    fig.update_yaxes(title_text="Pixels", range=[0, 10], row=2, col=2)
    return _fig_html(fig)


def make_cross_camera_charts(model_metrics, model_names):
    """Multiple cross-camera views: agreement bar + cumulative + deficit heatmap."""
    # 1. Standard agreement bar
    fig1 = go.Figure()
    for mi, name in enumerate(model_names):
        dist = model_metrics[name]["cross_camera"].get("cams_agree_dist", {})
        if not dist:
            continue
        n_cams = sorted(dist.keys())
        rates = [dist[c] for c in n_cams]
        fig1.add_trace(go.Bar(
            x=[f"{c}/5" for c in n_cams], y=rates, name=name,
            marker_color=COLORS[mi % len(COLORS)], opacity=0.85,
            hovertemplate="%{x} cams: %{y:.1%}<extra>%{fullData.name}</extra>",
        ))
    fig1.update_layout(
        title="Simultaneous Detection: How Many Cameras See the Mouse",
        yaxis_title="Fraction of Frames", yaxis=dict(tickformat=".0%"),
        barmode="group", template="plotly_white", height=400,
        legend=dict(font=dict(size=13), orientation='h', yanchor='bottom', y=1.02),
    )

    # 2. "Usable frames" — frames with ≥3 cameras (pipeline minimum)
    fig3 = go.Figure()
    for mi, name in enumerate(model_names):
        dist = model_metrics[name]["cross_camera"].get("cams_agree_dist", {})
        max_n = max(dist.keys()) if dist else 5
        usable_3 = sum(dist.get(k, 0) for k in range(3, max_n + 1))
        usable_4 = sum(dist.get(k, 0) for k in range(4, max_n + 1))
        usable_5 = dist.get(5, 0) + dist.get(max_n, 0) if max_n > 5 else dist.get(5, 0)
        fig3.add_trace(go.Bar(
            x=["≥3 cams (LP minimum)", "≥4 cams", "5/5 cams (ideal)"],
            y=[usable_3, usable_4, usable_5], name=name,
            marker_color=COLORS[mi % len(COLORS)], opacity=0.85,
            text=[f"{usable_3:.1%}", f"{usable_4:.1%}", f"{usable_5:.1%}"],
            textposition="outside",
        ))
    fig3.update_layout(
        title="Usable Frames for Pipeline (≥3 cameras required for LP, ≥5 ideal for 3D)",
        yaxis_title="Fraction of Frames", yaxis=dict(tickformat=".0%", range=[0, 1.1]),
        barmode="group", template="plotly_white", height=400,
        legend=dict(font=dict(size=13), orientation='h', yanchor='bottom', y=1.02),
    )

    return _fig_html(fig1) + _fig_html(fig3)


def make_cross_model_charts(cross_model, model_names):
    if not cross_model:
        return "<p>Only one model — no cross-model comparison.</p>"

    sections = []

    # Summary heatmap if 3+ models
    if len(model_names) >= 3:
        n = len(model_names)
        iou_matrix = np.full((n, n), 1.0)
        agree_matrix = np.full((n, n), 1.0)
        for key, cm in cross_model.items():
            a, b = key.split("_vs_")
            ai, bi = model_names.index(a), model_names.index(b)
            iou_matrix[ai, bi] = iou_matrix[bi, ai] = cm.get("iou_median", 0)
            agree_matrix[ai, bi] = agree_matrix[bi, ai] = cm.get("both_detect", 0)

        fig_heat = make_subplots(rows=1, cols=2,
                                 subplot_titles=["Both-Detect Rate", "Median IoU (when both detect)"],
                                 horizontal_spacing=0.12)
        fig_heat.add_trace(go.Heatmap(
            z=agree_matrix, x=model_names, y=model_names,
            colorscale="Greens", zmin=0.5, zmax=1,
            text=np.vectorize(lambda v: f"{v:.1%}")(agree_matrix),
            texttemplate="%{text}", textfont=dict(size=12),
        ), row=1, col=1)
        fig_heat.add_trace(go.Heatmap(
            z=iou_matrix, x=model_names, y=model_names,
            colorscale="Blues", zmin=0.5, zmax=1,
            text=np.vectorize(lambda v: f"{v:.3f}")(iou_matrix),
            texttemplate="%{text}", textfont=dict(size=12),
        ), row=1, col=2)
        fig_heat.update_layout(template="plotly_white", height=400,
                               title="Cross-Model Agreement Summary")
        sections.append(_fig_html(fig_heat))
    for key, cm in cross_model.items():
        a, b = key.split("_vs_")

        # Stat cards
        stats_html = f"""
        <div style="display:flex; gap:15px; margin:15px 0; flex-wrap:wrap;">
            <div style="text-align:center; padding:15px 25px; background:#d5f5e3; border-radius:8px;">
                <div style="font-size:28px; font-weight:bold; color:#27ae60;">{cm['both_detect']:.1%}</div>
                <div style="color:#666; font-size:12px;">Both Detect</div>
            </div>
            <div style="text-align:center; padding:15px 25px; background:#d6eaf8; border-radius:8px;">
                <div style="font-size:28px; font-weight:bold; color:#2980b9;">{cm.get('iou_median',0):.3f}</div>
                <div style="color:#666; font-size:12px;">Median IoU</div>
            </div>
            <div style="text-align:center; padding:15px 25px; background:#fadbd8; border-radius:8px;">
                <div style="font-size:28px; font-weight:bold; color:#c0392b;">{cm['a_only']:.1%}</div>
                <div style="color:#666; font-size:12px;">{a} Only</div>
            </div>
            <div style="text-align:center; padding:15px 25px; background:#d7bde2; border-radius:8px;">
                <div style="font-size:28px; font-weight:bold; color:#8e44ad;">{cm['b_only']:.1%}</div>
                <div style="color:#666; font-size:12px;">{b} Only</div>
            </div>
            <div style="text-align:center; padding:15px 25px; background:#f2f3f4; border-radius:8px;">
                <div style="font-size:28px; font-weight:bold; color:#95a5a6;">{cm['neither']:.1%}</div>
                <div style="color:#666; font-size:12px;">Neither</div>
            </div>
        </div>
        """

        # Pie charts: overall + by threshold
        fig1 = make_subplots(rows=1, cols=2,
                             subplot_titles=["Detection Agreement (conf > 0.3)",
                                             "Agreement by Category"],
                             specs=[[{"type": "pie"}, {"type": "xy"}]],
                             horizontal_spacing=0.1)

        fig1.add_trace(go.Pie(
            labels=["Both", f"{a} only", f"{b} only", "Neither"],
            values=[cm["both_detect"], cm["a_only"], cm["b_only"], cm["neither"]],
            marker_colors=["#2ecc71", "#e74c3c", "#3498db", "#bdc3c7"],
            hole=0.4, textinfo="percent+label", textfont=dict(size=12),
        ), row=1, col=1)

        # Stacked bar showing agreement breakdown
        categories = ["Both", f"{a} only", f"{b} only", "Neither"]
        values = [cm["both_detect"], cm["a_only"], cm["b_only"], cm["neither"]]
        colors = ["#2ecc71", "#e74c3c", "#3498db", "#bdc3c7"]
        for i, (cat, val, col) in enumerate(zip(categories, values, colors)):
            fig1.add_trace(go.Bar(
                x=["Agreement"], y=[val], name=cat,
                marker_color=col, showlegend=False,
                text=f"{val:.1%}", textposition="inside",
                textfont=dict(color="white", size=13),
            ), row=1, col=2)

        fig1.update_layout(template="plotly_white", height=380,
                           barmode="stack", title=f"Cross-Model: {a} vs {b}")
        fig1.update_yaxes(tickformat=".0%", row=1, col=2)

        # IoU histogram (cleaner)
        fig2 = go.Figure()
        iou_vals = cm.get("iou_values", np.array([]))
        if len(iou_vals) > 0:
            fig2.add_trace(go.Histogram(
                x=iou_vals, nbinsx=40,
                marker_color="#2ecc71", opacity=0.8,
                marker_line=dict(width=0.5, color="#27ae60"),
            ))
            med = np.median(iou_vals)
            fig2.add_shape(type="line", x0=med, x1=med, y0=0, y1=1,
                           yref="paper", line=dict(dash="dash", color="red", width=2))
            fig2.add_annotation(x=med, y=0.95, yref="paper",
                                text=f"median = {med:.3f}", showarrow=False,
                                font=dict(color="red", size=13))
        fig2.update_layout(
            title=f"Bbox IoU When Both Detect — How Much Do the Boxes Overlap?",
            xaxis_title="IoU (1.0 = identical boxes)",
            yaxis_title="Frame Count",
            template="plotly_white", height=380,
            xaxis=dict(range=[0, 1.05]),
        )

        # Detection gain/loss chart — what does each model uniquely find?
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=[f"{a} detects<br>{b} misses", f"{b} detects<br>{a} misses"],
            y=[cm["a_only"], cm["b_only"]],
            marker_color=[COLORS[0], COLORS[1]],
            text=[f"{cm['a_only']:.1%}", f"{cm['b_only']:.1%}"],
            textposition="outside", textfont=dict(size=14),
        ))
        fig3.update_layout(
            title="Unique Detections — What Each Model Finds That the Other Misses",
            yaxis_title="Fraction of Frames", yaxis=dict(tickformat=".1%"),
            template="plotly_white", height=350,
        )

        sections.append(stats_html + _fig_html(fig1) + _fig_html(fig2) + _fig_html(fig3))

    return "\n".join(sections)


def make_summary_table(model_metrics, model_names):
    headers = """<tr>
        <th>Model</th>
        <th>Det @0.3</th><th>Det @0.5</th><th>Det @0.7</th>
        <th>Conf Median</th><th>Conf Mean</th>
        <th>Zero Det %</th><th>Multi Det %</th>
        <th>Jitter med/p95</th>
        <th>Conf Stability</th>
        <th>Cams ≥3</th><th>Cams = 5</th>
    </tr>"""
    rows = ""
    for name in model_names:
        det = model_metrics[name]["detection"]
        temp = model_metrics[name]["temporal"]
        xcam = model_metrics[name]["cross_camera"]
        dist = xcam.get("cams_agree_dist", {})
        max_n = max(dist.keys()) if dist else 5
        usable_3 = sum(dist.get(k, 0) for k in range(3, max_n + 1))
        usable_5 = dist.get(5, 0)
        rows += f"""<tr>
            <td><b>{name}</b></td>
            <td>{det['det_rate'][0.3]:.1%}</td>
            <td>{det['det_rate'][0.5]:.1%}</td>
            <td>{det['det_rate'][0.7]:.1%}</td>
            <td>{det['conf_median']:.3f}</td>
            <td>{det['conf_mean']:.3f}</td>
            <td>{det['zero_det_rate']:.1%}</td>
            <td>{det['multi_det_rate']:.1%}</td>
            <td>{temp['jitter_median']:.1f} / {temp['jitter_p95']:.1f} px</td>
            <td>{temp['conf_stability']:.4f}</td>
            <td>{usable_3:.1%}</td>
            <td>{usable_5:.1%}</td>
        </tr>"""
    return f'<table class="summary-table"><thead>{headers}</thead><tbody>{rows}</tbody></table>'


def generate_report(model_metrics, cross_model, model_names, sessions, output_path):
    from datetime import datetime
    desc = lambda t: f'<p style="color:#666; font-size:14px; margin:5px 0 15px 0;">{t}</p>'
    n_frames = sum(m["detection"]["conf_values"].shape[0] for m in model_metrics.values())

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>YOLO Model Evaluation</title>
<script src="https://cdn.plot.ly/plotly-2.20.0.min.js"></script>
<style>
    body {{ font-family: Inter, -apple-system, sans-serif; margin: 20px 40px; background: #fafafa; color: #2c3e50; }}
    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
    h2 {{ color: #34495e; margin-top: 30px; }}
    .summary-table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }}
    .summary-table th {{ background: #2c3e50; color: white; padding: 8px 12px; text-align: center; white-space: nowrap; }}
    .summary-table td {{ padding: 6px 12px; text-align: center; border-bottom: 1px solid #eee; }}
    .summary-table tr:hover {{ background: #f0f8ff; }}
    .section {{ background: white; border-radius: 8px; padding: 20px 25px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .meta {{ color: #888; font-size: 13px; margin-bottom: 20px; }}
</style>
</head><body>

<h1>YOLO Model Evaluation Report</h1>
<div class="meta">
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} |
    Models: {', '.join(model_names)} |
    Sessions: {len(sessions)} |
    Total frames: {n_frames:,}
</div>

<div class="section">
<h2>Summary</h2>
{desc("All key metrics. Cams ≥3 = fraction of frames usable by LP pipeline. Cams = 5 = ideal for 3D triangulation.")}
{make_summary_table(model_metrics, model_names)}
</div>

<div class="section">
<h2>Detection Rate {_winner(model_metrics, model_names, 'det_rate_03')}</h2>
{desc("Left: full threshold range. Right: zoomed to high-confidence detections (0.5–0.95). The gap between models at high thresholds shows which model is more reliably confident.")}
{make_detection_rate_chart(model_metrics, model_names)}
</div>

<div class="section">
<h2>Confidence Distribution {_winner(model_metrics, model_names, 'conf_median')}</h2>
{desc("Three views of confidence: all frames (including misses at 0), detected frames above 0.3, and above 0.5. A better model pushes the distribution higher. Box shows quartiles, line shows median.")}
{make_confidence_chart(model_metrics, model_names)}
</div>

<div class="section">
<h2>Per-Camera Detection</h2>
{desc("Detection rate per camera view averaged across all sessions. A robust model should be consistent across all 5 views.")}
{make_per_cam_chart(model_metrics, model_names)}
</div>

<div class="section">
<h2>Temporal Stability {_winner(model_metrics, model_names, 'jitter', higher_is_better=False)}</h2>
{desc("Top row: frame-to-frame bbox center displacement (jitter). Bottom row: frame-to-frame bbox size change. Left: full distribution clipped to p95. Right: zoomed into the small-value range where differences matter. Lower = smoother tracking.")}
{make_jitter_charts(model_metrics, model_names)}
</div>

<div class="section">
<h2>Cross-Camera Agreement {_winner(model_metrics, model_names, 'cams_agree')}</h2>
{desc("Three views: (1) How many cameras detect simultaneously — 5/5 should dominate. (2) Cumulative — what fraction of frames have at least N cameras. (3) Pipeline usability — ≥3 cameras is the minimum for LP, 5/5 is ideal for 3D triangulation.")}
{make_cross_camera_charts(model_metrics, model_names)}
</div>

<div class="section">
<h2>Cross-Model Comparison</h2>
{desc("Head-to-head comparison. Agreement pie shows how often both/one/neither model detects. IoU histogram shows bbox overlap when both detect (1.0 = identical boxes). Unique detections chart shows what each model finds that the other misses.")}
{make_cross_model_charts(cross_model, model_names)}
</div>

</body></html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Report saved: {output_path}")
