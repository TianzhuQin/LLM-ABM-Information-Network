from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PLOTLY_AVAILABLE = False
BOKEH_AVAILABLE = False

try:  # pragma: no cover
    from plotly.subplots import make_subplots  # type: ignore
    import plotly.graph_objects as go  # type: ignore

    PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover
    from bokeh.plotting import figure  # type: ignore
    from bokeh.layouts import column  # type: ignore

    BOKEH_AVAILABLE = True
except Exception:
    pass


def _scores_for_round(entry: Dict[str, Any], metric_id: Optional[str]) -> Dict[int, float]:
    if metric_id and "scores_multi" in entry:
        multi = entry.get("scores_multi") or {}
        data = multi.get(metric_id)
        if isinstance(data, dict):
            return data
    return entry.get("scores", {})


def _mean_scores(history: List[Dict[str, Any]], metric_id: Optional[str]) -> List[float]:
    means: List[float] = []
    for entry in history:
        smap = _scores_for_round(entry, metric_id)
        if smap:
            arr = np.array(list(smap.values()), dtype=float)
            means.append(float(np.nanmean(arr)))
        else:
            means.append(float("nan"))
    return means


def _coverage_counts(history: List[Dict[str, Any]]) -> List[int]:
    return [len(entry.get("coverage", [])) for entry in history]


def _group_indices(personas: Optional[Sequence[Any]], attr: Optional[str]) -> Dict[str, List[int]]:
    if personas is None or attr is None:
        return {}

    mapping: Dict[str, List[int]] = {}
    for idx, persona in enumerate(personas):
        val = getattr(persona, attr, None)
        if val is None and getattr(persona, "extra", None):
            val = persona.extra.get(attr, None)
        key = str(val) if val is not None else "Unknown"
        mapping.setdefault(key, []).append(idx)
    return mapping


def _group_means(history: List[Dict[str, Any]], groups: Dict[str, List[int]], metric_id: Optional[str]) -> Dict[str, List[float]]:
    series: Dict[str, List[float]] = {name: [] for name in groups}
    for entry in history:
        smap = _scores_for_round(entry, metric_id)
        for name, idxs in groups.items():
            if not idxs:
                series[name].append(float("nan"))
                continue
            vals = [float(smap.get(i, np.nan)) for i in idxs]
            arr = np.array(vals, dtype=float)
            series[name].append(float(np.nanmean(arr)) if arr.size else float("nan"))
    return series


def _select_groups(personas: Optional[Sequence[Any]], attr: Optional[str], groups: Optional[Sequence[str]], metric_id: Optional[str], history: List[Dict[str, Any]]) -> Tuple[Dict[str, List[int]], Dict[str, List[float]]]:
    group_indices = _group_indices(personas, attr)
    if groups is not None and group_indices:
        selected = []
        for g in groups:
            if g in group_indices:
                selected.append((g, group_indices[g]))
        group_indices = dict(selected)
    group_means = _group_means(history, group_indices, metric_id) if group_indices else {}
    return group_indices, group_means


def _plotly_dashboard(  # pragma: no cover - thin wrapper around Plotly
    *,
    history: List[Dict[str, Any]],
    personas: Optional[Sequence[Any]],
    metric_label: str,
    metric_id: Optional[str],
    attr: Optional[str],
    groups: Optional[Sequence[str]],
) -> "go.Figure":
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly is not installed. Run `pip install plotly`.")

    rounds = list(range(len(history)))
    coverage = _coverage_counts(history)
    means = _mean_scores(history, metric_id)
    base_scores = _scores_for_round(history[-1], metric_id) or _scores_for_round(history[0], metric_id)
    n_nodes = len(base_scores)

    _, group_means = _select_groups(personas, attr, groups, metric_id, history)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(f"Coverage (max {max(coverage)} of {n_nodes})", f"{metric_label} Over Time"),
    )

    fig.add_trace(go.Scatter(x=rounds, y=coverage, mode="lines+markers", name="Coverage", line=dict(color="#2E6FBE")), row=1, col=1)
    fig.add_trace(go.Scatter(x=rounds, y=means, mode="lines+markers", name=f"Mean {metric_label}", line=dict(color="#D2453D")), row=2, col=1)

    palette = ["#FF6B6B", "#4ECDC4", "#FFA600", "#556EE6", "#1ABC9C", "#F06595", "#F4A261", "#2A9D8F"]
    for idx, (name, series) in enumerate(group_means.items()):
        fig.add_trace(
            go.Scatter(x=rounds, y=series, mode="lines", name=name, line=dict(color=palette[idx % len(palette)], width=2)),
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text="Round", row=2, col=1)
    fig.update_yaxes(title_text="# Nodes", row=1, col=1)
    fig.update_yaxes(title_text=f"{metric_label} (0-1)", row=2, col=1, range=[0, 1])
    fig.update_layout(height=720, title_text="LLM Society Interactive Dashboard", legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def _bokeh_dashboard(  # pragma: no cover - thin wrapper around Bokeh
    *,
    history: List[Dict[str, Any]],
    personas: Optional[Sequence[Any]],
    metric_label: str,
    metric_id: Optional[str],
    attr: Optional[str],
    groups: Optional[Sequence[str]],
) -> "Any":
    if not BOKEH_AVAILABLE:
        raise RuntimeError("Bokeh is not installed. Run `pip install bokeh`.")

    rounds = list(range(len(history)))
    coverage = _coverage_counts(history)
    means = _mean_scores(history, metric_id)
    base_scores = _scores_for_round(history[-1], metric_id) or _scores_for_round(history[0], metric_id)
    n_nodes = len(base_scores)

    _, group_means = _select_groups(personas, attr, groups, metric_id, history)

    cov_fig = figure(height=300, sizing_mode="stretch_width", title=f"Coverage (max {max(coverage)} of {n_nodes})", x_axis_label="Round", y_axis_label="# Nodes")
    cov_fig.line(rounds, coverage, line_width=3, color="#2E6FBE", legend_label="Coverage")
    cov_fig.circle(rounds, coverage, size=6, color="#2E6FBE")
    cov_fig.legend.location = "top_left"

    score_fig = figure(height=360, sizing_mode="stretch_width", title=f"{metric_label} Over Time", x_axis_label="Round", y_axis_label=f"{metric_label} (0-1)")
    score_fig.line(rounds, means, line_width=3, color="#D2453D", legend_label=f"Mean {metric_label}")
    palette = ["#FF6B6B", "#4ECDC4", "#FFA600", "#556EE6", "#1ABC9C", "#F06595", "#F4A261", "#2A9D8F"]
    for idx, (name, series) in enumerate(group_means.items()):
        score_fig.line(rounds, series, line_width=2, color=palette[idx % len(palette)], legend_label=name, line_dash="dashed")
    score_fig.legend.location = "top_left"
    score_fig.y_range.start = 0
    score_fig.y_range.end = 1

    return column(cov_fig, score_fig)


def build_dashboard(
    *,
    history: List[Dict[str, Any]],
    personas: Optional[Sequence[Any]],
    metric_label: str,
    metric_id: Optional[str] = None,
    engine: str = "plotly",
    attr: Optional[str] = None,
    groups: Optional[Sequence[str]] = None,
) -> Tuple[str, object]:
    """Return (engine, dashboard_obj) for interactive use."""
    if not history:
        raise ValueError("history is empty; run simulate() first.")

    engine_normalized = (engine or "plotly").strip().lower()
    if engine_normalized == "plotly":
        fig = _plotly_dashboard(
            history=history,
            personas=personas,
            metric_label=metric_label,
            metric_id=metric_id,
            attr=attr,
            groups=groups,
        )
        return "plotly", fig
    if engine_normalized == "bokeh":
        layout = _bokeh_dashboard(
            history=history,
            personas=personas,
            metric_label=metric_label,
            metric_id=metric_id,
            attr=attr,
            groups=groups,
        )
        return "bokeh", layout
    raise ValueError("engine must be 'plotly' or 'bokeh'")



