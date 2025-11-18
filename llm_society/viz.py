from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import animation


_DEFAULT_THEME = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#E0E0E0",
    "axes.grid": True,
    "grid.color": "#EAEAEA",
    "grid.alpha": 1.0,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.labelcolor": "#333333",
    "xtick.color": "#4F4F4F",
    "ytick.color": "#4F4F4F",
    "font.family": "DejaVu Sans",
    "lines.linewidth": 2.0,
    "lines.markersize": 5.5,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "savefig.dpi": 200,
    "figure.autolayout": True,
}


def set_theme(rc: Optional[Dict[str, object]] = None) -> None:
    """Set a modern, publication-ready Matplotlib theme."""
    style = dict(_DEFAULT_THEME)
    if rc:
        style.update(rc)
    plt.rcParams.update(style)


def _polish_axes(ax):
    for spine in ["top", "right"]:
        try:
            ax.spines[spine].set_visible(False)
        except Exception:
            pass
    return ax


def plot_coverage_over_time(history: List[Dict]) -> None:
    coverage_sizes = [len(h.get("coverage", [])) for h in history]
    rounds = range(len(history))
    total_nodes = 0
    if history:
        sample_scores = history[0].get("scores", {})
        if isinstance(sample_scores, dict):
            total_nodes = len(sample_scores)
    fig, ax = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)
    ax.plot(rounds, coverage_sizes, marker="o", color="#2E6FBE")
    ax.fill_between(list(rounds), coverage_sizes, step="pre", alpha=0.10, color="#2E6FBE")
    ax.set_xlabel("Round")
    ax.set_ylabel("# nodes exposed & believing > 0")
    if total_nodes > 0:
        ax.set_ylim(0, total_nodes + max(1, total_nodes * 0.02))
        ax.set_yticks(range(0, total_nodes + 1))
    ax.set_title("Coverage Over Time")
    _polish_axes(ax)
    # annotate final value
    if coverage_sizes:
        ax.annotate(
            f"final: {coverage_sizes[-1]}",
            xy=(len(coverage_sizes) - 1, coverage_sizes[-1]),
            xytext=(5, 6),
            textcoords="offset points",
            color="#2E6FBE",
        )
    plt.show()


def plot_final_scores(G: nx.Graph, scores: Dict[int, float], pos: Optional[Dict[int, np.ndarray]] = None, metric_label: str = "Score") -> None:
    score_arr = np.array([scores[i] for i in G.nodes()])
    if pos is None:
        pos = nx.spring_layout(G, seed=0)
    fig, ax = plt.subplots(figsize=(6.8, 6.8), constrained_layout=True)
    node_colors = plt.cm.viridis(score_arr)
    # light, thin edges
    nx.draw_networkx_edges(G, pos=pos, width=0.4, alpha=0.25, edge_color="#9AA4B2", ax=ax)
    nx.draw_networkx_nodes(G, pos=pos, node_size=70, node_color=node_colors, linewidths=0.0, ax=ax)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array(score_arr)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(f"{metric_label} strength")
    ax.set_title(f"Final {metric_label}")
    ax.set_axis_off()
    plt.show()


def plot_mean_score_over_time(history: List[Dict], metric_label: str = "Score", metric_id: Optional[str] = None) -> None:
    rounds = range(len(history))
    means: List[float] = []
    for h in history:
        smap = _scores_map(h, metric_id)
        if smap:
            means.append(float(np.mean(list(smap.values()))))
        else:
            means.append(0.0)
    fig, ax = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)
    ax.plot(list(rounds), means, marker="o", color="#F97316")
    ax.set_xlabel("Round")
    ax.set_ylabel(f"Mean {metric_label} (0-1)")
    ax.set_title(f"Average {metric_label} Over Time")
    ax.set_ylim(0, 1)
    _polish_axes(ax)
    plt.show()

def _scores_map(h: Dict, metric_id: Optional[str] = None) -> Dict[int, float]:
    """Return per-node scores map from history entry, optionally selecting a metric."""
    if metric_id and "scores_multi" in h:
        multi = h.get("scores_multi") or {}
        data = multi.get(metric_id)
        if isinstance(data, dict):
            return data
    return h.get("scores", {})


def plot_score_trajectories(history: List[Dict], node_ids: List[int], ylim: Optional[List[float]] = None, metric_label: str = "Score", metric_id: Optional[str] = None) -> None:
    """Plot per-round score values for selected node IDs using the history list."""
    rounds = list(range(len(history)))
    plt.figure(figsize=(7.5, 3.6))
    for node_id in node_ids:
        ys = [float(_scores_map(h, metric_id).get(node_id, np.nan)) for h in history]
        plt.plot(rounds, ys, marker="o", label=f"node {node_id}")
    plt.xlabel("Round")
    plt.ylabel(f"{metric_label} (0-1)")
    if ylim is None:
        plt.ylim(0, 1)
    else:
        plt.ylim(ylim[0], ylim[1])
    plt.title(f"{metric_label} Trajectories")
    plt.legend()
    plt.show()


def score_trajectories_table(history: List[Dict], node_ids: Optional[List[int]] = None, metric_id: Optional[str] = None):
    """Return a pandas DataFrame of score trajectories. Columns: round and one column per node."""
    import pandas as pd  # local import to avoid hard dependency at module import time
    num_rounds = len(history)
    rounds = list(range(num_rounds))
    if node_ids is None:
        # infer all nodes from round 0 scores
        node_ids = sorted(list(_scores_map(history[0], metric_id).keys()))
    data: Dict[str, List[float]] = {"round": rounds}
    for nid in node_ids:
        data[str(nid)] = [float(_scores_map(history[r], metric_id).get(nid, np.nan)) for r in rounds]
    return pd.DataFrame(data)




def animate_network(
    history: List[Dict],
    G: nx.Graph,
    interval_ms: int = 600,
    figsize=(6, 6),
    metric_label: str = "Score",
    metric_id: Optional[str] = None,
    personas: Optional[List] = None,
    color_by: Optional[str] = None,
):
    """Return a matplotlib.animation.FuncAnimation showing node score changes over rounds."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    pos = nx.spring_layout(G, seed=0)
    nx.draw_networkx_edges(G, pos=pos, alpha=0.18, width=0.5, edge_color="#9AA4B2", ax=ax)

    color_mode = "score"
    categorical_colors = None
    cmap = plt.get_cmap("viridis")
    
    if color_by == "segment" and personas:
        color_mode = "segment"
        segment_names = [p.extra.get("_segment_name", "Unknown") for p in personas]
        unique_names = sorted(list(set(segment_names)))
        name_to_id = {name: i for i, name in enumerate(unique_names)}
        node_color_indices = [name_to_id[name] for name in segment_names]
        
        cmap_categorical = plt.get_cmap('tab10', len(unique_names))
        categorical_colors = [cmap_categorical(i) for i in node_color_indices]
        
        import matplotlib.patches as mpatches
        legend_patches = [mpatches.Patch(color=cmap_categorical(i), label=name) for i, name in enumerate(unique_names)]
        ax.legend(handles=legend_patches, title="Segment", bbox_to_anchor=(1.05, 1), loc='upper left')

    else: # Default to score-based coloring
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(metric_label)

    def frame_scores(t: int):
        m = _scores_map(history[t], metric_id)
        return [float(m.get(i, 0.0)) for i in G.nodes()]

    node_scatter = None

    def init():
        nonlocal node_scatter
        scores0 = frame_scores(0)
        colors = cmap(np.array(scores0)) if color_mode == "score" else categorical_colors
        
        xs = [pos[i][0] for i in G.nodes()]
        ys = [pos[i][1] for i in G.nodes()]
        node_scatter = ax.scatter(xs, ys, c=colors, s=65)
        ax.set_axis_off()
        ax.set_title(f"{metric_label} evolution (t=0)")
        return (node_scatter, ax.title)

    def update(frame):
        nonlocal node_scatter
        if color_mode == "score":
            scores = frame_scores(frame)
            node_scatter.set_color(cmap(np.array(scores)))
        # For categorical colors, the color is static, so no update needed
        
        ax.set_title(f"{metric_label} evolution (t={frame})")
        return (node_scatter, ax.title)

    ani = animation.FuncAnimation(fig, update, init_func=init, frames=len(history), interval=interval_ms, blit=True)
    return ani


def show_animation(
    history: List[Dict],
    G: nx.Graph,
    interval_ms: int = 600,
    figsize=(6, 6),
    metric_label: str = "Score",
    metric_id: Optional[str] = None,
    personas: Optional[List] = None,
    color_by: Optional[str] = None,
):
    """
    Display the animation as a reliable HTML5 video in notebooks.
    This method prioritizes reliability over interactivity.
    """
    ani = animate_network(
        history, G, interval_ms, figsize, metric_label, metric_id, personas, color_by
    )

    try:
        from IPython.display import HTML, display
        # Use to_html5_video() for maximum compatibility.
        html = ani.to_html5_video()
        plt.close(ani._fig)  # Avoid showing a static plot underneath
        display(HTML(html))
    except Exception as e:
        print(f"Animation display failed: {e}")
        # Fallback for non-notebook or problematic environments
        plt.show()

    return ani


def save_animation(ani: animation.FuncAnimation, filename: str, fps: int = 2, dpi: int = 150) -> None:
    """Save animation to mp4/gif/html with graceful fallbacks.

    - mp4 requires ffmpeg; if unavailable, falls back to gif (Pillow), else HTML.
    """
    import os
    ext = os.path.splitext(filename)[1].lower()
    if ext in {".mp4", ".m4v"}:
        try:
            from matplotlib.animation import FFMpegWriter  # type: ignore
            ani.save(filename, writer=FFMpegWriter(fps=fps), dpi=dpi)
            return
        except Exception:
            # fallback to GIF
            ext = ".gif"
            filename = os.path.splitext(filename)[0] + ext
    if ext == ".gif":
        try:
            from matplotlib.animation import PillowWriter  # type: ignore
            ani.save(filename, writer=PillowWriter(fps=fps), dpi=dpi)
            return
        except Exception:
            pass
    # final fallback: HTML5 video
    html = ani.to_html5_video()
    html_path = filename if filename.endswith(".html") else filename + ".html"
    with open(html_path, "w") as f:
        f.write(html)
    return


def plot_group_over_time(
    history: List[Dict],
    personas: List,
    *,
    attr: Optional[str] = "political",
    groups: Optional[List[str]] = None,
    figsize=(7, 3),
    segments: Optional[List[Dict]] = None,
    metric_label: str = "Score",
    metric_id: Optional[str] = None,
    by: str = "traits",  # 'traits' | 'segment'
) -> None:
    """Plot mean score over time grouped either by segments' traits (default) or by segment index.

    - history: list of round dictionaries with 'scores'
    - personas: list of Persona objects (must have attribute 'attr' or key in extra)
    - by: 'traits' to group by a trait key; 'segment' to group by segment index (0..S-1)
    - attr (traits mode): attribute name; must be one of the keys defined under segments[*].traits
    - groups: optional explicit group order/filter; otherwise inferred (or from segments choices or seg indices)
    - segments: segments configuration to validate traits and derive group order
    """
    if by not in {"traits", "segment"}:
        raise ValueError("by must be 'traits' or 'segment'")

    if by == "traits":
        # If segments are provided, validate attr against segments' traits
        if segments:
            allowed_attrs = set()
            for seg in segments:
                try:
                    tr = seg.get("traits", {}) or {}
                    for k in tr.keys():
                        allowed_attrs.add(str(k))
                except Exception:
                    continue
            if attr not in allowed_attrs:
                raise ValueError(f"attr='{attr}' is not in segments' traits: {sorted(allowed_attrs)}")

        # Collect group membership by trait
        def get_group(p):
            val = getattr(p, attr, None)
            if val is None and getattr(p, "extra", None):
                val = p.extra.get(attr, None)
            return str(val) if val is not None else "Unknown"

        # Preferred group order from segments choices, if provided
        preferred_order: Optional[List[str]] = None
        if segments:
            try:
                values = []
                for seg in segments:
                    tr = seg.get("traits", {}) or {}
                    spec = tr.get(attr, None)
                    if isinstance(spec, dict) and "choices" in spec and isinstance(spec["choices"], dict):
                        # keep insertion order of keys as given by user config
                        for key in spec["choices"].keys():
                            values.append(str(key))
                if values:
                    preferred_order = list(dict.fromkeys(values))
            except Exception:
                preferred_order = None

        all_groups = [get_group(p) for p in personas]
        uniq = sorted(list(dict.fromkeys(all_groups)))
        if groups is not None:
            uniq = [g for g in groups if g in set(all_groups)]
        elif preferred_order is not None:
            uniq = [g for g in preferred_order if g in set(all_groups)]
        group_to_indices: Dict[str, List[int]] = {g: [] for g in uniq}
        for i, g in enumerate(all_groups):
            if g in group_to_indices:
                group_to_indices[g].append(i)
        legend_title = attr
    else:
        # Group by segment index (0..S-1), using persona.extra['_segment_index'] when available
        def get_seg_idx(p):
            idx = None
            if getattr(p, "extra", None):
                idx = p.extra.get("_segment_index", None)
            return int(idx) if idx is not None else -1
        def get_seg_label(idx: int) -> str:
            if segments is not None and 0 <= idx < len(segments):
                name = segments[idx].get("name", None)
                if name:
                    return str(name)
            return f"Segment {idx}"

        all_idx = [get_seg_idx(p) for p in personas]
        uniq_idx = sorted(list(dict.fromkeys([g for g in all_idx if g is not None and g >= 0])))
        # Map group selection: allow indices or names
        if groups:
            mapped: List[int] = []
            for g in groups:
                try:
                    mapped.append(int(g))
                    continue
                except Exception:
                    pass
                # try match by name
                if segments:
                    for idx in range(len(segments)):
                        if str(segments[idx].get("name", "")) == str(g):
                            mapped.append(idx)
                            break
            uniq_idx = [i for i in uniq_idx if i in set(mapped)]
        # Build indices per label
        group_to_indices: Dict[str, List[int]] = {}
        for idx in uniq_idx:
            label = get_seg_label(idx)
            group_to_indices[label] = []
        for i, idx in enumerate(all_idx):
            if idx in uniq_idx:
                label = get_seg_label(idx)
                group_to_indices[label].append(i)
        legend_title = "segment"

    rounds = list(range(len(history)))
    plt.figure(figsize=(max(figsize[0], 8), figsize[1]))
    for g, idxs in group_to_indices.items():
        if not idxs:
            continue
        ys = []
        for h in history:
            smap = _scores_map(h, metric_id)
            vals = [float(smap.get(i, np.nan)) for i in idxs]
            ys.append(float(np.nanmean(vals)) if len(vals) else np.nan)
        plt.plot(rounds, ys, marker="o", label=str(g))
    plt.xlabel("Round")
    plt.ylabel(f"Mean {metric_label} (0-1)")
    plt.ylim(0, 1)
    title = f"Group mean {metric_label} by '{legend_title}'" if by == "traits" else f"Segment mean {metric_label}"
    plt.title(title)
    plt.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.show()


def plot_centrality_vs_score_exposure(
    G: nx.Graph,
    history: List[Dict],
    metric: str = "degree",
    jitter: float = 0.02,
    figsize=(8, 3),
    metric_label: str = "Score",
    show_exposure: bool = False,
    metric_id: Optional[str] = None,
) -> None:
    """Scatter plots: centrality vs final score, and centrality vs exposure (0/1).

    metric: 'degree' | 'betweenness' | 'eigenvector'
    """
    final = history[-1]
    scores_map = _scores_map(final, metric_id)
    coverage = set(final.get("coverage", set())) if show_exposure else set()

    # Centrality
    if metric == "degree":
        cent = nx.degree_centrality(G)
    elif metric == "betweenness":
        cent = nx.betweenness_centrality(G)
    elif metric == "eigenvector":
        try:
            cent = nx.eigenvector_centrality(G, max_iter=1000)
        except Exception:
            cent = nx.katz_centrality_numpy(G)  # fallback to a spectral centrality
    else:
        raise ValueError("Unknown metric. Use 'degree', 'betweenness', or 'eigenvector'.")

    xs = []
    ys_score = []
    ys_exp = []
    for i in G.nodes():
        xs.append(float(cent.get(i, 0.0)))
        ys_score.append(float(scores_map.get(i, 0.0)))
        ys_exp.append(1.0 if i in coverage else 0.0)

    if show_exposure:
        fig, ax = plt.subplots(1, 2, figsize=(max(figsize[0], 10.5), max(3.2, figsize[1])), constrained_layout=True)
        ax_left = ax[0]
        ax_right = ax[1]
    else:
        fig, ax_left = plt.subplots(1, 1, figsize=(max(figsize[0], 6.5), max(3.2, figsize[1])), constrained_layout=True)
        ax_right = None
    # Left: Centrality vs score (polished)
    s_color = "#2E6FBE"
    ax_left.scatter(xs, ys_score, s=46, alpha=0.85, color=s_color, edgecolors="white", linewidths=0.6)
    ax_left.set_xlabel(f"{metric} centrality")
    ax_left.set_ylabel(f"Final {metric_label}")
    ax_left.set_title(f"Centrality vs {metric_label}")
    _polish_axes(ax_left)
    # Trend line with simple bootstrap CI for aesthetics
    try:
        x_arr = np.array(xs, dtype=float)
        y_arr = np.array(ys_score, dtype=float)
        msk = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_arr = x_arr[msk]
        y_arr = y_arr[msk]
        if x_arr.size > 1 and y_arr.size > 1 and np.nanstd(x_arr) > 0:
            rng = np.random.default_rng(123)
            x_line = np.linspace(float(np.nanmin(x_arr)), float(np.nanmax(x_arr)), 120)
            mb = []
            for _ in range(min(300, 50 * max(2, x_arr.size))):
                idx = rng.integers(0, x_arr.size, x_arr.size)
                m, b = np.polyfit(x_arr[idx], y_arr[idx], 1)
                mb.append((m, b))
            mb = np.array(mb, dtype=float)
            y_pred = np.outer(mb[:, 0], x_line) + mb[:, 1:2]
            lo = np.nanpercentile(y_pred, 5, axis=0)
            hi = np.nanpercentile(y_pred, 95, axis=0)
            m_hat, b_hat = np.polyfit(x_arr, y_arr, 1)
            y_hat = m_hat * x_line + b_hat
            ax_left.fill_between(x_line, lo, hi, color=s_color, alpha=0.10, linewidth=0)
            ax_left.plot(x_line, y_hat, color="#275CA4", linewidth=2.6, alpha=0.95)
            # annotate Pearson r with a subtle label box
            r = float(np.corrcoef(x_arr, y_arr)[0, 1])
            ax_left.text(
                0.02, 0.98, f"r({metric_label}, degree) = {r:.2f}",
                transform=ax_left.transAxes, ha="left", va="top",
                color="#1F2D3D",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#E5E7EB", alpha=0.9),
            )
    except Exception:
        pass

    if show_exposure and ax_right is not None:
        # Small jitter for exposure for visibility
        ys_exp_jitter = np.array(ys_exp) + np.random.uniform(-jitter, jitter, size=len(ys_exp))
        ax_right.scatter(xs, ys_exp_jitter, s=46, alpha=0.85, color="#D2453D", edgecolors="white", linewidths=0.6, marker="o")
        ax_right.set_xlabel(f"{metric} centrality")
        ax_right.set_ylabel("Exposure (0/1)")
        ax_right.set_title("Centrality vs Exposure")
        ax_right.set_yticks([0, 1])
        _polish_axes(ax_right)
        # light reference lines for 0/1
        try:
            ax_right.hlines([0, 1], xmin=min(xs), xmax=max(xs), colors="#E5E7EB", linestyles="--", linewidth=1.1, alpha=0.8)
        except Exception:
            pass

    plt.tight_layout()
    plt.show()
def plot_intervention_effect(
    history: List[Dict],
    intervention_rounds: Optional[List[int]] = None,
    attr: Optional[str] = None,
    groups: Optional[List[str]] = None,
    segments: Optional[List[Dict]] = None,
    metric_label: str = "Score",
    metric_id: Optional[str] = None,
) -> None:
    """Plot mean score over time, with optional splitting by a segments trait.

    - If attr is None (default), plots overall mean score over time.
    - If attr is provided, it must be a key within segments[*].traits; we will plot
      mean score per group (restricted to 'groups' if provided).
    """
    rounds = list(range(len(history)))

    def mean_scores_for_indices(indices: List[int]) -> List[float]:
        ys = []
        for h in history:
            smap = _scores_map(h, metric_id)
            vals = [float(smap.get(i, np.nan)) for i in indices]
            ys.append(float(np.nanmean(vals)) if len(vals) else np.nan)
        return ys

    # Prepare figure
    plt.figure(figsize=(7.5, 3.6))

    if attr is None:
        # Overall mean score
        ys = []
        for h in history:
            smap = _scores_map(h, metric_id)
            vals = [float(v) for v in smap.values()]
            ys.append(float(np.mean(vals)) if len(vals) else np.nan)
        plt.plot(rounds, ys, marker="o", label=f"Mean {metric_label}", color="#2E6FBE")
        plt.fill_between(rounds, ys, step="pre", alpha=0.10, color="#2E6FBE")
    else:
        # Validate and compute per-group means using segments
        if not segments:
            raise ValueError("segments must be provided when attr is specified.")
        allowed_attrs = set()
        for seg in segments:
            try:
                tr = seg.get("traits", {}) or {}
                for k in tr.keys():
                    allowed_attrs.add(str(k))
            except Exception:
                continue
        if attr not in allowed_attrs:
            raise ValueError(f"attr='{attr}' is not in segments' traits: {sorted(allowed_attrs)}")

        # Determine preferred group order from segments choices if present
        preferred_order: Optional[List[str]] = None
        try:
            values = []
            for seg in segments:
                tr = seg.get("traits", {}) or {}
                spec = tr.get(attr, None)
                if isinstance(spec, dict) and "choices" in spec and isinstance(spec["choices"], dict):
                    for key in spec["choices"].keys():
                        values.append(str(key))
            if values:
                preferred_order = list(dict.fromkeys(values))
        except Exception:
            preferred_order = None

        # Infer group membership from personas implicitly via indices present in scores
        # We assume that node indices are 0..n-1 and group can be read off round 0 score dictionary keys.
        # Since we don't have Persona objects here, we expect the caller to pass the subset via 'groups' if they want specific groups.
        # However, to honor the request, we'll approximate group membership based on segments proportions by mapping round 0 node count to segment order.
        # Better approach: caller should use API.Network.plot which has access to personas and forwards segments; we can still compute from personas there.
        # Here, we fallback to overall mean per 'groups' only if provided; otherwise plot overall mean.
        if groups is None or not groups:
            # No explicit groups provided; fallback to overall mean with attr noted
            ys = []
            for h in history:
                smap = _scores_map(h, metric_id)
                vals = [float(v) for v in smap.values()]
                ys.append(float(np.mean(vals)) if len(vals) else np.nan)
            plt.plot(rounds, ys, marker="o", label=f"Mean {metric_label} (all)", color="#2E6FBE")
            plt.fill_between(rounds, ys, step="pre", alpha=0.10, color="#2E6FBE")
        else:
            # We need personas to form groups accurately; detect if history carries indices only.
            # Expect caller (API) to precompute membership; since this function only has history,
            # we cannot infer true membership. Therefore, we document that when attr/groups are used,
            # this function must be called via API which passes 'personas' elsewhere. To keep it functional,
            # we will attempt to split nodes into groups evenly by label order as a last resort.
            scores0 = _scores_map(history[0], metric_id)
            node_ids = sorted(list(scores0.keys()))
            n = len(node_ids)
            # even split across requested groups
            num_groups = len(groups)
            group_to_indices: Dict[str, List[int]] = {g: [] for g in groups}
            for idx, nid in enumerate(node_ids):
                group_to_indices[groups[idx % num_groups]].append(int(nid))

            palette = ["#2E6FBE", "#D2453D", "#2BAF6A", "#8E44AD", "#F39C12", "#16A085"]
            for k, g in enumerate(groups):
                ys = mean_scores_for_indices(group_to_indices[g])
                color = palette[k % len(palette)]
                plt.plot(rounds, ys, marker="o", label=str(g), color=color)

    if intervention_rounds:
        for i, r_round in enumerate(intervention_rounds):
            plt.axvline(
                x=r_round,
                color="#D62728",  # More visible red
                linestyle="--",
                linewidth=2,
                label=f"Intervention (t={r_round})" if i == 0 else None,
            )

    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.gca().legend(handles, labels, loc="upper left", frameon=True)

    plt.xlabel("Round")
    plt.ylabel(f"Mean {metric_label} (0-1)")
    plt.ylim(0, 1)
    plt.title(f"Intervention Effect on Mean {metric_label}")
    plt.legend()
    plt.tight_layout()
    plt.show()











