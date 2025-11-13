from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import animation


def plot_coverage_over_time(history: List[Dict]) -> None:
    coverage_sizes = [len(h["coverage"]) for h in history]
    plt.figure(figsize=(6, 3))
    plt.plot(range(len(history)), coverage_sizes, marker="o")
    plt.xlabel("Round")
    plt.ylabel("# nodes exposed & believing > 0")
    plt.title("Coverage Over Time")
    plt.grid(alpha=0.3)
    plt.show()


def plot_final_beliefs(G: nx.Graph, beliefs: Dict[int, float], pos: Optional[Dict[int, np.ndarray]] = None) -> None:
    belief_arr = np.array([beliefs[i] for i in G.nodes()])
    if pos is None:
        pos = nx.spring_layout(G, seed=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    node_colors = plt.cm.viridis(belief_arr)
    nx.draw_networkx(G, pos=pos, with_labels=False, node_size=60, node_color=node_colors, width=0.3, ax=ax)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array(belief_arr)
    fig.colorbar(sm, ax=ax, label="Belief strength")
    ax.set_title("Final Beliefs")
    ax.set_axis_off()
    plt.show()


def plot_belief_trajectories(history: List[Dict], node_ids: List[int], ylim: Optional[List[float]] = None) -> None:
    """Plot per-round belief values for selected node IDs using the history list."""
    rounds = list(range(len(history)))
    plt.figure(figsize=(6, 3))
    for node_id in node_ids:
        ys = [float(h.get("beliefs", {}).get(node_id, np.nan)) for h in history]
        plt.plot(rounds, ys, marker="o", label=f"node {node_id}")
    plt.xlabel("Round")
    plt.ylabel("Belief (0-1)")
    if ylim is None:
        plt.ylim(0, 1)
    else:
        plt.ylim(ylim[0], ylim[1])
    plt.title("Belief Trajectories")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


def belief_trajectories_table(history: List[Dict], node_ids: Optional[List[int]] = None):
    """Return a pandas DataFrame of belief trajectories. Columns: round and one column per node."""
    import pandas as pd  # local import to avoid hard dependency at module import time
    num_rounds = len(history)
    rounds = list(range(num_rounds))
    if node_ids is None:
        # infer all nodes from round 0 beliefs
        node_ids = sorted(list(history[0].get("beliefs", {}).keys()))
    data: Dict[str, List[float]] = {"round": rounds}
    for nid in node_ids:
        data[str(nid)] = [float(history[r].get("beliefs", {}).get(nid, np.nan)) for r in rounds]
    return pd.DataFrame(data)


def animate_network(history: List[Dict], G: nx.Graph, interval_ms: int = 600, figsize=(6, 6)):
    """Return a matplotlib.animation.FuncAnimation showing node belief changes over rounds.

    Node color encodes belief [0,1] using viridis; edges drawn lightly. Pass the
    final `G` and the full `history` list (round 0..T).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=0)
    edges = nx.draw_networkx_edges(G, pos=pos, alpha=0.2, width=0.5, ax=ax)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, label="Belief")

    def frame_beliefs(t: int):
        return [float(history[t]["beliefs"].get(i, 0.0)) for i in G.nodes()]

    node_scatter = None

    def init():
        nonlocal node_scatter
        beliefs0 = frame_beliefs(0)
        colors = plt.cm.viridis(np.array(beliefs0))
        xs = [pos[i][0] for i in G.nodes()]
        ys = [pos[i][1] for i in G.nodes()]
        node_scatter = ax.scatter(xs, ys, c=colors, s=60)
        ax.set_axis_off()
        ax.set_title("Belief evolution (t=0)")
        return node_scatter,

    def update(frame):
        nonlocal node_scatter
        beliefs = frame_beliefs(frame)
        node_scatter.set_color(plt.cm.viridis(np.array(beliefs)))
        ax.set_title(f"Belief evolution (t={frame})")
        return node_scatter,

    ani = animation.FuncAnimation(fig, update, init_func=init, frames=len(history), interval=interval_ms, blit=True)
    return ani


def show_animation(history: List[Dict], G: nx.Graph, interval_ms: int = 600, figsize=(6, 6)):
    """Display the animation; uses JS HTML in notebooks, falls back to plt.show()."""
    ani = animate_network(history, G, interval_ms=interval_ms, figsize=figsize)
    try:
        from IPython.display import HTML, display  # type: ignore
        html = ani.to_jshtml()
        import matplotlib.pyplot as plt
        try:
            plt.close(ani._fig)  # avoid duplicate static figure display
        except Exception:
            pass
        display(HTML(html))
    except Exception:
        import matplotlib.pyplot as plt
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


def plot_group_beliefs_over_time(history: List[Dict], personas: List, attr: str = "political", groups: Optional[List[str]] = None, figsize=(7, 3)) -> None:
    """Plot mean belief over time for groups defined by a Persona attribute (e.g., 'political').

    - history: list of round dictionaries with 'beliefs'
    - personas: list of Persona objects (must have attribute 'attr' or key in extra)
    - attr: attribute name on Persona (falls back to extra dict if missing)
    - groups: optional explicit group order/filter; otherwise inferred from personas
    """
    # Collect group membership
    def get_group(p):
        val = getattr(p, attr, None)
        if val is None and getattr(p, "extra", None):
            val = p.extra.get(attr, None)
        return str(val) if val is not None else "Unknown"

    all_groups = [get_group(p) for p in personas]
    uniq = sorted(list(dict.fromkeys(all_groups)))
    if groups is not None:
        uniq = [g for g in groups if g in set(all_groups)]
    group_to_indices: Dict[str, List[int]] = {g: [] for g in uniq}
    for i, g in enumerate(all_groups):
        if g in group_to_indices:
            group_to_indices[g].append(i)

    rounds = list(range(len(history)))
    plt.figure(figsize=figsize)
    for g, idxs in group_to_indices.items():
        if not idxs:
            continue
        ys = []
        for h in history:
            beliefs = h.get("beliefs", {})
            vals = [float(beliefs.get(i, np.nan)) for i in idxs]
            ys.append(float(np.nanmean(vals)) if len(vals) else np.nan)
        plt.plot(rounds, ys, marker="o", label=str(g))
    plt.xlabel("Round")
    plt.ylabel("Mean belief (0-1)")
    plt.ylim(0, 1)
    plt.title(f"Group mean beliefs by '{attr}'")
    plt.grid(alpha=0.3)
    plt.legend(title=attr, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_centrality_vs_belief_exposure(G: nx.Graph, history: List[Dict], metric: str = "degree", jitter: float = 0.02, figsize=(8, 3)) -> None:
    """Scatter plots: centrality vs final belief, and centrality vs exposure (0/1).

    metric: 'degree' | 'betweenness' | 'eigenvector'
    """
    final = history[-1]
    beliefs = final.get("beliefs", {})
    coverage = set(final.get("coverage", set()))

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
    ys_belief = []
    ys_exp = []
    for i in G.nodes():
        xs.append(float(cent.get(i, 0.0)))
        ys_belief.append(float(beliefs.get(i, 0.0)))
        ys_exp.append(1.0 if i in coverage else 0.0)

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].scatter(xs, ys_belief, s=20, alpha=0.7)
    ax[0].set_xlabel(f"{metric} centrality")
    ax[0].set_ylabel("Final belief")
    ax[0].set_title("Centrality vs Belief")
    ax[0].grid(alpha=0.3)

    # Small jitter for exposure for visibility
    ys_exp_jitter = np.array(ys_exp) + np.random.uniform(-jitter, jitter, size=len(ys_exp))
    ax[1].scatter(xs, ys_exp_jitter, s=20, alpha=0.7, color="crimson")
    ax[1].set_xlabel(f"{metric} centrality")
    ax[1].set_ylabel("Exposure (0/1)")
    ax[1].set_title("Centrality vs Exposure")
    ax[1].set_yticks([0, 1])
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_intervention_effect(history: List[Dict], intervention_round: Optional[int] = None, figsize=(6, 3)) -> None:
    """Plot coverage with a vertical line at the intervention round and annotate pre/post slopes."""
    cov = [len(h.get("coverage", set())) for h in history]
    rounds = list(range(len(history)))

    def slope(x, y):
        if len(x) < 2:
            return np.nan
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)
        A = np.vstack([x_arr, np.ones_like(x_arr)]).T
        m, _b = np.linalg.lstsq(A, y_arr, rcond=None)[0]
        return float(m)

    plt.figure(figsize=figsize)
    plt.plot(rounds, cov, marker="o", label="coverage")
    if intervention_round is not None and 0 <= intervention_round < len(history):
        plt.axvline(intervention_round, color="red", linestyle="--", label=f"intervention at t={intervention_round}")
        pre_x = list(range(0, intervention_round + 1))
        post_x = list(range(intervention_round, len(history)))
        pre_s = slope(pre_x, cov[: len(pre_x)])
        post_s = slope(post_x, cov[intervention_round :])
        plt.text(intervention_round + 0.1, max(cov) * 0.2, f"pre-slope≈{pre_s:.2f}\npost-slope≈{post_s:.2f}", color="red")
    plt.xlabel("Round")
    plt.ylabel("# nodes exposed & believing > 0")
    plt.title("Intervention effect on coverage")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


