from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


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


