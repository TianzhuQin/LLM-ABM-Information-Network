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


