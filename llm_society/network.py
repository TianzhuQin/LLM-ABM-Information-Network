import random
import networkx as nx
import numpy as np


def build_random_network(n: int, mean_degree: int, seed: int = 7) -> nx.Graph:
    p = min(0.99, max(0.0, mean_degree / max(1, n - 1)))
    G = nx.gnp_random_graph(n, p, seed=seed)
    if not nx.is_connected(G) and n > 1:
        comps = [list(c) for c in nx.connected_components(G)]
        for i in range(len(comps) - 1):
            u = random.choice(comps[i])
            v = random.choice(comps[i + 1])
            G.add_edge(u, v)
    for u, v in G.edges():
        G[u][v]["weight"] = float(0.2 + 0.8 * np.random.beta(2, 2))
    return G



