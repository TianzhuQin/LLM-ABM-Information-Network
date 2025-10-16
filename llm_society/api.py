from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import DEFAULTS, load_config
from .simulation import run_simulation, iterate_simulation
from . import viz


class NodeProxy:
    def __init__(self, net: "Network", node_id: int) -> None:
        self._net = net
        self.id = node_id

    def plot(self) -> None:
        if self._net._history is None:
            raise RuntimeError("Call simulate() before plotting node trajectories.")
        viz.plot_belief_trajectories(self._net._history, [self.id])


class Network:
    def __init__(
        self,
        *,
        information: str,
        n: int = DEFAULTS["n"],
        degree: int = DEFAULTS["edge_mean_degree"],
        rounds: int = DEFAULTS["rounds"],
        depth: float = DEFAULTS["depth"],
        depth_max: int = DEFAULTS["max_convo_turns"],
        edge_frac: float = DEFAULTS["edge_sample_frac"],
        seeds: Optional[List[int]] = None,
        seed_belief: float = DEFAULTS["seed_belief"],
        talk_prob: float = DEFAULTS["talk_information_prob"],
        mode: str = DEFAULTS["contagion_mode"],
        complex_k: int = DEFAULTS["complex_threshold_k"],
        stop_when_stable: bool = DEFAULTS["stop_when_stable"],
        stability_tol: float = DEFAULTS["stability_tol"],
        rng: int = DEFAULTS["rng_seed"],
        api_key_file: str = DEFAULTS["api_key_file"],
        segments: Optional[List[Dict[str, Any]]] = None,
        model: str = DEFAULTS["model"],
        print_conversations: bool = DEFAULTS["print_conversations"],
        print_belief_updates: bool = DEFAULTS["print_belief_updates"],
        print_round_summaries: bool = DEFAULTS["print_round_summaries"],
        print_all_conversations: bool = DEFAULTS["print_all_conversations"],
    ) -> None:
        if not isinstance(information, str) or information.strip() == "":
            raise ValueError("'information' must be a non-empty string.")
        self.information = information.strip()

        self.n = int(n)
        self.degree = int(degree)
        self.rounds = int(rounds)
        # depth: 0-1 intensity for conversation length tendency
        self.depth = int(depth_max)
        self.depth_intensity = float(max(0.0, min(1.0, depth)))
        self.edge_frac = float(edge_frac)
        self.seeds = list(seeds) if seeds is not None else list(DEFAULTS["seed_nodes"])  # copy
        self.seed_belief = float(seed_belief)
        self.talk_prob = float(talk_prob)
        self.mode = str(mode)
        self.complex_k = int(complex_k)
        self.stop_when_stable = bool(stop_when_stable)
        self.stability_tol = float(stability_tol)
        self.rng = int(rng)
        self.api_key_file = str(api_key_file)
        self.segments = list(segments) if segments is not None else []
        self.model = str(model)
        self.print_conversations = bool(print_conversations)
        self.print_belief_updates = bool(print_belief_updates)
        self.print_round_summaries = bool(print_round_summaries)
        self.print_all_conversations = bool(print_all_conversations)

        self._result: Optional[Dict[str, Any]] = None
        self._history: Optional[List[Dict[str, Any]]] = None
        self._beliefs: Optional[Dict[int, float]] = None
        self._G = None
        self.nodes: List[NodeProxy] = []

    def _make_cfg(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "n": self.n,
            "edge_mean_degree": self.degree,
            "rounds": self.rounds,
            "depth": self.depth_intensity,
            "max_convo_turns": self.depth,
            "edge_sample_frac": self.edge_frac,
            "seed_nodes": list(self.seeds),
            "seed_belief": self.seed_belief,
            "information_text": self.information,
            "talk_information_prob": self.talk_prob,
            "contagion_mode": self.mode,
            "complex_threshold_k": self.complex_k,
            "stop_when_stable": self.stop_when_stable,
            "stability_tol": self.stability_tol,
            "rng_seed": self.rng,
            "api_key_file": self.api_key_file,
            "persona_segments": list(self.segments),
            "print_conversations": self.print_conversations,
            "print_belief_updates": self.print_belief_updates,
            "print_round_summaries": self.print_round_summaries,
            "print_all_conversations": self.print_all_conversations,
        }

    def simulate(self) -> None:
        cfg = self._make_cfg()
        self._result = run_simulation(cfg)
        self._history = self._result["history"]
        self._beliefs = self._result["beliefs"]
        self._G = self._result["G"]
        self.nodes = [NodeProxy(self, i) for i in range(self.n)]

    def step(self) -> bool:
        """Advance the simulation by one round, preserving accumulated history.

        Returns True if a new step was produced, False if finished.
        """
        if getattr(self, "_iter", None) is None:
            self._iter = iterate_simulation(self._make_cfg())
            self._history = []
        try:
            state = next(self._iter)
        except StopIteration:
            return False
        self._G = state["G"]
        self._beliefs = dict(state["beliefs"])
        if self._history is None:
            self._history = []
        self._history.append(state["history_entry"])
        if not self.nodes:
            self.nodes = [NodeProxy(self, i) for i in range(self.n)]
        return True

    def plot(self, save: Optional[str] = None) -> None:
        if self._history is None or self._beliefs is None or self._G is None:
            raise RuntimeError("Call simulate() before plot().")
        ani = viz.show_animation(self._history, self._G)
        if save:
            viz.save_animation(ani, save)

    @property
    def history(self) -> List[Dict[str, Any]]:
        if self._history is None:
            raise RuntimeError("Call simulate() first to populate history.")
        return self._history

    @property
    def beliefs(self) -> Dict[int, float]:
        if self._beliefs is None:
            raise RuntimeError("Call simulate() first to populate beliefs.")
        return self._beliefs


def network(*, information: str, config: Optional[Dict[str, Any]] = None, config_file: Optional[str] = None, **kwargs: Any) -> Network:
    """Factory supporting three call styles:
    - network(information=..., n=..., degree=..., ...)
    - network(information=..., config={...})
    - network(information=..., config_file="path.yaml")

    'information' is required and must be a non-empty string.
    """
    if not isinstance(information, str) or information.strip() == "":
        raise ValueError("'information' must be a non-empty string.")

    # Back-compat: allow claim in kwargs but enforce non-empty
    if "claim" in kwargs and not kwargs.get("information"):
        claim_val = str(kwargs.pop("claim"))
        if claim_val.strip() != "":
            kwargs["information"] = claim_val

    if config_file is not None or config is not None:
        src = config_file if config_file is not None else config  # type: ignore
        cfg = load_config(src)
        return Network(
            information=information,
            n=int(cfg["n"]),
            degree=int(cfg["edge_mean_degree"]),
            rounds=int(cfg["rounds"]),
            depth=float(cfg.get("depth", cfg.get("convo_depth_p", DEFAULTS["depth"]))),
            depth_max=int(cfg["max_convo_turns"]),
            edge_frac=float(cfg["edge_sample_frac"]),
            seeds=list(cfg["seed_nodes"]),
            seed_belief=float(cfg["seed_belief"]),
            talk_prob=float(cfg.get("talk_information_prob", DEFAULTS["talk_information_prob"])),
            mode=str(cfg["contagion_mode"]),
            complex_k=int(cfg["complex_threshold_k"]),
            stop_when_stable=bool(cfg["stop_when_stable"]),
            stability_tol=float(cfg["stability_tol"]),
            rng=int(cfg["rng_seed"]),
            api_key_file=str(cfg["api_key_file"]),
            segments=list(cfg.get("persona_segments", [])),
            model=str(cfg["model"]),
            print_conversations=bool(cfg["print_conversations"]),
            print_belief_updates=bool(cfg["print_belief_updates"]),
            print_round_summaries=bool(cfg["print_round_summaries"]),
            print_all_conversations=bool(cfg["print_all_conversations"]),
        )

    # kwargs style
    return Network(information=information, **kwargs)


