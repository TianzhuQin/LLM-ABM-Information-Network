from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import DEFAULTS, load_config
from .simulation import run_simulation, iterate_simulation
from . import viz
import networkx as nx
import json


class NodeProxy:
    def __init__(self, net: "Network", node_id: int) -> None:
        self._net = net
        self.id = node_id

    def plot(self) -> None:
        if self._net._history is None:
            raise RuntimeError("Call simulate() before plotting node trajectories.")
        viz.plot_belief_trajectories(self._net._history, [self.id], metric_label=self._net.metric_name.capitalize())


class Network:
    def __init__(
        self,
        *,
        information: str,
        n: int = DEFAULTS["n"],
        metric_name: str = DEFAULTS["metric_name"],
        metric_prompt: str = DEFAULTS["metric_prompt"],
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
        intervention_round: Optional[int] = DEFAULTS.get("intervention_round", None),
        intervention_nodes: Optional[List[int]] = None,
        intervention_content: str = DEFAULTS.get("intervention_content", ""),
        graph: Optional[nx.Graph] = None,
    ) -> None:
        if not isinstance(information, str) or information.strip() == "":
            raise ValueError("'information' must be a non-empty string.")
        self.information = information.strip()

        # If a custom graph is provided, override n from graph
        self._custom_graph = graph
        self.n = int(graph.number_of_nodes()) if graph is not None else int(n)
        self.metric_name = str(metric_name)
        self.metric_prompt = str(metric_prompt)
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
        self.intervention_round = intervention_round
        self.intervention_nodes = list(intervention_nodes) if intervention_nodes is not None else []
        self.intervention_content = str(intervention_content or "")

        self._result: Optional[Dict[str, Any]] = None
        self._history: Optional[List[Dict[str, Any]]] = None
        self._scores: Optional[Dict[int, float]] = None
        self._G = None
        self._personas: Optional[List[Any]] = None
        self.nodes: List[NodeProxy] = []

    def _make_cfg(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "n": self.n,
            "metric_name": self.metric_name,
            "metric_prompt": self.metric_prompt,
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
            "intervention_round": self.intervention_round,
            "intervention_nodes": list(self.intervention_nodes),
            "intervention_content": self.intervention_content,
            "G": self._custom_graph,
        }

    def simulate(self) -> None:
        cfg = self._make_cfg()
        self._result = run_simulation(cfg)
        self._history = self._result["history"]
        self._scores = self._result.get("scores", self._result.get("beliefs"))
        self._G = self._result["G"]
        self._personas = self._result["personas"]
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
        # prefer scores, fallback to beliefs for backward compatibility
        self._scores = dict(state.get("scores", state.get("beliefs", {})))
        self._personas = state["personas"]
        if self._history is None:
            self._history = []
        self._history.append(state["history_entry"])
        if not self.nodes:
            self.nodes = [NodeProxy(self, i) for i in range(self.n)]
        return True

    def plot(self, save: Optional[str] = None) -> None:
        if self._history is None or self._scores is None or self._G is None:
            raise RuntimeError("Call simulate() before plot().")
        ani = viz.show_animation(self._history, self._G, metric_label=self.metric_name.capitalize())
        if save:
            viz.save_animation(ani, save)

    def plot(
        self,
        type: str = "animation",
        save: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Plot different visualizations.

        type:
          - "animation": animated network (default; save supports mp4/gif/html)
          - "coverage": coverage over time
          - "final_beliefs": final belief heat map on the graph
          - "group_beliefs": mean belief over time by persona attribute
              kwargs: attr="political", groups=[...]
          - "centrality": centrality vs final belief/exposure scatter
              kwargs: metric="degree"|"betweenness"|"eigenvector"
          - "intervention_effect": coverage with intervention marker
              kwargs: intervention_round=int (optional; auto-detected if omitted)

        Backward-compat:
          - Accept legacy keyword 'plot_type' and map it to 'type' if provided.
          - If the first arg historically looked like a filename and not a known type,
            we treat it as 'save' and default to animation.
        """
        if self._history is None or self._scores is None or self._G is None:
            raise RuntimeError("Call simulate() before plot().")

        # Legacy support: allow plot_type kw to set 'type'
        if "plot_type" in kwargs and isinstance(kwargs["plot_type"], str):
            # only override if caller didn't explicitly pass a non-default 'type'
            if type == "animation":
                type = kwargs.pop("plot_type")
            else:
                kwargs.pop("plot_type")

        allowed = {"animation", "coverage", "final_scores", "final_beliefs", "group", "group_beliefs", "centrality", "intervention_effect"}
        # Back-compat rescue: if first arg looks like a filename and not a known type
        if type not in allowed and save is None:
            # treat plot_type as 'save' and default to animation
            save = str(type)
            type = "animation"

        if type == "animation":
            ani = viz.show_animation(self._history, self._G, metric_label=self.metric_name.capitalize())
            if save:
                viz.save_animation(ani, save)
            return

        if type == "coverage":
            viz.plot_coverage_over_time(self._history)
            return

        if type == "final_beliefs":
            # backward-compat: map to final_scores
            type = "final_scores"
        if type == "final_scores":
            viz.plot_final_scores(self._G, self._scores, metric_label=self.metric_name.capitalize())
            return

        if type in {"group", "group_beliefs"}:
            # 'group' is preferred; 'group_beliefs' kept for backward-compat
            by = kwargs.get("by", "traits")  # 'traits' | 'segment'
            attr = kwargs.get("attr", "political")
            groups = kwargs.get("groups", None)
            if getattr(self, "_personas", None) is None:
                raise RuntimeError("Personas are required for group plots. Run simulate() first.")
            viz.plot_group_over_time(
                self._history,
                self._personas,
                attr=attr,
                groups=groups,
                segments=self.segments,
                metric_label=self.metric_name.capitalize(),
                by=by,
            )
            return

        if type == "centrality":
            metric = kwargs.get("metric", "degree")
            show_exposure = kwargs.get("show_exposure", False)
            viz.plot_centrality_vs_score_exposure(self._G, self._history, metric=metric, metric_label=self.metric_name.capitalize(), show_exposure=show_exposure)
            return

        if type == "intervention_effect":
            # try to auto-detect intervention round from history metadata
            intervention_round = kwargs.get("intervention_round", None)
            if intervention_round is None:
                for h in self._history:
                    if h.get("intervention_active"):
                        intervention_round = h.get("intervention_round", None)
                        break
            attr = kwargs.get("attr", None)
            groups = kwargs.get("groups", None)
            viz.plot_intervention_effect(
                self._history,
                intervention_round=intervention_round,
                attr=attr,
                groups=groups,
                segments=self.segments,
                metric_label=self.metric_name.capitalize(),
            )
            return

        raise ValueError(f"Unknown type: {type}")

    @property
    def history(self) -> List[Dict[str, Any]]:
        if self._history is None:
            raise RuntimeError("Call simulate() first to populate history.")
        return self._history

    @property
    def beliefs(self) -> Dict[int, float]:
        if self._scores is None:
            raise RuntimeError("Call simulate() first to populate beliefs.")
        return self._scores

    @property
    def scores(self) -> Dict[int, float]:
        if self._scores is None:
            raise RuntimeError("Call simulate() first to populate scores.")
        return self._scores

    @property
    def graph(self):
        if self._G is None:
            raise RuntimeError("Call simulate() first to populate graph.")
        return self._G

    @property
    def personas(self):
        if self._personas is None:
            raise RuntimeError("Call simulate() first to populate personas.")
        return self._personas

    # Conversations and summaries/export
    def conversations(self, round: Optional[int] = None):
        """Return conversations for a round or all rounds."""
        if self._history is None:
            raise RuntimeError("Call simulate() first.")
        if round is None:
            return {h["round"]: h.get("conversations", []) for h in self._history}
        r = int(round)
        if not (0 <= r < len(self._history)):
            raise ValueError("round out of range")
        return self._history[r].get("conversations", [])

    def get_conversation(self, round: int, u: int, v: int):
        """Return the conversation record between nodes u and v at a given round, or None."""
        convos = self.conversations(round)
        uv = {int(u), int(v)}
        for rec in convos:
            if {int(rec.get("u", -1)), int(rec.get("v", -1))} == uv:
                return rec
        return None

    def summary(self) -> Dict[str, Any]:
        """Return quick metrics summary for the run."""
        if self._history is None or self._scores is None:
            raise RuntimeError("Call simulate() first.")
        n = self.n
        cov_series = [len(h.get("coverage", [])) for h in self._history]
        final_cov = cov_series[-1] if cov_series else 0
        mean_belief = float(sum(self._scores.values()) / max(1, len(self._scores)))
        # t_50: first round where coverage >= 50% of n
        t_50 = None
        target = 0.5 * n
        for h in self._history:
            if len(h.get("coverage", [])) >= target:
                t_50 = h["round"]
                break
        # optional polarization by political
        pol_gap = None
        try:
            groups: Dict[str, List[int]] = {}
            for i, p in enumerate(self._personas or []):
                key = getattr(p, "political", None) or (p.extra.get("political") if getattr(p, "extra", None) else None)
                key = str(key) if key is not None else None
                if key:
                    groups.setdefault(key, []).append(i)
            if groups.get("Democrat") and groups.get("Republican"):
                import numpy as np  # local import
                dem = np.array([self._scores.get(i, float("nan")) for i in groups["Democrat"]], dtype=float)
                rep = np.array([self._scores.get(i, float("nan")) for i in groups["Republican"]], dtype=float)
                pol_gap = float(np.nanmean(rep) - np.nanmean(dem))
        except Exception:
            pass
        return {
            "rounds": len(self._history) - 1,
            "final_coverage": final_cov,
            "mean_belief": mean_belief,
            "t_50": t_50,
            "polarization_gap_rep_minus_dem": pol_gap,
        }

    def summary_report(self, width: int = 88) -> str:
        """Return a professional, OLS-like text report summarizing the run."""
        if self._history is None or self._scores is None:
            raise RuntimeError("Call simulate() first.")
        import numpy as np  # local import
        metric_label = str(self.metric_name).capitalize()

        def line(char: str = "─") -> str:
            return char * max(20, width)

        def kv(label: str, value: str, sep: str = ":") -> str:
            left = f"{label}{sep} "
            pad = max(1, 28 - len(label))
            return f"{label}{sep}{' ' * pad}{value}"

        n = int(self.n)
        rounds = int(max(0, len(self._history) - 1))
        cov_series = [len(h.get("coverage", [])) for h in self._history]
        final_cov = cov_series[-1] if cov_series else 0
        # time-to-50% coverage
        t_50 = None
        target = 0.5 * n
        for h in self._history:
            if len(h.get("coverage", [])) >= target:
                t_50 = int(h["round"])
                break

        # belief statistics
        b = np.array([float(self._scores.get(i, np.nan)) for i in sorted(self._scores.keys())], dtype=float)
        mean_b = float(np.nanmean(b)) if b.size else float("nan")
        med_b = float(np.nanmedian(b)) if b.size else float("nan")
        std_b = float(np.nanstd(b, ddof=1)) if b.size > 1 else float("nan")
        p10 = float(np.nanpercentile(b, 10)) if b.size else float("nan")
        p90 = float(np.nanpercentile(b, 90)) if b.size else float("nan")
        share_hi = float(np.nanmean(b >= 0.7)) if b.size else float("nan")
        share_lo = float(np.nanmean(b <= 0.3)) if b.size else float("nan")

        # polarization (if political groups available)
        pol = {"Democrat": float("nan"), "Republican": float("nan"), "gap_rep_minus_dem": float("nan")}
        try:
            groups: Dict[str, List[int]] = {}
            for i, p in enumerate(self._personas or []):
                key = getattr(p, "political", None) or (p.extra.get("political") if getattr(p, "extra", None) else None)
                key = str(key) if key is not None else None
                if key:
                    groups.setdefault(key, []).append(i)
            if groups.get("Democrat"):
                dem = np.array([self._beliefs.get(i, np.nan) for i in groups["Democrat"]], dtype=float)
                pol["Democrat"] = float(np.nanmean(dem))
            if groups.get("Republican"):
                rep = np.array([self._beliefs.get(i, np.nan) for i in groups["Republican"]], dtype=float)
                pol["Republican"] = float(np.nanmean(rep))
            if np.isfinite(pol["Democrat"]) and np.isfinite(pol["Republican"]):
                pol["gap_rep_minus_dem"] = float(pol["Republican"] - pol["Democrat"])
        except Exception:
            pass

        # correlations with degree centrality (quick read)
        corr_degree = float("nan")
        try:
            import networkx as nx  # local import
            cent = nx.degree_centrality(self._G)
            x = np.array([float(cent.get(i, 0.0)) for i in sorted(self._scores.keys())], dtype=float)
            y = np.array([float(self._scores.get(i, np.nan)) for i in sorted(self._scores.keys())], dtype=float)
            if np.isfinite(x).sum() > 1 and np.isfinite(y).sum() > 1:
                corr_degree = float(np.corrcoef(x, y)[0, 1])
        except Exception:
            pass

        # model and config info
        model_name = str(self.model)
        mode = str(self.mode)
        info = [
            kv("Model", model_name),
            kv("Mode", mode),
            kv("Agents (n)", str(n)),
            kv("Rounds", str(rounds)),
            kv("Mean degree", str(self.degree)),
            kv("Edge sampling frac", f"{self.edge_frac:.2f}"),
            kv("Talk prob", f"{self.talk_prob:.2f}"),
            kv("Convo depth intensity", f"{self.depth_intensity:.2f}"),
        ]
        if self.intervention_round is not None:
            info.extend(
                [
                    kv("Intervention round", str(self.intervention_round)),
                    kv("Intervention nodes", f"{len(self.intervention_nodes)} selected"),
                ]
            )

        # assemble report
        rows: List[str] = []
        rows.append("LLM Society Simulation Summary".center(max(20, width)))
        rows.append(line("═"))
        rows.append("Model Information")
        rows.append(line())
        rows += info
        rows.append("")
        rows.append("Fit Statistics")
        rows.append(line())
        rows.append(kv("Final coverage", f"{final_cov} ({final_cov/n:.2%} of n)"))
        rows.append(kv("Time to 50% coverage", "—" if t_50 is None else f"t = {t_50}"))
        rows.append(kv(f"{metric_label} mean (std)", f"{mean_b:.3f} ({std_b:.3f})"))
        rows.append(kv(f"{metric_label} median [p10, p90]", f"{med_b:.3f} [{p10:.3f}, {p90:.3f}]"))
        rows.append(kv(f"Share {metric_label.lower()} ≥ 0.7", f"{share_hi:.2%}"))
        rows.append(kv(f"Share {metric_label.lower()} ≤ 0.3", f"{share_lo:.2%}"))
        rows.append(kv(f"Corr({metric_label.lower()}, degree)", f"{corr_degree:.2f}"))
        # group diff section (only if available)
        if np.isfinite(pol.get("Democrat", float("nan"))) or np.isfinite(pol.get("Republican", float("nan"))):
            rows.append("")
            rows.append("Group Differences (political)")
            rows.append(line())
            if np.isfinite(pol.get("Democrat", float("nan"))):
                rows.append(kv(f"Mean {metric_label.lower()} (Democrat)", f"{pol['Democrat']:.3f}"))
            if np.isfinite(pol.get("Republican", float("nan"))):
                rows.append(kv(f"Mean {metric_label.lower()} (Republican)", f"{pol['Republican']:.3f}"))
            if np.isfinite(pol.get("gap_rep_minus_dem", float("nan"))):
                rows.append(kv("Gap (Rep - Dem)", f"{pol['gap_rep_minus_dem']:.3f}"))
        # last round LLM sentence, if present
        txt = str(self._history[-1].get("summary", "") or "").strip()
        if txt:
            rows.append("")
            rows.append("LLM Narrative Summary (last round)")
            rows.append(line())
            rows.append(txt)
        rows.append(line("═"))
        return "\n".join(rows)

    def export(self, history_csv: Optional[str] = None, beliefs_csv: Optional[str] = None, conversations_jsonl: Optional[str] = None) -> None:
        """Export results to files. CSV requires pandas."""
        if self._history is None or self._scores is None:
            raise RuntimeError("Call simulate() first.")
        if history_csv or beliefs_csv:
            import pandas as pd  # local import
        if history_csv:
            rows = []
            for h in self._history:
                rows.append({
                    "round": h["round"],
                    "coverage": len(h.get("coverage", [])),
                    "summary": h.get("summary", ""),
                })
            pd.DataFrame(rows).to_csv(history_csv, index=False)
        if beliefs_csv:
            rounds = [h["round"] for h in self._history]
            data: Dict[str, Any] = {"round": rounds}
            last_map = self._history[-1].get("scores", self._history[-1].get("beliefs", {}))
            node_ids = sorted(list(last_map.keys()))
            for nid in node_ids:
                data[str(nid)] = [float((h.get("scores", h.get("beliefs", {}))).get(nid, float("nan"))) for h in self._history]
            pd.DataFrame(data).to_csv(beliefs_csv, index=False)
        if conversations_jsonl:
            with open(conversations_jsonl, "w", encoding="utf-8") as f:
                for h in self._history:
                    t = h["round"]
                    for rec in h.get("conversations", []):
                        out = {"round": t, **rec}
                        try:
                            f.write(json.dumps(out, ensure_ascii=False) + "\n")
                        except Exception:
                            f.write(str(out) + "\n")


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
            metric_name=str(cfg.get("metric_name", DEFAULTS["metric_name"])),
            metric_prompt=str(cfg.get("metric_prompt", DEFAULTS["metric_prompt"])),
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
            intervention_round=cfg.get("intervention_round", DEFAULTS.get("intervention_round")),
            intervention_nodes=list(cfg.get("intervention_nodes", [])),
            intervention_content=str(cfg.get("intervention_content", DEFAULTS.get("intervention_content", ""))),
            graph=kwargs.get("graph", None),  # allow graph passed alongside config kwargs
        )

    # kwargs style
    return Network(information=information, **kwargs)


