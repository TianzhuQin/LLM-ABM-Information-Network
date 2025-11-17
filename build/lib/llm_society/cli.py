#!/usr/bin/env python3
import argparse
from importlib import resources
from typing import Dict, Any, List
import yaml

from llm_society.config import load_config, DEFAULTS
from llm_society.simulation import run_simulation


def write_example_config(dest_path: str) -> None:
    with resources.files("llm_society").joinpath("data/example.yaml").open("rb") as rf:
        data = rf.read()
    with open(dest_path, "wb") as wf:
        wf.write(data)


def _parse_seeds(val: str) -> List[int]:
    return [int(x) for x in val.split(",") if x.strip() != ""]


def main() -> None:
    p = argparse.ArgumentParser(description="Run LLM Society simulation")
    # Config IO
    p.add_argument("--config", type=str, help="Path to YAML/JSON config file", default=None)
    p.add_argument("--write-example-config", type=str, metavar="PATH", help="Write packaged example config to PATH and exit")
    # Unified API-style flags (all optional; override config/defaults if provided)
    p.add_argument("--information", type=str, help="Information text (claim) to simulate", default=None)
    p.add_argument("--n", type=int, help="Number of agents", default=None)
    p.add_argument("--degree", type=int, help="Mean degree (edge_mean_degree)", default=None)
    p.add_argument("--rounds", type=int, help="Number of rounds", default=None)
    p.add_argument("--depth", type=float, help="Conversation depth intensity [0-1]", default=None)
    p.add_argument("--depth-max", type=int, help="Max conversation turns", default=None)
    p.add_argument("--edge-frac", type=float, help="Fraction of edges to sample each round", default=None)
    p.add_argument("--conversation-scope", type=str, choices=["edges", "all"], help="Conversation sampling scope for LLM mode", default=None)
    p.add_argument("--pair-weight-epsilon", type=float, help="Minimum tie weight used when sampling all node pairs", default=None)
    p.add_argument("--seeds", type=str, help="Comma-separated seed node indices", default=None)
    p.add_argument("--seed-score", type=float, help="Initial score for seed nodes [0-1]", default=None)
    p.add_argument("--talk-prob", type=float, help="Probabilty of discussing the information in a convo", default=None)
    p.add_argument("--mode", type=str, choices=["llm", "simple", "complex"], help="Contagion mode", default=None)
    p.add_argument("--complex-k", type=int, help="Threshold k for complex contagion", default=None)
    p.add_argument("--stop-when-stable", action="store_true", help="Stop early when stable")
    p.add_argument("--stability-tol", type=float, help="Tolerance for stability check", default=None)
    p.add_argument("--dynamic-network", action="store_true", help="Enable dynamic network restructuring")
    p.add_argument("--link-sever-threshold", type=float, help="Weight threshold below which links are severed", default=None)
    p.add_argument("--link-formation-threshold", type=float, help="Weight threshold above which new links are formed", default=None)
    p.add_argument("--rng", type=int, help="RNG seed", default=None)
    p.add_argument("--events-file", type=str, help="Path to events YAML file", default=None)
    # persona / segments
    p.add_argument("--segments-file", type=str, help="Path to persona segments YAML file", default=None)
    p.add_argument("--api-key-file", type=str, help="Path to API key file", default=None)
    p.add_argument("--model", type=str, help="Model name for LLM calls", default=None)
    # Interventions
    p.add_argument("--intervention-round", type=int, help="Round at which to start intervention (inject content for selected nodes)", default=None)
    p.add_argument("--intervention-nodes", type=str, help="Comma-separated node ids to apply intervention to", default=None)
    p.add_argument("--intervention-content", type=str, help="Intervention content prompt to inject into targeted agents' system messages", default=None)
    p.add_argument("--memory-turns-per-agent", type=int, help="How many recent utterances each agent remembers in their system prompt (LLM mode)", default=None)

    args = p.parse_args()

    if args.write_example_config:
        write_example_config(args.write_example_config)
        print(f"Wrote example config to {args.write_example_config}")
        return

    # Base config: from file if provided, else defaults
    if args.config:
        cfg: Dict[str, Any] = load_config(args.config)
    else:
        cfg = dict(DEFAULTS)

    # Apply overrides from flags if provided
    if args.information is not None:
        cfg["information_text"] = args.information
    if args.n is not None:
        cfg["n"] = int(args.n)
    if args.degree is not None:
        cfg["edge_mean_degree"] = int(args.degree)
    if args.rounds is not None:
        cfg["rounds"] = int(args.rounds)
    if args.depth is not None:
        cfg["depth"] = float(args.depth)
    if args.depth_max is not None:
        cfg["max_convo_turns"] = int(args.depth_max)
    if args.edge_frac is not None:
        cfg["edge_sample_frac"] = float(args.edge_frac)
    if args.conversation_scope is not None:
        cfg["conversation_scope"] = str(args.conversation_scope)
    if args.pair_weight_epsilon is not None:
        cfg["pair_weight_epsilon"] = float(args.pair_weight_epsilon)
    if args.seeds is not None:
        cfg["seed_nodes"] = _parse_seeds(args.seeds)
    if args.seed_score is not None:
        cfg["seed_score"] = float(args.seed_score)
    if args.talk_prob is not None:
        cfg["talk_information_prob"] = float(args.talk_prob)
    if args.mode is not None:
        cfg["contagion_mode"] = str(args.mode)
    if args.complex_k is not None:
        cfg["complex_threshold_k"] = int(args.complex_k)
    if args.stop_when_stable:
        cfg["stop_when_stable"] = True
    if args.stability_tol is not None:
        cfg["stability_tol"] = float(args.stability_tol)
    if args.dynamic_network:
        cfg["dynamic_network"] = True
    if args.link_sever_threshold is not None:
        cfg["link_sever_threshold"] = float(args.link_sever_threshold)
    if args.link_formation_threshold is not None:
        cfg["link_formation_threshold"] = float(args.link_formation_threshold)
    if args.rng is not None:
        cfg["rng_seed"] = int(args.rng)
    if args.events_file:
        with open(args.events_file) as f:
            cfg["events"] = yaml.safe_load(f)
    if args.segments_file:
        with open(args.segments_file) as f:
            cfg["persona_segments"] = yaml.safe_load(f)
    if args.api_key_file is not None:
        cfg["api_key_file"] = str(args.api_key_file)
    if args.model is not None:
        cfg["model"] = str(args.model)
    if args.intervention_round is not None:
        cfg["intervention_round"] = int(args.intervention_round)
    if args.intervention_nodes is not None:
        cfg["intervention_nodes"] = _parse_seeds(args.intervention_nodes)
    if args.intervention_content is not None:
        cfg["intervention_content"] = str(args.intervention_content)
    if args.memory_turns_per_agent is not None:
        cfg["memory_turns_per_agent"] = int(max(0, args.memory_turns_per_agent))

    # Validate required information
    if not str(cfg.get("information_text", "")).strip():
        p.error("--information is required unless provided in --config")

    result = run_simulation(cfg)
    history = result["history"]
    print(f"Rounds: {len(history) - 1}")
    print(f"Round 0 summary: {history[0]['summary']}")
    print(f"Final coverage: {len(history[-1]['coverage'])}")


if __name__ == "__main__":
    main()


