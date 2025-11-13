import os
import json
from typing import Any, Dict, Union


def _as_bool(val: Any, default: bool) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        v = val.strip().lower()
        if v in {"1", "true", "t", "yes", "y"}:
            return True
        if v in {"0", "false", "f", "no", "n"}:
            return False
    return default


DEFAULTS: Dict[str, Any] = {
    "model": "gpt-4.1",
    "n": 20,
    "edge_mean_degree": 4,
    "rounds": 10,
    # Metric settings
    # Default metric is "credibility" scored as a probability in [0,1]
    "metric_name": "credibility",
    "metric_prompt": (
        "On a 0.0 to 1.0 scale, rate the perceived credibility of the following claim "
        "as a probability that it is true. Return ONLY a single number between 0 and 1."
    ),
    "depth": 0.6,  # 0-1 intensity: 0=very shallow, 1=very deep
    "max_convo_turns": 4,
    "edge_sample_frac": 0.5,
    "seed_nodes": [0, 1],
    "seed_belief": 0.98,  # legacy name
    "seed_score": 0.98,   # preferred name
    "information_text": "5G towers cause illness.",
    "talk_information_prob": 0.25,
    "contagion_mode": "llm",  # 'llm' | 'simple' | 'complex'
    "complex_threshold_k": 2,
    "stop_when_stable": False,
    "stability_tol": 1e-4,
    "rng_seed": 0,
    "api_key_file": "api-key.txt",
    "persona_segments": [],
    # intervention controls (for LLM mode)
    "intervention_round": None,  # int or None
    "intervention_nodes": [],  # list of node ids
    # intervention content prompt to inject into targeted agents' system messages from intervention_round onward
    "intervention_content": "",
    "casual_topics": [
        "weekend plans",
        "favorite foods",
        "movies or TV shows",
        "music you enjoy",
        "travel dreams",
        "recent hobbies",
        "sports",
        "weather today",
    ],
    # printing controls
    "print_conversations": True,
    "print_belief_updates": True,
    "print_round_summaries": True,
    "print_all_conversations": True,
}


def load_config(source: Union[str, Dict[str, Any], None]) -> Dict[str, Any]:
    """Load config from a dict, JSON/YAML file path, or None for defaults."""
    cfg: Dict[str, Any] = {}
    if source is None:
        cfg = {}
    elif isinstance(source, dict):
        cfg = dict(source)
    elif isinstance(source, str):
        path = source
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        _, ext = os.path.splitext(path)
        if ext.lower() in {".yml", ".yaml"}:
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise RuntimeError("PyYAML is required to load YAML configs. Install pyyaml.") from e
            with open(path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        elif ext.lower() in {".json"}:
            with open(path, "r") as f:
                cfg = json.load(f)
        else:
            # attempt YAML first, then JSON
            try:
                import yaml  # type: ignore
                with open(path, "r") as f:
                    cfg = yaml.safe_load(f) or {}
            except Exception:
                with open(path, "r") as f:
                    cfg = json.load(f)
    else:
        raise TypeError("Unsupported config source type")

    merged = dict(DEFAULTS)
    for k, v in (cfg or {}).items():
        merged[k] = v

    # Backward compatibility: allow legacy 'convo_depth_p' to set 'depth'
    if "depth" not in merged and "convo_depth_p" in merged:
        try:
            merged["depth"] = float(merged["convo_depth_p"])  # type: ignore[arg-type]
        except Exception:
            pass

    # Accept 'degree' as alias for 'edge_mean_degree'
    if "degree" in merged:
        try:
            merged["edge_mean_degree"] = int(merged["degree"])  # type: ignore[arg-type]
        except Exception:
            pass

    # Normalize certain types/keys
    merged["contagion_mode"] = str(merged.get("contagion_mode", "llm")).lower()
    merged["stop_when_stable"] = _as_bool(merged.get("stop_when_stable"), DEFAULTS["stop_when_stable"]) 
    # seed nodes: accept comma-separated string
    seeds = merged.get("seed_nodes")
    if isinstance(seeds, str):
        merged["seed_nodes"] = [int(x) for x in seeds.split(",") if x.strip() != ""]
    # normalize seed score key (prefer 'seed_score', accept legacy 'seed_belief')
    if "seed_score" not in merged:
        if "seed_belief" in merged:
            try:
                merged["seed_score"] = float(merged["seed_belief"])  # type: ignore[arg-type]
            except Exception:
                merged["seed_score"] = float(DEFAULTS["seed_score"])
        else:
            merged["seed_score"] = float(DEFAULTS["seed_score"])
    return merged



