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
    "convo_depth_p": 0.6,
    "max_convo_turns": 4,
    "edge_sample_frac": 0.5,
    "seed_nodes": [0, 1],
    "seed_belief": 0.98,
    "information_text": "5G towers cause illness.",
    "talk_information_prob": 0.25,
    "contagion_mode": "llm",  # 'llm' | 'simple' | 'complex'
    "complex_threshold_k": 2,
    "stop_when_stable": False,
    "stability_tol": 1e-4,
    "rng_seed": 0,
    "api_key_file": "api-key.txt",
    "persona_segments": [],
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

    # Normalize certain types/keys
    merged["contagion_mode"] = str(merged.get("contagion_mode", "llm")).lower()
    merged["stop_when_stable"] = _as_bool(merged.get("stop_when_stable"), DEFAULTS["stop_when_stable"]) 
    # seed nodes: accept comma-separated string
    seeds = merged.get("seed_nodes")
    if isinstance(seeds, str):
        merged["seed_nodes"] = [int(x) for x in seeds.split(",") if x.strip() != ""]
    return merged



