import os
import json
import re
from typing import Any, Dict, Union, List


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


def _slugify_metric_id(raw: str, fallback: str = "metric") -> str:
    base = str(raw or "").strip().lower()
    if not base:
        base = fallback
    slug = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return slug or fallback


DEFAULTS: Dict[str, Any] = {
    "model": "gpt-4.1",
    "n": 20,
    "edge_mean_degree": 4,
    "rounds": 10,
    "conversation_scope": "edges",  # 'edges' | 'all'
    "pair_weight_epsilon": 0.02,
    # Metric settings
    # Default metric is "credibility" scored as a probability in [0,1]
    "metric_name": "credibility",
    "metric_prompt": (
        "On a 0.0 to 1.0 scale, rate the perceived credibility of the following claim "
        "as a probability that it is true. Return ONLY a single number between 0 and 1."
    ),
    "depth": 0.6,  # 0-1 intensity: 0=very shallow, 1=very deep
    "max_convo_turns": 6,
    "memory_turns_per_agent": 0,
    "edge_sample_frac": 0.5,
    "seed_nodes": [0],
    "seed_score": 1.0,
    "information": "5G towers cause illness.",
    "talk_information_prob": 0.25,
    "metrics": None,
    "contagion_mode": "llm",  # 'llm' | 'simple' | 'complex'
    "complex_threshold_k": 2,
    "stop_when_stable": False,
    "stability_tol": 0.001,
    "dynamic_network": False,
    "link_sever_threshold": 0.1,
    "link_formation_threshold": 0.3,
    "rng_seed": 42,
    "api_key_file": "api-key.txt",
    "events": [], # List of external shock events
    "persona_segments": [],
    # intervention controls (for LLM mode) - DEPRECATED in favor of events
    "intervention_round": None,  # int or None
    "intervention_nodes": [],  # list of node ids
    # intervention content prompt to inject into targeted agents' system messages from intervention_round onward
    "intervention_content": "",
    # printing controls
    "print_conversations": True,
    "print_score_updates": True,
    "print_round_summaries": True,
    "print_all_conversations": True,
}

def normalize_metrics(raw_metrics: Any, default_name: str, default_prompt: str) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if isinstance(raw_metrics, list):
        for idx, entry in enumerate(raw_metrics):
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("label", entry.get("name", default_name))).strip() or default_name
            prompt = str(entry.get("prompt", default_prompt))
            m_id = _slugify_metric_id(entry.get("id") or label or f"metric_{idx}", fallback=f"metric_{idx}")
            normalized.append({
                "id": m_id,
                "label": label,
                "prompt": prompt,
            })
    if not normalized:
        normalized = [{
            "id": _slugify_metric_id(default_name, "metric"),
            "label": default_name,
            "prompt": default_prompt,
        }]
    return normalized


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
    merged["conversation_scope"] = str(merged.get("conversation_scope", DEFAULTS["conversation_scope"])).lower()
    try:
        merged["pair_weight_epsilon"] = max(0.0, float(merged.get("pair_weight_epsilon", DEFAULTS["pair_weight_epsilon"])))
    except Exception:
        merged["pair_weight_epsilon"] = float(DEFAULTS["pair_weight_epsilon"])
    merged["stop_when_stable"] = _as_bool(merged.get("stop_when_stable"), DEFAULTS["stop_when_stable"]) 
    # seed nodes: accept comma-separated string
    seeds = merged.get("seed_nodes")
    if isinstance(seeds, str):
        merged["seed_nodes"] = [int(x) for x in seeds.split(",") if x.strip() != ""]
    if "seed_score" not in merged:
        merged["seed_score"] = float(DEFAULTS["seed_score"])
    try:
        merged["memory_turns_per_agent"] = max(0, int(merged.get("memory_turns_per_agent", DEFAULTS["memory_turns_per_agent"])))
    except Exception:
        merged["memory_turns_per_agent"] = int(DEFAULTS["memory_turns_per_agent"])
    if "print_score_updates" not in merged:
        merged["print_score_updates"] = _as_bool(merged.get("print_score_updates"), DEFAULTS["print_score_updates"])
    merged["metrics"] = normalize_metrics(
        merged.get("metrics"),
        str(merged.get("metric_name", DEFAULTS["metric_name"])),
        str(merged.get("metric_prompt", DEFAULTS["metric_prompt"])),
    )
    merged["metric_name"] = str(merged["metrics"][0]["label"])
    merged["metric_prompt"] = str(merged["metrics"][0]["prompt"])
    return merged



