import json
import random
from typing import Dict, List, Tuple, Iterator, Any, Optional

import numpy as np
import networkx as nx

from .persona import Persona, sample_personas, persona_to_text, personas_from_graph
from .network import build_random_network
from .llm import call_chat, build_client
from .config import DEFAULTS
from collections import defaultdict
import concurrent.futures


def _clip_01(x: float) -> float:
    try:
        return float(min(1.0, max(0.0, float(x))))
    except Exception:
        return 0.0


def _parse_first_float(text: str) -> float:
    import re
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text or "")
    if not matches:
        raise ValueError("no float found")
    return float(matches[0])


def _parse_two_floats(text: str) -> Tuple[float, float]:
    import re
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text or "")
    if len(matches) >= 2:
        return float(matches[0]), float(matches[1])
    if len(matches) == 1:
        v = float(matches[0])
        return v, v
    raise ValueError("no floats found")


def _edge_key(u: int, v: int) -> Tuple[int, int]:
    i, j = int(u), int(v)
    return (i, j) if i <= j else (j, i)


def _init_weight_cache(G: nx.Graph) -> Dict[Tuple[int, int], float]:
    cache: Dict[Tuple[int, int], float] = {}
    for u, v, data in G.edges(data=True):
        cache[_edge_key(u, v)] = float(np.clip(data.get("weight", 0.5), 0.0, 1.0))
    return cache


def _get_weight(cache: Dict[Tuple[int, int], float], u: int, v: int) -> float:
    return float(cache.get(_edge_key(u, v), 0.0))


def _update_weight_after_convo(
    cache: Dict[Tuple[int, int], float],
    G: nx.Graph,
    u: int,
    v: int,
    *,
    talked_about_info: bool,
    conversation_turns: int,
    max_turns: int,
    allow_new_edges: bool,
) -> float:
    key = _edge_key(u, v)
    current = float(cache.get(key, 0.0))
    turn_pairs = max(1, conversation_turns // 2)
    turn_norm = min(1.0, turn_pairs / max(1, max_turns))
    if talked_about_info:
        delta = 0.08 + 0.5 * turn_norm
    else:
        delta = 0.02 + 0.25 * turn_norm
    new_weight = float(max(0.0, min(1.0, current + delta)))
    cache[key] = new_weight
    if allow_new_edges and not G.has_edge(u, v) and new_weight > 0.0:
        G.add_edge(u, v, weight=new_weight)
    if G.has_edge(u, v):
        G[u][v]["weight"] = new_weight
    return new_weight


def _all_node_pairs(n: int) -> List[Tuple[int, int]]:
    if n < 2:
        return []
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _parse_turn_line(line: str) -> Tuple[Optional[int], str]:
    try:
        speaker_text, utterance = line.split(":", 1)
        speaker = int(speaker_text.strip())
        return speaker, utterance.strip()
    except Exception:
        return None, line.strip()


def _format_memory(memory_store: Optional[Dict[int, List[str]]], node_id: int, max_entries: int) -> str:
    if memory_store is None or max_entries <= 0:
        return ""
    entries = memory_store.get(int(node_id), [])
    if not entries:
        return ""
    return "Recent conversation snippets:\n" + "\n".join(entries[-max_entries:])


def _update_memory_store(
    memory_store: Optional[Dict[int, List[str]]],
    node_id: int,
    turns: List[str],
    max_entries: int,
) -> None:
    if memory_store is None or max_entries <= 0 or not turns:
        return
    entries = memory_store.setdefault(int(node_id), [])
    for line in turns:
        speaker, utterance = _parse_turn_line(line)
        if speaker is None or not utterance:
            continue
        if speaker == node_id:
            label = "You said"
        else:
            label = f"Node {speaker} said"
        entries.append(f"{label}: {utterance}")
    if len(entries) > max_entries:
        del entries[:-max_entries]


def llm_multi_metric_updates(
    model: str,
    topic_text: str,
    metrics: List[Dict[str, Any]],
    prior_scores_i: Dict[str, float],
    prior_scores_j: Dict[str, float],
    tie_weight: float,
    convo_turns: List[str],
    dynamic_network: bool,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Return updated scores per metric for two agents and a new tie strength."""
    client = build_client()
    convo_text = "\n".join(convo_turns[-8:]) if convo_turns else ""
    metric_lines = []
    priors_lines = []
    metric_ids: List[str] = []
    for idx, metric in enumerate(metrics):
        mid = str(metric.get("id", f"metric_{idx}"))
        metric_ids.append(mid)
        label = str(metric.get("label", mid))
        prompt = str(metric.get("prompt", "Provide a score between 0 and 1."))
        metric_lines.append(f"- {mid} ({label}): {prompt}")
        priors_lines.append(
            f"{mid}: A={prior_scores_i.get(mid, 0.0):.3f}, B={prior_scores_j.get(mid, 0.0):.3f}"
        )

    sys_base = (
        "You are updating multiple metrics (each in [0,1]) for two people after their conversation about a topic. "
        "Return ONLY a valid JSON object. Do not add commentary."
    )
    if dynamic_network:
        sys = (
            sys_base + ' The object should contain two keys: "scores" and "new_tie_strength". '
            'The "scores" value should be an object where each key is a metric id and each value is '
            'an object like {"A": 0.xx, "B": 0.yy}. The "new_tie_strength" value should be a single '
            "number between 0.0 (very weak) and 1.0 (very strong)."
        )
    else:
        sys = sys_base.replace("JSON object", 'JSON object where each key is a metric id and each value is an object like {"A": 0.xx, "B": 0.yy}')

    prompt_extra = ""
    if dynamic_network:
        prompt_extra = (
            "\nConsidering their current tie strength and this conversation, what is their new tie strength (0.0 to 1.0)? "
            'Return this as "new_tie_strength" in the top-level JSON object.'
    )

    score_bias_hint = (
        "Each agent trusts their own 'inside information' and prior stance. Base your scores on that "
        "trusted lens and only reduce belief if the conversation itself includes compelling evidence "
        "that this agent would personally accept."
    )

    prompt = (
        f"Topic: {topic_text}\n"
        f"Metrics and prompts:\n" + "\n".join(metric_lines) + "\n\n"
        f"Prior scores:\n" + "\n".join(priors_lines) + "\n"
        f"Current tie strength (0-1): {float(np.clip(tie_weight, 0.0, 1.0)):.3f}\n"
        f"Recent conversation turns:\n{convo_text}\n\n"
        f"{score_bias_hint}\n"
        f"Return JSON with one entry per metric id.{prompt_extra}"
    )
    out = call_chat(client, model, [{"role": "system", "content": sys}, {"role": "user", "content": prompt}], max_tokens_requested=128)
    updates_i = {mid: float(prior_scores_i.get(mid, 0.0)) for mid in metric_ids}
    updates_j = {mid: float(prior_scores_j.get(mid, 0.0)) for mid in metric_ids}
    new_tie_strength = tie_weight
    try:
        data = json.loads(out)
        scores_data = data.get("scores") if dynamic_network else data
        if dynamic_network:
            new_tie_strength = float(data.get("new_tie_strength", tie_weight))

        if scores_data and isinstance(scores_data, dict):
            for mid in metric_ids:
                if mid in scores_data and isinstance(scores_data[mid], dict):
                    updates_i[mid] = float(scores_data[mid].get("A", prior_scores_i.get(mid, 0.0)))
                    updates_j[mid] = float(scores_data[mid].get("B", prior_scores_j.get(mid, 0.0)))
    except Exception:
        # fallback for malformed JSON or unexpected structure
        pass

    return updates_i, updates_j, new_tie_strength


def llm_conversation_and_scores(
    model: str,
    p_i: Persona,
    p_j: Persona,
    topic_text: str,
    depth_intensity: float,
    talk_about_topic: bool,
    prior_scores_i: Dict[str, float],
    prior_scores_j: Dict[str, float],
    tie_weight: float,
    max_turns: int,
    *,
    extra_system_i: str = "",
    extra_system_j: str = "",
    metrics: Optional[List[Dict[str, Any]]] = None,
    memory_text_i: str = "",
    memory_text_j: str = "",
    dynamic_network: bool = False,
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]], List[str], bool, float]:
    metrics = metrics or [{"id": "credibility", "label": "Credibility", "prompt": "Provide a score between 0 and 1."}]
    client = build_client()
    # Tie-adjusted depth intensity and cap
    tie_weight = float(max(0.0, min(1.0, tie_weight)))
    depth_factor = 0.5 + 0.5 * tie_weight
    eff_depth_intensity = float(max(0.0, min(1.0, depth_intensity * depth_factor)))
    max_turns_scaled = int(max(1, round(max_turns * depth_factor)))
    # Map intensity [0,1] -> geometric parameter p in (0,1]
    p_geo = float(max(0.05, min(1.0, 1.0 - 0.95 * eff_depth_intensity)))
    depth = int(np.random.geometric(p=p_geo))
    depth = min(depth, max_turns_scaled)
    style_hint = (
        "Chat casually like two friends. Use 1-2 plain sentences. No markdown, no bullet points, "
        "no headings, no numbered lists, no bold/italics. Keep it natural and conversational. "
        "Stay confident and consistent with your persona's beliefs; avoid hedging statements like "
        "'I'm not sure' unless the other agent provides trusted evidence you personally accept."
    )

    def metric_guidance(priors: Dict[str, float]) -> str:
        parts = []
        for metric in metrics:
            mid = str(metric.get("id"))
            label = str(metric.get("label", mid))
            parts.append(f"{label}={priors.get(mid, 0.0):.2f}")
        return "; ".join(parts)

    topic_scope_hint = (
        "Base your statements only on the provided topic description and what the other agent says. "
        "Do not invent references to outside authorities, official statements, or media reports unless they are explicitly mentioned in the conversation."
    )

    topic_text = str(topic_text or "").strip()
    discuss_topic = bool(talk_about_topic and topic_text)
    base_sys_i = f"You are in a casual conversation. {style_hint} Demographics: {persona_to_text(p_i)}."
    base_sys_j = f"You are in a casual conversation. {style_hint} Demographics: {persona_to_text(p_j)}."
    if discuss_topic:
        guidance_i = (
            f"Topic: {topic_text}. {topic_scope_hint} Your current metric scores are: {metric_guidance(prior_scores_i)}. "
            "If this topic is discussed, express views consistent with these scores."
        )
        guidance_j = (
            f"Topic: {topic_text}. {topic_scope_hint} Your current metric scores are: {metric_guidance(prior_scores_j)}. "
            "If this topic is discussed, express views consistent with these scores."
        )
    else:
        guidance_i = guidance_j = ""
    sys_i = f"{base_sys_i} {guidance_i}".strip()
    sys_j = f"{base_sys_j} {guidance_j}".strip()
    if extra_system_i:
        sys_i = f"{sys_i} Intervention instruction: {extra_system_i}"
    if extra_system_j:
        sys_j = f"{sys_j} Intervention instruction: {extra_system_j}"
    if memory_text_i:
        sys_i = f"{sys_i} {memory_text_i}"
    if memory_text_j:
        sys_j = f"{sys_j} {memory_text_j}"

    if discuss_topic:
        last = f"Let's talk about this topic: {topic_text}"
    else:
        last = "Let's just chat about something else."

    turns: List[str] = []
    for _ in range(depth):
        out_i = call_chat(client, model, [{"role": "system", "content": sys_i}, {"role": "user", "content": last}], max_tokens_requested=160)
        turns.append(f"{p_i.pid}: {out_i}")
        out_j = call_chat(client, model, [{"role": "system", "content": sys_j}, {"role": "user", "content": out_i}], max_tokens_requested=160)
        turns.append(f"{p_j.pid}: {out_j}")
        last = out_j

    if discuss_topic:
        b_i, b_j, new_tie = llm_multi_metric_updates(
            model, topic_text, metrics, prior_scores_i, prior_scores_j, tie_weight, turns, dynamic_network
        )
        return b_i, b_j, turns, True, new_tie
    return None, None, turns, False, tie_weight


def llm_score_summary(model: str, topic_text: str, scores: List[float], metric_name: str) -> str:
    client = build_client()
    arr = np.array(scores, dtype=float)
    mean_b = float(np.mean(arr))
    med_b = float(np.median(arr))
    hi = float(np.mean(arr >= 0.7))
    lo = float(np.mean(arr <= 0.3))
    stats = f"mean={mean_b:.2f}, median={med_b:.2f}, share>=0.7={hi:.2f}, share<=0.3={lo:.2f}"
    topic_text = topic_text.strip() or "No specific topic"
    prompt = (
        f"Given these stats for '{metric_name}', write ONE short sentence summarizing the distribution. "
        "Do not repeat numbers.\n" + f"Topic: {topic_text}\nStatistics: {stats}"
    )
    return call_chat(build_client(), model, [{"role": "user", "content": prompt}], max_tokens_requested=64)


# A sentinel value for "no score"
NO_SCORE = -1.0


def iterate_simulation(cfg: Dict) -> Iterator[Dict[str, Any]]:
    model = cfg["model"]
    metrics = cfg.get("metrics") or [{
        "id": "credibility",
        "label": str(cfg.get("metric_name", "credibility")),
        "prompt": str(cfg.get("metric_prompt", DEFAULTS["metric_prompt"])),
    }]
    metric_ids = [str(m.get("id")) for m in metrics]
    primary_metric = metrics[0]
    primary_metric_id = str(primary_metric.get("id"))
    metric_name = str(primary_metric.get("label", cfg.get("metric_name", "credibility")))
    # If a custom graph is provided, prefer its size for n
    custom_G = cfg.get("G", None)
    if custom_G is not None:
        try:
            n = int(custom_G.number_of_nodes())
        except Exception:
            n = int(cfg["n"])
    else:
        n = int(cfg["n"])
    mean_deg = int(cfg["edge_mean_degree"])
    rounds = int(cfg["rounds"])
    depth_intensity = float(cfg.get("depth", cfg.get("convo_depth_p", DEFAULTS["depth"])))
    edge_sample_frac = float(cfg["edge_sample_frac"])
    conversation_scope = str(cfg.get("conversation_scope", DEFAULTS.get("conversation_scope", "edges"))).lower()
    pair_weight_epsilon = float(max(0.0, cfg.get("pair_weight_epsilon", DEFAULTS.get("pair_weight_epsilon", 0.01))))
    seed_nodes = list(cfg["seed_nodes"])
    seed_score = float(cfg["seed_score"])
    information_text = str(cfg.get("information", "")).strip()
    intervention_content = str(cfg.get("intervention_content", "")).strip()
    discuss_prob = float(cfg.get("talk_information_prob", 0.0))
    contagion_mode = str(cfg.get("contagion_mode", "llm"))
    complex_k = int(cfg.get("complex_threshold_k", 2))
    stop_when_stable = bool(cfg.get("stop_when_stable", False))
    stability_tol = float(cfg.get("stability_tol", 1e-4))
    rng_seed = int(cfg.get("rng_seed", 0))
    api_key_file = str(cfg.get("api_key_file", "api-key.txt"))
    segments = cfg.get("persona_segments", [])
    max_convo_turns_cfg = int(cfg.get("max_convo_turns", DEFAULTS.get("max_convo_turns", 4)))
    print_convos = bool(cfg.get("print_conversations", True))
    print_updates = bool(cfg.get("print_score_updates", True))
    print_rounds = bool(cfg.get("print_round_summaries", True))
    print_all_convos = bool(cfg.get("print_all_conversations", True))
    # interventions (LLM mode)
    intervention_round = cfg.get("intervention_round", None)
    intervention_nodes = set(cfg.get("intervention_nodes", []))
    intervention_content = str(cfg.get("intervention_content", "")).strip()
    if intervention_round is not None:
        try:
            intervention_round_int = int(intervention_round)
        except Exception:
            intervention_round_int = None
    else:
        intervention_round_int = None
    summary_topic = information_text or intervention_content or "Intervention-only treatment"
    memory_turns_per_agent = int(max(0, cfg.get("memory_turns_per_agent", DEFAULTS.get("memory_turns_per_agent", 0))))

    random.seed(rng_seed)
    np.random.seed(rng_seed)
    # propagate api key file to LLM loader via env var
    import os
    if api_key_file:
        os.environ["OPENAI_API_KEY_FILE"] = api_key_file

    # Build or adopt graph
    if custom_G is not None:
        try:
            import networkx as nx  # local import
            G = custom_G.copy()
            # Relabel nodes to 0..n-1 if needed
            nodes = list(G.nodes())
            if len(nodes) != n or set(nodes) != set(range(n)):
                mapping = {old: i for i, old in enumerate(nodes)}
                G = nx.relabel_nodes(G, mapping, copy=True)
                n = G.number_of_nodes()
            # Ensure weights
            for u, v in G.edges():
                if "weight" not in G[u][v]:
                    G[u][v]["weight"] = float(0.2 + 0.8 * np.random.beta(2, 2))
        except Exception:
            # Fallback to random if custom graph invalid
            G = build_random_network(n, mean_deg, seed=rng_seed + 7)
    else:
        G = build_random_network(n, mean_deg, seed=rng_seed + 7)

    # Personas: if custom graph and no segments provided, derive from node attributes; else sample
    if custom_G is not None and (not segments):
        try:
            personas = personas_from_graph(G)
        except Exception:
            personas = sample_personas(n, [])
        # normalize length just in case
        if len(personas) != n:
            if len(personas) < n:
                personas.extend(sample_personas(n - len(personas)))
            personas = personas[:n]
    else:
        personas = sample_personas(n, segments)

    seed_set = set(seed_nodes)
    scores_by_metric: Dict[str, Dict[int, float]] = {
        mid: {i: (seed_score if i in seed_set else 0.0) for i in range(n)} for mid in metric_ids
    }
    scores = scores_by_metric[primary_metric_id]
    exposed = {i: (i in seed_set) for i in range(n)}

    arr0 = [scores[i] for i in range(n)]
    sum0 = llm_score_summary(model, summary_topic, arr0, metric_name) if contagion_mode == "llm" else ""
    if print_rounds and contagion_mode == "llm":
        print(f"Round 0 summary: {sum0}")
    coverage0 = {i for i in range(n) if exposed[i]}
    active0 = {i for i in range(n) if exposed[i] and any(scores_by_metric[mid][i] > 0 for mid in metric_ids)}
    history_entry = {
        "round": 0,
        "coverage": coverage0,
        "active_believers": active0,
        "scores": scores.copy(),
        "scores_multi": {mid: scores_by_metric[mid].copy() for mid in metric_ids},
        "summary": sum0,
        "metrics": metrics,
    }
    history = [history_entry]
    yield {
        "t": 0,
        "G": G,
        "personas": personas,
        "scores": scores,
        "scores_multi": {mid: scores_by_metric[mid].copy() for mid in metric_ids},
        "exposed": exposed,
        "history_entry": history_entry,
        "full_history": history,
    }

    memory_store = {i: [] for i in range(n)} if memory_turns_per_agent > 0 else None

    # Main simulation loop
    if contagion_mode == "llm":
        # --- LLM Contagion Mode ---
        weight_cache = _init_weight_cache(G)
        # if conversation_scope="all", we pre-calculate all pairs to avoid re-computing
        # inside the loop; this is memory-intensive for large n but faster.
        all_pairs = _all_node_pairs(n) if conversation_scope == "all" else None
        allow_new_edges = conversation_scope == "all"
        dynamic_network = bool(cfg.get("dynamic_network", False))

        # Pre-process events for quick lookup
        events_by_round = defaultdict(list)
        for event in cfg.get("events", []):
            if "round" in event and "type" in event:
                events_by_round[int(event["round"])].append(event)

        # Convert deprecated intervention params to an event
        intervention_round_cfg = cfg.get("intervention_round")
        if intervention_round_cfg is not None:
            events_by_round[int(intervention_round_cfg)].append({
                "type": "intervention",
                "nodes": cfg.get("intervention_nodes", []),
                "content": cfg.get("intervention_content", "")
            })

        # Main simulation loop
        for t in range(1, rounds + 1):
            # Trigger events for the current round
            if t in events_by_round:
                for event in events_by_round[t]:
                    event_type = event.get("type")
                    if event_type == "intervention":
                        # This will be handled inside the loop where intervention_prompts is available
                        pass
                    elif event_type == "information_injection":
                        nodes = event.get("nodes", [])
                        score = float(event.get("seed_score", 1.0))
                        metric_id = str(event.get("metric", primary_metric_id))
                        if not isinstance(nodes, list): # handle "all" case
                            nodes = list(range(n))
                        for node_id in nodes:
                            if 0 <= node_id < n:
                                scores_by_metric[metric_id][node_id] = score
                                exposed[node_id] = True

            prev_scores = scores.copy()
            convos_for_round: List[Dict[str, Any]] = []
            if conversation_scope == "all":
                sample_pairs = all_pairs or []
            else:  # 'edges'
                # a random sample of edges to activate this round
                sample_pairs = list(G.edges())
            total_pairs = len(sample_pairs)
            if total_pairs == 0:
                break
            k = max(1, int(total_pairs * edge_sample_frac))
            k = min(total_pairs, k)
            weights_arr = np.array([_get_weight(weight_cache, u, v) for (u, v) in sample_pairs], dtype=float)
            if conversation_scope == "all":
                weights_arr = weights_arr + pair_weight_epsilon
            sum_w = float(weights_arr.sum())
            if sum_w > 0:
                probs = weights_arr / sum_w
                chosen_idx = np.random.choice(total_pairs, size=k, replace=False, p=probs)
            else:
                chosen_idx = np.random.choice(total_pairs, size=k, replace=False)
            rnd = [sample_pairs[idx] for idx in chosen_idx]
            for u, v in rnd:
                w = float(_get_weight(weight_cache, u, v))
                # Determine if this pair should talk about the topic.
                should_talk = (
                    random.random() < discuss_prob and
                    information_text != ""
                )

                # Get prior scores for each agent on each metric
                prior_scores_u = {mid: scores_by_metric[mid][u] for mid in metric_ids}
                prior_scores_v = {mid: scores_by_metric[mid][v] for mid in metric_ids}
                prev_u_primary = prior_scores_u.get(primary_metric_id, scores[u])
                prev_v_primary = prior_scores_v.get(primary_metric_id, scores[v])
                extra_i = intervention_content if (intervention_round_int is not None and t >= intervention_round_int and u in intervention_nodes and intervention_content) else ""
                extra_j = intervention_content if (intervention_round_int is not None and t >= intervention_round_int and v in intervention_nodes and intervention_content) else ""
                memory_text_i = _format_memory(memory_store, u, memory_turns_per_agent)
                memory_text_j = _format_memory(memory_store, v, memory_turns_per_agent)
                b_i, b_j, turns, did_talk, new_tie_strength = llm_conversation_and_scores(
                    model,
                    personas[u],
                    personas[v],
                    information_text, # Always pass the main topic
                    depth_intensity,
                    should_talk,
                    prior_scores_u,
                    prior_scores_v,
                    w,
                    max_convo_turns_cfg,
                    extra_system_i=extra_i,
                    extra_system_j=extra_j,
                    metrics=metrics,
                    memory_text_i=memory_text_i,
                    memory_text_j=memory_text_j,
                    dynamic_network=dynamic_network,
                )
                if print_convos and (print_all_convos or did_talk):
                    print(f"\n=== Conversation {u} <-> {v} ===")
                    for line in turns:
                        print(line)
                    if not did_talk:
                        print("(No information discussed; scores unchanged.)")
                    print(f"=== End Conversation {u} <-> {v} ===\n")
                if did_talk:
                    if print_updates:
                        try:
                            print(
                                f"Score update {u}<->{v} ({metric_name}): {u} {prev_u_primary:.2f} -> {scores[u]:.2f}, {v} {prev_v_primary:.2f} -> {scores[v]:.2f}"
                            )
                        except Exception:
                            print(
                                f"Score update {u}<->{v}: {u} {prev_u_primary} -> {scores[u]}, {v} {prev_v_primary} -> {scores[v]}"
                            )
                    for mid in metric_ids:
                        if b_i is not None:
                            scores_by_metric[mid][u] = b_i[mid]
                        if b_j is not None:
                            scores_by_metric[mid][v] = b_j[mid]
                
                # update state
                exposed[u] = exposed[u] or did_talk
                exposed[v] = exposed[v] or did_talk
                if memory_store is not None and did_talk:
                    _update_memory_store(memory_store, u, turns, memory_turns_per_agent)
                    _update_memory_store(memory_store, v, turns, memory_turns_per_agent)

                if dynamic_network and did_talk:
                    _set_edge_weight(weight_cache, u, v, new_tie_strength)

                # record conversation (or non-conversation) for this pair
                try:
                    convos_for_round.append({
                        "u": int(u),
                        "v": int(v),
                        "did_talk": bool(did_talk),
                        "turns": list(turns),
                    })
                except Exception:
                    convos_for_round.append({
                        "u": int(u),
                        "v": int(v),
                        "did_talk": bool(did_talk),
                        "turns": [str(x) for x in turns],
                    })
            cov = {i for i in range(n) if exposed[i]}
            active_cov = {i for i in range(n) if exposed[i] and any(scores_by_metric[mid][i] > 0 for mid in metric_ids)}
            arr_t = [scores[i] for i in range(n)]
            sum_t = llm_score_summary(model, summary_topic, arr_t, metric_name)
            if print_rounds:
                print(f"Round {t}: {len(cov)}/{n} exposed")
                print(f"Round {t} active believers (>0 score): {len(active_cov)}")
                print(f"Round {t} summary: {sum_t}")

            history_entry = {
                "round": t,
                "coverage": cov,
                "active_believers": active_cov,
                "scores": scores.copy(),
                "scores_multi": {mid: scores_by_metric[mid].copy() for mid in metric_ids},
                "summary": sum_t,
                "conversations": list(convos_for_round),
                "metrics": metrics,
            }
            history.append(history_entry)
            yield {
                "t": t,
                "G": G,
                "personas": personas,
                "scores": scores,
                "scores_multi": {mid: scores_by_metric[mid].copy() for mid in metric_ids},
                "exposed": exposed,
                "history_entry": history_entry,
                "full_history": history,
            }

            if stop_when_stable:
                max_diff = max(abs(scores[i] - prev_scores[i]) for i in range(n))
                if max_diff <= stability_tol:
                    break
    else:  # simple or complex contagion
        # --- Simple/Complex Contagion Mode ---
        for t in range(1, rounds + 1):
            prev_scores = scores.copy()
            prev_exposed = exposed.copy()
            next_exposed = exposed.copy()
            for i in G.nodes():
                if prev_exposed[i]:
                    continue
                num_exposed_neighbors = sum(1 for j in G.neighbors(i) if prev_exposed[j])
                if contagion_mode == "simple":
                    if num_exposed_neighbors >= 1:
                        next_exposed[i] = True
                else:
                    k = int(max(1, complex_k))
                    if num_exposed_neighbors >= k:
                        next_exposed[i] = True
            for i in range(n):
                if not exposed[i] and next_exposed[i]:
                    scores[i] = float(np.clip(max(scores[i], seed_score), 0.0, 1.0))
                    for mid in metric_ids:
                        scores_by_metric[mid][i] = scores[i]
            exposed = next_exposed
            cov = {i for i in range(n) if exposed[i]}
            active_cov = {i for i in range(n) if exposed[i] and scores[i] > 0}
            arr_t = [scores[i] for i in range(n)]
            sum_t = ""
            history_entry = {
                "round": t,
                "coverage": cov,
                "active_believers": active_cov,
                "scores": scores.copy(),
                "scores_multi": {mid: scores_by_metric[mid].copy() for mid in metric_ids},
                "summary": sum_t,
                "conversations": [],
                "metrics": metrics,
            }
            history.append(history_entry)
            yield {
                "t": t,
                "G": G,
                "personas": personas,
                "scores": scores,
                "scores_multi": {mid: scores_by_metric[mid].copy() for mid in metric_ids},
                "exposed": exposed,
                "history_entry": history_entry,
                "full_history": history,
            }
            if stop_when_stable:
                max_diff = max(abs(scores[i] - prev_scores[i]) for i in range(n))
                if max_diff <= stability_tol:
                    break


def run_simulation(cfg: Dict) -> Dict:
    # a bit of a monolith...
    final_state = {}
    for state in iterate_simulation(cfg):
        final_state = state
    G = final_state["G"]
    scores = final_state["scores"]
    scores_multi = {mid: dict(vals) for mid, vals in final_state.get("scores_multi", {}).items()}
    history = final_state["full_history"]
    personas = final_state["personas"]
    return {"G": G, "scores": scores, "scores_multi": scores_multi, "history": history, "personas": personas}


def _init_weight_cache(G) -> Dict[Tuple[int, int], float]:
    # For 'all' mode, this also serves to store potential (unrealized) edge weights.
    # It starts with all realized edges.
    return {(u, v): float(d.get("weight", 0.5)) for u, v, d in G.edges(data=True)}


def _get_weight(cache, u, v) -> float:
    key = tuple(sorted((u, v)))
    return cache.get(key, 0.0)


def _set_edge_weight(
    cache: Dict[Tuple[int, int], float],
    u: int,
    v: int,
    new_weight: float,
) -> None:
    """Directly set the edge weight in the cache."""
    key = tuple(sorted((u, v)))
    cache[key] = max(0.0, min(1.0, new_weight))



