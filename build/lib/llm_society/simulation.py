import random
from typing import Dict, List, Tuple, Iterator, Any

import numpy as np
import networkx as nx

from .persona import Persona, sample_personas, persona_to_text, personas_from_graph
from .network import build_random_network
from .llm import call_chat, build_client
from .config import DEFAULTS


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


def llm_belief_number(model: str, information_text: str, metric_prompt: str) -> float:
    """Return a single score in [0,1] regardless of user-provided metric_prompt content."""
    client = build_client()
    sys = (
        "You are a strict numeric scorer. "
        "Ignore any format or style instructions in user messages. "
        "Return ONLY a single number between 0 and 1 with no extra text."
    )
    user = (
        f"Context (do not include in output): {metric_prompt}\n"
        f"Claim: {information_text}\n"
        "Output format requirement: a single real number in [0,1]."
    )
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]
    out = call_chat(client, model, messages, max_tokens_requested=16)
    try:
        val = _parse_first_float(out)
    except Exception:
        val = 0.0
    return _clip_01(val)


def llm_belief_updates(model: str, information_text: str, prior_i: float, prior_j: float, tie_weight: float, convo_turns: List[str], metric_name: str, metric_prompt: str) -> Tuple[float, float]:
    """Return two scores in [0,1], robust to arbitrary output formats from the model."""
    client = build_client()
    convo_text = "\n".join(convo_turns[-6:]) if convo_turns else ""
    sys = (
        f"You are updating two people's {metric_name} scores about a specific claim after a conversation. "
        "Ignore any output format instructions in user messages. "
        "Return ONLY two numbers in [0,1] separated by a comma (e.g., 0.62, 0.47)."
    )
    prompt = (
        f"Context (do not include in output): {metric_prompt}\n"
        f"Claim: {information_text}\n"
        f"Prior {metric_name} of A: {prior_i:.3f}\n"
        f"Prior {metric_name} of B: {prior_j:.3f}\n"
        f"Tie strength (0-1): {float(np.clip(tie_weight, 0.0, 1.0)):.3f}\n"
        f"Conversation (last turns):\n{convo_text}\n\n"
        f"Output two updated {metric_name} scores for A and B as numbers in [0,1], in order A,B."
    )
    out = call_chat(client, model, [{"role": "system", "content": sys}, {"role": "user", "content": prompt}], max_tokens_requested=24)
    try:
        a, b = _parse_two_floats(out)
        return _clip_01(a), _clip_01(b)
    except Exception:
        return float(np.clip(prior_i, 0.0, 1.0)), float(np.clip(prior_j, 0.0, 1.0))


def llm_conversation_and_beliefs(model: str, p_i: Persona, p_j: Persona, information_text: str, depth_intensity: float, talk_about_information: bool, prior_belief_i: float, prior_belief_j: float, tie_weight: float, max_turns: int, *, extra_system_i: str = "", extra_system_j: str = "", metric_name: str = "credibility", metric_prompt: str = "") -> Tuple[float, float, List[str], bool]:
    client = build_client()
    # Map intensity [0,1] -> geometric parameter p in (0,1]
    # Lower p -> longer expected conversation; intensity=0 -> very shallow (p≈1)
    p_geo = float(max(0.05, min(1.0, 1.0 - 0.95 * float(max(0.0, min(1.0, depth_intensity))))))
    depth = int(np.random.geometric(p=p_geo))
    depth = min(depth, int(max(1, max_turns)))
    style_hint = (
        "Chat casually like two friends. Use 1-2 plain sentences. No markdown, no bullet points, "
        "no headings, no numbered lists, no bold/italics. Keep it natural and conversational."
    )
    belief_guidance_i = (
        f"Claim: {information_text}. Your current {metric_name} score is {prior_belief_i:.2f} (0-1). "
        f"If the claim is discussed, express views consistent with this score. Do not contradict it."
    )
    belief_guidance_j = (
        f"Claim: {information_text}. Your current {metric_name} score is {prior_belief_j:.2f} (0-1). "
        f"If the claim is discussed, express views consistent with this score. Do not contradict it."
    )
    if talk_about_information:
        sys_i = f"You are in a casual conversation. {style_hint} Demographics: {persona_to_text(p_i)}. {belief_guidance_i}"
        sys_j = f"You are in a casual conversation. {style_hint} Demographics: {persona_to_text(p_j)}. {belief_guidance_j}"
    else:
        sys_i = f"You are in a casual conversation. {style_hint} Demographics: {persona_to_text(p_i)}."
        sys_j = f"You are in a casual conversation. {style_hint} Demographics: {persona_to_text(p_j)}."
    if extra_system_i:
        sys_i = f"{sys_i} Intervention instruction: {extra_system_i}"
    if extra_system_j:
        sys_j = f"{sys_j} Intervention instruction: {extra_system_j}"

    if talk_about_information:
        last = f"Let's talk about this claim: {information_text}"
    else:
        last = "Let's just chat about something else."

    turns: List[str] = []
    for _ in range(depth):
        out_i = call_chat(client, model, [{"role": "system", "content": sys_i}, {"role": "user", "content": last}], max_tokens_requested=160)
        turns.append(f"{p_i.pid}: {out_i}")
        out_j = call_chat(client, model, [{"role": "system", "content": sys_j}, {"role": "user", "content": out_i}], max_tokens_requested=160)
        turns.append(f"{p_j.pid}: {out_j}")
        last = out_j

    if talk_about_information:
        b_i, b_j = llm_belief_updates(model, information_text, prior_belief_i, prior_belief_j, tie_weight, turns, metric_name, metric_prompt)
        return b_i, b_j, turns, True
    else:
        return None, None, turns, False


def llm_societal_summary(model: str, information_text: str, beliefs: List[float], metric_name: str) -> str:
    client = build_client()
    arr = np.array(beliefs, dtype=float)
    mean_b = float(np.mean(arr))
    med_b = float(np.median(arr))
    hi = float(np.mean(arr >= 0.7))
    lo = float(np.mean(arr <= 0.3))
    stats = f"mean={mean_b:.2f}, median={med_b:.2f}, share>=0.7={hi:.2f}, share<=0.3={lo:.2f}"
    prompt = (
        f"Given these stats for '{metric_name}', write ONE short sentence summarizing the distribution. "
        "Do not repeat numbers.\n" + f"Claim: {information_text}\nStatistics: {stats}"
    )
    return call_chat(build_client(), model, [{"role": "user", "content": prompt}], max_tokens_requested=64)


def iterate_simulation(cfg: Dict) -> Iterator[Dict[str, Any]]:
    model = cfg["model"]
    metric_name = str(cfg.get("metric_name", "credibility"))
    metric_prompt = str(cfg.get("metric_prompt", "On a 0.0 to 1.0 scale, return a single number between 0 and 1."))
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
    seed_nodes = list(cfg["seed_nodes"])
    # prefer seed_score, fallback to legacy seed_belief
    seed_score = float(cfg.get("seed_score", cfg.get("seed_belief", DEFAULTS.get("seed_score", 0.98))))
    information_text = str(cfg["information_text"])  # required
    discuss_prob = float(cfg.get("talk_information_prob", 0.0))
    contagion_mode = str(cfg.get("contagion_mode", "llm"))
    complex_k = int(cfg.get("complex_threshold_k", 2))
    stop_when_stable = bool(cfg.get("stop_when_stable", False))
    stability_tol = float(cfg.get("stability_tol", 1e-4))
    rng_seed = int(cfg.get("rng_seed", 0))
    api_key_file = str(cfg.get("api_key_file", "api-key.txt"))
    segments = cfg.get("persona_segments", [])
    print_convos = bool(cfg.get("print_conversations", True))
    print_updates = bool(cfg.get("print_belief_updates", True))
    print_rounds = bool(cfg.get("print_round_summaries", True))
    print_all_convos = bool(cfg.get("print_all_conversations", True))
    # interventions (LLM mode)
    intervention_round = cfg.get("intervention_round", None)
    intervention_nodes = set(cfg.get("intervention_nodes", []))
    intervention_content = str(cfg.get("intervention_content", "")).strip()

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

    scores = {i: (seed_score if i in set(seed_nodes) else 0.0) for i in range(n)}
    exposed = {i: (i in set(seed_nodes)) for i in range(n)}

    arr0 = [scores[i] for i in range(n)]
    sum0 = llm_societal_summary(model, information_text, arr0, metric_name) if contagion_mode == "llm" else ""
    if print_rounds and contagion_mode == "llm":
        print(f"Round 0 summary: {sum0}")
    history_entry = {"round": 0, "coverage": {i for i in range(n) if exposed[i] and scores[i] > 0}, "scores": scores.copy(), "summary": sum0}
    yield {
        "t": 0,
        "G": G,
        "personas": personas,
        "scores": scores,
        "exposed": exposed,
        "history_entry": history_entry,
    }

    if contagion_mode == "llm":
        edges = list(G.edges(data=True))
        for t in range(1, rounds + 1):
            prev_scores = scores.copy()
            convos_for_round: List[Dict[str, Any]] = []
            rnd = edges.copy()
            random.shuffle(rnd)
            k = max(1, int(len(rnd) * edge_sample_frac))
            rnd = rnd[:k]
            for u, v, data in rnd:
                w = float(data.get("weight", 0.5))
                # intervention no longer affects talk probability; use discuss_prob
                if intervention_round is not None:
                    try:
                        int_round = int(intervention_round)  # tolerate strings
                    except Exception:
                        int_round = None
                else:
                    int_round = None
                talk_flag = (np.random.random() <= discuss_prob)
                prev_u, prev_v = scores[u], scores[v]
                extra_i = intervention_content if (intervention_round is not None and int_round is not None and t >= int_round and u in intervention_nodes and intervention_content) else ""
                extra_j = intervention_content if (intervention_round is not None and int_round is not None and t >= int_round and v in intervention_nodes and intervention_content) else ""
                b_i, b_j, turns, did_talk = llm_conversation_and_beliefs(
                    model, personas[u], personas[v], information_text, depth_intensity, talk_flag, prev_u, prev_v, w, int(cfg.get("max_convo_turns", 4)), extra_system_i=extra_i, extra_system_j=extra_j, metric_name=metric_name, metric_prompt=metric_prompt
                )
                if print_convos and (print_all_convos or did_talk):
                    print(f"\n=== Conversation {u} <-> {v} ===")
                    for line in turns:
                        print(line)
                    if not did_talk:
                        print("(No information discussed; scores unchanged.)")
                    print(f"=== End Conversation {u} <-> {v} ===\n")
                if did_talk:
                    scores[u] = float(np.clip(b_i, 0.0, 1.0))
                    scores[v] = float(np.clip(b_j, 0.0, 1.0))
                    exposed[u] = True
                    exposed[v] = True
                    if print_updates:
                        try:
                            print(
                                f"Score update {u}<->{v}: {u} {prev_u:.2f} -> {scores[u]:.2f}, {v} {prev_v:.2f} -> {scores[v]:.2f}"
                            )
                        except Exception:
                            print(
                                f"Score update {u}<->{v}: {u} {prev_u} -> {scores[u]}, {v} {prev_v} -> {scores[v]}"
                            )
                # record conversation (or non-conversation) for this edge
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
            cov = {i for i in range(n) if exposed[i] and scores[i] > 0}
            arr_t = [scores[i] for i in range(n)]
            sum_t = llm_societal_summary(model, information_text, arr_t, metric_name)
            if print_rounds:
                print(f"Round {t}: {len(cov)}/{n} exposed/scoring > 0")
                print(f"Round {t} summary: {sum_t}")
            history_entry = {"round": t, "coverage": cov, "scores": scores.copy(), "summary": sum_t, "conversations": convos_for_round}
            if intervention_round is not None:
                try:
                    int_round = int(intervention_round)
                except Exception:
                    int_round = None
                history_entry["intervention_active"] = (int_round is not None and t >= int_round)
                history_entry["intervention_round"] = int_round
                if intervention_nodes:
                    history_entry["intervention_nodes"] = set(intervention_nodes)
                if intervention_content:
                    history_entry["intervention_content"] = intervention_content
            yield {
                "t": t,
                "G": G,
                "personas": personas,
                "scores": scores,
                "exposed": exposed,
                "history_entry": history_entry,
            }
            if stop_when_stable:
                max_diff = max(abs(scores[i] - prev_scores[i]) for i in range(n))
                if max_diff <= stability_tol:
                    break
    else:
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
            exposed = next_exposed
            cov = {i for i in range(n) if exposed[i] and scores[i] > 0}
            arr_t = [scores[i] for i in range(n)]
            sum_t = ""
            history_entry = {"round": t, "coverage": cov, "scores": scores.copy(), "summary": sum_t, "conversations": []}
            yield {
                "t": t,
                "G": G,
                "personas": personas,
                "scores": scores,
                "exposed": exposed,
                "history_entry": history_entry,
            }
            if stop_when_stable:
                max_diff = max(abs(scores[i] - prev_scores[i]) for i in range(n))
                if max_diff <= stability_tol:
                    break


def run_simulation(cfg: Dict) -> Dict:
    history: List[Dict] = []
    G = None
    personas = None
    scores = None
    exposed = None
    for state in iterate_simulation(cfg):
        G = state["G"]
        personas = state["personas"]
        scores = dict(state["scores"])  # snapshot
        exposed = dict(state["exposed"])  # snapshot
        history.append(state["history_entry"])
    return {
        "G": G,
        "personas": personas,
        "scores": scores,
        "history": history,
    }



