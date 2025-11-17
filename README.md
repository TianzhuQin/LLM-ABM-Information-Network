### LLM Society — LLM-driven Information Diffusion

A Python package to simulate information diffusion with LLM-based agent conversations. It supports metric scoring in [0,1], segments-based personas, interventions, polished visualizations, and a simple CLI.

## Links
- Tutorial (LLM network, segments, interventions, custom graphs, export): `docs/TUTORIAL.ipynb`

## Features
- Segment-based persona configuration (proportions, flexible trait specs; optional segment names)
- Random network generation with tie strengths, or use your own NetworkX graph
- LLM-driven conversations and numeric scoring in [0,1] (metric-based), or simple/complex contagion modes
- Tie strengths influence edge sampling, talk probability, conversation depth, and can grow/decay over time (even for non-adjacent pairs via all-pairs mode)
- Optional agent memory to keep recent utterances in-context for longer-term continuity
- Multi-metric scoring per topic (e.g., credibility, emotion, action intent) with user-defined prompts and joint JSON outputs
- Interactive dashboards (Plotly/Bokeh) for rapid, shareable analysis
- Group plots (by traits or by segment), intervention effect plots, centrality plots, animations
- YAML/JSON config + CLI; exporting history/scores/conversations

## Installation
1) Python 3.10+
2) Install
```bash
pip install -r requirements.txt
```
3) Provide OpenAI key (LLM mode)
```bash
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
# or use a file (first line)
echo "<YOUR_OPENAI_API_KEY>" > api-key.txt
```

## Quickstart (Notebook)
```python
from llm_society.api import network
from llm_society.viz import set_theme

set_theme()
net = network(
  information="5G towers cause illness.",
  n=20, degree=4, rounds=10,
  talk_prob=0.25, mode="llm", complex_k=2, rng=0
)
net.simulate()             # conversations, score updates, summaries
net.plot(type="final")
net.plot(type="centrality", metric="degree", show_exposure=False)
```

## Plotting
- final: final node scores heatmap on the graph
- coverage: coverage (exposed & score>0) over time
- group: mean score by group (by="traits" with attr in segments' traits; or by="segment")
- centrality: centrality vs final score; optionally add exposure panel via show_exposure=True
- intervention: mean score over time with intervention marker; optionally group by traits
- animation: animated score evolution

## Advanced Capabilities
- Grouping
  - Traits: `net.plot(type="group", by="traits", attr="political")`
  - Segment: `net.plot(type="group", by="segment", groups=["High-Dem", "High-Rep"])`
- Interventions
  ```python
  net = network(..., intervention_round=6, intervention_nodes=[0,1,2], intervention_content="Be skeptical...")
  net.simulate()
  net.plot(type="intervention", attr="political", groups=["Democrat","Republican"])
  ```
- Custom Graph Personas
  - If you pass `graph=G` and omit `segments`, personas are built from node attributes (`gender`, `race`, `age`, `religion`, `political`; others go to `extra`).
- All-pairs conversations (LLM mode)
  - Set `conversation_scope="all_pairs"` (or CLI `--conversation-scope all_pairs`) to allow any node pair to chat.
  - Pairs without edges start at weight 0 but still get a small selection chance; repeated conversations strengthen their tie and add the edge into the network.
- Multi-metric scoring
  - Define `metrics` (list of `{id,label,prompt}`) so each conversation returns structured JSON with coordinated scores for both speakers.
  - The first metric acts as the "primary" score used by legacy APIs/plots; additional metrics are stored in `history[*].scores_multi` and can be visualized via `net.plot(..., metric="emotion")`.
- Intervention-only runs
  - Leave `information=""` and configure `intervention_round`, `intervention_nodes`, and `intervention_content`.
  - Agents chat casually until the intervention round starts, after which conversations probabilistically focus on the treatment content.
- Agent memory
  - Set `memory_turns_per_agent > 0` (e.g., 4–8) to inject that many recent utterances (self + partners) into each agent’s system prompt so they can recall past exchanges.

## Exporting
```python
net.export(
  history_csv="history.csv",
  scores_csv="scores_by_round.csv",
  conversations_jsonl="conversations.jsonl",
)

# interactive dashboard inside notebooks
fig = net.dashboard(engine="plotly", attr="political", metric="credibility")
fig

# or save to HTML manually
html = net.dashboard(engine="plotly", attr="political", metric="credibility", to_html=True)
Path("dashboard.html").write_text(html, encoding="utf-8")
```

## CLI
```bash
# write an example config
llm-society --write-example-config my-config.yaml
# run with a config
llm-society --config my-config.yaml
# or run fully via flags
llm-society \
  --information "Claim text" --n 20 --degree 4 --rounds 10 \
  --depth 0.6 --depth-max 6 --edge-frac 0.5 --conversation-scope all \
  --seeds 0,1 --talk-prob 0.25 --mode llm --complex-k 2 --rng 0
```

## Configuration (overview)
- Core: `n`, `degree`, `rounds`, `depth` (0–1), `max_convo_turns`, `edge_sample_frac`
- Seeds: `seed_nodes`, `seed_score`  
- Info/LLM: `information` (may be blank if an intervention is configured), `talk_information_prob`, `model`, `metric_name`, `metric_prompt`
- Metrics: optional `metrics=[{id,label,prompt}, ...]` to request multi-dimensional scoring (first metric remains the default for legacy APIs)
- Modes: `contagion_mode` in {llm, simple, complex}, `complex_threshold_k`
- Conversation scope: `conversation_scope` in {edges, all}, `pair_weight_epsilon` (minimum sampling weight boost for zero-tie pairs)
- Memory: `memory_turns_per_agent` (0 disables memory)
- Personas: `persona_segments` (with `proportion`, `traits`, optional `name`)

## License
MIT


