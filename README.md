### LLM Society — LLM-driven Information Diffusion

A Python package to simulate information diffusion with LLM-based agent conversations. It supports metric scoring in [0,1], segments-based personas, interventions, polished visualizations, and a simple CLI.

## Links
- Quickstart (tiny LLM network, n=8): `docs/QUICKSTART_TINY.ipynb`
- Advanced tutorial (segments, interventions, custom graphs, export): `docs/ADVANCED_TUTORIAL.ipynb`

## Features
- Segment-based persona configuration (proportions, flexible trait specs; optional segment names)
- Random network generation with tie strengths, or use your own NetworkX graph
- LLM-driven conversations and numeric scoring in [0,1] (metric-based), or simple/complex contagion modes
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
net.plot(type="final_scores")
net.plot(type="centrality", metric="degree", show_exposure=False)
```

## Plotting
- final_scores: final node scores heatmap on the graph
- coverage: coverage (exposed & score>0) over time
- group: mean score by group (by="traits" with attr in segments' traits; or by="segment")
- centrality: centrality vs final score; optionally add exposure panel via show_exposure=True
- intervention_effect: mean score over time with intervention marker; optionally group by traits
- animation: animated score evolution

## Advanced Capabilities
- Grouping
  - Traits: `net.plot(type="group", by="traits", attr="political")`
  - Segment: `net.plot(type="group", by="segment", groups=["High-Dem", "High-Rep"])`
- Interventions
  ```python
  net = network(..., intervention_round=6, intervention_nodes=[0,1,2], intervention_content="Be skeptical...")
  net.simulate()
  net.plot(type="intervention_effect", attr="political", groups=["Democrat","Republican"])
  ```
- Custom Graph Personas
  - If you pass `graph=G` and omit `segments`, personas are built from node attributes (`gender`, `race`, `age`, `religion`, `political`; others go to `extra`).

## Exporting
```python
net.export(
  history_csv="history.csv",
  beliefs_csv="scores_by_round.csv",    # backward-compatible argument name
  conversations_jsonl="conversations.jsonl",
)
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
  --depth 0.6 --depth-max 6 --edge-frac 0.5 \
  --seeds 0,1 --talk-prob 0.25 --mode llm --complex-k 2 --rng 0
```

## Configuration (overview)
- Core: `n`, `degree`, `rounds`, `depth` (0–1), `max_convo_turns`, `edge_sample_frac`
- Seeds: `seed_nodes`, `seed_score` (or legacy `seed_belief`)  
- Info/LLM: `information_text`, `talk_information_prob`, `model`, `metric_name`, `metric_prompt`
- Modes: `contagion_mode` in {llm, simple, complex}, `complex_threshold_k`
- Personas: `persona_segments` (with `proportion`, `traits`, optional `name`)

## License
MIT


