LLM Society Information Diffusion Simulation

A modular repo to simulate information diffusion using LLM-based agent conversations.

Features
- Segment-based persona configuration (proportions, flexible trait specs)
- Random network generation with tie strengths
- LLM-driven conversations and belief updates, or simple/complex contagion modes
- YAML/JSON config + CLI
- Visualization utilities and notebook integration

Install
1. Python 3.10+
2. Install the package (from PyPI when published, or locally):
```bash
pip install llm-society
# or for local development
pip install -e .
```
3. Set OpenAI API key via env var or add `api-key.txt` (single line):
```bash
# Preferred: environment variable
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
# Or: file (gitignored)
echo "<YOUR_OPENAI_API_KEY>" > api-key.txt
```

Run from CLI
```bash
# write an example config to a path
llm-society --write-example-config my-config.yaml

# run the simulation with your config
llm-society --config my-config.yaml

# override selected parameters via flags
llm-society --config my-config.yaml --depth 0.8 --rounds 20

# run fully via flags (no config file)
llm-society \
  --information "5G towers cause illness." \
  --n 20 --degree 4 --rounds 10 \
  --depth 0.6 --depth-max 6 --edge-frac 0.5 \
  --seeds 0,1 --seed-belief 0.98 --talk-prob 0.25 \
  --mode llm --complex-k 2 --rng 0 --model gpt-4.1
```

Use in Notebook
Example snippet:

```python
from llm_society import network

net = network(n=5, degree=2, rounds=10, depth=0.6, depth_max=6,
             edge_frac=0.5, seeds=[0,1], seed_belief=0.98,
             information="5G towers cause illness.", talk_prob=0.25,
             mode="llm", complex_k=2, rng=0)
net.simulate()   # prints conversations, belief updates, summaries
net.plot()       # coverage curve + final beliefs graph
net.nodes[1].plot()  # single-node belief trajectory
```

Config Schema
See `llm_society/data/example.yaml`. Key fields:
- `n`, `degree`, `rounds`, `depth` (0-1), `max_convo_turns`, `edge_sample_frac`
- `seed_nodes`, `seed_belief`, `information_text`, `talk_information_prob`
- `contagion_mode`: `llm` | `simple` | `complex`; `complex_threshold_k`
- `persona_segments`: list of segments with `proportion` and `traits`.
  - Trait values can be fixed strings, weighted `choices`, or numeric distributions (`dist: normal`, or `uniform: [a,b]`).
  - Extra traits allowed and included in prompts.

Depth interpretation:
- `depth` ∈ [0,1] controls conversation length tendency. Higher means longer conversations.
- Internally mapped to a geometric distribution; `depth=0` → very short; `depth=1` → near the `max_convo_turns` cap often.

License
MIT


