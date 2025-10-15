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
export OPENAI_API_KEY=sk-...your_key...
```

Run from CLI
```bash
# write an example config to a path
llm-society --write-example-config my-config.yaml

# run the simulation with your config
llm-society --config my-config.yaml
```

Use in Notebook
Example snippet:
```python
from llm_society.config import load_config
from llm_society.simulation import run_simulation
from llm_society.viz import plot_coverage_over_time, plot_final_beliefs
from llm_society import network

cfg = load_config('config/example.yaml')
result = run_simulation(cfg)
plot_coverage_over_time(result['history'])
plot_final_beliefs(result['G'], result['beliefs'])
```

Object-oriented API
```python
from llm_society import network

net = network(n=5, degree=2, rounds=10, depth=0.6, depth_max=6,
              edge_frac=0.5, seeds=[0,1], seed_belief=0.98,
              claim="5G towers cause illness.", talk_prob=0.25,
              mode="llm", complex_k=2, rng=0)
net.simulate()   # prints conversations, belief updates, summaries
net.plot()       # coverage curve + final beliefs graph
net.nodes[1].plot()  # single-node belief trajectory
```

Config Schema
See `config/example.yaml`. Key fields:
- `n`, `edge_mean_degree`, `rounds`, `convo_depth_p`, `edge_sample_frac`
- `seed_nodes`, `seed_belief`, `information_text`, `talk_information_prob`
- `contagion_mode`: `llm` | `simple` | `complex`; `complex_threshold_k`
- `persona_segments`: list of segments with `proportion` and `traits`.
  - Trait values can be fixed strings, weighted `choices`, or numeric distributions (`dist: normal`, or `uniform: [a,b]`).
  - Extra traits allowed and included in prompts.

License
MIT


