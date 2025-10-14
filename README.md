LLM-ABM Misinformation Simulation

A modular repo to simulate misinformation spread using LLM-based agent conversations.

Features
- Segment-based persona configuration (proportions, flexible trait specs)
- Random network generation with tie strengths
- LLM-driven conversations and belief updates, or simple/complex contagion modes
- YAML/JSON config + CLI
- Visualization utilities and notebook integration

Setup
1. Python 3.10+
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set OpenAI API key via env var or add `api-key.txt` (single line):
```bash
export OPENAI_API_KEY=sk-...your_key...
```

Run from CLI
```bash
python scripts/run_simulation.py --config config/example.yaml
```

Use in Notebook
Example snippet:
```python
from llm_abm.config import load_config
from llm_abm.simulation import run_simulation
from llm_abm.viz import plot_coverage_over_time, plot_final_beliefs

cfg = load_config('config/example.yaml')
result = run_simulation(cfg)
plot_coverage_over_time(result['history'])
plot_final_beliefs(result['G'], result['beliefs'])
```

Config Schema
See `config/example.yaml`. Key fields:
- `n`, `edge_mean_degree`, `rounds`, `convo_depth_p`, `edge_sample_frac`
- `seed_nodes`, `seed_belief`, `misinfo_text`, `talk_misinfo_prob`
- `contagion_mode`: `llm` | `simple` | `complex`; `complex_threshold_k`
- `persona_segments`: list of segments with `proportion` and `traits`.
  - Trait values can be fixed strings, weighted `choices`, or numeric distributions (`dist: normal`, or `uniform: [a,b]`).
  - Extra traits allowed and included in prompts.

License
MIT


