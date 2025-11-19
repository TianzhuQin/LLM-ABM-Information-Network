"""llm_society package: LLM-driven agent-based information diffusion simulation.

Modules:
- config: load configuration from YAML/JSON/dict
- persona: Persona dataclass and segment-based sampling
- network: random network builder
- llm: LLM client utilities and prompts
- simulation: run the simulation based on config
- viz: visualization helpers
"""

__version__ = "0.3.2"

from .config import load_config  # noqa: F401
from .persona import Persona, sample_personas, persona_to_text  # noqa: F401
from .network import build_random_network  # noqa: F401
from .llm import build_client  # noqa: F401
from .simulation import run_simulation  # noqa: F401
from . import viz  # noqa: F401

# OO API
from .api import Network, network  # noqa: F401


