#!/usr/bin/env python3
import argparse
from llm_abm.config import load_config
from llm_abm.simulation import run_simulation


def main() -> None:
    p = argparse.ArgumentParser(description="Run LLM ABM simulation from config file")
    p.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config file")
    args = p.parse_args()
    cfg = load_config(args.config)
    result = run_simulation(cfg)
    # minimal console output
    history = result["history"]
    print(f"Rounds: {len(history) - 1}")
    print(f"Round 0 summary: {history[0]['summary']}")
    print(f"Final coverage: {len(history[-1]['coverage'])}")


if __name__ == "__main__":
    main()


