#!/usr/bin/env python3
import argparse
from importlib import resources

from llm_society.config import load_config
from llm_society.simulation import run_simulation


def write_example_config(dest_path: str) -> None:
    with resources.files("llm_society").joinpath("data/example.yaml").open("rb") as rf:
        data = rf.read()
    with open(dest_path, "wb") as wf:
        wf.write(data)


def main() -> None:
    p = argparse.ArgumentParser(description="Run LLM Society simulation")
    p.add_argument("--config", type=str, help="Path to YAML/JSON config file")
    p.add_argument(
        "--write-example-config",
        type=str,
        metavar="PATH",
        help="Write packaged example config to PATH and exit",
    )
    args = p.parse_args()

    if args.write_example_config:
        write_example_config(args.write_example_config)
        print(f"Wrote example config to {args.write_example_config}")
        return

    if not args.config:
        p.error("--config is required unless --write-example-config is provided")

    cfg = load_config(args.config)
    result = run_simulation(cfg)
    history = result["history"]
    print(f"Rounds: {len(history) - 1}")
    print(f"Round 0 summary: {history[0]['summary']}")
    print(f"Final coverage: {len(history[-1]['coverage'])}")


if __name__ == "__main__":
    main()


