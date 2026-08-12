from __future__ import annotations

import argparse
from pathlib import Path

from massive_activations_hla.data.datasets import load_dataset_prompts
from massive_activations_hla.data.prompts import Prompt, load_prompts


def add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompts", help="Prompt YAML, e.g. configs/datasets/prompts.yaml.")
    parser.add_argument("--datasets", help="Dataset YAML, e.g. configs/datasets/five_domains.yaml.")
    parser.add_argument("--limit", type=int, help="Limit loaded dataset examples per dataset.")


def load_inputs(prompts: str | None, datasets: str | None, limit: int | None) -> list[Prompt]:
    """Load exactly one prompt source for command-line scripts."""
    if bool(prompts) == bool(datasets):
        raise SystemExit("Pass exactly one of --prompts or --datasets.")
    if prompts:
        return load_prompts(prompts)
    assert datasets is not None
    return load_dataset_prompts(datasets, limit)


def prepare_output_root(output: str | Path) -> Path:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    return out
