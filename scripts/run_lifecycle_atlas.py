#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import add_input_args, load_inputs, prepare_output_root

from massive_activations_hla.analysis.lifecycle import save_component_trace
from massive_activations_hla.capture.component_capture import capture_component_trace
from massive_activations_hla.capture.model_loading import load_model
from massive_activations_hla.models.registry import ModelRegistry
from massive_activations_hla.plotting.lifecycle_atlas import plot_lifecycle_atlas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture and plot systematic-outlier lifecycle atlas."
    )
    p.add_argument("--models", required=True, help="Model registry YAML.")
    p.add_argument("--model-names", nargs="+", required=True)
    add_input_args(p)
    p.add_argument("--token-index", type=int, default=0)
    p.add_argument("--max-length", type=int)
    p.add_argument("--output", default="outputs/lifecycle_atlas")
    p.add_argument("--capture-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    registry = ModelRegistry.from_yaml(args.models)
    specs = [registry.get(name) for name in args.model_names]
    inputs = load_inputs(args.prompts, args.datasets, args.limit)
    outroot = prepare_output_root(args.output)
    for spec in specs:
        print(f"[lifecycle] loading {spec.name}")
        loaded = load_model(spec)
        for prompt in inputs:
            print(f"[lifecycle] capturing {spec.name} / {prompt.name}")
            trace = capture_component_trace(
                loaded.model,
                loaded.tokenizer,
                prompt.text,
                model_name=spec.name,
                prompt_name=prompt.name,
                max_length=args.max_length,
            )
            stem = outroot / spec.name / prompt.name / f"token_{args.token_index:03d}_atlas"
            save_component_trace(trace, stem.with_suffix(".npz"))
            if not args.capture_only:
                out = plot_lifecycle_atlas(
                    trace=trace,
                    model_spec=spec,
                    token_index=args.token_index,
                    output=stem.with_suffix(".png"),
                )
                print(f"Saved {out}")
        del loaded


if __name__ == "__main__":
    main()
