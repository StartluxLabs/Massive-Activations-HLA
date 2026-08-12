#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import add_input_args, load_inputs, prepare_output_root

from massive_activations_hla.analysis.morphology import save_activation_trace
from massive_activations_hla.capture.activation_capture import capture_token_layer_max
from massive_activations_hla.capture.model_loading import load_model
from massive_activations_hla.models.registry import ModelRegistry
from massive_activations_hla.plotting.morphology import plot_token_trace


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trace and plot layerwise massive activations for PAS/ISP morphology."
    )
    p.add_argument("--models", required=True, help="Model registry YAML.")
    p.add_argument("--model-names", nargs="+", help="Specific model names from the registry.")
    p.add_argument("--model-group", help="Select models with metadata.group == this value.")
    add_input_args(p)
    p.add_argument("--token-index", type=int, default=0)
    p.add_argument("--max-length", type=int)
    p.add_argument("--output", default="outputs/morphology")
    p.add_argument("--capture-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    registry = ModelRegistry.from_yaml(args.models)
    specs = registry.select(names=args.model_names, group=args.model_group)
    if not specs:
        raise SystemExit("No models selected.")

    inputs = load_inputs(args.prompts, args.datasets, args.limit)

    outroot = prepare_output_root(args.output)
    for spec in specs:
        print(f"[morphology] loading {spec.name}")
        loaded = load_model(spec)
        for prompt in inputs:
            print(f"[morphology] capturing {spec.name} / {prompt.name}")
            trace = capture_token_layer_max(
                loaded.model,
                loaded.tokenizer,
                prompt.text,
                model_name=spec.name,
                prompt_name=prompt.name,
                max_length=args.max_length,
            )
            stem = outroot / spec.name / prompt.name / f"token_{args.token_index:03d}"
            save_activation_trace(trace, stem.with_suffix(".npz"))
            if not args.capture_only:
                plot_token_trace(
                    trace,
                    spec,
                    token_index=args.token_index,
                    output=stem.with_suffix(".png"),
                )
        del loaded


if __name__ == "__main__":
    main()
