#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from massive_activations_hla.analysis.attention_maps import load_attention_maps
from massive_activations_hla.analysis.morphology import load_activation_trace
from massive_activations_hla.analysis.pas_isp_metrics import (
    fixed_sink_tokens,
    max_activation_sink_tokens,
    pas_isp_summary,
    sink_tokens_from_attention_maps,
)
from massive_activations_hla.models.registry import ModelRegistry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute PAS/ISP metrics from captured activation traces."
    )
    p.add_argument("--models", required=True, help="Model registry YAML.")
    p.add_argument("--model-names", nargs="+", required=True)
    p.add_argument("--trace-root", required=True, help="Root produced by run_morphology.py.")
    p.add_argument(
        "--attention-root",
        help=(
            "Optional root containing full-attention maps. Expected files are searched under "
            "<attention-root>/<model>/<prompt>/layer_<L>.npy or .npz, with L as a 1-based layer index. "
            "Arrays should have shape [heads, query, source] or [query, source]."
        ),
    )
    p.add_argument("--prompt-names", nargs="+", required=True)
    p.add_argument("--sink-token", type=int, default=0, help="Consensus sink token index for fixed-token metrics.")
    p.add_argument(
        "--trace-token-index",
        type=int,
        default=0,
        help="Token-index stem used by run_morphology.py when saving token_<idx>.npz. The trace file stores all tokens.",
    )
    p.add_argument(
        "--sink-mode",
        choices=["fixed", "max_activation"],
        default="fixed",
        help=(
            "How to choose sink tokens for PAS alignment when attention maps are not supplied. "
            "`fixed` uses --sink-token; `max_activation` uses each pre-attention layer's max-activation token."
        ),
    )
    p.add_argument("--output", default="outputs/metrics/pas_isp_metrics.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    registry = ModelRegistry.from_yaml(args.models)
    rows = []
    for name in args.model_names:
        spec = registry.get(name)
        for prompt_name in args.prompt_names:
            trace_path = Path(args.trace_root) / name / prompt_name / f"token_{args.trace_token_index:03d}.npz"
            trace = load_activation_trace(trace_path)
            # The trace stores all-token layer values; metrics use the selected sink token.
            layer_values = np.asarray(trace.layer_values)
            attention_maps = load_attention_maps(
                args.attention_root,
                model=name,
                prompt=prompt_name,
                full_layers=spec.full_attention_layers,
            )
            if attention_maps:
                sink_tokens = sink_tokens_from_attention_maps(attention_maps)
                retention_token = args.sink_token
                resolved_sink_mode = "attention"
            elif args.sink_mode == "fixed":
                sink_tokens = fixed_sink_tokens(spec.full_attention_layers, args.sink_token)
                retention_token = args.sink_token
                resolved_sink_mode = "fixed"
            else:
                sink_tokens = max_activation_sink_tokens(layer_values, spec.full_attention_layers)
                retention_token = int(np.argmax(layer_values.max(axis=0)))
                resolved_sink_mode = "max_activation"
            metrics = pas_isp_summary(
                layer_values=layer_values,
                full_attention_layers=spec.full_attention_layers,
                sink_tokens_by_full_layer=sink_tokens,
                retention_token=retention_token,
            )
            rows.append(
                {
                    "model": name,
                    "prompt": prompt_name,
                    "trace_token_index": args.trace_token_index,
                    "sink_mode": resolved_sink_mode,
                    "retention_token": retention_token,
                    "num_attention_maps": len(attention_maps),
                    **metrics,
                }
            )
    if not rows:
        raise SystemExit("No metric rows were produced. Check --model-names and --prompt-names.")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
