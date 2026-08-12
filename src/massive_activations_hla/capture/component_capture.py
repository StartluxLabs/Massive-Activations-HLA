from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from massive_activations_hla.capture.activation_capture import decode_token
from massive_activations_hla.capture.layer_hooks import get_decoder_layers, tensor_from_output

COMPONENT_KEYS = ["residual_input", "attention_update", "mlp_update", "block_output"]
COMPONENT_LABELS = ["Residual input", "Attention update", "MLP update", "Block output"]


@dataclass
class ComponentTrace:
    model_name: str
    prompt_name: str
    token_ids: list[int]
    token_labels: list[str]
    layer_values: np.ndarray  # [num_layers, seq_len], max abs block output per token
    component_values: np.ndarray  # [seq_len, 4, num_layers], signed values at each layer's dominant feature
    target_features: np.ndarray  # [seq_len, num_layers]


def get_attention_module(block: Any) -> Any | None:
    for name in ("self_attn", "attention", "attn", "linear_attn", "mixer"):
        if hasattr(block, name):
            return getattr(block, name)
    return None


def get_mlp_module(block: Any) -> Any | None:
    for name in ("mlp", "feed_forward", "ffn"):
        if hasattr(block, name):
            return getattr(block, name)
    return None


@contextmanager
def capture_component_outputs(layers: list[Any]) -> Iterator[dict[int, dict[str, Any]]]:
    records: dict[int, dict[str, Any]] = {i: {} for i in range(len(layers))}
    handles = []

    def layer_pre(i: int):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            records[i]["residual_input"] = args[0].detach().float().cpu()

        return hook

    def layer_post(i: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            records[i]["block_output"] = tensor_from_output(output).detach().float().cpu()

        return hook

    def component_post(i: int, key: str):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            records[i][key] = tensor_from_output(output).detach().float().cpu()

        return hook

    for i, block in enumerate(layers):
        handles.append(block.register_forward_pre_hook(layer_pre(i)))
        handles.append(block.register_forward_hook(layer_post(i)))
        attn = get_attention_module(block)
        if attn is not None:
            handles.append(attn.register_forward_hook(component_post(i, "attention_update")))
        mlp = get_mlp_module(block)
        if mlp is not None:
            handles.append(mlp.register_forward_hook(component_post(i, "mlp_update")))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


def capture_component_trace(
    model: Any,
    tokenizer: Any,
    text: str,
    model_name: str,
    prompt_name: str,
    max_length: int | None = None,
) -> ComponentTrace:
    import torch

    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids
    if max_length is not None:
        input_ids = input_ids[:, :max_length].contiguous()
    input_ids = input_ids.to(next(model.parameters()).device)
    layers = get_decoder_layers(model)
    with capture_component_outputs(layers) as records:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False, return_dict=True)

    num_layers = len(layers)
    seq_len = int(input_ids.shape[1])
    layer_values = np.zeros((num_layers, seq_len), dtype=np.float32)
    component_values = np.full((seq_len, len(COMPONENT_KEYS), num_layers), np.nan, dtype=np.float32)
    target_features = np.full((seq_len, num_layers), -1, dtype=np.int64)

    for layer_idx in range(num_layers):
        block_output = records[layer_idx]["block_output"][0]  # [T, H]
        layer_values[layer_idx] = block_output.abs().amax(dim=-1).numpy()
        for token_idx in range(seq_len):
            feature = int(block_output[token_idx].abs().argmax().item())
            target_features[token_idx, layer_idx] = feature
            for row_idx, key in enumerate(COMPONENT_KEYS):
                if key in records[layer_idx]:
                    component_values[token_idx, row_idx, layer_idx] = float(
                        records[layer_idx][key][0, token_idx, feature].item()
                    )

    token_ids = input_ids[0].detach().cpu().tolist()
    return ComponentTrace(
        model_name=model_name,
        prompt_name=prompt_name,
        token_ids=[int(x) for x in token_ids],
        token_labels=[decode_token(tokenizer, int(x)) for x in token_ids],
        layer_values=layer_values,
        component_values=component_values,
        target_features=target_features,
    )
