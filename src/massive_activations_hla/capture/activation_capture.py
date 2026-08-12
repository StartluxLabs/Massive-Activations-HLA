from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from massive_activations_hla.capture.layer_hooks import capture_layer_outputs, get_decoder_layers


@dataclass
class ActivationTrace:
    model_name: str
    prompt_name: str
    token_ids: list[int]
    token_labels: list[str]
    layer_values: np.ndarray  # [num_layers, seq_len], max abs activation per token


def decode_token(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    return text.replace("\n", "\\n").replace("\t", "\\t")


def capture_token_layer_max(
    model: Any,
    tokenizer: Any,
    text: str,
    model_name: str,
    prompt_name: str,
    max_length: int | None = None,
) -> ActivationTrace:
    import torch

    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids
    if max_length is not None:
        input_ids = input_ids[:, :max_length].contiguous()
    input_ids = input_ids.to(next(model.parameters()).device)
    layers = get_decoder_layers(model)
    with capture_layer_outputs(layers) as outputs:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False, return_dict=True)
    values = []
    for out in outputs:
        # [B, T, H] -> [T]
        values.append(out[0].abs().amax(dim=-1).numpy())
    token_ids = input_ids[0].detach().cpu().tolist()
    return ActivationTrace(
        model_name=model_name,
        prompt_name=prompt_name,
        token_ids=[int(x) for x in token_ids],
        token_labels=[decode_token(tokenizer, int(x)) for x in token_ids],
        layer_values=np.stack(values, axis=0),
    )
