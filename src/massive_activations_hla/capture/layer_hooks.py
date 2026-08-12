from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def get_decoder_layers(model: Any) -> list[Any]:
    """Best-effort decoder layer discovery for HF causal LMs."""
    candidates = [
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        obj = model
        ok = True
        for part in path:
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return list(obj)
    raise AttributeError("Could not locate decoder layers. Add a model adapter for this architecture.")


def tensor_from_output(output: Any) -> Any:
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return tensor_from_output(output[0])
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"Cannot extract tensor from output type {type(output)!r}")


@contextmanager
def capture_layer_outputs(layers: list[Any]) -> Iterator[list[Any]]:
    records: list[Any] = []
    handles = []

    def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
        records.append(tensor_from_output(output).detach().float().cpu())

    for layer in layers:
        handles.append(layer.register_forward_hook(hook))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()
