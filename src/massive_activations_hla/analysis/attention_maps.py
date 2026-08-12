from __future__ import annotations

from pathlib import Path

import numpy as np


def load_attention_array(path: str | Path) -> np.ndarray:
    """Load one attention array from .npy or .npz."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        if "attention" in arr:
            return np.asarray(arr["attention"])
        if "attn" in arr:
            return np.asarray(arr["attn"])
        first_key = list(arr.keys())[0]
        return np.asarray(arr[first_key])
    return np.asarray(arr)


def load_attention_maps(
    attention_root: str | Path | None,
    model: str,
    prompt: str,
    full_layers: list[int],
) -> dict[int, np.ndarray]:
    """Load full-attention maps arranged by model/prompt/layer."""
    if attention_root is None:
        return {}
    base = Path(attention_root) / model / prompt
    maps: dict[int, np.ndarray] = {}
    for layer in full_layers:
        candidates = [
            base / f"layer_{layer}.npy",
            base / f"layer_{layer:02d}.npy",
            base / f"layer_{layer}.npz",
            base / f"layer_{layer:02d}.npz",
        ]
        for path in candidates:
            if path.exists():
                maps[int(layer)] = load_attention_array(path)
                break
    return maps
