from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from massive_activations_hla.capture.activation_capture import ActivationTrace


def save_activation_trace(trace: ActivationTrace, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        layer_values=trace.layer_values,
        token_ids=np.array(trace.token_ids, dtype=np.int64),
        token_labels=np.array(trace.token_labels, dtype=object),
    )
    meta = {
        "model_name": trace.model_name,
        "prompt_name": trace.prompt_name,
        "num_layers": int(trace.layer_values.shape[0]),
        "num_tokens": int(trace.layer_values.shape[1]),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def load_activation_trace(path: str | Path) -> ActivationTrace:
    path = Path(path)
    arr = np.load(path, allow_pickle=True)
    meta = json.loads(path.with_suffix(".json").read_text())
    return ActivationTrace(
        model_name=meta["model_name"],
        prompt_name=meta["prompt_name"],
        token_ids=[int(x) for x in arr["token_ids"].tolist()],
        token_labels=[str(x) for x in arr["token_labels"].tolist()],
        layer_values=arr["layer_values"],
    )
