from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from massive_activations_hla.capture.component_capture import ComponentTrace


def save_component_trace(trace: ComponentTrace, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        layer_values=trace.layer_values,
        component_values=trace.component_values,
        target_features=trace.target_features,
        token_ids=np.array(trace.token_ids, dtype=np.int64),
        token_labels=np.array(trace.token_labels, dtype=object),
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "model_name": trace.model_name,
                "prompt_name": trace.prompt_name,
                "num_layers": int(trace.layer_values.shape[0]),
                "num_tokens": int(trace.layer_values.shape[1]),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_component_trace(path: str | Path) -> ComponentTrace:
    path = Path(path)
    arr = np.load(path, allow_pickle=True)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return ComponentTrace(
        model_name=meta["model_name"],
        prompt_name=meta["prompt_name"],
        token_ids=[int(x) for x in arr["token_ids"].tolist()],
        token_labels=[str(x) for x in arr["token_labels"].tolist()],
        layer_values=arr["layer_values"],
        component_values=arr["component_values"],
        target_features=arr["target_features"],
    )
