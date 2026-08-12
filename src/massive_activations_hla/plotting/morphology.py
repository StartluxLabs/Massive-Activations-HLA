from __future__ import annotations

from pathlib import Path

import numpy as np

from massive_activations_hla.capture.activation_capture import ActivationTrace
from massive_activations_hla.models.registry import ModelSpec
from massive_activations_hla.plotting.lifecycle_atlas import pretty_model_title
from massive_activations_hla.plotting.style import set_paper_style


def plot_token_trace(
    trace: ActivationTrace,
    model_spec: ModelSpec,
    token_index: int = 0,
    output: str | Path | None = None,
    title: str | None = None,
) -> Path | None:
    import matplotlib.pyplot as plt

    set_paper_style()
    y = trace.layer_values[:, token_index]
    x = np.arange(1, len(y) + 1)
    fig, ax = plt.subplots(figsize=(10.5, 3.0))
    ax.set_facecolor("#eef6ff")
    ax.plot(x, y, color="#0877c9", lw=3.0, marker="o", ms=5.5, mec="white", mew=0.9)
    for f in model_spec.full_attention_layers:
        if f > 1:
            ax.axvspan(f - 1 - 0.35, f - 1 + 0.35, color="#f2b8b5", alpha=0.28, lw=0)
        ax.axvspan(f - 0.35, f + 0.35, color="#f4d35e", alpha=0.28, lw=0)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Max |activation|")
    token_label = trace.token_labels[token_index] if token_index < len(trace.token_labels) else str(token_index)
    default_title = f"{pretty_model_title(model_spec)} · token {token_index}: {token_label!r}"
    ax.set_title(title or default_title, pad=10)
    ax.grid(axis="y", alpha=0.35)
    ax.set_xlim(0.5, len(y) + 0.5)
    fig.tight_layout()
    if output is None:
        return None
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    return output
