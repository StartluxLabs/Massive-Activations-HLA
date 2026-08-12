from __future__ import annotations

from pathlib import Path

import numpy as np

from massive_activations_hla.capture.component_capture import COMPONENT_LABELS, ComponentTrace
from massive_activations_hla.models.registry import ModelSpec
from massive_activations_hla.plotting.style import set_paper_style


def pretty_model_title(model_spec: ModelSpec) -> str:
    family = model_spec.family or model_spec.name
    family = {
        "GatedDeltaNet": "GDN",
        "DeltaNet": "DeltaNet",
        "Transformer": "Transformer",
    }.get(family, family)
    bits = [family]
    if model_spec.scale:
        bits.append(model_spec.scale)
    if model_spec.hybrid_ratio:
        bits.append(f"Hybrid {model_spec.hybrid_ratio}")
    return " · ".join(bits)


def plot_lifecycle_atlas(
    trace: ComponentTrace,
    model_spec: ModelSpec,
    token_index: int,
    output: str | Path,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    set_paper_style()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    y = trace.layer_values[:, token_index]
    mat = trace.component_values[token_index]
    x = np.arange(1, len(y) + 1)

    fig = plt.figure(figsize=(14.6, 6.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.18], hspace=0.58)
    ax_top = fig.add_subplot(gs[0])
    ax_map = fig.add_subplot(gs[1])

    pre_color = "#EFA6A6"
    full_color = "#F3D36B"
    for ax in (ax_top, ax_map):
        for f in model_spec.full_attention_layers:
            if f > 1:
                ax.axvspan(f - 1 - 0.35, f - 1 + 0.35, color=pre_color, alpha=0.26, zorder=0)
            ax.axvspan(f - 0.35, f + 0.35, color=full_color, alpha=0.30, zorder=0)

    ax_top.set_facecolor("#eef6ff")
    ax_top.plot(x, y, color="#0877c9", lw=3.2, marker="o", ms=6.0, mec="white", mew=0.9)
    ax_top.set_xlim(0.5, len(y) + 0.5)
    ax_top.set_ylabel("Max |activation|", fontweight="bold")
    ax_top.grid(axis="y", alpha=0.30)
    ax_top.set_title("Layerwise Activation Magnitude", pad=6)

    vmax = np.nanpercentile(np.abs(mat), 98)
    vmax = max(float(vmax), 1e-6)
    im = ax_map.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax_map.set_yticks(np.arange(len(COMPONENT_LABELS)))
    ax_map.set_yticklabels(COMPONENT_LABELS)
    ax_map.set_xticks(np.arange(len(y)))
    ax_map.set_xticklabels(x)
    ax_map.set_xlabel("Decoder layer", fontweight="bold", labelpad=36)
    ax_map.set_title("Signed Components of Each Layer's Dominant Feature", pad=8)

    for f in model_spec.full_attention_layers:
        if f > 1:
            ax_map.axvline(f - 2, color="#C44E52", linewidth=1.7, alpha=0.75)
            ax_map.text(
                f - 2,
                len(COMPONENT_LABELS) + 0.18,
                "Pre\nattn",
                color="#C44E52",
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
                clip_on=False,
            )
        ax_map.axvline(f - 1, color="#9C7A00", linewidth=1.7, alpha=0.85)
        ax_map.text(
            f - 1,
            len(COMPONENT_LABELS) + 0.18,
            "Full\nattn",
            color="#9C7A00",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            clip_on=False,
        )

    cbar = fig.colorbar(im, ax=ax_map, fraction=0.026, pad=0.018)
    cbar.set_label("signed value", fontweight="bold")

    handles = [
        Patch(facecolor=pre_color, alpha=0.45, label="Pre-attention layer"),
        Patch(facecolor=full_color, alpha=0.55, label="Full-attention layer"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2)
    token_label = trace.token_labels[token_index] if token_index < len(trace.token_labels) else str(token_index)
    title = f"{pretty_model_title(model_spec)} · token {token_index}: {token_label!r}"
    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.905)
    fig.subplots_adjust(left=0.13, right=0.94, bottom=0.14, top=0.80, hspace=0.60)
    fig.savefig(output)
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    return output
