from __future__ import annotations

import numpy as np


def fixed_sink_tokens(full_attention_layers: list[int], sink_token: int) -> dict[int, int]:
    """Use the same sink token for every full-attention event."""
    return {int(f): int(sink_token) for f in full_attention_layers}


def max_activation_sink_tokens(layer_values: np.ndarray, full_attention_layers: list[int]) -> dict[int, int]:
    """Choose the pre-attention layer's max-activation token for each FA event.

    This is not an attention-sink metric; it is a useful fallback/control when
    attention maps are unavailable.
    """
    out: dict[int, int] = {}
    for f in full_attention_layers:
        if f <= 1 or f - 2 >= layer_values.shape[0]:
            continue
        out[int(f)] = int(np.argmax(layer_values[f - 2]))
    return out


def sink_spike_alignment_rate(
    sink_tokens_by_full_layer: dict[int, int],
    layer_values: np.ndarray,
    full_attention_layers: list[int],
) -> tuple[float, int, int]:
    """Compute PAS alignment rate for one input.

    `layer_values` is [num_layers, seq_len] and uses 1-based layer indexing in
    `full_attention_layers`. For a full-attention layer f, the corresponding
    pre-attention layer is f-1.
    """
    hits = 0
    total = 0
    prev_full = 0
    for f in full_attention_layers:
        if f <= 1:
            prev_full = f
            continue
        token = sink_tokens_by_full_layer.get(f)
        if token is None or token >= layer_values.shape[1]:
            prev_full = f
            continue
        block = list(range(prev_full + 1, f))
        if not block:
            prev_full = f
            continue
        block_idx = [x - 1 for x in block]
        pre_idx = f - 2
        values = layer_values[block_idx, token]
        hits += int(layer_values[pre_idx, token] == np.max(values))
        total += 1
        prev_full = f
    return (float(hits / total) if total else float("nan"), hits, total)


def dominant_sink_token_from_attention(attention: np.ndarray) -> int:
    """Identify the source token with highest mean received causal attention.

    Parameters
    ----------
    attention:
        Attention array with shape [heads, query, source] or [query, source].
        The function averages over heads and over valid future queries q > t.
    """
    if attention.ndim == 2:
        attention = attention[None, :, :]
    if attention.ndim != 3:
        raise ValueError("attention must have shape [heads, query, source] or [query, source]")
    _, q_len, s_len = attention.shape
    scores = np.full(s_len, -np.inf, dtype=np.float64)
    for t in range(s_len):
        queries = np.arange(q_len) > t
        if np.any(queries):
            scores[t] = float(attention[:, queries, t].mean())
    return int(np.argmax(scores))


def sink_tokens_from_attention_maps(attentions_by_full_layer: dict[int, np.ndarray]) -> dict[int, int]:
    """Map full-attention layer index to dominant sink token from attention maps."""
    return {int(f): dominant_sink_token_from_attention(np.asarray(attn)) for f, attn in attentions_by_full_layer.items()}


def inter_spike_retention_score(
    consensus_sink_token: int,
    layer_values: np.ndarray,
    full_attention_layers: list[int],
) -> float:
    """Compute inter-spike retention between adjacent PAS events."""
    if consensus_sink_token < 0 or consensus_sink_token >= layer_values.shape[1]:
        return float("nan")
    pre_layers = [f - 1 for f in full_attention_layers if f > 1]
    if len(pre_layers) < 2:
        return float("nan")
    scores: list[float] = []
    for left, right in zip(pre_layers[:-1], pre_layers[1:]):  # noqa: B905
        intervening = [ell for ell in range(left + 1, right) if ell not in full_attention_layers]
        if not intervening:
            continue
        denom = min(
            layer_values[left - 1, consensus_sink_token],
            layer_values[right - 1, consensus_sink_token],
        )
        if denom <= 0:
            continue
        vals = [min(1.0, float(layer_values[ell - 1, consensus_sink_token] / denom)) for ell in intervening]
        scores.append(float(np.mean(vals)))
    return float(np.mean(scores)) if scores else float("nan")


def pas_isp_summary(
    layer_values: np.ndarray,
    full_attention_layers: list[int],
    sink_tokens_by_full_layer: dict[int, int],
    retention_token: int,
) -> dict[str, float | int]:
    """Compute the paper-facing PAS/ISP metrics for one trace."""
    align, hits, total = sink_spike_alignment_rate(
        sink_tokens_by_full_layer=sink_tokens_by_full_layer,
        layer_values=layer_values,
        full_attention_layers=full_attention_layers,
    )
    retention = inter_spike_retention_score(
        consensus_sink_token=retention_token,
        layer_values=layer_values,
        full_attention_layers=full_attention_layers,
    )
    return {
        "sink_spike_alignment": align,
        "alignment_hits": hits,
        "alignment_events": total,
        "inter_spike_retention": retention,
    }
