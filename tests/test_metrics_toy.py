from __future__ import annotations

import numpy as np

from massive_activations_hla.analysis.pas_isp_metrics import (
    dominant_sink_token_from_attention,
    inter_spike_retention_score,
    sink_spike_alignment_rate,
)


def test_inter_spike_retention_toy() -> None:
    # layers x tokens. Full attention layers are 3 and 6, so PAS are 2 and 5.
    layer_values = np.array(
        [
            [1.0],
            [10.0],
            [4.0],
            [5.0],
            [8.0],
            [3.0],
        ]
    )
    score = inter_spike_retention_score(
        consensus_sink_token=0,
        layer_values=layer_values,
        full_attention_layers=[3, 6],
    )
    # Intervening layers between PAS 2 and PAS 5 are 3 and 4, but layer 3 is
    # full attention and excluded. Retention = min(1, layer4 / min(10, 8)).
    assert np.isclose(score, 5.0 / 8.0)


def test_sink_spike_alignment_toy() -> None:
    layer_values = np.array(
        [
            [1.0, 2.0],
            [10.0, 3.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [8.0, 7.0],
            [3.0, 8.0],
        ]
    )
    score, hits, total = sink_spike_alignment_rate(
        sink_tokens_by_full_layer={3: 0, 6: 1},
        layer_values=layer_values,
        full_attention_layers=[3, 6],
    )
    assert hits == 2
    assert total == 2
    assert np.isclose(score, 1.0)


def test_dominant_sink_token_from_attention() -> None:
    attn = np.zeros((2, 4, 4), dtype=float)
    attn[:, 1:, 0] = 0.5
    attn[:, 2:, 1] = 0.2
    assert dominant_sink_token_from_attention(attn) == 0
