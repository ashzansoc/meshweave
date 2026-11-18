"""Placement chooses a split by predicted latency, not by dividing evenly.

The estimator is checked against the one measurement we have: the deployed
12/12 split ran at 310.8 ms/token on the real link. The prediction should land
near it, and the ranking must be honest about being nearly flat between similar
machines.
"""

from __future__ import annotations

import pytest

from meshweave.planner.placement import (
    DEFAULT_MS_PER_LAYER,
    LinkCost,
    NodeSpeed,
    choose_split,
    estimate,
    layer_splits,
    speeds_from_records,
)
from meshweave.runtime.model_config import DenseConfig

CONFIG_JSON = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "intermediate_size": 4864,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
    "attention_bias": True,
    "torch_dtype": "bfloat16",
}


@pytest.fixture()
def config() -> DenseConfig:
    return DenseConfig.from_json(CONFIG_JSON)


def _measured_speeds() -> list[NodeSpeed]:
    """The two nodes' measured single-node rates from benchmarks/vdc/results.json."""
    return [
        NodeSpeed("gpu-node-1", 38.8 / 24, measured=True),
        NodeSpeed("brisk-thistle-9ae2", 41.0 / 24, measured=True),
    ]


def _links() -> dict[tuple[str, str], LinkCost]:
    a, b = "gpu-node-1", "brisk-thistle-9ae2"
    return {
        (a, b): LinkCost(a, b, rtt_ms=226.0, mbps=8.0),
        (b, a): LinkCost(b, a, rtt_ms=226.0, mbps=8.0),
    }


# --------------------------------------------------------------------------- #
# candidate generation
# --------------------------------------------------------------------------- #


def test_every_layer_is_assigned_exactly_once(config: DenseConfig) -> None:
    for sizes in layer_splits(24, 3, limit=1000):
        assert sum(sizes) == 24
        assert all(size > 0 for size in sizes)


def test_the_candidate_count_matches_the_composition_count(config: DenseConfig) -> None:
    # 24 into 2 positive parts is 23; the even split is already in that set.
    assert len(list(layer_splits(24, 2, limit=1000))) == 23
    assert len(list(layer_splits(24, 1, limit=10))) == 1


def test_the_search_can_be_truncated_without_losing_the_even_split(
    config: DenseConfig,
) -> None:
    _, candidates, truncated = choose_split(
        config, _measured_speeds(), _links(), limit=5
    )
    assert truncated
    assert any(c.bounds == [(0, 12), (12, 24)] for c in candidates)


# --------------------------------------------------------------------------- #
# the estimate
# --------------------------------------------------------------------------- #


def test_the_estimate_lands_near_the_measured_run(config: DenseConfig) -> None:
    """Measured: 12/12 across these two nodes ran at 310.8 ms/token."""
    candidate = estimate([(0, 12), (12, 24)], _measured_speeds(), _links(), 1792)
    assert 0.6 * 310.8 < candidate.predicted_ms_per_token < 1.6 * 310.8


def test_network_dominates_when_the_link_is_slow(config: DenseConfig) -> None:
    fast_link = estimate([(0, 12), (12, 24)], _measured_speeds(), _links(), 1792)
    slow = {
        (a, b): LinkCost(a, b, rtt_ms=5.0, mbps=200.0)
        for a, b in (("gpu-node-1", "brisk-thistle-9ae2"), ("brisk-thistle-9ae2", "gpu-node-1"))
    }
    fast = estimate([(0, 12), (12, 24)], _measured_speeds(), slow, 1792)
    assert fast_link.network_ms > fast.network_ms * 10


def test_a_missing_link_is_an_assumption_not_a_silent_zero(config: DenseConfig) -> None:
    candidate = estimate([(0, 12), (12, 24)], _measured_speeds(), {}, 1792)
    assert any("unmeasured" in note for note in candidate.assumptions)


def test_unmeasured_speed_is_declared(config: DenseConfig) -> None:
    speeds = [NodeSpeed("a", DEFAULT_MS_PER_LAYER), NodeSpeed("b", DEFAULT_MS_PER_LAYER)]
    candidate = estimate([(0, 12), (12, 24)], speeds, _links(), 1792)
    assert sum(1 for note in candidate.assumptions if "assuming" in note) == 2


def test_a_single_stage_has_no_network_cost(config: DenseConfig) -> None:
    candidate = estimate([(0, 24)], [NodeSpeed("a", 38.8 / 24, True)], {}, 1792)
    assert candidate.network_ms == 0.0
    assert candidate.predicted_ms_per_token == pytest.approx(38.8, abs=0.5)


# --------------------------------------------------------------------------- #
# choosing
# --------------------------------------------------------------------------- #


def test_two_similar_machines_keep_the_even_split(config: DenseConfig) -> None:
    """A 1 ms predicted gain on a 275 ms estimate is inside the model's error."""
    best, _, _ = choose_split(config, _measured_speeds(), _links(), hidden_bytes_per_token=1792)
    assert best.bounds == [(0, 12), (12, 24)]


def test_a_much_faster_node_gets_more_layers(config: DenseConfig) -> None:
    speeds = [
        NodeSpeed("fast", 10.0 / 24, measured=True),
        NodeSpeed("slow", 60.0 / 24, measured=True),
    ]
    best, _, _ = choose_split(config, speeds, _links(), hidden_bytes_per_token=1792)
    fast_layers = best.bounds[0][1]
    slow_layers = best.bounds[1][1] - best.bounds[1][0]
    assert fast_layers > slow_layers


def test_the_choice_is_reported_with_its_arithmetic(config: DenseConfig) -> None:
    best, alternatives, _ = choose_split(
        config, _measured_speeds(), _links(), hidden_bytes_per_token=1792
    )
    assert best.compute_ms > 0
    assert best.network_ms > 0
    assert len(alternatives) >= 2
    assert alternatives == sorted(alternatives, key=lambda c: c.predicted_ms_per_token)


def test_speeds_from_records_falls_back_and_says_so() -> None:
    from tests.test_deployments import node

    speeds = speeds_from_records([node("a"), node("b")], measured={"a": 38.8}, layers=24)
    by_id = {s.node_id: s for s in speeds}
    assert by_id["a"].measured and by_id["a"].ms_per_layer == pytest.approx(38.8 / 24)
    assert not by_id["b"].measured
    assert by_id["b"].ms_per_layer == DEFAULT_MS_PER_LAYER