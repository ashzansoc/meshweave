"""The model adapter must produce correct memory and partition arithmetic.

These tests use the real published Qwen2.5 configurations, not invented ones, so
a regression in the math shows up as a wrong GiB figure against a known model.
"""

from __future__ import annotations

import pytest
from meshweave.errors import ModelLoadError
from meshweave.models.base import ModelSpec
from meshweave.models.qwen2_5 import Qwen25Adapter

GIB = 1024**3

QWEN_7B = {
    "hidden_size": 3584,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 18944,
    "vocab_size": 152064,
    "tie_word_embeddings": False,
}


def make(**overrides: object) -> Qwen25Adapter:
    adapter = Qwen25Adapter()
    spec_kwargs = {
        "architecture": "Qwen2ForCausalLM",
        "max_position_embeddings": 32768,
        "rope_theta": 1e6,
        "rms_norm_eps": 1e-6,
        **QWEN_7B,
    }
    spec_kwargs.update(overrides)
    adapter._spec = ModelSpec(**spec_kwargs)  # type: ignore[arg-type]
    return adapter


def test_7b_weight_size_matches_published_model() -> None:
    adapter = make()
    bf16_gib = adapter.weight_bytes_total("bf16") / GIB
    # Qwen2.5-7B is ~7.6B params; bf16 should land near 14.2 GiB.
    assert 13.5 < bf16_gib < 15.0, bf16_gib


def test_int4_is_one_quarter_of_bf16() -> None:
    adapter = make()
    ratio = adapter.weight_bytes_total("int4") / adapter.weight_bytes_total("bf16")
    assert ratio == pytest.approx(0.25)


def test_activation_is_independent_of_layer_count() -> None:
    """The fact pipeline parallelism depends on: activations are tiny."""
    adapter = make()
    one = adapter.activation_bytes(1, 1)
    assert one == 3584 * 2
    assert one == 7168


def test_kv_cache_accounts_for_gqa() -> None:
    """Only kv heads are cached, so GQA cuts the KV cache substantially."""
    adapter = make()
    # 4 kv heads, not 28 attention heads
    per_token_per_layer = adapter.kv_cache_bytes_per_token_per_layer("bf16")
    assert per_token_per_layer == 2 * 4 * 128 * 2
    mha_equivalent = 2 * 28 * 128 * 2
    assert per_token_per_layer < mha_equivalent / 4


def test_kv_cache_shrinks_with_quantization() -> None:
    adapter = make()
    fp16 = adapter.kv_cache_bytes(2048)
    int8 = adapter.kv_cache_bytes(2048, precision_bytes="int8")
    assert int8 < fp16


def test_partition_across_two_gpus_is_balanced() -> None:
    adapter = make()
    partition = adapter.partition_for_memory([22 * GIB, 22 * GIB])
    assert partition.total_layers() == 28
    counts = [p.count for p in partition.layers.values()]
    assert max(counts) - min(counts) <= 1


def test_single_gpu_partition_is_one_stage() -> None:
    adapter = make()
    partition = adapter.partition_for_memory([22 * GIB])
    assert partition.stage_count == 1
    assert partition.layers[0].start == 0
    assert partition.layers[0].end == 28


def test_impossible_model_is_rejected() -> None:
    """A 32B model cannot fit in one 24 GiB GPU. The planner must say so."""
    adapter = make(
        hidden_size=5120,
        num_hidden_layers=64,
        num_attention_heads=40,
        num_key_value_heads=8,
        intermediate_size=27648,
    )
    with pytest.raises(ModelLoadError, match="short by"):
        adapter.partition_for_memory([22 * GIB])


def test_rejection_explains_the_deficit() -> None:
    adapter = make(
        hidden_size=5120,
        num_hidden_layers=64,
        num_attention_heads=40,
        num_key_value_heads=8,
        intermediate_size=27648,
    )
    with pytest.raises(ModelLoadError) as excinfo:
        adapter.partition_for_memory([22 * GIB])
    message = str(excinfo.value)
    assert "GiB" in message
    assert "Use a smaller model" in message


def test_partition_by_layers_covers_every_layer() -> None:
    adapter = make()
    partition = adapter.partition_by_layers(4)
    covered = sum(p.count for p in partition.layers.values())
    assert covered == 28
    # contiguous, no gaps
    ranges = list(partition.layers.values())
    for previous, nxt in zip(ranges, ranges[1:], strict=False):
        assert previous.end == nxt.start


def test_tied_embeddings_free_the_head() -> None:
    tied = make(tie_word_embeddings=True)
    untied = make(tie_word_embeddings=False)
    assert tied.head_bytes("bf16") == 0
    assert untied.head_bytes("bf16") > 0


def test_unknown_precision_is_rejected() -> None:
    adapter = make()
    with pytest.raises(ModelLoadError):
        adapter.weight_bytes_total("fp3")


def test_no_stage_gets_zero_layers() -> None:
    """An empty stage still pays a pipeline boundary; it must be collapsed.

    Uneven capacities push the proportional split toward zero for the small GPU;
    the partitioner must rebalance rather than emit an empty stage.
    """
    adapter = make(
        hidden_size=5120,
        num_hidden_layers=64,
        num_attention_heads=40,
        num_key_value_heads=8,
        intermediate_size=27648,
    )
    partition = adapter.partition_for_memory([14 * GIB, 24 * GIB, 24 * GIB])
    assert all(p.count > 0 for p in partition.layers.values())
    assert partition.total_layers() == 64
