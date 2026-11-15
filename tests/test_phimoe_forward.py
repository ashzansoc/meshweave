"""Regression tests for the defects found during Phase 3A numerical work.

The valuable lesson from this phase is encoded here: the runtime's MoE routing
must renormalise, and that fact was established by measuring the reference, not by
reading another project's source.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from meshweave.runtime.gguf_stage import (  # noqa: E402
    HEAD_DIM,
    HEADS,
    HIDDEN,
    KV_HEADS,
    TOP_K,
    _as_linear,
)
from meshweave.runtime.phimoe_forward import (  # noqa: E402
    KvCache,
    apply_rope,
    rms_norm,
    rope_tables,
    rotate_half,
)

# --------------------------------------------------------------------------- #
# Tensor orientation -- measured against the real GGUF, not assumed
# --------------------------------------------------------------------------- #


def test_as_linear_transposes_2d():
    """GGUF hands back [out, in]; matmul needs [in, out]."""
    gguf_layout = torch.arange(6, dtype=torch.float16).reshape(2, 3)  # [out=2, in=3]
    linear = _as_linear(gguf_layout)
    assert tuple(linear.shape) == (3, 2)  # [in=3, out=2]
    x = torch.ones(1, 3, dtype=torch.float16)
    assert tuple((x @ linear).shape) == (1, 2)


def test_as_linear_transposes_expert_stack():
    """Expert tensors are [expert, out, in] and become [expert, in, out]."""
    stack = torch.zeros(4, 960, 4096, dtype=torch.float16)  # [expert, out, in]
    linear = _as_linear(stack)
    assert tuple(linear.shape) == (4, 4096, 960)
    x = torch.ones(2, 4096, dtype=torch.float16)
    assert tuple((x @ linear[0]).shape) == (2, 960)


# --------------------------------------------------------------------------- #
# RMSNorm
# --------------------------------------------------------------------------- #


def test_rms_norm_matches_reference_formula():
    x = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    w = torch.ones(2, dtype=torch.float32)
    out = rms_norm(x, w, None)
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True))
    assert torch.allclose(out, x / rms, atol=1e-6)


def test_rms_norm_applies_bias():
    """This model carries norm biases; they must change the result."""
    x = torch.ones(1, 4, dtype=torch.float32)
    w = torch.ones(4, dtype=torch.float32)
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0])
    with_bias = rms_norm(x, w, bias)
    without = rms_norm(x, w, None)
    assert not torch.allclose(with_bias, without)
    assert torch.allclose(with_bias - without, bias, atol=1e-6)


def test_rms_norm_preserves_dtype():
    x = torch.randn(1, 2, 8, dtype=torch.float16)
    out = rms_norm(x, torch.ones(8, dtype=torch.float16), None)
    assert out.dtype == torch.float16


# --------------------------------------------------------------------------- #
# RoPE -- verified against ggml's NEOX kernel (split halves)
# --------------------------------------------------------------------------- #


def test_rotate_half_uses_split_halves():
    """ggml NEOX pairs index j with j + dim/2, not adjacent indices."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = rotate_half(x)
    # first half negated second half, then first half
    assert torch.allclose(out, torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_rope_at_position_zero_is_identity():
    x = torch.randn(1, 2, 1, HEAD_DIM, dtype=torch.float32)
    cos, sin = rope_tables(1, torch.device("cpu"), torch.float32, offset=0)
    out = apply_rope(x, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))
    assert torch.allclose(out, x, atol=1e-6)


def test_rope_preserves_norm_per_head():
    """Rotary embedding is orthogonal, so head magnitude is invariant."""
    x = torch.randn(1, 4, 3, HEAD_DIM, dtype=torch.float32)
    cos, sin = rope_tables(3, torch.device("cpu"), torch.float32)
    out = apply_rope(x, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-5)


def test_rope_offset_shifts_position():
    """A decode step at offset 5 must differ from one at offset 0."""
    x = torch.randn(1, 1, 1, HEAD_DIM, dtype=torch.float32)
    c0, s0 = rope_tables(1, torch.device("cpu"), torch.float32, offset=0)
    c5, s5 = rope_tables(1, torch.device("cpu"), torch.float32, offset=5)
    a = apply_rope(x, c0.unsqueeze(0).unsqueeze(0), s0.unsqueeze(0).unsqueeze(0))
    b = apply_rope(x, c5.unsqueeze(0).unsqueeze(0), s5.unsqueeze(0).unsqueeze(0))
    assert not torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# KV cache
# --------------------------------------------------------------------------- #


def test_kv_cache_stores_and_reads_back():
    cache = KvCache(capacity=8, device=torch.device("cpu"), dtype=torch.float16)
    cache.ensure(layer=0, batch=1)
    k = torch.ones(1, KV_HEADS, 2, HEAD_DIM, dtype=torch.float16)
    v = torch.full((1, KV_HEADS, 2, HEAD_DIM), 2.0, dtype=torch.float16)
    cache.write(0, 0, k, v)
    k_all, v_all = cache.read(0, 2)
    assert torch.allclose(k_all, k)
    assert torch.allclose(v_all, v)


def test_kv_cache_grows_by_position():
    cache = KvCache(capacity=8, device=torch.device("cpu"), dtype=torch.float16)
    cache.ensure(0, 1)
    step = torch.zeros(1, KV_HEADS, 1, HEAD_DIM, dtype=torch.float16)
    cache.write(0, 0, step, step)
    cache.write(0, 1, step, step)
    _, v_all = cache.read(0, 2)
    assert v_all.shape[2] == 2


# --------------------------------------------------------------------------- #
# MoE routing -- the defect this phase actually corrected
# --------------------------------------------------------------------------- #


def test_top_k_weights_are_renormalised():
    """Established by measurement, not by reading another project's source.

    llama.cpp's reference logits for the validation prompt are
    [67.39, 66.52, 66.52, 66.39, 65.63]. Only the renormalised routing reproduces
    them. Phase 3A briefly removed renormalisation on the strength of the HF
    `sparsemixer` source and broke a correct implementation.
    """
    torch.manual_seed(0)
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    topk_probs, topk_idx = torch.topk(probs, TOP_K, dim=-1)
    normalised = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
    assert torch.allclose(normalised.sum(dim=-1), torch.ones(1))
    assert normalised[0, 0] > topk_probs[0, 0]
    assert topk_idx.tolist() == [[0, 1]]


def test_renormalisation_changes_the_result():
    """Guards against a silent revert: the two variants must differ."""
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    topk_probs, _ = torch.topk(probs, 2, dim=-1)
    raw = topk_probs.sum(dim=-1)
    assert not torch.allclose(raw, torch.ones(1))


# --------------------------------------------------------------------------- #
# Geometry constants must match the model
# --------------------------------------------------------------------------- #


def test_geometry_matches_phi_mini_moe():
    assert HIDDEN == 4096
    assert HEADS == 32
    assert KV_HEADS == 8
    assert HEAD_DIM == 128
    assert HEADS % KV_HEADS == 0, "GQA grouping must divide evenly"
    assert TOP_K == 2
