"""Known model definitions.

Planning must work without downloading weights, so the shape facts for supported
models live here. These are the published Qwen2.5 configurations; they are used
for memory and partition arithmetic only. The authoritative source at runtime is
still the model's own ``config.json`` when it is present on disk.

Adding a model here is a deliberate act: it declares that MESHWEAVE knows how to
reason about that architecture.
"""

from __future__ import annotations

from meshweave.errors import ModelLoadError
from meshweave.models.base import ModelSpec

_ARCH_QWEN2 = "Qwen2ForCausalLM"


def _qwen2(
    *,
    hidden: int,
    layers: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    intermediate: int,
    vocab: int,
    tied: bool,
    max_pos: int = 32768,
    rope_theta: float = 1000000.0,
) -> ModelSpec:
    return ModelSpec(
        architecture=_ARCH_QWEN2,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate,
        vocab_size=vocab,
        max_position_embeddings=max_pos,
        rope_theta=rope_theta,
        rms_norm_eps=1e-6,
        tie_word_embeddings=tied,
    )


QWEN2_5: dict[str, ModelSpec] = {
    "Qwen2.5-0.5B-Instruct": _qwen2(
        hidden=896,
        layers=24,
        heads=14,
        kv_heads=2,
        head_dim=64,
        intermediate=4864,
        vocab=151936,
        tied=True,
    ),
    "Qwen2.5-1.5B-Instruct": _qwen2(
        hidden=1536,
        layers=28,
        heads=12,
        kv_heads=2,
        head_dim=128,
        intermediate=8960,
        vocab=151936,
        tied=True,
    ),
    "Qwen2.5-3B-Instruct": _qwen2(
        hidden=2048,
        layers=36,
        heads=16,
        kv_heads=2,
        head_dim=128,
        intermediate=11008,
        vocab=151936,
        tied=True,
    ),
    "Qwen2.5-7B-Instruct": _qwen2(
        hidden=3584,
        layers=28,
        heads=28,
        kv_heads=4,
        head_dim=128,
        intermediate=18944,
        vocab=152064,
        tied=False,
    ),
    "Qwen2.5-14B-Instruct": _qwen2(
        hidden=5120,
        layers=48,
        heads=40,
        kv_heads=8,
        head_dim=128,
        intermediate=13824,
        vocab=152064,
        tied=False,
    ),
    "Qwen2.5-32B-Instruct": _qwen2(
        hidden=5120,
        layers=64,
        heads=40,
        kv_heads=8,
        head_dim=128,
        intermediate=27648,
        vocab=152064,
        tied=False,
    ),
    "Qwen2.5-72B-Instruct": _qwen2(
        hidden=8192,
        layers=80,
        heads=64,
        kv_heads=8,
        head_dim=128,
        intermediate=29568,
        vocab=152064,
        tied=False,
    ),
}


def resolve(model: str) -> ModelSpec:
    """Return the spec for a known model id.

    Accepts a bare name or a Hugging Face repo id (``Qwen/Qwen2.5-7B-Instruct``);
    the last path segment is what matters.
    """
    key = model.strip()
    if key in QWEN2_5:
        return QWEN2_5[key]
    short = key.rstrip("/").split("/")[-1]
    if short in QWEN2_5:
        return QWEN2_5[short]
    known = ", ".join(sorted(QWEN2_5))
    raise ModelLoadError(
        f"model {model!r} is not in the MESHWEAVE registry. Known models: {known}. "
        "Pass --model-path to a local directory containing config.json to plan "
        "against an unlisted model."
    )


def is_known(model: str) -> bool:
    try:
        resolve(model)
        return True
    except ModelLoadError:
        return False
