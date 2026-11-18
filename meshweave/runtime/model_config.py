"""Model geometry read from ``config.json``, with no torch dependency.

The coordinator needs to size a deployment before anything is downloaded, so the
arithmetic lives here and both the planner and the runtime use it. Every field is
read from the model's own config; nothing is assumed from the parameter count.

Only architectures whose attention and MLP maths this runtime actually
implements are accepted. A model it does not know is refused by name rather than
executed wrongly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from meshweave.errors import ModelLoadError

SUPPORTED_ARCHITECTURES = {
    # Qwen2 and Qwen2.5 dense variants share one architecture definition.
    "Qwen2ForCausalLM": "qwen2",
}

DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
}


@dataclass(frozen=True)
class DenseConfig:
    """Geometry of a dense decoder-only causal LM."""

    architecture: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    vocab: int
    eps: float
    rope_theta: float
    tie_embeddings: bool
    attn_bias: bool
    dtype: str

    # ------------------------------------------------------------- parsing

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> DenseConfig:
        architectures = raw.get("architectures") or []
        arch = architectures[0] if architectures else ""
        if arch not in SUPPORTED_ARCHITECTURES:
            supported = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
            raise ModelLoadError(
                f"architecture {arch or 'unknown'!r} is not implemented by this "
                f"runtime. Supported: {supported}. Refusing rather than running a "
                "model whose attention or MLP maths has not been verified."
            )

        scaling = raw.get("rope_scaling")
        if scaling:
            raise ModelLoadError(
                f"rope_scaling={scaling!r} is not implemented. Positions would be "
                "computed wrongly, so this model is refused rather than approximated."
            )

        # transformers v5 moved these into ``rope_parameters``; older checkpoints
        # carry ``rope_theta`` at the top level. Reading only the old location
        # silently fell back to theta=10000 on a 1e6 model, which changes every
        # rotary angle and produces wrong attention while looking plausible.
        rope_params = raw.get("rope_parameters") or {}
        rope_type = rope_params.get("rope_type")
        if rope_type not in (None, "default"):
            raise ModelLoadError(f"rope_type={rope_type!r} is not implemented")
        rope_theta = float(
            raw.get("rope_theta")
            or rope_params.get("rope_theta")
            or 10000.0
        )
        if float(raw.get("partial_rotary_factor", 1.0)) != 1.0:
            raise ModelLoadError("partial_rotary_factor is not implemented")

        hidden = int(raw["hidden_size"])
        heads = int(raw["num_attention_heads"])
        head_dim = int(raw.get("head_dim") or hidden // heads)
        if head_dim * heads != int(raw.get("head_dim") or hidden):
            # Some configs carry head_dim explicitly while hidden != heads*head_dim;
            # that is legal, so only sanity-check the q projection width.
            pass

        return cls(
            architecture=arch,
            hidden=hidden,
            layers=int(raw["num_hidden_layers"]),
            heads=heads,
            kv_heads=int(raw.get("num_key_value_heads") or heads),
            head_dim=head_dim,
            ffn=int(raw["intermediate_size"]),
            vocab=int(raw["vocab_size"]),
            eps=float(raw.get("rms_norm_eps", 1e-6)),
            rope_theta=rope_theta,
            tie_embeddings=bool(raw.get("tie_word_embeddings", False)),
            # Qwen2 always carries q/k/v biases, and the published config.json
            # omits the key entirely -- Qwen/Qwen2.5-0.5B-Instruct's config has
            # no attention_bias field, while its checkpoint has q_proj.bias,
            # k_proj.bias and v_proj.bias. Defaulting this to False computed
            # attention without them and produced fluent-looking gibberish.
            attn_bias=bool(raw.get("attention_bias", True)),
            dtype=str(raw.get("torch_dtype") or raw.get("dtype") or "bfloat16"),
        )

    @classmethod
    def from_file(cls, path: Path) -> DenseConfig:
        try:
            return cls.from_json(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise ModelLoadError(f"cannot read model config {path}: {exc}") from exc

    @classmethod
    def from_json_text(cls, text: str) -> DenseConfig:
        try:
            return cls.from_json(json.loads(text))
        except (json.JSONDecodeError, KeyError) as exc:
            raise ModelLoadError(f"invalid config.json: {exc}") from exc

    # --------------------------------------------------------------- sizes

    def bytes_per_layer(self, dtype_bytes: int) -> int:
        """Weights of one decoder layer, including q/k/v biases when present."""
        q = self.hidden * self.heads * self.head_dim
        kv = self.hidden * self.kv_heads * self.head_dim
        o = self.heads * self.head_dim * self.hidden
        attn = q + 2 * kv + o
        if self.attn_bias:
            attn += self.heads * self.head_dim + 2 * self.kv_heads * self.head_dim
        mlp = 3 * self.hidden * self.ffn  # gate, up, down
        norms = 2 * self.hidden
        return (attn + mlp + norms) * dtype_bytes

    def embedding_bytes(self, dtype_bytes: int) -> int:
        return self.vocab * self.hidden * dtype_bytes

    def head_bytes(self, dtype_bytes: int) -> int:
        # When embeddings are tied, the head reuses the embedding matrix: it must
        # be resident, so it is still counted.
        return self.vocab * self.hidden * dtype_bytes

    def stage_bytes(
        self,
        start: int,
        end: int,
        *,
        owns_embed: bool,
        owns_head: bool,
        dtype_bytes: int,
    ) -> int:
        total = (end - start) * self.bytes_per_layer(dtype_bytes)
        if owns_embed:
            total += self.embedding_bytes(dtype_bytes)
        if owns_head:
            total += self.head_bytes(dtype_bytes)
        return total

    def checkpoint_bytes(self, dtype_bytes: int) -> int:
        """Memory needed to *download* the model: every layer, one full copy.

        Each node holds the whole checkpoint on disk even though it keeps only
        its own layers in VRAM, because the safetensors shards are not split
        finely enough to fetch one layer range by itself.
        """
        return (
            self.layers * self.bytes_per_layer(dtype_bytes)
            + self.embedding_bytes(dtype_bytes)
            + (0 if self.tie_embeddings else self.head_bytes(dtype_bytes))
            + 2 * self.hidden
        )

    def kv_bytes_per_token(self, dtype_bytes: int) -> int:
        """KV for the whole model: one layer's worth per layer."""
        return 2 * self.layers * self.kv_heads * self.head_dim * dtype_bytes

    def kv_bytes_per_stage_token(self, start: int, end: int, dtype_bytes: int) -> int:
        """KV for the layers one stage owns.

        A stage caches only its own layers, so reserving the whole model's KV on
        every node over-reserved by the node count and refused deployments that
        would have fitted.
        """
        return 2 * max(0, end - start) * self.kv_heads * self.head_dim * dtype_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "hidden": self.hidden,
            "layers": self.layers,
            "heads": self.heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "ffn": self.ffn,
            "vocab": self.vocab,
            "tie_embeddings": self.tie_embeddings,
            "attn_bias": self.attn_bias,
            "dtype": self.dtype,
        }


def dtype_bytes_for(name: str) -> int:
    try:
        return DTYPE_BYTES[name]
    except KeyError as exc:
        raise ModelLoadError(f"unsupported dtype {name!r}") from exc


def partition_layers(config: DenseConfig, nodes: int) -> list[tuple[int, int]]:
    """Split layers across stages as evenly as possible, front stages first.

    24 layers over 2 nodes gives (0,12) and (12,24). Over 5 nodes it gives
    (0,5), (5,10), (10,15), (15,20), (20,24) -- every layer exactly once.
    """
    if nodes < 1:
        raise ValueError("nodes must be >= 1")
    if nodes > config.layers:
        raise ValueError(
            f"{config.layers} layers cannot be spread over {nodes} nodes without "
            "leaving a stage with no work"
        )
    base, extra = divmod(config.layers, nodes)
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(nodes):
        size = base + (1 if index < extra else 0)
        bounds.append((start, start + size))
        start += size
    return bounds
