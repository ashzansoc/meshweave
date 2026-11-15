"""Model adapter interface.

MESHWEAVE supports one architecture first (Qwen2.5) and adds adapters behind
this interface. The adapter is the only place that knows where a transformer
block boundary is, how large an activation is, or how the KV cache is laid out.

Anything that wants to reason about layers, memory or activation sizes asks the
adapter. Nothing hardcodes Qwen's shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class LayerPlan:
    """A contiguous half-open range of transformer layers, [start, end)."""

    start: int
    end: int

    @property
    def count(self) -> int:
        return self.end - self.start

    def __contains__(self, index: int) -> bool:
        return self.start <= index < self.end


@dataclass(frozen=True)
class ModelSpec:
    """Shape facts that determine memory and activation sizes."""

    architecture: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool

    @property
    def is_gqa(self) -> bool:
        return self.num_key_value_heads != self.num_attention_heads

    @property
    def kv_heads_per_group(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


@dataclass
class Partition:
    """How a model is split across nodes.

    ``layers`` maps a stage index to the layer range that stage owns. Stage 0
    also owns the embedding; the final stage owns the final norm and LM head.
    """

    layers: dict[int, LayerPlan]
    embedding_stage: int = 0
    head_stage: int = 0
    node_for_stage: dict[int, str] = field(default_factory=dict)

    @property
    def stage_count(self) -> int:
        return len(self.layers)

    def ordered_stages(self) -> list[int]:
        return sorted(self.layers)

    def total_layers(self) -> int:
        return sum(p.count for p in self.layers.values())

    def node_for(self, stage: int) -> str | None:
        return self.node_for_stage.get(stage)


@runtime_checkable
class ModelAdapter(Protocol):
    """Everything MESHWEAVE needs to know about a model family."""

    spec: ModelSpec

    def load_spec(self, model_path: str) -> ModelSpec:
        """Read the config and return the shape facts.

        Reads a local directory or a HF repo id. Raises ``ModelLoadError`` when
        the config is absent or unparseable.
        """
        ...

    # ------------------------------------------------------------ memory

    def weight_bytes_per_layer(self, precision_bytes: float) -> int:
        """Bytes of weight storage for one transformer layer."""
        ...

    def weight_bytes_total(self, precision_bytes: float) -> int:
        """Bytes of weight storage for the whole model, excluding embeddings."""
        ...

    def embedding_bytes(self, precision_bytes: float) -> int:
        """Embedding table bytes (tied or not)."""
        ...

    def head_bytes(self, precision_bytes: float) -> int:
        """LM head bytes (zero when tied to the embedding)."""
        ...

    def kv_cache_bytes_per_token_per_layer(self, precision_bytes: float = 2.0) -> int:
        """KV bytes for one token in one layer.

        Accounts for GQA: only ``num_key_value_heads`` heads are cached.
        """
        ...

    def kv_cache_bytes(
        self, num_tokens: int, stages: int | None = None, precision_bytes: float = 2.0
    ) -> int:
        """KV bytes for ``num_tokens`` across all layers, or one stage's layers."""
        ...

    def activation_bytes(
        self, batch: int, seq_len: int, precision_bytes: float = 2.0
    ) -> int:
        """Bytes of one hidden-state activation tensor.

        This is what crosses a pipeline boundary. It does not scale with model
        size, which is why pipeline parallelism is bandwidth-cheap.
        """
        ...

    # -------------------------------------------------------- partitioning

    def partition_by_layers(self, num_stages: int) -> Partition:
        """Split layers evenly. Callers rebalance for heterogeneity."""
        ...

    def partition_for_memory(
        self,
        capacities_bytes: list[int],
        precision_bytes: float = 2.0,
        kv_tokens: int = 2048,
    ) -> Partition:
        """Assign layers to stages so each stage's weights fit its GPU.

        Greedy from the end: give each stage as many layers as fit, leaving the
        remainder to earlier stages. Raises if the model cannot be split across
        the given capacities at all.
        """
        ...
