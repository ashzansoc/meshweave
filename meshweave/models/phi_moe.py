"""Phi-mini-MoE adapter.

Verified against the published configuration (microsoft/Phi-mini-MoE-instruct),
not from memory:

    architecture   PhiMoEForCausalLM
    total params   7,647,632,704  (7.65B) -- ALL experts
    active params  ~2.23B per token
    layers         32, every layer is MoE (no dense prefix)
    experts        16 per layer, top-2 routing, no shared expert
    hidden         4096
    expert FFN     960 (SwiGLU, three matrices w1/w2/w3, no bias)
    attention      GQA, 32 heads / 8 KV heads, head_dim 128, qkv bias
    embedding      untied, and lm_head carries a bias
    context        4096 max

The critical fact for partitioning: **weight memory scales with TOTAL parameters,
not active parameters.** At bf16 the weights are 14.24 GiB even though each token
only touches ~2.23B of them. An experiment that sizes memory from the active count
will be wrong by 3.4x.

Memory budget at bf16, by component:

    experts      6,039,797,760 params  = 11.25 GiB
    attention    1,342,504,960 params  =  2.50 GiB
    embedding      131,334,144 params  =  0.24 GiB
    lm_head        131,334,144 params  =  0.24 GiB
    routers+norms    2,629,632 params  =  0.00 GiB
    --------------------------------------------
    total        7,647,632,704 params  = 14.24 GiB

KV cache: 2 x 8 KV heads x 128 head_dim x 2 bytes = 4 KiB per token per layer,
128 KiB per token across all 32 layers.

This adapter also models the GGUF quantisation actually used on the hosts (Q6_K),
because that is what fits in the available disk and VRAM. The bf16 figures are
kept for reference and for the report's memory accounting.
"""

from __future__ import annotations

from dataclasses import dataclass

from meshweave.errors import ModelLoadError
from meshweave.models.base import LayerPlan, ModelSpec, Partition

# Verified from config.json and the safetensors index.
PHI_MINI_MOE_PARAMS = 7_647_632_704
PHI_MINI_MOE_EXPERTS = 16
PHI_MINI_MOE_ACTIVE_EXPERTS = 2
PHI_MINI_MOE_LAYERS = 32
PHI_MINI_MOE_HIDDEN = 4096
PHI_MINI_MOE_EXPERT_FFN = 960
PHI_MINI_MOE_KV_HEADS = 8
PHI_MINI_MOE_HEADS = 32
PHI_MINI_MOE_HEAD_DIM = 128
PHI_MINI_MOE_VOCAB = 32064
PHI_MINI_MOE_MAX_POS = 4096
PHI_MINI_MOE_INTERMEDIATE = 960  # attention-only intermediate; experts use this too

# Measured file sizes for the published GGUF quants (gabriellarson/Phi-mini-MoE-instruct-GGUF).
# Bytes per parameter derived from those file sizes.
GGUF_BYTES_PER_PARAM = {
    "F16": 2.0,
    "Q8_0": 7.576 * 1024**3 / PHI_MINI_MOE_PARAMS,
    "Q6_K": 6.306 * 1024**3 / PHI_MINI_MOE_PARAMS,
    "Q4_K_M": 4.650 * 1024**3 / PHI_MINI_MOE_PARAMS,
    "Q4_0": 4.063 * 1024**3 / PHI_MINI_MOE_PARAMS,
    "Q3_K_M": 3.694 * 1024**3 / PHI_MINI_MOE_PARAMS,
    "Q2_K": 2.928 * 1024**3 / PHI_MINI_MOE_PARAMS,
}

# Fraction of the model that is expert FFN weights. Used to size an expert-parallel
# placement, where only experts move between nodes.
EXPERT_FRACTION = 6_039_797_760 / PHI_MINI_MOE_PARAMS


@dataclass(frozen=True)
class GgufQuant:
    name: str
    bytes_per_param: float

    @property
    def total_bytes(self) -> int:
        return int(PHI_MINI_MOE_PARAMS * self.bytes_per_param)


def available_quants() -> dict[str, float]:
    """GGUF quant names with their measured GiB size."""
    return {k: v * PHI_MINI_MOE_PARAMS / 1024**3 for k, v in GGUF_BYTES_PER_PARAM.items()}


class PhiMoEAdapter:
    """Memory and partitioning model for Phi-mini-MoE."""

    architecture = "PhiMoEForCausalLM"
    total_parameters = PHI_MINI_MOE_PARAMS
    num_experts = PHI_MINI_MOE_EXPERTS
    active_experts = PHI_MINI_MOE_ACTIVE_EXPERTS
    num_layers = PHI_MINI_MOE_LAYERS
    hidden_size = PHI_MINI_MOE_HIDDEN
    head_dim = PHI_MINI_MOE_HEAD_DIM
    num_kv_heads = PHI_MINI_MOE_KV_HEADS
    vocab_size = PHI_MINI_MOE_VOCAB

    # ------------------------------------------------------------ memory

    def weight_bytes(self, quant: str = "Q6_K") -> int:
        """Total weight bytes for a GGUF quantisation."""
        try:
            bpp = GGUF_BYTES_PER_PARAM[quant]
        except KeyError as exc:
            raise ModelLoadError(
                f"unknown quant {quant!r}; known: {sorted(GGUF_BYTES_PER_PARAM)}"
            ) from exc
        return int(PHI_MINI_MOE_PARAMS * bpp)

    def weight_bytes_bf16(self) -> int:
        return int(PHI_MINI_MOE_PARAMS * 2.0)

    def kv_bytes_per_token(self) -> int:
        """KV cache for one token across all 32 layers."""
        return 2 * self.num_kv_heads * self.head_dim * 2 * self.num_layers

    def kv_bytes(self, num_tokens: int) -> int:
        return self.kv_bytes_per_token() * num_tokens

    def per_layer_bytes(self, quant: str = "Q6_K") -> int:
        """Average bytes for one layer, used to size a layer partition."""
        return self.weight_bytes(quant) // self.num_layers

    def expert_bytes_per_layer(self, quant: str = "Q6_K") -> int:
        """Bytes of expert FFN weights in one layer."""
        return int(self.weight_bytes(quant) * EXPERT_FRACTION) // self.num_layers

    # ------------------------------------------------------- partitioning

    def plan_partition(
        self,
        capacities_bytes: list[int],
        quant: str = "Q6_K",
        kv_tokens: int = 2048,
    ) -> Partition:
        """Split layers across nodes so each node's share fits its VRAM.

        Reserves KV cache headroom on every node, plus the embedding on the first
        and the lm_head on the last. Layer counts are proportional to spare
        capacity, not equal, because the GPUs differ.
        """
        if not capacities_bytes:
            raise ModelLoadError("no GPU capacities supplied")

        n = len(capacities_bytes)
        if n > self.num_layers:
            raise ModelLoadError(
                f"{self.num_layers} layers cannot be spread over {n} nodes"
            )

        embedding_bytes = int(131_334_144 * (GGUF_BYTES_PER_PARAM[quant]))
        head_bytes = int(131_334_144 * GGUF_BYTES_PER_PARAM[quant])

        kv_reserve = self.kv_bytes(kv_tokens) // max(1, n)

        overhead = [0] * n
        overhead[0] += embedding_bytes
        overhead[-1] += head_bytes

        usable = [
            max(0, capacities_bytes[i] - kv_reserve - overhead[i]) for i in range(n)
        ]
        total_usable = sum(usable)
        per_layer = self.per_layer_bytes(quant)
        total_layers_bytes = per_layer * self.num_layers

        if total_usable < total_layers_bytes:
            deficit = (total_layers_bytes - total_usable) / 1024**3
            raise ModelLoadError(
                f"{quant} weights need {total_layers_bytes / 1024**3:.2f} GiB plus "
                f"{kv_reserve / 1024**3:.2f} GiB KV headroom per node, but only "
                f"{total_usable / 1024**3:.2f} GiB is available (short {deficit:.2f} GiB). "
                "Use a smaller quantisation or fewer KV tokens."
            )

        raw = [usable[i] / total_usable * self.num_layers for i in range(n)]
        alloc = [int(x) for x in raw]
        remainder = self.num_layers - sum(alloc)
        order = sorted(range(n), key=lambda i: raw[i] - alloc[i], reverse=True)
        for i in order[:remainder]:
            alloc[i] += 1

        # Never emit an empty stage: it still pays a full network boundary.
        alloc = _merge_empty(alloc)

        layers: dict[int, LayerPlan] = {}
        cursor = 0
        for stage, count in enumerate(alloc):
            layers[stage] = LayerPlan(start=cursor, end=cursor + count)
            cursor += count
        return Partition(layers=layers, embedding_stage=0, head_stage=len(alloc) - 1)

    def plan_expert_split(self, expert_counts: list[int]) -> dict[int, list[int]]:
        """Assign experts to nodes for expert parallelism.

        Returns node index -> list of expert ids. Only expert FFN weights move
        between nodes in this scheme; attention and norms are replicated. The
        split is by count, not by bytes, because all experts are the same size.
        """
        total = sum(expert_counts)
        if total != self.num_experts:
            raise ModelLoadError(
                f"expert counts sum to {total}, expected {self.num_experts}"
            )
        out: dict[int, list[int]] = {}
        cursor = 0
        for node_index, count in enumerate(expert_counts):
            out[node_index] = list(range(cursor, cursor + count))
            cursor += count
        return out

    def attention_replication_bytes(self, quant: str = "Q6_K") -> int:
        """Bytes of non-expert weights every node must hold in expert parallel."""
        non_expert = 1.0 - EXPERT_FRACTION
        return int(self.weight_bytes(quant) * non_expert)


def _merge_empty(alloc: list[int]) -> list[int]:
    out: list[int] = []
    for count in alloc:
        if count == 0 and out:
            continue
        out.append(count)
    return out or [1]


def spec_for_report() -> ModelSpec:
    """A ModelSpec describing Phi-mini-MoE, for the planner and reports."""
    return ModelSpec(
        architecture="PhiMoEForCausalLM",
        hidden_size=PHI_MINI_MOE_HIDDEN,
        num_hidden_layers=PHI_MINI_MOE_LAYERS,
        num_attention_heads=PHI_MINI_MOE_HEADS,
        num_key_value_heads=PHI_MINI_MOE_KV_HEADS,
        head_dim=PHI_MINI_MOE_HEAD_DIM,
        intermediate_size=PHI_MINI_MOE_EXPERT_FFN,
        vocab_size=PHI_MINI_MOE_VOCAB,
        max_position_embeddings=PHI_MINI_MOE_MAX_POS,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
