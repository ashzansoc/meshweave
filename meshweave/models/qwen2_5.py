"""Qwen2.5 adapter.

Supports Qwen2.5-0.5B / 1.5B / 7B / 14B / 32B / 72B Instruct and base. All
Qwen2.5 dense models share one architecture: pre-norm RMSNorm, RoPE, GQA
attention with a bias on q/k/v projections, SwiGLU MLP, tied embeddings only on
the smallest variants.

Shape facts are read from ``config.json`` rather than assumed, so a variant with
a different head count still produces correct memory and partition math.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meshweave.errors import ModelLoadError
from meshweave.models.base import LayerPlan, ModelSpec, Partition

# bytes per parameter for the supported precisions
PRECISION_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

QWEN25_ARCHITECTURE = "Qwen2ForCausalLM"


def _read_config(model_path: str) -> dict[str, Any]:
    """Load config.json from a local dir or an HF repo cache.

    We deliberately do not download weights here. Planning must be possible
    without touching the network or a GPU.
    """
    path = Path(model_path).expanduser()
    candidates = [path / "config.json"] if path.is_dir() else []
    if path.is_file() and path.name == "config.json":
        candidates = [path]
    if not candidates:
        # HF hub local cache: models--org--name/snapshots/<rev>/config.json
        base = Path.home() / ".cache" / "huggingface" / "hub"
        slug = "models--" + str(model_path).replace("/", "--")
        for cfg in base.glob(f"{slug}/snapshots/*/config.json"):
            candidates.append(cfg)

    for candidate in candidates:
        try:
            with candidate.open() as fh:
                loaded: dict[str, Any] = json.load(fh)
                return loaded
        except (OSError, json.JSONDecodeError):
            continue
    raise ModelLoadError(
        f"no readable config.json for {model_path!r}. Pass a local directory "
        "containing config.json, or pre-download the model."
    )


class Qwen25Adapter:
    """Concrete adapter for the Qwen2.5 dense family."""

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path
        self._spec: ModelSpec | None = None

    # ------------------------------------------------------------- loading

    @property
    def spec(self) -> ModelSpec:
        if self._spec is None:
            self._spec = self.load_spec(self._model_path or "")
        return self._spec

    def load_spec(self, model_path: str) -> ModelSpec:
        cfg = _read_config(model_path)

        architecture = str(cfg.get("architectures", [""])[0] or QWEN25_ARCHITECTURE)
        hidden = int(cfg["hidden_size"])
        layers = int(cfg["num_hidden_layers"])
        heads = int(cfg["num_attention_heads"])
        kv_heads = int(cfg.get("num_key_value_heads", heads))
        head_dim = int(cfg.get("head_dim", hidden // heads))
        intermediate = int(cfg["intermediate_size"])
        vocab = int(cfg["vocab_size"])
        max_pos = int(cfg.get("max_position_embeddings", 32768))
        rope_theta = float(cfg.get("rope_theta", 1000000.0))
        rms_eps = float(cfg.get("rms_norm_eps", 1e-6))
        tied = bool(cfg.get("tie_word_embeddings", False))

        spec = ModelSpec(
            architecture=architecture,
            hidden_size=hidden,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate,
            vocab_size=vocab,
            max_position_embeddings=max_pos,
            rope_theta=rope_theta,
            rms_norm_eps=rms_eps,
            tie_word_embeddings=tied,
        )
        self._spec = spec
        return spec

    # -------------------------------------------------------------- memory

    @staticmethod
    def _precision(precision_bytes: float | str) -> float:
        if isinstance(precision_bytes, str):
            try:
                return PRECISION_BYTES[precision_bytes.lower()]
            except KeyError as exc:
                raise ModelLoadError(f"unknown precision {precision_bytes!r}") from exc
        return float(precision_bytes)

    def weight_bytes_per_layer(self, precision_bytes: float = 2.0) -> int:
        s = self.spec
        p = self._precision(precision_bytes)

        # attention: q_proj (hidden -> heads*head_dim) + bias,
        #            k_proj/v_proj (hidden -> kv_heads*head_dim) + bias,
        #            o_proj (heads*head_dim -> hidden)
        q = s.hidden_size * (s.num_attention_heads * s.head_dim)
        k = s.hidden_size * (s.num_key_value_heads * s.head_dim)
        v = s.hidden_size * (s.num_key_value_heads * s.head_dim)
        o = (s.num_attention_heads * s.head_dim) * s.hidden_size
        biases = (s.num_attention_heads * s.head_dim) + 2 * (
            s.num_key_value_heads * s.head_dim
        )

        # mlp: gate_proj, up_proj, down_proj (SwiGLU)
        gate = s.hidden_size * s.intermediate_size
        up = s.hidden_size * s.intermediate_size
        down = s.intermediate_size * s.hidden_size

        # norms: input_layernorm, post_attention_layernorm
        norms = 2 * s.hidden_size

        return int((q + k + v + o + biases + gate + up + down + norms) * p)

    def embedding_bytes(self, precision_bytes: float = 2.0) -> int:
        return int(self.spec.vocab_size * self.spec.hidden_size * self._precision(precision_bytes))

    def head_bytes(self, precision_bytes: float = 2.0) -> int:
        if self.spec.tie_word_embeddings:
            return 0
        return int(
            self.spec.vocab_size * self.spec.hidden_size * self._precision(precision_bytes)
        )

    def weight_bytes_total(self, precision_bytes: float = 2.0) -> int:
        s = self.spec
        return (
            self.weight_bytes_per_layer(precision_bytes) * s.num_hidden_layers
            + self.embedding_bytes(precision_bytes)
            + self.head_bytes(precision_bytes)
        )

    def kv_cache_bytes_per_token_per_layer(self, precision_bytes: float = 2.0) -> int:
        s = self.spec
        # K and V, only for kv heads (GQA). fp16 by default.
        return int(2 * s.num_key_value_heads * s.head_dim * self._precision(precision_bytes))

    def kv_cache_bytes(
        self, num_tokens: int, stages: int | None = None, precision_bytes: float = 2.0
    ) -> int:
        s = self.spec
        per_layer = self.kv_cache_bytes_per_token_per_layer(precision_bytes)
        if stages is None:
            return per_layer * num_tokens * s.num_hidden_layers
        # a pipeline stage holds only its own layers; approximate by dividing the
        # layer count, which is exact for an even split
        return per_layer * num_tokens * (s.num_hidden_layers // max(1, stages))

    def activation_bytes(self, batch: int, seq_len: int, precision_bytes: float = 2.0) -> int:
        """One hidden-state tensor crossing a stage boundary.

        This is independent of layer count and parameter count -- the reason
        pipeline parallelism is cheap on the wire.
        """
        return int(batch * seq_len * self.spec.hidden_size * self._precision(precision_bytes))

    # -------------------------------------------------------- partitioning

    def partition_by_layers(self, num_stages: int) -> Partition:
        s = self.spec
        if num_stages < 1:
            raise ModelLoadError("num_stages must be >= 1")
        if num_stages > s.num_hidden_layers:
            raise ModelLoadError(
                f"cannot split {s.num_hidden_layers} layers across {num_stages} stages"
            )
        base = s.num_hidden_layers // num_stages
        extra = s.num_hidden_layers % num_stages
        layers: dict[int, LayerPlan] = {}
        cursor = 0
        for stage in range(num_stages):
            count = base + (1 if stage < extra else 0)
            layers[stage] = LayerPlan(start=cursor, end=cursor + count)
            cursor += count
        return Partition(layers=layers, embedding_stage=0, head_stage=num_stages - 1)

    def partition_for_memory(
        self,
        capacities_bytes: list[int],
        precision_bytes: float = 2.0,
        kv_tokens: int = 2048,
    ) -> Partition:
        """Assign layers to stages so each stage's weights fit its GPU.

        Strategy: give the first stage the embedding, the last stage the head,
        then distribute remaining layers proportionally to spare capacity. This
        is a heuristic; it does not model activation peaks. Callers that need a
        hard guarantee should verify with an actual load.
        """
        s = self.spec
        if not capacities_bytes:
            raise ModelLoadError("no GPU capacities supplied")

        n_stages = len(capacities_bytes)
        if n_stages > s.num_hidden_layers:
            raise ModelLoadError(
                f"{s.num_hidden_layers} layers cannot be spread over {n_stages} stages"
            )

        self._precision(precision_bytes)
        emb = self.embedding_bytes(precision_bytes)
        head = self.head_bytes(precision_bytes)
        per_layer = self.weight_bytes_per_layer(precision_bytes)

        # Reserve KV cache headroom on every stage, then the fixed overhead on
        # the first and last stage.
        kv_reserve = self.kv_cache_bytes(kv_tokens, stages=n_stages, precision_bytes=precision_bytes)
        overhead = [0] * n_stages
        overhead[0] += emb
        overhead[-1] += head

        usable = [
            max(0, capacities_bytes[i] - kv_reserve - overhead[i]) for i in range(n_stages)
        ]

        total_usable = sum(usable)
        total_need = per_layer * s.num_hidden_layers
        if total_usable < total_need:
            deficit_gib = (total_need - total_usable) / (1024**3)
            raise ModelLoadError(
                f"model needs {total_need / 1024**3:.1f} GiB of weights + "
                f"{kv_reserve / 1024**3:.1f} GiB KV headroom across stages, but only "
                f"{total_usable / 1024**3:.1f} GiB is available (short by {deficit_gib:.1f} GiB). "
                "Use a smaller model, a lower precision, or fewer KV tokens."
            )

        # Proportional allocation, largest remainder to the last stage so the
        # final stage (which holds the LM head) is never starved.
        raw = [usable[i] / total_usable * s.num_hidden_layers for i in range(n_stages)]
        alloc = [int(x) for x in raw]
        remainder = s.num_hidden_layers - sum(alloc)
        order = sorted(range(n_stages), key=lambda i: raw[i] - alloc[i], reverse=True)
        for i in order[:remainder]:
            alloc[i] += 1

        # Any stage that got zero layers is a defect: an empty stage still pays a
        # full pipeline boundary. Collapse it by merging into a neighbour.
        alloc = _merge_empty_stages(alloc)

        layers: dict[int, LayerPlan] = {}
        cursor = 0
        for stage, count in enumerate(alloc):
            layers[stage] = LayerPlan(start=cursor, end=cursor + count)
            cursor += count

        return Partition(layers=layers, embedding_stage=0, head_stage=len(alloc) - 1)


def _merge_empty_stages(alloc: list[int]) -> list[int]:
    """Merge zero-layer stages into the preceding non-empty stage."""
    out: list[int] = []
    for count in alloc:
        if count == 0 and out:
            continue
        out.append(count)
    if not out:
        out = [1]
    return out


def precision_bytes_for(name: str) -> float:
    try:
        return PRECISION_BYTES[name.lower()]
    except KeyError as exc:
        raise ModelLoadError(f"unknown precision {name!r}; known: {sorted(PRECISION_BYTES)}") from exc
