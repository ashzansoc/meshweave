"""Load a layer range of Phi-mini-MoE from a local GGUF into PyTorch.

This is the piece that makes local weight ownership possible. Each host reads its
own copy of the GGUF and materialises only the tensors it owns, so no weights
cross the network.

Why not use ``transformers``: the published checkpoint is bf16 (14.24 GiB), which
does not fit in the 16 GB disk available on these hosts, and PhiMoE needs
``trust_remote_code``. The GGUF Q6_K file (6.31 GiB) is already on every host and
is self-describing, so it is the practical source of truth.

GGUF tensor layout, verified against the real file:

    token_embd.weight            [vocab, hidden]
    blk.N.attn_norm.{weight,bias}       RMSNorm with a bias
    blk.N.attn_q.weight  [hidden, hidden]      + bias
    blk.N.attn_k.weight  [hidden, kv_dim]      + bias      (GQA)
    blk.N.attn_v.weight  [hidden, kv_dim]      + bias
    blk.N.attn_output.weight [hidden, hidden]  + bias
    blk.N.ffn_norm.{weight,bias}        RMSNorm with a bias
    blk.N.ffn_gate_inp.weight   [hidden, n_experts]   router, F32
    blk.N.ffn_gate_exps.weight  [hidden, ffn, n_experts]   w1
    blk.N.ffn_up_exps.weight    [hidden, ffn, n_experts]   w3
    blk.N.ffn_down_exps.weight  [ffn, hidden, n_experts]   w2
    output_norm.{weight,bias}
    output.weight [vocab, hidden] + output.bias

Note the expert tensors are stacked across all experts on the last axis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from meshweave.errors import ModelLoadError

# Phi-mini-MoE geometry, verified from config.json and the GGUF metadata.
HIDDEN = 4096
LAYERS = 32
HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
KV_DIM = KV_HEADS * HEAD_DIM  # 1024
EXPERTS = 16
TOP_K = 2
FFN = 960
ROPE_THETA = 10000.0
RMS_EPS = 1e-5


@dataclass
class StageWeights:
    """The tensors one stage owns, already on a device.

    ``embedding`` is present only on the first stage, ``lm_head`` only on the last.
    ``layers`` maps absolute layer index -> that layer's tensors.
    """

    layer_range: tuple[int, int]
    layers: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    embedding: torch.Tensor | None = None
    lm_head_weight: torch.Tensor | None = None
    lm_head_bias: torch.Tensor | None = None
    final_norm_weight: torch.Tensor | None = None
    final_norm_bias: torch.Tensor | None = None

    @property
    def owns_embedding(self) -> bool:
        return self.embedding is not None

    @property
    def owns_head(self) -> bool:
        return self.lm_head_weight is not None

    def summary(self) -> dict[str, Any]:
        n_params = sum(
            t.numel() for layer in self.layers.values() for t in layer.values()
        )
        if self.embedding is not None:
            n_params += self.embedding.numel()
        if self.lm_head_weight is not None:
            n_params += self.lm_head_weight.numel()
        return {
            "layer_range": list(self.layer_range),
            "layers_owned": len(self.layers),
            "parameters": n_params,
            "owns_embedding": self.owns_embedding,
            "owns_head": self.owns_head,
            "bytes": n_params * 2,  # bf16
        }


def _dequantise(tensor: Any) -> torch.Tensor:
    """Dequantise one GGUF tensor to fp16, in GGUF's native layout.

    The native layout is [out, in] for matrices and [expert, out, in] for stacked
    experts -- the reverse of what PyTorch matmul expects. Measured, not assumed:

        blk.0.attn_k.weight   declared [4096,1024] -> array (1024,4096)
        ffn_gate_exps.weight  declared [4096,960,16] -> array (16,960,4096)
        token_embd.weight     declared [4096,32064] -> array (32064,4096)  [vocab,hidden]
    """
    import gguf

    if tensor.tensor_type.name == "F32":
        return torch.from_numpy(tensor.data.astype("float32")).to(torch.float16)
    if tensor.tensor_type.name == "F16":
        return torch.from_numpy(tensor.data.astype("float16"))
    arr = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
    return torch.from_numpy(arr.astype("float32")).to(torch.float16)


def _as_linear(t: torch.Tensor) -> torch.Tensor:
    """Reorient a weight so ``x @ W`` computes the linear projection.

    2D matrices become [in, out]; 3D expert stacks become [expert, in, out].
    """
    if t.dim() == 2:
        return t.t()
    if t.dim() == 3:
        return t.transpose(-1, -2)
    raise ModelLoadError(f"unexpected weight rank {t.dim()}")


def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in tensor.shape)


class GgufStageLoader:
    """Reads assigned tensors from a local GGUF.

    Only the requested layer range, plus optionally the embedding and the head,
    are dequantised. Everything else is skipped, which is what keeps each host's
    memory use proportional to its share rather than the whole model.
    """

    def __init__(self, path: str | Path, device: str = "cuda") -> None:
        self.path = Path(path)
        self.device = device
        if not self.path.exists():
            raise ModelLoadError(f"model file not found: {self.path}")
        self._reader: Any = None

    def _open(self) -> Any:
        if self._reader is None:
            import gguf

            self._reader = gguf.GGUFReader(str(self.path))
            self._by_name = {t.name: t for t in self._reader.tensors}
        return self._reader

    def tensor_names(self) -> list[str]:
        self._open()
        return list(self._by_name)

    def metadata(self) -> dict[str, Any]:
        """Model geometry as declared by the file, for verification."""
        reader = self._open()
        fields: dict[str, Any] = {}
        for key in (
            "general.architecture",
            "phimoe.block_count",
            "phimoe.embedding_length",
            "phimoe.attention.head_count",
            "phimoe.attention.head_count_kv",
            "phimoe.expert_count",
            "phimoe.expert_used_count",
            "phimoe.feed_forward_length",
        ):
            try:
                fields[key] = reader.get_field(key).contents()
            except KeyError:
                fields[key] = None
        return fields

    def _get(self, name: str, *, linear: bool = False) -> torch.Tensor | None:
        """Fetch one tensor. ``linear=True`` reorients it for ``x @ W``."""
        self._open()
        tensor = self._by_name.get(name)
        if tensor is None:
            return None
        out = _dequantise(tensor)
        if linear:
            out = _as_linear(out)
        return out.to(self.device)

    def _require(self, name: str, *, linear: bool = False) -> torch.Tensor:
        tensor = self._get(name, linear=linear)
        if tensor is None:
            raise ModelLoadError(f"required tensor {name!r} absent from {self.path}")
        return tensor

    def load(
        self,
        layer_start: int,
        layer_end: int,
        *,
        with_embedding: bool = False,
        with_head: bool = False,
    ) -> StageWeights:
        """Materialise one stage.

        Raises when a required tensor is missing, rather than proceeding with a
        partially-initialised layer: a silent zero-weight layer would produce
        plausible but wrong output.
        """
        if layer_start < 0 or layer_end > LAYERS or layer_start >= layer_end:
            raise ModelLoadError(
                f"invalid layer range [{layer_start}, {layer_end}) for {LAYERS} layers"
            )

        stage = StageWeights(layer_range=(layer_start, layer_end))

        for index in range(layer_start, layer_end):
            prefix = f"blk.{index}."
            stage.layers[index] = {
                "attn_norm_w": self._require(prefix + "attn_norm.weight"),
                "attn_norm_b": self._require(prefix + "attn_norm.bias"),
                "wq": self._require(prefix + "attn_q.weight", linear=True),
                "bq": self._require(prefix + "attn_q.bias"),
                "wk": self._require(prefix + "attn_k.weight", linear=True),
                "bk": self._require(prefix + "attn_k.bias"),
                "wv": self._require(prefix + "attn_v.weight", linear=True),
                "bv": self._require(prefix + "attn_v.bias"),
                "wo": self._require(prefix + "attn_output.weight", linear=True),
                "bo": self._require(prefix + "attn_output.bias"),
                "ffn_norm_w": self._require(prefix + "ffn_norm.weight"),
                "ffn_norm_b": self._require(prefix + "ffn_norm.bias"),
                "router": self._require(prefix + "ffn_gate_inp.weight"),
                "w_gate": self._require(prefix + "ffn_gate_exps.weight", linear=True),
                "w_up": self._require(prefix + "ffn_up_exps.weight", linear=True),
                "w_down": self._require(prefix + "ffn_down_exps.weight", linear=True),
            }

            expected = {
                "wq": (HIDDEN, HIDDEN),
                "wk": (HIDDEN, KV_DIM),
                "wv": (HIDDEN, KV_DIM),
                "wo": (HIDDEN, HIDDEN),
                "router": (EXPERTS, HIDDEN),
                "w_gate": (EXPERTS, HIDDEN, FFN),
                "w_up": (EXPERTS, HIDDEN, FFN),
                "w_down": (EXPERTS, FFN, HIDDEN),
            }
            for key, want in expected.items():
                got = tuple(stage.layers[index][key].shape)
                if got != want:
                    raise ModelLoadError(
                        f"layer {index} tensor {key}: shape {got} != expected {want}. "
                        "The file does not match the Phi-mini-MoE geometry this "
                        "runtime implements."
                    )

        if with_embedding:
            stage.embedding = self._require("token_embd.weight")
        if with_head:
            stage.final_norm_weight = self._require("output_norm.weight")
            stage.final_norm_bias = self._require("output_norm.bias")
            stage.lm_head_weight = self._require("output.weight")
            stage.lm_head_bias = self._get("output.bias")

        return stage


def checksum(path: str | Path) -> str:
    """SHA-256 of the model file, for verifying every host has the same revision."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(stage: StageWeights, path: str | Path) -> None:
    """Record what this stage owns, so a run can be audited afterwards."""
    Path(path).write_text(json.dumps(stage.summary(), indent=2))
