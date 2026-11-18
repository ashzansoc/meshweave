"""Dense decoder-only stage: Qwen2 / Qwen2.5 maths over an owned layer range.

Each stage holds only the layers it was assigned, on the device it was given.
The pipeline moves hidden states between stages; nothing else is transferred.

Geometry and dropout-free inference follow the reference implementation:

    pre-norm RMSNorm (weight only), RoPE applied to the full head_dim using
    rotate_half (NEOX layout), GQA with q/k/v biases when the config sets
    attention_bias, SwiGLU MLP, no bias in o_proj or the MLP.

Weights come from safetensors as ``[out, in]`` and are stored transposed, so the
forward pass is a plain ``x @ w``. Only the keys this stage owns are read from
disk, which is what makes a stage fit on a card that cannot hold the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from meshweave.errors import ModelLoadError
from meshweave.runtime.model_config import DenseConfig

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


# --------------------------------------------------------------------------- #
# maths
# --------------------------------------------------------------------------- #


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm exactly as the reference computes it.

    The reference narrows back to the input dtype *before* applying the weight,
    so the multiply happens at model precision. Multiplying in fp32 and
    narrowing afterwards is more accurate but is not the same function, and the
    difference is visible in bf16.
    """
    dtype = x.dtype
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    normed = (x32 * torch.rsqrt(variance + eps)).to(dtype)
    return weight * normed


def rope_tables(
    seq_len: int, config: DenseConfig, device: torch.device, dtype: torch.dtype, offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, config.head_dim, 2, device=device, dtype=torch.float32) / config.head_dim)
    )
    freqs = torch.outer(positions, inv_freq)  # [seq, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [seq, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


@dataclass
class KvCache:
    """KV cache for the layers this stage owns only."""

    capacity: int
    config: DenseConfig
    device: torch.device
    dtype: torch.dtype
    k: dict[int, torch.Tensor] = field(default_factory=dict)
    v: dict[int, torch.Tensor] = field(default_factory=dict)
    length: int = 0

    def ensure(self, layer: int) -> None:
        if layer in self.k:
            return
        shape = (1, self.config.kv_heads, self.capacity, self.config.head_dim)
        self.k[layer] = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.v[layer] = torch.zeros(shape, device=self.device, dtype=self.dtype)

    def write(self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        end = start + k.shape[2]
        if end > self.capacity:
            raise ModelLoadError(
                f"sequence of {end} exceeds the KV capacity {self.capacity}; "
                "raise --max-seq or shorten the prompt"
            )
        self.k[layer][:, :, start:end, :] = k
        self.v[layer][:, :, start:end, :] = v

    def read(self, layer: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k[layer][:, :, :length, :], self.v[layer][:, :, :length, :]

    def reset(self) -> None:
        self.length = 0

    @property
    def bytes_allocated(self) -> int:
        return sum(t.numel() * t.element_size() for t in list(self.k.values()) + list(self.v.values()))


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #


def _weight_map(model_dir: Path) -> dict[str, str] | None:
    """param name -> shard file. ``None`` means the model is a single file."""
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        try:
            import json

            return dict(json.loads(index.read_text())["weight_map"])
        except (OSError, KeyError, ValueError) as exc:
            raise ModelLoadError(f"unreadable safetensors index in {model_dir}: {exc}") from exc
    if (model_dir / "model.safetensors").exists():
        return None
    raise ModelLoadError(
        f"no model.safetensors or model.safetensors.index.json in {model_dir}"
    )


def _read_tensors(model_dir: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    index = _weight_map(model_dir)
    found: dict[str, torch.Tensor] = {}

    if index is None:
        with safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") as fh:
            available = set(fh.keys())
            for key in keys:
                if key in available:
                    found[key] = fh.get_tensor(key)
            missing = [k for k in keys if k not in found]
            if missing:
                raise ModelLoadError(f"checkpoint is missing {missing[0]}")
        return found

    by_file: dict[str, list[str]] = {}
    for key in keys:
        shard = index.get(key)
        if shard is None:
            raise ModelLoadError(
                f"checkpoint {model_dir} has no entry for {key}. The model layout "
                "differs from the supported Qwen2 layout."
            )
        by_file.setdefault(shard, []).append(key)

    for shard, shard_keys in by_file.items():
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as fh:
            for key in shard_keys:
                found[key] = fh.get_tensor(key)
    return found


def _linear(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if t.ndim != 2:
        raise ModelLoadError(f"expected a 2-D projection, got shape {tuple(t.shape)}")
    return t.t().to(dtype).contiguous()


def layer_keys(config: DenseConfig, index: int) -> list[str]:
    prefix = f"model.layers.{index}"
    keys = [
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.v_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.mlp.gate_proj.weight",
        f"{prefix}.mlp.up_proj.weight",
        f"{prefix}.mlp.down_proj.weight",
    ]
    if config.attn_bias:
        keys += [
            f"{prefix}.self_attn.q_proj.bias",
            f"{prefix}.self_attn.k_proj.bias",
            f"{prefix}.self_attn.v_proj.bias",
        ]
    return keys


@dataclass
class StageWeights:
    config: DenseConfig
    layers: dict[int, dict[str, torch.Tensor]]
    embedding: torch.Tensor | None = None
    final_norm: torch.Tensor | None = None
    lm_head: torch.Tensor | None = None
    owns_embed: bool = False
    owns_head: bool = False

    @property
    def bytes(self) -> int:
        total = 0
        for layer in self.layers.values():
            for tensor in layer.values():
                total += tensor.numel() * tensor.element_size()
        for tensor in (self.embedding, self.final_norm, self.lm_head):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total


def load_stage(
    model_dir: Path,
    config: DenseConfig,
    start: int,
    end: int,
    *,
    owns_embed: bool,
    owns_head: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> StageWeights:
    """Read only this stage's tensors, and only its layers, from the checkpoint."""
    keys: list[str] = []
    for index in range(start, end):
        keys += layer_keys(config, index)
    if owns_embed:
        keys.append("model.embed_tokens.weight")
    if owns_head:
        keys.append("model.norm.weight")
        if not config.tie_embeddings:
            keys.append("lm_head.weight")
    elif config.tie_embeddings:
        pass

    raw = _read_tensors(model_dir, keys)

    layers: dict[int, dict[str, torch.Tensor]] = {}
    for index in range(start, end):
        prefix = f"model.layers.{index}"
        weights = {
            "ln1_w": raw[f"{prefix}.input_layernorm.weight"].to(dtype),
            "ln2_w": raw[f"{prefix}.post_attention_layernorm.weight"].to(dtype),
            "wq": _linear(raw[f"{prefix}.self_attn.q_proj.weight"], dtype),
            "wk": _linear(raw[f"{prefix}.self_attn.k_proj.weight"], dtype),
            "wv": _linear(raw[f"{prefix}.self_attn.v_proj.weight"], dtype),
            "wo": _linear(raw[f"{prefix}.self_attn.o_proj.weight"], dtype),
            "gate": _linear(raw[f"{prefix}.mlp.gate_proj.weight"], dtype),
            "up": _linear(raw[f"{prefix}.mlp.up_proj.weight"], dtype),
            "down": _linear(raw[f"{prefix}.mlp.down_proj.weight"], dtype),
        }
        if config.attn_bias:
            weights["bq"] = raw[f"{prefix}.self_attn.q_proj.bias"].to(dtype)
            weights["bk"] = raw[f"{prefix}.self_attn.k_proj.bias"].to(dtype)
            weights["bv"] = raw[f"{prefix}.self_attn.v_proj.bias"].to(dtype)
        layers[index] = {k: v.to(device) for k, v in weights.items()}

    embedding = None
    if owns_embed:
        embedding = raw["model.embed_tokens.weight"].to(device=device, dtype=dtype)

    final_norm = None
    lm_head = None
    if owns_head:
        final_norm = raw["model.norm.weight"].to(device=device, dtype=dtype)
        if config.tie_embeddings:
            if embedding is None:
                tied = _read_tensors(model_dir, ["model.embed_tokens.weight"])[
                    "model.embed_tokens.weight"
                ]
                embedding = tied.to(device=device, dtype=dtype)
            lm_head = _linear(embedding, dtype)
        else:
            lm_head = _linear(raw["lm_head.weight"], dtype).to(device)

    return StageWeights(
        config=config,
        layers=layers,
        embedding=embedding,
        final_norm=final_norm,
        lm_head=lm_head,
        owns_embed=owns_embed,
        owns_head=owns_head,
    )


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #


@dataclass
class StageOutput:
    hidden: torch.Tensor
    logits: torch.Tensor | None = None


class DenseStage:
    """Runs this stage's layers. One instance per stage."""

    def __init__(
        self,
        weights: StageWeights,
        device: torch.device,
        *,
        capacity: int = 2048,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.weights = weights
        self.config = weights.config
        self.device = device
        self.dtype = dtype
        self.kv = KvCache(capacity=capacity, config=self.config, device=device, dtype=dtype)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.weights.embedding is None:
            raise ModelLoadError("this stage does not own the embedding")
        return self.weights.embedding[token_ids].to(self.dtype)

    @torch.no_grad()
    def forward(self, hidden: torch.Tensor, *, position_offset: int = 0) -> StageOutput:
        x = hidden.to(self.dtype)
        seq_len = x.shape[1]
        cos, sin = rope_tables(seq_len, self.config, self.device, self.dtype, position_offset)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        for index in sorted(self.weights.layers):
            x = self._layer(index, self.weights.layers[index], x, cos, sin, position_offset)

        logits = None
        if self.weights.owns_head:
            if self.weights.final_norm is None or self.weights.lm_head is None:
                raise ModelLoadError("stage owns the head but is missing norm or lm_head")
            normed = rms_norm(x, self.weights.final_norm, self.config.eps)
            logits = (normed.to(self.dtype) @ self.weights.lm_head).float()

        self.kv.length += seq_len
        return StageOutput(hidden=x, logits=logits)

    def _layer(
        self,
        index: int,
        w: dict[str, torch.Tensor],
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        offset: int,
    ) -> torch.Tensor:
        cfg = self.config
        batch, seq, _ = x.shape

        residual = x
        normed = rms_norm(x, w["ln1_w"], cfg.eps)

        q = normed @ w["wq"]
        k = normed @ w["wk"]
        v = normed @ w["wv"]
        if cfg.attn_bias:
            q = q + w["bq"]
            k = k + w["bk"]
            v = v + w["bv"]

        q = q.view(batch, seq, cfg.heads, cfg.head_dim).transpose(1, 2)
        k = k.view(batch, seq, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        v = v.view(batch, seq, cfg.kv_heads, cfg.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        self.kv.ensure(index)
        self.kv.write(index, offset, k, v)
        k_all, v_all = self.kv.read(index, offset + seq)

        group = cfg.heads // cfg.kv_heads
        k_expanded = k_all.repeat_interleave(group, dim=1)
        v_expanded = v_all.repeat_interleave(group, dim=1)

        scores = (q.float() @ k_expanded.float().transpose(-1, -2)) / math.sqrt(cfg.head_dim)
        scores = scores + self._causal_mask(seq, k_all.shape[2], offset)
        probs = torch.softmax(scores, dim=-1).to(self.dtype)

        out = probs @ v_expanded
        out = out.transpose(1, 2).reshape(batch, seq, cfg.heads * cfg.head_dim)
        x = residual + (out @ w["wo"])

        residual = x
        normed = rms_norm(x, w["ln2_w"], cfg.eps)
        gate = normed @ w["gate"]
        up = normed @ w["up"]
        down = (torch.nn.functional.silu(gate) * up) @ w["down"]
        return residual + down

    def _causal_mask(self, seq: int, key_len: int, offset: int) -> torch.Tensor:
        rows = torch.arange(seq, device=self.device).unsqueeze(1) + offset
        cols = torch.arange(key_len, device=self.device).unsqueeze(0)
        mask = torch.zeros(seq, key_len, device=self.device, dtype=torch.float32)
        return mask.masked_fill(cols > rows, float("-inf"))

    def reset(self) -> None:
        self.kv.reset()


@torch.no_grad()
def sample_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None = None,
) -> int:
    """Pick the next token from the last position's logits."""
    row = logits[0, -1].float()
    if temperature <= 0:
        return int(torch.argmax(row).item())

    scaled = row / temperature
    probs = torch.softmax(scaled, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative - sorted_probs <= top_p
        keep[0] = True
        filtered = torch.zeros_like(probs)
        filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
        probs = filtered / filtered.sum()

    return int(torch.multinomial(probs, 1, generator=generator).item())
