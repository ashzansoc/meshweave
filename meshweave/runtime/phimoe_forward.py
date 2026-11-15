"""Phi-mini-MoE forward pass over an owned layer range.

This is the model computation, not a tensor forwarder. Each stage runs its own
layers for real; the pipeline only moves hidden states between stages.

Geometry verified against the GGUF and config:

    hidden 4096, 32 layers, 32 query heads / 8 KV heads (GQA), head_dim 128
    experts 16, top-2 routing, no shared expert
    RMSNorm carries a bias (unusual -- most models do not)
    SwiGLU FFN per expert: down( silu(gate(x)) * up(x) )

Attention maths follow the reference: q/k/v projections with bias, RoPE over the
full head_dim using theta=10000, causal masking, then the output projection with
bias. KV cache is per-stage and holds only the layers that stage owns.

Numeric strategy: activations are fp16 on the wire and in the KV cache, but the
attention softmax and the router softmax run in fp32, because both are sensitive
to accumulated error and a mismatch there shows up as wrong tokens rather than
small drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from meshweave.runtime.gguf_stage import (
    HEAD_DIM,
    HEADS,
    KV_HEADS,
    RMS_EPS,
    ROPE_THETA,
    TOP_K,
    StageWeights,
)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """RMSNorm with an optional affine bias, computed in fp32."""
    original = x.dtype
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    normed = x32 * torch.rsqrt(variance + RMS_EPS)
    normed = normed * weight.to(torch.float32)
    if bias is not None:
        normed = normed + bias.to(torch.float32)
    return normed.to(original)


def rope_tables(
    seq_len: int, device: torch.device, dtype: torch.dtype, offset: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin for rotary position embedding.

    ``offset`` lets a decode step start at the right absolute position without
    recomputing the whole table.
    """
    positions = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, device=device, dtype=torch.float32) / HEAD_DIM)
    )
    freqs = torch.outer(positions, inv_freq)  # [seq, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [seq, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to a [batch, heads, seq, head_dim] tensor."""
    return (x * cos) + (rotate_half(x) * sin)


@dataclass
class KvCache:
    """Per-stage KV cache. Only the stage's own layers are stored.

    Shape per layer: [batch, kv_heads, capacity, head_dim].
    """

    capacity: int
    device: torch.device
    dtype: torch.dtype = torch.float16
    k: dict[int, torch.Tensor] = field(default_factory=dict)
    v: dict[int, torch.Tensor] = field(default_factory=dict)
    length: int = 0

    def ensure(self, layer: int, batch: int) -> None:
        if layer in self.k:
            return
        self.k[layer] = torch.zeros(
            batch, KV_HEADS, self.capacity, HEAD_DIM, device=self.device, dtype=self.dtype
        )
        self.v[layer] = torch.zeros(
            batch, KV_HEADS, self.capacity, HEAD_DIM, device=self.device, dtype=self.dtype
        )

    def write(self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        end = start + k.shape[2]
        self.k[layer][:, :, start:end, :] = k
        self.v[layer][:, :, start:end, :] = v

    def read(self, layer: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k[layer][:, :, :length, :], self.v[layer][:, :, :length, :]

    def reset(self) -> None:
        self.length = 0

    @property
    def bytes_allocated(self) -> int:
        return sum(t.numel() * t.element_size() for t in list(self.k.values()) + list(self.v.values()))


@dataclass
class ForwardResult:
    hidden: torch.Tensor
    logits: torch.Tensor | None = None
    router_choices: dict[int, torch.Tensor] = field(default_factory=dict)


class StageExecutor:
    """Runs the layers a stage owns.

    One instance per stage; it holds the weights on the device and the KV cache for
    its own layers.
    """

    def __init__(
        self,
        weights: StageWeights,
        device: torch.device,
        kv_capacity: int = 4096,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.weights = weights
        self.device = device
        self.dtype = dtype
        self.kv = KvCache(capacity=kv_capacity, device=device, dtype=dtype)

    # ---------------------------------------------------------------- embed

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map token ids to hidden states. First stage only."""
        if self.weights.embedding is None:
            raise RuntimeError("this stage does not own the embedding")
        return self.weights.embedding[token_ids].to(self.dtype)

    # ------------------------------------------------------------- forward

    @torch.no_grad()
    def forward(
        self,
        hidden: torch.Tensor,
        *,
        position_offset: int = 0,
        use_cache: bool = True,
        record_router: bool = False,
    ) -> ForwardResult:
        """Run this stage's layers on ``hidden``.

        ``hidden`` is [1, seq, 4096] in fp16. Returns the transformed hidden state,
        plus logits when this stage owns the head.
        """
        x = hidden.to(self.dtype)
        router_choices: dict[int, torch.Tensor] = {}
        seq_len = x.shape[1]
        cos, sin = rope_tables(seq_len, self.device, self.dtype, offset=position_offset)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,seq,head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)

        for index in sorted(self.weights.layers):
            layer = self.weights.layers[index]
            x = self._decoder_layer(
                index,
                layer,
                x,
                cos=cos,
                sin=sin,
                position_offset=position_offset,
                use_cache=use_cache,
                router_choices=router_choices if record_router else None,
            )

        logits = None
        if self.weights.owns_head:
            if (
                self.weights.final_norm_weight is None
                or self.weights.lm_head_weight is None
            ):
                raise RuntimeError(
                    "stage claims to own the head but is missing output_norm or output"
                )
            normed = rms_norm(
                x, self.weights.final_norm_weight, self.weights.final_norm_bias
            )
            logits = normed.to(self.dtype) @ self.weights.lm_head_weight.T
            if self.weights.lm_head_bias is not None:
                logits = logits + self.weights.lm_head_bias
            logits = logits.float()

        if use_cache:
            self.kv.length += seq_len

        return ForwardResult(hidden=x, logits=logits, router_choices=router_choices)

    # --------------------------------------------------------- one layer

    def _decoder_layer(
        self,
        index: int,
        w: dict[str, torch.Tensor],
        x: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_offset: int,
        use_cache: bool,
        router_choices: dict[int, torch.Tensor] | None,
    ) -> torch.Tensor:
        # --- attention block ---
        residual = x
        normed = rms_norm(x, w["attn_norm_w"], w["attn_norm_b"])
        attn_out = self._attention(
            index, w, normed, cos=cos, sin=sin,
            position_offset=position_offset, use_cache=use_cache,
        )
        x = residual + attn_out

        # --- MoE block ---
        residual = x
        normed = rms_norm(x, w["ffn_norm_w"], w["ffn_norm_b"])
        moe_out = self._moe(index, w, normed, router_choices=router_choices)
        return residual + moe_out

    def _attention(
        self,
        index: int,
        w: dict[str, torch.Tensor],
        x: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_offset: int,
        use_cache: bool,
    ) -> torch.Tensor:
        batch, seq, _ = x.shape

        q = (x @ w["wq"]) + w["bq"]
        k = (x @ w["wk"]) + w["bk"]
        v = (x @ w["wv"]) + w["bv"]

        # [batch, seq, heads*dim] -> [batch, heads, seq, dim]
        q = q.view(batch, seq, HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(batch, seq, KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(batch, seq, KV_HEADS, HEAD_DIM).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if use_cache:
            self.kv.ensure(index, batch)
            self.kv.write(index, position_offset, k, v)
            k_all, v_all = self.kv.read(index, position_offset + seq)
        else:
            k_all, v_all = k, v

        # GQA: repeat each KV head for its group of query heads.
        group = HEADS // KV_HEADS
        k_expanded = k_all.repeat_interleave(group, dim=1)
        v_expanded = v_all.repeat_interleave(group, dim=1)

        # fp32 for the softmax: fp16 accumulation visibly shifts token choice.
        scores = (q.float() @ k_expanded.float().transpose(-1, -2)) / math.sqrt(HEAD_DIM)
        scores = scores + self._causal_mask(seq, k_all.shape[2], position_offset)
        probs = torch.softmax(scores, dim=-1).to(self.dtype)

        out = probs @ v_expanded
        out = out.transpose(1, 2).reshape(batch, seq, HEADS * HEAD_DIM)
        return (out @ w["wo"]) + w["bo"]

    def _causal_mask(self, seq: int, key_len: int, offset: int) -> torch.Tensor:
        """Additive mask blocking future positions.

        Row ``i`` may attend to keys ``<= offset + i``.
        """
        rows = torch.arange(seq, device=self.device).unsqueeze(1) + offset
        cols = torch.arange(key_len, device=self.device).unsqueeze(0)
        blocked = cols > rows
        mask = torch.zeros(seq, key_len, device=self.device, dtype=torch.float32)
        return mask.masked_fill(blocked, float("-inf"))

    def _moe(
        self,
        index: int,
        w: dict[str, torch.Tensor],
        x: torch.Tensor,
        *,
        router_choices: dict[int, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Top-k routed mixture of SwiGLU experts."""
        batch, seq, hidden = x.shape
        flat = x.reshape(batch * seq, hidden)

        # Router logits in fp32; the top-k choice must not flip on fp16 noise.
        logits = flat.float() @ w["router"].float().t()
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, TOP_K, dim=-1)
        # Renormalise over the selected experts.
        #
        # This was briefly removed during Phase 3A on the strength of the HF
        # `sparsemixer` source, which uses the raw softmax values. That was wrong:
        # llama.cpp's reference logits for the validation prompt are
        # [67.39, 66.52, 66.52, 66.39, 65.63], which is exactly what the
        # renormalised version produces. The reference implementation must be
        # trusted over a reading of a different codebase.
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        if router_choices is not None:
            router_choices[index] = topk_idx.detach()

        out = torch.zeros_like(flat)
        for slot in range(TOP_K):
            expert_ids = topk_idx[:, slot]  # [tokens]
            weight = topk_probs[:, slot].unsqueeze(-1)  # [tokens,1]
            for expert in torch.unique(expert_ids).tolist():
                mask = expert_ids == expert
                if not bool(mask.any()):
                    continue
                selected = flat[mask]
                gate = selected @ w["w_gate"][expert]
                up = selected @ w["w_up"][expert]
                activated = torch.nn.functional.silu(gate) * up
                down = activated @ w["w_down"][expert]
                out[mask] += down * weight[mask]

        return out.reshape(batch, seq, hidden)


def reference_positions(seq_len: int) -> list[int]:
    """Token positions for a prompt, used by tests and the coordinator."""
    return list(range(seq_len))
