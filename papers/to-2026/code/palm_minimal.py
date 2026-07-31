"""A small, auditable PyTorch implementation of PaLM's core architecture.

This is an educational model, not Google's training stack or a 540B replica.
It implements the architectural ideas disclosed in the PaLM paper:

* decoder-only causal language modeling;
* bias-free pre-normalization;
* parallel attention and SwiGLU branches;
* multi-query attention (one shared key/value head);
* rotary position embeddings (RoPE);
* shared input/output embeddings; and
* the auxiliary softmax z-loss used for training stability.

Run:
    python3 papers/to-2026/code/palm_minimal.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PaLMConfig:
    vocab_size: int = 256
    d_model: int = 64
    d_ff: int = 256
    n_layers: int = 2
    n_heads: int = 4
    head_dim: int = 16
    max_seq_len: int = 128
    rope_base: float = 10_000.0
    layer_norm_eps: float = 1e-5
    z_loss_weight: float = 1e-4

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "head_dim": self.head_dim,
            "max_seq_len": self.max_seq_len,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.rope_base <= 0 or self.layer_norm_eps <= 0:
            raise ValueError("rope_base and layer_norm_eps must be positive")
        if self.z_loss_weight < 0:
            raise ValueError("z_loss_weight must be non-negative")


class BiasFreeLayerNorm(nn.Module):
    """Ordinary LayerNorm with a learned scale but no learned shift.

    The PaLM paper calls this operation LayerNorm and states that layer norms
    have no bias.  Do not silently replace it with RMSNorm when reproducing the
    paper: RMSNorm omits mean subtraction and is a different operation.
    """

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulating statistics in float32 avoids avoidable fp16/bf16 error.
        source_dtype = x.dtype
        x_float = x.float()
        mean = x_float.mean(dim=-1, keepdim=True)
        variance = (x_float - mean).square().mean(dim=-1, keepdim=True)
        normalized = (x_float - mean) * torch.rsqrt(variance + self.eps)
        return normalized.to(source_dtype) * self.scale


def rotate_adjacent_pairs(x: torch.Tensor) -> torch.Tensor:
    """Map every adjacent pair ``(x0, x1)`` to ``(-x1, x0)``."""

    pairs = x.reshape(*x.shape[:-1], -1, 2)
    even, odd = pairs.unbind(dim=-1)
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    base: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate queries and keys.

    Shapes:
        q: [batch, query_heads, sequence, head_dim]
        k: [batch, 1,           sequence, head_dim]
    """

    sequence_length, head_dim = q.shape[-2:]
    if k.shape[0] != q.shape[0] or k.shape[-2:] != q.shape[-2:]:
        raise ValueError("q and k must share batch, sequence, and head_dim")
    if head_dim % 2:
        raise ValueError("RoPE requires an even head_dim")

    device = q.device
    pair_indices = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inverse_frequencies = 1.0 / (base ** (pair_indices / head_dim))
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    phase = torch.outer(positions, inverse_frequencies).repeat_interleave(2, dim=-1)
    cos = phase.cos().to(q.dtype)[None, None, :, :]
    sin = phase.sin().to(q.dtype)[None, None, :, :]
    return (
        q * cos + rotate_adjacent_pairs(q) * sin,
        k * cos + rotate_adjacent_pairs(k) * sin,
    )


class MultiQueryAttention(nn.Module):
    """Causal attention with H query heads and one shared key/value head."""

    def __init__(self, config: PaLMConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.rope_base = config.rope_base
        attention_width = config.n_heads * config.head_dim

        self.q_proj = nn.Linear(config.d_model, attention_width, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.head_dim, bias=False)
        self.out_proj = nn.Linear(attention_width, config.d_model, bias=False)

    def project_qkv(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expose the head shapes so MQA's sharing is easy to inspect."""

        batch, sequence, _ = x.shape
        q = self.q_proj(x).view(batch, sequence, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)  # [B, H, T, Dh]
        k = self.k_proj(x).view(batch, sequence, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, sequence, 1, self.head_dim).transpose(1, 2)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.project_qkv(x)
        q, k = apply_rope(q, k, base=self.rope_base)

        # The head dimension of k/v is 1. Broadcasting shares the same K/V
        # sequence across all query heads without changing the MQA semantics.
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        sequence = x.size(1)
        causal_mask = torch.ones(
            sequence,
            sequence,
            device=x.device,
            dtype=torch.bool,
        ).tril()
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        context = torch.matmul(probabilities, v)

        batch = x.size(0)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch, sequence, self.n_heads * self.head_dim)
        return self.out_proj(context)


class SwiGLU(nn.Module):
    """PaLM MLP: ``SiLU(xW) * (xV)`` followed by an output projection."""

    def __init__(self, config: PaLMConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.value_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.out_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.silu(self.gate_proj(x)) * self.value_proj(x))


class ParallelPaLMBlock(nn.Module):
    """Compute attention and MLP from the same normalized input."""

    def __init__(self, config: PaLMConfig) -> None:
        super().__init__()
        self.norm = BiasFreeLayerNorm(config.d_model, config.layer_norm_eps)
        self.attention = MultiQueryAttention(config)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        return x + self.attention(normalized) + self.mlp(normalized)


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    negative_log_likelihood: torch.Tensor
    z_loss: torch.Tensor


class TinyPaLM(nn.Module):
    """A tiny PaLM-style decoder with a tied embedding/language-model head."""

    def __init__(self, config: PaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            ParallelPaLMBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = BiasFreeLayerNorm(config.d_model, config.layer_norm_eps)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.size(1) > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")

        hidden = self.token_embedding(token_ids)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)

        # Weight tying: the same matrix embeds inputs and projects outputs.
        # PaLM scales pre-softmax logits because its embeddings are not normalized.
        return F.linear(hidden, self.token_embedding.weight) / math.sqrt(
            self.config.d_model
        )

    def language_model_loss(self, token_ids: torch.Tensor) -> LossBreakdown:
        """Next-token cross-entropy plus PaLM's ``1e-4 * log(Z)^2`` loss."""

        if token_ids.size(1) < 2:
            raise ValueError("language-model training requires at least two tokens")
        next_token_logits = self(token_ids)[:, :-1, :].float()
        next_tokens = token_ids[:, 1:]

        nll = F.cross_entropy(
            next_token_logits.reshape(-1, self.config.vocab_size),
            next_tokens.reshape(-1),
        )
        log_z = torch.logsumexp(next_token_logits, dim=-1)
        z_loss = self.config.z_loss_weight * log_z.square().mean()
        return LossBreakdown(total=nll + z_loss, negative_log_likelihood=nll, z_loss=z_loss)


def kv_cache_elements_per_layer(
    *,
    batch_size: int,
    sequence_length: int,
    n_heads: int,
    head_dim: int,
    multi_query: bool,
) -> int:
    """Count cached K and V scalar elements for one attention layer."""

    if min(batch_size, sequence_length, n_heads, head_dim) <= 0:
        raise ValueError("cache dimensions must be positive")
    kv_heads = 1 if multi_query else n_heads
    return 2 * batch_size * sequence_length * kv_heads * head_dim


def _check_invariants() -> None:
    torch.manual_seed(7)
    config = PaLMConfig()
    model = TinyPaLM(config).eval()
    token_ids = torch.randint(0, config.vocab_size, (2, 12))

    logits = model(token_ids)
    assert logits.shape == (2, 12, config.vocab_size)

    # No Linear or normalization bias parameter should exist.
    assert not any(name.endswith("bias") for name, _ in model.named_parameters())

    # MQA keeps H query heads but only one key/value head.
    q, k, v = model.blocks[0].attention.project_qkv(model.token_embedding(token_ids))
    assert q.shape == (2, config.n_heads, 12, config.head_dim)
    assert k.shape == v.shape == (2, 1, 12, config.head_dim)

    # Causality: changing future tokens cannot change earlier-position logits.
    changed = token_ids.clone()
    changed[:, 6:] = (changed[:, 6:] + 1) % config.vocab_size
    torch.testing.assert_close(model(token_ids)[:, :6], model(changed)[:, :6])

    losses = model.language_model_loss(token_ids)
    assert torch.isfinite(losses.total) and losses.z_loss >= 0
    losses.total.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    mha_cache = kv_cache_elements_per_layer(
        batch_size=1,
        sequence_length=2048,
        n_heads=48,
        head_dim=256,
        multi_query=False,
    )
    mqa_cache = kv_cache_elements_per_layer(
        batch_size=1,
        sequence_length=2048,
        n_heads=48,
        head_dim=256,
        multi_query=True,
    )
    assert mha_cache == 48 * mqa_cache

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print("All tiny PaLM checks passed.")
    print(f"logits: {tuple(logits.shape)}; parameters: {parameters:,}")
    print(
        f"loss={losses.total.item():.4f} "
        f"(nll={losses.negative_log_likelihood.item():.4f}, "
        f"z={losses.z_loss.item():.6f})"
    )
    print(f"PaLM-540B-style MQA cache reduction per layer: {mha_cache // mqa_cache}x")


if __name__ == "__main__":
    _check_invariants()
