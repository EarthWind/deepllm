"""A small, readable RoPE implementation using the paper's interleaved layout.

Run:
    python3 papers/to-2026/code/rope_minimal.py

Tensor convention:
    q, k: [batch, heads, sequence, head_dim]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """Turn every adjacent pair (x0, x1) into (-x1, x0)."""
    if x.size(-1) % 2 != 0:
        raise ValueError("RoPE requires an even rotary dimension")

    pairs = x.reshape(*x.shape[:-1], -1, 2)
    x_even, x_odd = pairs.unbind(dim=-1)
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def build_rope_cache(
    position_ids: torch.Tensor,
    rotary_dim: int,
    *,
    base: float = 10_000.0,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin with shape [batch, 1, sequence, rotary_dim].

    Trigonometric functions are evaluated in float32 for numerical stability,
    then cast to the model dtype.
    """
    if rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be even")
    if position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.ndim != 2:
        raise ValueError("position_ids must have shape [sequence] or [batch, sequence]")

    position_ids = position_ids.to(device=device)
    pair_index = torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (pair_index / rotary_dim))
    phase = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq

    # The paper pairs adjacent channels: (0,1), (2,3), ...
    phase = phase.repeat_interleave(2, dim=-1)
    return phase.cos().unsqueeze(1).to(dtype), phase.sin().unsqueeze(1).to(dtype)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    rotary_dim: int | None = None,
    base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to all or a prefix of each query/key head."""
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape [batch, heads, sequence, head_dim]")
    if q.shape[0] != k.shape[0] or q.shape[-2:] != k.shape[-2:]:
        raise ValueError("q and k must share batch, sequence, and head dimensions")
    if q.device != k.device or q.dtype != k.dtype:
        raise ValueError("q and k must share device and dtype")
    if position_ids.ndim == 1:
        valid_positions = position_ids.numel() == q.size(-2)
    elif position_ids.ndim == 2:
        valid_positions = (
            position_ids.size(-1) == q.size(-2)
            and position_ids.size(0) in (1, q.size(0))
        )
    else:
        valid_positions = False
    if not valid_positions:
        raise ValueError("position_ids must broadcast to [batch, sequence]")

    head_dim = q.size(-1)
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    if not 0 < rotary_dim <= head_dim or rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be positive, even, and <= head_dim")

    cos, sin = build_rope_cache(
        position_ids,
        rotary_dim,
        base=base,
        device=q.device,
        dtype=q.dtype,
    )

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = q_rot * cos + rotate_pairs(q_rot) * sin
    k_rot = k_rot * cos + rotate_pairs(k_rot) * sin
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


def rope_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    is_causal: bool = True,
) -> torch.Tensor:
    """The place where RoPE sits inside standard scaled dot-product attention."""
    q, k = apply_rope(q, k, position_ids)
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def _check_invariants() -> None:
    torch.manual_seed(7)

    # 1. Rotation is orthogonal, so it preserves every vector norm.
    q = torch.randn(2, 4, 6, 8)
    k = torch.randn(2, 2, 6, 8)  # GQA is fine: q_heads may differ from k_heads.
    positions = torch.arange(6)
    q_rope, k_rope = apply_rope(q, k, positions)
    torch.testing.assert_close(q.norm(dim=-1), q_rope.norm(dim=-1))
    torch.testing.assert_close(k.norm(dim=-1), k_rope.norm(dim=-1))

    # 2. Shifting both absolute positions equally keeps the dot product fixed.
    q_one = torch.randn(1, 1, 1, 8)
    k_one = torch.randn(1, 1, 1, 8)
    q_a, _ = apply_rope(q_one, k_one, torch.tensor([3]))
    _, k_a = apply_rope(q_one, k_one, torch.tensor([11]))
    q_b, _ = apply_rope(q_one, k_one, torch.tensor([103]))
    _, k_b = apply_rope(q_one, k_one, torch.tensor([111]))
    score_a = (q_a * k_a).sum()
    score_b = (q_b * k_b).sum()
    torch.testing.assert_close(score_a, score_b, rtol=1e-4, atol=1e-4)

    print("All RoPE checks passed.")
    print(f"norm before/after: {q[0, 0, 0].norm():.6f} / {q_rope[0, 0, 0].norm():.6f}")
    print(f"same relative offset scores: {score_a:.6f} / {score_b:.6f}")


if __name__ == "__main__":
    _check_invariants()
