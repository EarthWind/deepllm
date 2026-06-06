import math

import torch
from torch import nn


class CLAAttention(nn.Module):
    """
    Cross-Layer Attention.

    The current layer always computes its own query states, while key/value
    states can either be projected from hidden states or reused from a previous
    layer. This makes the module suitable for CLA-style KV sharing across
    adjacent layers.
    """

    def __init__(self, d_model, q_heads, kv_heads=1, dropout=0.0, bias=True):
        super().__init__()

        if d_model % q_heads != 0:
            raise ValueError("d_model must be divisible by q_heads")
        if q_heads % kv_heads != 0:
            raise ValueError("q_heads must be divisible by kv_heads")

        self.d_model = d_model
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = d_model // q_heads
        self.group_size = q_heads // kv_heads

        kv_dim = self.kv_heads * self.head_dim
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, kv_dim, bias=bias)
        self.w_v = nn.Linear(d_model, kv_dim, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def compute_kv(self, k, v=None):
        """
        Project hidden states into reusable KV states.

        Returns tensors shaped as [batch, kv_heads, seq_len, head_dim].
        """
        if v is None:
            v = k

        key_states = self._split_kv(self.w_k(k))
        value_states = self._split_kv(self.w_v(v))
        return key_states, value_states

    def forward(self, q, k=None, v=None, mask=None, need_weights=False):
        q_input = q
        batch_size, q_len, _ = q_input.size()
        q = self._split_q(self.w_q(q_input))

        if k is None and v is None:
            k, v = self.compute_kv(q_input)
            k_len = q_len
        elif k is None or v is None:
            raise ValueError("k and v must both be provided, or both be omitted")
        elif k.dim() == 3 and v.dim() == 3:
            _, k_len, _ = k.size()
            k, v = self.compute_kv(k, v)
        elif k.dim() == 4 and v.dim() == 4:
            k_len = self._validate_shared_kv(k, v, batch_size)
        else:
            raise ValueError("k and v must both be 3D hidden states or 4D shared KV states")

        k = self._expand_kv(k)
        v = self._expand_kv(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = self._apply_mask(scores, mask, batch_size, q_len, k_len)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        out = self.w_o(out)

        if need_weights:
            return out, attn
        return out

    def _split_q(self, tensor):
        batch_size, length, _ = tensor.size()
        return tensor.view(batch_size, length, self.q_heads, self.head_dim).transpose(1, 2)

    def _split_kv(self, tensor):
        batch_size, length, _ = tensor.size()
        return tensor.view(batch_size, length, self.kv_heads, self.head_dim).transpose(1, 2)

    def _expand_kv(self, tensor):
        if self.kv_heads == self.q_heads:
            return tensor
        return tensor.repeat_interleave(self.group_size, dim=1)

    def _validate_shared_kv(self, key_states, value_states, batch_size):
        if key_states.shape != value_states.shape:
            raise ValueError("shared key and value states must have the same shape")
        if key_states.dim() != 4:
            raise ValueError("shared key/value states must be 4D")

        expected_prefix = (batch_size, self.kv_heads)
        if tuple(key_states.shape[:2]) != expected_prefix:
            raise ValueError(
                "shared key/value states must have shape [B, kv_heads, K, head_dim]"
            )
        if key_states.shape[-1] != self.head_dim:
            raise ValueError(
                "shared key/value states must have the same head_dim as the query heads"
            )
        return key_states.size(-2)

    def _apply_mask(self, scores, mask, batch_size, q_len, k_len):
        if mask.dim() == 2:
            mask = mask[:, None, None, :]
        elif mask.dim() == 3:
            mask = mask[:, None, :, :]
        elif mask.dim() != 4:
            raise ValueError("mask must have 2, 3 or 4 dimensions")

        expected_shapes = (
            (batch_size, 1, 1, k_len),
            (batch_size, 1, q_len, k_len),
            (batch_size, self.q_heads, q_len, k_len),
        )
        if tuple(mask.shape) not in expected_shapes:
            raise ValueError(
                "mask shape must be [B, K], [B, Q, K], [B, 1, Q, K] or [B, q_heads, Q, K]"
            )

        return scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)


if __name__ == "__main__":
    x = torch.randn(2, 16, 512)
    cla = CLAAttention(d_model=512, q_heads=8, kv_heads=1, dropout=0.1)

    shared_k, shared_v = cla.compute_kv(x)
    producer_out = cla(x, shared_k, shared_v)
    consumer_out, attn = cla(x, shared_k, shared_v, need_weights=True)

    print("shared key:", shared_k.shape)
    print("producer output:", producer_out.shape)
    print("consumer output:", consumer_out.shape)
    print("attention:", attn.shape)
