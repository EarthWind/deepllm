import math

import torch
from torch import nn


class MQAAttention(nn.Module):
    """
    Multi-Query Attention.

    Query uses multiple heads, while all heads share a single key head and a
    single value head.
    """

    def __init__(self, d_model, q_heads, dropout=0.0, bias=True):
        super().__init__()

        if d_model % q_heads != 0:
            raise ValueError("d_model must be divisible by q_heads")

        self.d_model = d_model
        self.q_heads = q_heads
        self.kv_heads = 1
        self.head_dim = d_model // q_heads

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, self.head_dim, bias=bias)
        self.w_v = nn.Linear(d_model, self.head_dim, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, need_weights=False):
        batch_size, q_len, _ = q.size()
        _, k_len, _ = k.size()

        q = self._split_q(self.w_q(q))
        k = self._split_kv(self.w_k(k))
        v = self._split_kv(self.w_v(v))

        # Reuse the single shared key/value head for every query head.
        k = k.expand(-1, self.q_heads, -1, -1)
        v = v.expand(-1, self.q_heads, -1, -1)

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
    mqa = MQAAttention(d_model=512, q_heads=8, dropout=0.1)
    y, attn = mqa(x, x, x, need_weights=True)
    print("output:", y.shape)
    print("attention:", attn.shape)
