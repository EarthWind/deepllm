#!/usr/bin/env python3
"""Llama 3 系统设计的零依赖教学实现。

对应《The Llama 3 Herd of Models》的几项可独立验证机制：

1. 8B / 70B / 405B 架构配置与稠密训练 FLOPs 近似；
2. 8 个 KV heads 的 GQA head 映射和 KV Cache 容量；
3. RoPE(theta=500,000) 的二维旋转与范数保持；
4. 长序列 packing 时阻止跨文档注意力的 causal document mask；
5. K 个候选的 reward-model rejection sampling；
6. 屏蔽格式 token、附加 chosen NLL 正则的 DPO 损失；
7. 0.1% 长上下文 SFT 样本的配额核算。

它不是 Meta 官方实现，也不加载模型权重。所有输出都是由论文披露的配置和公式
计算出的容量/损失示例，不是对论文 benchmark 的复现。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ModelSpec:
    name: str
    parameters: int
    layers: int
    model_dim: int
    ffn_dim: int
    query_heads: int
    kv_heads: int = 8
    vocab_size: int = 128_000
    rope_base: float = 500_000.0

    def __post_init__(self) -> None:
        if self.model_dim % self.query_heads != 0:
            raise ValueError("model_dim must be divisible by query_heads")
        if self.query_heads % self.kv_heads != 0:
            raise ValueError("query_heads must be divisible by kv_heads")

    @property
    def head_dim(self) -> int:
        return self.model_dim // self.query_heads

    @property
    def queries_per_kv_head(self) -> int:
        return self.query_heads // self.kv_heads


LLAMA3_SPECS: Mapping[str, ModelSpec] = {
    "8B": ModelSpec(
        name="8B",
        parameters=8_000_000_000,
        layers=32,
        model_dim=4_096,
        ffn_dim=14_336,
        query_heads=32,
    ),
    "70B": ModelSpec(
        name="70B",
        parameters=70_000_000_000,
        layers=80,
        model_dim=8_192,
        ffn_dim=28_672,
        query_heads=64,
    ),
    "405B": ModelSpec(
        name="405B",
        parameters=405_000_000_000,
        layers=126,
        model_dim=16_384,
        ffn_dim=53_248,
        query_heads=128,
    ),
}


def dense_training_flops(parameters: int, tokens: int) -> float:
    """标准 dense Transformer 训练计算近似 C ≈ 6ND。"""

    if parameters <= 0 or tokens <= 0:
        raise ValueError("parameters and tokens must be positive")
    return 6.0 * parameters * tokens


def bf16_weight_bytes(spec: ModelSpec) -> int:
    """只计算 BF16 权重，不含 KV、激活和运行时缓冲区。"""

    return spec.parameters * 2


def kv_cache_bytes(
    spec: ModelSpec,
    context_length: int,
    *,
    batch_size: int = 1,
    bytes_per_element: int = 2,
    use_mha: bool = False,
) -> int:
    """计算 decoder-only Transformer 的 K+V cache 理论容量。

    bytes = batch * layers * sequence * 2(K,V) * heads * head_dim * dtype_bytes
    use_mha=True 用 query-head 数模拟传统 MHA，便于和 8-head GQA 对比。
    """

    if context_length <= 0 or batch_size <= 0 or bytes_per_element <= 0:
        raise ValueError("context, batch, and element size must be positive")
    heads = spec.query_heads if use_mha else spec.kv_heads
    return (
        batch_size
        * spec.layers
        * context_length
        * 2
        * heads
        * spec.head_dim
        * bytes_per_element
    )


def gibibytes(byte_count: int) -> float:
    if byte_count < 0:
        raise ValueError("byte_count must be non-negative")
    return byte_count / (1024**3)


def kv_head_for_query(spec: ModelSpec, query_head: int) -> int:
    """返回某个 query head 在 GQA 中共享的 KV head 编号。"""

    if not 0 <= query_head < spec.query_heads:
        raise IndexError("query_head out of range")
    return query_head // spec.queries_per_kv_head


def estimated_tokens(characters: int, characters_per_token: float) -> float:
    if characters < 0 or characters_per_token <= 0:
        raise ValueError("invalid tokenizer inputs")
    return characters / characters_per_token


def tokenizer_savings(
    old_chars_per_token: float = 3.17,
    new_chars_per_token: float = 3.94,
) -> tuple[float, float]:
    """同一英文字符数下的 token 与全注意力 pair 理论降幅。"""

    if old_chars_per_token <= 0 or new_chars_per_token <= 0:
        raise ValueError("compression rates must be positive")
    sequence_ratio = old_chars_per_token / new_chars_per_token
    token_reduction = 1.0 - sequence_ratio
    quadratic_attention_reduction = 1.0 - sequence_ratio**2
    return token_reduction, quadratic_attention_reduction


def rope_rotate(
    vector: Sequence[float],
    position: int,
    *,
    base: float = 500_000.0,
) -> tuple[float, ...]:
    """对一个 head vector 应用基础 RoPE；相邻两个维度组成旋转平面。"""

    if len(vector) == 0 or len(vector) % 2:
        raise ValueError("vector length must be a positive even number")
    if position < 0 or base <= 1.0:
        raise ValueError("position must be non-negative and base > 1")

    result: list[float] = []
    dimension = len(vector)
    for pair_index in range(dimension // 2):
        x0 = float(vector[2 * pair_index])
        x1 = float(vector[2 * pair_index + 1])
        inverse_frequency = base ** (-(2.0 * pair_index) / dimension)
        angle = position * inverse_frequency
        cosine = math.cos(angle)
        sine = math.sin(angle)
        result.extend((x0 * cosine - x1 * sine, x0 * sine + x1 * cosine))
    return tuple(result)


def squared_norm(vector: Sequence[float]) -> float:
    return sum(float(value) ** 2 for value in vector)


def document_causal_mask(document_ids: Sequence[int]) -> tuple[tuple[bool, ...], ...]:
    """只允许看到同一文档中当前位置及其之前的 token。

    True 表示可注意，False 表示屏蔽。这样把多个文档 pack 进一条 128K
    序列时，不会让后一个文档读取前一个文档的内容。
    """

    if not document_ids:
        raise ValueError("document_ids must not be empty")
    return tuple(
        tuple(
            key_index <= query_index
            and document_ids[key_index] == document_ids[query_index]
            for key_index in range(len(document_ids))
        )
        for query_index in range(len(document_ids))
    )


@dataclass(frozen=True)
class Candidate:
    text: str
    reward: float


def rejection_sample(candidates: Sequence[Candidate]) -> Candidate:
    """论文 RS 的核心：同一 prompt 采样 K 个回答，RM 选择最高分。"""

    if not candidates:
        raise ValueError("candidates must not be empty")
    return max(candidates, key=lambda candidate: (candidate.reward, candidate.text))


def masked_logprob_sum(
    token_logprobs: Sequence[float],
    is_formatting_token: Sequence[bool],
) -> float:
    """DPO 中忽略 header / termination 等聊天协议 token。"""

    if len(token_logprobs) != len(is_formatting_token):
        raise ValueError("logprobs and mask must have the same length")
    return sum(
        logprob
        for logprob, is_formatting in zip(token_logprobs, is_formatting_token)
        if not is_formatting
    )


def log_sigmoid(value: float) -> float:
    """数值稳定的 log(sigmoid(x))。"""

    if value >= 0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def llama3_dpo_loss(
    *,
    policy_chosen_logprob: float,
    policy_rejected_logprob: float,
    reference_chosen_logprob: float,
    reference_rejected_logprob: float,
    chosen_nll: float,
    beta: float = 0.1,
    nll_coefficient: float = 0.2,
) -> float:
    """论文披露的 DPO + 0.2 chosen-NLL 正则教学表达。"""

    if beta <= 0 or chosen_nll < 0 or nll_coefficient < 0:
        raise ValueError("invalid DPO hyperparameters")
    policy_margin = policy_chosen_logprob - policy_rejected_logprob
    reference_margin = reference_chosen_logprob - reference_rejected_logprob
    preference_loss = -log_sigmoid(beta * (policy_margin - reference_margin))
    return preference_loss + nll_coefficient * chosen_nll


def long_context_sft_quota(total_examples: int, fraction: float = 0.001) -> int:
    """0.1% long-context SFT mix 的最小整数配额。"""

    if total_examples <= 0 or not 0.0 <= fraction <= 1.0:
        raise ValueError("invalid quota inputs")
    return math.ceil(total_examples * fraction)


def render_mask(mask: Sequence[Sequence[bool]]) -> str:
    return "\n".join(" ".join("●" if allowed else "·" for allowed in row) for row in mask)


def demo() -> None:
    spec = LLAMA3_SPECS["405B"]
    flops = dense_training_flops(spec.parameters, 15_600_000_000_000)
    weights = gibibytes(bf16_weight_bytes(spec))
    gqa_cache = gibibytes(kv_cache_bytes(spec, 131_072))
    mha_cache = gibibytes(kv_cache_bytes(spec, 131_072, use_mha=True))
    token_reduction, attention_reduction = tokenizer_savings()

    print("Llama 3 405B disclosed-configuration arithmetic:")
    print(f"  dense training FLOPs ≈ {flops:.4e}")
    print(f"  BF16 weights only   ≈ {weights:.1f} GiB")
    print(f"  128K GQA KV cache   ≈ {gqa_cache:.1f} GiB / batch item")
    print(f"  128K MHA KV cache   ≈ {mha_cache:.1f} GiB / batch item")
    print(f"  GQA cache reduction = {mha_cache / gqa_cache:.0f}x")
    print("Tokenizer compression (same English character count):")
    print(f"  tokens reduced      ≈ {token_reduction:.1%}")
    print(f"  dense attention pairs reduced ≈ {attention_reduction:.1%}")

    doc_ids = (0, 0, 0, 1, 1)
    print("Document-aware causal mask for ids", doc_ids)
    print(render_mask(document_causal_mask(doc_ids)))

    candidates = [
        Candidate("candidate-a", 0.62),
        Candidate("candidate-b", 0.91),
        Candidate("candidate-c", 0.74),
    ]
    print("Rejection-sampling winner:", rejection_sample(candidates).text)

    dpo = llama3_dpo_loss(
        policy_chosen_logprob=-2.1,
        policy_rejected_logprob=-3.0,
        reference_chosen_logprob=-2.4,
        reference_rejected_logprob=-2.8,
        chosen_nll=0.7,
    )
    print(f"Toy regularized DPO loss: {dpo:.4f}")
    print("Long-context quota in 1M SFT examples:", long_context_sft_quota(1_000_000))

    rotated = rope_rotate((1.0, 2.0, 3.0, 4.0), position=128_000)
    assert math.isclose(squared_norm(rotated), 30.0, rel_tol=1e-12)
    assert math.isclose(flops, 3.7908e25)
    assert math.isclose(gqa_cache, 63.0)
    assert math.isclose(mha_cache, 1008.0)
    assert spec.queries_per_kv_head == 16
    assert kv_head_for_query(spec, 127) == 7
    assert document_causal_mask(doc_ids)[4][2] is False
    assert rejection_sample(candidates).text == "candidate-b"


if __name__ == "__main__":
    demo()
