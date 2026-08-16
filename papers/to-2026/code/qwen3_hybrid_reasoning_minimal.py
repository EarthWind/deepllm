#!/usr/bin/env python3
"""Qwen3 Technical Report 的零依赖机制教学实现。

这个文件只复现论文中可以脱离真实模型验证的结构与算术：

1. 八个 Qwen3-2504 模型的 dense / MoE、激活参数和 GQA 比例；
2. QK-Norm 的简化向量计算与 MoE Top-8/128 路由；
3. ``/think``、``/no_think``、硬开关及空 thinking block 的模式语义；
4. 达到最大 thinking budget 后中断思考并切换到最终回答；
5. GRPO 组内相对 advantage 与 teacher-logit KL 蒸馏；
6. 论文 Table 21/22 的 GPU-hour、能力收益和退化算术。

它不是官方训练或推理代码，不下载模型权重，也不声称复现 benchmark 分数。
Qwen3 的路由器、GRPO/off-policy 细节和完整超参数并未在技术报告中公开；
相应函数仅用于把已披露机制变成可检查的小例子。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ModelSpec:
    name: str
    architecture: str
    total_params_b: float
    active_params_b: float
    layers: int
    query_heads: int
    kv_heads: int
    experts_total: int | None = None
    experts_active: int | None = None
    paper_context_k: int = 128

    @property
    def active_parameter_fraction(self) -> float:
        return self.active_params_b / self.total_params_b

    @property
    def kv_head_fraction(self) -> float:
        """仅比较 KV head 与 query head 数，不等于完整推理 FLOPs 比例。"""

        return self.kv_heads / self.query_heads

    @property
    def active_expert_fraction(self) -> float | None:
        if self.experts_total is None or self.experts_active is None:
            return None
        return self.experts_active / self.experts_total


QWEN3_2504_MODELS = (
    ModelSpec("Qwen3-0.6B", "dense", 0.6, 0.6, 28, 16, 8, paper_context_k=32),
    ModelSpec("Qwen3-1.7B", "dense", 1.7, 1.7, 28, 16, 8, paper_context_k=32),
    ModelSpec("Qwen3-4B", "dense", 4.0, 4.0, 36, 32, 8),
    ModelSpec("Qwen3-8B", "dense", 8.0, 8.0, 36, 32, 8),
    ModelSpec("Qwen3-14B", "dense", 14.0, 14.0, 40, 40, 8),
    ModelSpec("Qwen3-32B", "dense", 32.0, 32.0, 64, 64, 8),
    ModelSpec("Qwen3-30B-A3B", "moe", 30.0, 3.0, 48, 32, 4, 128, 8),
    ModelSpec("Qwen3-235B-A22B", "moe", 235.0, 22.0, 94, 64, 4, 128, 8),
)


def rms(vector: Sequence[float], epsilon: float = 1e-6) -> float:
    if not vector:
        raise ValueError("vector must not be empty")
    return math.sqrt(sum(value * value for value in vector) / len(vector) + epsilon)


def rms_normalize(vector: Sequence[float], epsilon: float = 1e-6) -> tuple[float, ...]:
    scale = rms(vector, epsilon)
    return tuple(value / scale for value in vector)


def qk_normalized_logit(
    query: Sequence[float],
    key: Sequence[float],
    *,
    epsilon: float = 1e-6,
) -> float:
    """QK-Norm 后的单个 attention logit 教学版。

    论文只说明在 attention 中引入 QK-Norm 以稳定训练；真实模型还包含
    learned scale、RoPE、分头投影和具体 kernel，这里不冒充逐行复现。
    """

    if len(query) != len(key) or not query:
        raise ValueError("query and key must have equal non-zero dimensions")
    q_hat = rms_normalize(query, epsilon)
    k_hat = rms_normalize(key, epsilon)
    return sum(q * k for q, k in zip(q_hat, k_hat)) / math.sqrt(len(query))


def softmax(values: Sequence[float], temperature: float = 1.0) -> tuple[float, ...]:
    if not values:
        raise ValueError("values must not be empty")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    maximum = max(values)
    exps = [math.exp((value - maximum) / temperature) for value in values]
    total = sum(exps)
    return tuple(value / total for value in exps)


@dataclass(frozen=True)
class Route:
    expert_indices: tuple[int, ...]
    normalized_weights: tuple[float, ...]


def top_k_expert_route(gate_logits: Sequence[float], k: int = 8) -> Route:
    """选择 gate logit 最大的 k 个专家，并在入选专家内归一化。

    这是 Top-k 稀疏路由的最小示意。报告没有公开 router 逐行实现，因此这里
    不包含 capacity、token dropping、auxiliary/global-batch loss 等系统细节。
    """

    if not 0 < k <= len(gate_logits):
        raise ValueError("k must be within the number of experts")
    indices = tuple(
        sorted(range(len(gate_logits)), key=lambda index: (-gate_logits[index], index))[:k]
    )
    weights = softmax(tuple(gate_logits[index] for index in indices))
    return Route(indices, weights)


def routing_load_cv(routes: Iterable[Route], num_experts: int) -> float:
    """用专家 token-count 的变异系数诊断负载不均，不是论文损失公式。"""

    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    counts = [0] * num_experts
    num_routes = 0
    for route in routes:
        num_routes += 1
        for index in route.expert_indices:
            if not 0 <= index < num_experts:
                raise ValueError("expert index out of range")
            counts[index] += 1
    if num_routes == 0:
        raise ValueError("routes must not be empty")
    mean = sum(counts) / num_experts
    if mean == 0:
        return 0.0
    variance = sum((count - mean) ** 2 for count in counts) / num_experts
    return math.sqrt(variance) / mean


class ThinkingMode(str, Enum):
    THINKING = "thinking"
    NON_THINKING = "non-thinking"


@dataclass(frozen=True)
class Message:
    role: str
    content: str


_FLAG_PATTERN = re.compile(r"/(?:no_)?think\b")


def latest_soft_mode(messages: Sequence[Message]) -> ThinkingMode:
    """返回对话中最后出现的软模式指令；未出现时默认 thinking。"""

    mode = ThinkingMode.THINKING
    for message in messages:
        for match in _FLAG_PATTERN.finditer(message.content):
            mode = (
                ThinkingMode.NON_THINKING
                if match.group() == "/no_think"
                else ThinkingMode.THINKING
            )
    return mode


def resolve_mode(
    messages: Sequence[Message],
    *,
    enable_thinking: bool = True,
) -> ThinkingMode:
    """硬开关关闭时严格 non-thinking，否则由最后一个软 flag 决定。"""

    if not enable_thinking:
        return ThinkingMode.NON_THINKING
    return latest_soft_mode(messages)


def fusion_training_target(
    query: str,
    response: str,
    *,
    mode: ThinkingMode,
    thinking_content: str = "",
) -> str:
    """论文 Table 9 的两种 assistant target 的紧凑表示。"""

    if not query or not response:
        raise ValueError("query and response must not be empty")
    if mode is ThinkingMode.NON_THINKING and thinking_content:
        raise ValueError("non-thinking target must keep an empty thinking block")
    flag = "/think" if mode is ThinkingMode.THINKING else "/no_think"
    return (
        f"<|im_start|>user\n{query} {flag}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n{thinking_content}\n</think>\n"
        f"{response}<|im_end|>"
    )


STOP_THINKING_INSTRUCTION = (
    "Considering the limited time by the user, I have to give the solution "
    "based on the thinking directly now.\n</think>\n\n"
)


@dataclass(frozen=True)
class BudgetedAnswer:
    retained_thinking: tuple[str, ...]
    answer: tuple[str, ...]
    forced_stop: bool
    inserted_instruction: str


def apply_max_thinking_budget(
    thinking_tokens: Sequence[str],
    answer_tokens: Sequence[str],
    *,
    budget: int,
) -> BudgetedAnswer:
    """到达最大 thinking token 阈值后中断并转入回答。

    Qwen3 报告的 budget 机制是最大预算/截断式控制，不是 s1 那种通过忽略
    stop token 来强迫模型继续思考的 scale-up。
    """

    if budget < 0:
        raise ValueError("budget must be non-negative")
    if len(thinking_tokens) <= budget:
        return BudgetedAnswer(tuple(thinking_tokens), tuple(answer_tokens), False, "")
    return BudgetedAnswer(
        tuple(thinking_tokens[:budget]),
        tuple(answer_tokens),
        True,
        STOP_THINKING_INSTRUCTION,
    )


def group_relative_advantages(
    rewards: Sequence[float],
    *,
    epsilon: float = 1e-8,
) -> tuple[float, ...]:
    """GRPO 的组内标准化 reward 教学版。"""

    if not rewards:
        raise ValueError("rewards must not be empty")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    standard_deviation = math.sqrt(variance)
    if standard_deviation < epsilon:
        return tuple(0.0 for _ in rewards)
    return tuple((reward - mean) / (standard_deviation + epsilon) for reward in rewards)


def teacher_student_kl(
    teacher_logits: Sequence[float],
    student_logits: Sequence[float],
    *,
    temperature: float = 1.0,
) -> float:
    """在同一 student-generated prefix 上计算 KL(teacher || student)。"""

    if len(teacher_logits) != len(student_logits) or not teacher_logits:
        raise ValueError("teacher and student logits must have equal non-zero size")
    teacher = softmax(teacher_logits, temperature)
    student = softmax(student_logits, temperature)
    return sum(p * math.log(p / q) for p, q in zip(teacher, student))


def standard_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """无放回组合形式的标准 pass@k 估计器。"""

    if not 0 <= num_correct <= num_samples:
        raise ValueError("num_correct must be between zero and num_samples")
    if not 0 < k <= num_samples:
        raise ValueError("k must be within num_samples")
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


def relative_compute_reduction(baseline_gpu_hours: float, new_gpu_hours: float) -> float:
    if baseline_gpu_hours <= 0 or not 0 <= new_gpu_hours <= baseline_gpu_hours:
        raise ValueError("GPU-hour values are invalid")
    return 1.0 - new_gpu_hours / baseline_gpu_hours


def demo() -> None:
    flagship = next(model for model in QWEN3_2504_MODELS if model.name.endswith("A22B"))
    small_moe = next(model for model in QWEN3_2504_MODELS if model.name.endswith("A3B"))

    gate_logits = tuple((index % 17) / 10 for index in range(128))
    route = top_k_expert_route(gate_logits)
    attention_logit = qk_normalized_logit((2.0, 0.0, 1.0, -1.0), (1.0, 1.0, 0.0, -1.0))

    conversation = (
        Message("system", "Answer directly. /no_think"),
        Message("user", "This turn needs a careful proof. /think"),
    )
    soft_mode = resolve_mode(conversation)
    hard_mode = resolve_mode(conversation, enable_thinking=False)
    non_thinking_target = fusion_training_target(
        "Summarize this paragraph.",
        "A concise summary.",
        mode=ThinkingMode.NON_THINKING,
    )

    budgeted = apply_max_thinking_budget(
        ("plan", "derive", "check", "retry", "verify", "simplify"),
        ("final", "answer"),
        budget=4,
    )
    advantages = group_relative_advantages((0.0, 1.0, 1.0, 0.5))
    distill_kl = teacher_student_kl((2.0, 1.0, 0.0), (1.6, 1.1, 0.1))

    rl_gpu_hours = 17_920.0
    distill_gpu_hours = 1_800.0
    aime_delta = 81.4 - 83.8
    tool_use_delta = 85.5 - 63.3
    ruler_mode_gap = 95.0 - 92.2

    print("Qwen3 disclosed-mechanism arithmetic:")
    print("  released Qwen3-2504 models       =", len(QWEN3_2504_MODELS))
    print(f"  235B-A22B active parameter ratio = {flagship.active_parameter_fraction:.2%}")
    print(f"  30B-A3B active parameter ratio   = {small_moe.active_parameter_fraction:.1%}")
    print(f"  active experts per token         = {flagship.experts_active}/{flagship.experts_total} ({flagship.active_expert_fraction:.2%})")
    print(f"  flagship KV/Q head ratio         = {flagship.kv_heads}/{flagship.query_heads} ({flagship.kv_head_fraction:.2%})")
    print("  selected expert indices          =", route.expert_indices)
    print(f"  QK-normalized example logit      = {attention_logit:.6f}")
    print("  latest soft mode                 =", soft_mode.value)
    print("  hard enable_thinking=False       =", hard_mode.value)
    print("  non-thinking keeps empty block   =", "<think>\n\n</think>" in non_thinking_target)
    print("  retained thinking / forced stop  =", len(budgeted.retained_thinking), "/", budgeted.forced_stop)
    print("  GRPO-style advantages            =", tuple(round(value, 3) for value in advantages))
    print(f"  teacher-student KL               = {distill_kl:.6f}")
    print(f"  RL / on-policy distill GPU-hours = {rl_gpu_hours / distill_gpu_hours:.2f}x")
    print(f"  distillation compute reduction   = {relative_compute_reduction(rl_gpu_hours, distill_gpu_hours):.1%}")
    print(f"  Stage 2 -> 4 AIME / ToolUse      = {aime_delta:+.1f} / {tool_use_delta:+.1f} points")
    print(f"  RULER non-think - think average  = {ruler_mode_gap:+.1f} points")

    assert len(QWEN3_2504_MODELS) == 8
    assert sum(model.architecture == "dense" for model in QWEN3_2504_MODELS) == 6
    assert sum(model.architecture == "moe" for model in QWEN3_2504_MODELS) == 2
    assert flagship.experts_active == 8 and flagship.experts_total == 128
    assert len(route.expert_indices) == 8
    assert math.isclose(sum(route.normalized_weights), 1.0)
    assert soft_mode is ThinkingMode.THINKING
    assert hard_mode is ThinkingMode.NON_THINKING
    assert "<think>\n\n</think>" in non_thinking_target
    assert budgeted.forced_stop and len(budgeted.retained_thinking) == 4
    assert STOP_THINKING_INSTRUCTION in budgeted.inserted_instruction
    assert math.isclose(sum(advantages), 0.0, abs_tol=1e-7)
    assert distill_kl >= 0.0
    assert math.isclose(relative_compute_reduction(17_920, 1_800), 0.8995535714285714)
    assert math.isclose(standard_pass_at_k(10, 2, 1), 0.2)


if __name__ == "__main__":
    demo()
