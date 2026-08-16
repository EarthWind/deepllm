#!/usr/bin/env python3
"""s1《Simple Test-Time Scaling》的零依赖教学实现。

这个文件复现论文中可以脱离真实 32B 模型验证的机制算术：

1. quality / difficulty / diversity 三阶段数据筛选；
2. 域内按推理长度排名并使用 ``2**(-rank)`` 加权抽样；
3. 8-gram 去污染与 completion-only SFT label mask；
4. budget forcing 的“截短思考”和“忽略结束符 + Wait 续写”；
5. Control、Scaling、Performance 三个测试时扩展指标；
6. majority vote、按长度 rejection sampling 与 GPU-hour 对照。

它不是官方训练脚本，不加载 Qwen2.5-32B，也不会生成真实推理。官方代码需要
Transformers、TRL、FSDP/vLLM 和模型权重；这里用可检查的离散对象讲清数据选择、
解码控制和指标。原版 s1 使用 Gemini 轨迹；s1.1 才换用同一批问题的 R1 轨迹。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
from typing import Hashable, Iterable, Sequence


IGNORE_INDEX = -100


@dataclass(frozen=True)
class QuestionTrace:
    """论文 24,496 候选池中一条问题—轨迹的最小特征。"""

    qid: str
    domain: str
    thinking_tokens: int
    api_ok: bool = True
    clean_format: bool = True
    qwen7_correct: bool = False
    qwen32_correct: bool = False


def passes_quality_and_difficulty(example: QuestionTrace) -> bool:
    """保留格式可用、API 成功且两个 Qwen 基线都未解出的样本。

    论文会删除“任一基线模型答对”的题，因此保留条件是两者都答错。
    这里的正确性在原流程中由 Claude 对照参考解答判断。
    """

    if example.thinking_tokens <= 0:
        raise ValueError("thinking_tokens must be positive")
    return (
        example.api_ok
        and example.clean_format
        and not example.qwen7_correct
        and not example.qwen32_correct
    )


def length_rank_probabilities(
    examples: Sequence[QuestionTrace],
) -> tuple[float, ...]:
    """返回与输入顺序对齐的 ``p_i ∝ 2**(-rank_i)``。

    最长轨迹 rank=0、次长 rank=1；同长度时用 qid 稳定打破平局。长度只是
    难度代理，不代表轨迹一定正确，也不保证越长越好。
    """

    if not examples:
        raise ValueError("examples must not be empty")
    if any(example.thinking_tokens <= 0 for example in examples):
        raise ValueError("thinking_tokens must be positive")

    order = sorted(
        range(len(examples)),
        key=lambda index: (-examples[index].thinking_tokens, examples[index].qid),
    )
    ranks = [0] * len(examples)
    for rank, index in enumerate(order):
        ranks[index] = rank
    weights = [2.0 ** (-rank) for rank in ranks]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def _weighted_index(probabilities: Sequence[float], rng: random.Random) -> int:
    point = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if point < cumulative:
            return index
    return len(probabilities) - 1


def curate_subset(
    pool: Sequence[QuestionTrace],
    *,
    target_size: int,
    seed: int = 0,
    preselected_ids: Iterable[str] = (),
) -> tuple[QuestionTrace, ...]:
    """论文 Algorithm 1 的小规模版本：均匀选域，域内偏向长轨迹。

    ``preselected_ids`` 对应论文先加入的高质量 AIME / GPQA，以及足够长且
    正确的 MATH 轨迹。真实 s1K 目标为 1,000；demo 使用更小的池。
    """

    if target_size <= 0:
        raise ValueError("target_size must be positive")
    eligible = [example for example in pool if passes_quality_and_difficulty(example)]
    if target_size > len(eligible):
        raise ValueError("target_size exceeds eligible pool")

    by_id = {example.qid: example for example in eligible}
    if len(by_id) != len(eligible):
        raise ValueError("qids must be unique")

    selected: list[QuestionTrace] = []
    selected_ids: set[str] = set()
    for qid in preselected_ids:
        if qid not in by_id:
            raise ValueError(f"preselected id is not eligible: {qid}")
        if qid not in selected_ids:
            selected.append(by_id[qid])
            selected_ids.add(qid)
    if len(selected) > target_size:
        raise ValueError("too many preselected examples")

    remaining: dict[str, list[QuestionTrace]] = {}
    for example in eligible:
        if example.qid not in selected_ids:
            remaining.setdefault(example.domain, []).append(example)

    rng = random.Random(seed)
    while len(selected) < target_size:
        domains = sorted(domain for domain, values in remaining.items() if values)
        if not domains:
            raise RuntimeError("candidate pool exhausted")
        domain = rng.choice(domains)  # 域均匀，而非按域大小采样
        candidates = remaining[domain]
        probabilities = length_rank_probabilities(candidates)
        index = _weighted_index(probabilities, rng)
        chosen = candidates.pop(index)
        selected.append(chosen)
        selected_ids.add(chosen.qid)
    return tuple(selected)


def word_ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    """论文去污染规则的教学版 word n-gram 集合。"""

    if n <= 0:
        raise ValueError("n must be positive")
    words = text.lower().split()
    return {tuple(words[index : index + n]) for index in range(len(words) - n + 1)}


def has_ngram_overlap(train_question: str, eval_question: str, n: int = 8) -> bool:
    """两道题是否共享至少一个 n-gram。"""

    return bool(word_ngrams(train_question, n) & word_ngrams(eval_question, n))


def completion_only_labels(
    token_ids: Sequence[int],
    *,
    assistant_start: int,
) -> tuple[int, ...]:
    """只对 assistant 的 reasoning + answer 计算 next-token loss。

    官方 TRL collator 把 user/question 一侧设为 -100。这里直接展示 mask 后的
    label；``assistant_start`` 是第一个 assistant token 的位置。
    """

    if not 0 <= assistant_start <= len(token_ids):
        raise ValueError("assistant_start is outside token_ids")
    return tuple(
        IGNORE_INDEX if index < assistant_start else token
        for index, token in enumerate(token_ids)
    )


@dataclass(frozen=True)
class ReasoningPass:
    """模型在一次尝试结束思考符之前产生的抽象 token。"""

    tokens: tuple[str, ...]


@dataclass(frozen=True)
class BudgetResult:
    thinking_tokens: tuple[str, ...]
    ignored_end_markers: int
    forced_exit: bool
    answer_prefix: tuple[str, ...]

    @property
    def token_count(self) -> int:
        return len(self.thinking_tokens)


def budget_force(
    passes: Sequence[ReasoningPass],
    *,
    max_thinking_tokens: int,
    ignore_end_times: int = 0,
    continuation: tuple[str, ...] = ("Wait",),
    answer_prefix: tuple[str, ...] = ("<END_THINK>", "Final", "Answer:"),
) -> BudgetResult:
    """模拟 s1 的两种解码干预。

    - 到达 ``max_thinking_tokens``：截断思考并转入 answer（scale down）；
    - 模型尝试结束但仍有 ignore 配额：移除结束符、追加 ``Wait`` 并继续
      下一段（scale up）。

    ``passes`` 是脚本化模型输出，不是真实语言模型；真实实现需要在每次停止后
    把旧 token 和 continuation 重新送入解码器。
    """

    if max_thinking_tokens <= 0:
        raise ValueError("max_thinking_tokens must be positive")
    if ignore_end_times < 0:
        raise ValueError("ignore_end_times must be non-negative")
    if not passes:
        raise ValueError("passes must not be empty")

    output: list[str] = []
    ignored = 0
    forced_exit = False

    for reasoning_pass in passes:
        remaining = max_thinking_tokens - len(output)
        if remaining <= 0:
            forced_exit = True
            break

        take = reasoning_pass.tokens[:remaining]
        output.extend(take)
        if len(take) < len(reasoning_pass.tokens):
            forced_exit = True
            break

        # 该 pass 自然尝试产生 END_THINK。若仍有配额，就压掉它并续写。
        if ignored < ignore_end_times:
            remaining = max_thinking_tokens - len(output)
            if remaining < len(continuation):
                forced_exit = True
                break
            output.extend(continuation)
            ignored += 1
            continue
        break
    else:
        # 脚本已经耗尽，但调用者还要求继续；真实模型可能循环或耗尽上下文。
        forced_exit = ignored < ignore_end_times

    return BudgetResult(tuple(output), ignored, forced_exit, answer_prefix)


def control_rate(
    observed_compute: Sequence[float],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
) -> float:
    """论文公式 (1)：落在目标 compute 区间内的运行比例。"""

    size = len(observed_compute)
    if size == 0 or len(lower_bounds) != size or len(upper_bounds) != size:
        raise ValueError("all inputs must have equal non-zero size")
    if any(low > high for low, high in zip(lower_bounds, upper_bounds)):
        raise ValueError("lower bound exceeds upper bound")
    hits = sum(
        low <= observed <= high
        for observed, low, high in zip(observed_compute, lower_bounds, upper_bounds)
    )
    return hits / size


def average_pairwise_scaling(
    compute: Sequence[float],
    performance: Sequence[float],
) -> float:
    """论文公式 (2)：所有 compute 点对之间斜率的平均值。"""

    if len(compute) < 2 or len(compute) != len(performance):
        raise ValueError("need at least two aligned points")
    pairs = sorted(zip(compute, performance))
    if len({point[0] for point in pairs}) != len(pairs):
        raise ValueError("compute values must be unique")
    slopes = [
        (pairs[right][1] - pairs[left][1])
        / (pairs[right][0] - pairs[left][0])
        for left in range(len(pairs))
        for right in range(left + 1, len(pairs))
    ]
    return sum(slopes) / len(slopes)


def peak_performance(performance: Sequence[float]) -> float:
    """论文公式 (3)：观测 operating points 中的最佳成绩。"""

    if not performance:
        raise ValueError("performance must not be empty")
    return max(performance)


def majority_vote(answers: Sequence[Hashable]) -> Hashable:
    """并行扩展基线；平票时按首次出现顺序稳定选择。"""

    if not answers:
        raise ValueError("answers must not be empty")
    counts = Counter(answers)
    best_count = max(counts.values())
    return next(answer for answer in answers if counts[answer] == best_count)


def rejection_sample_by_length(
    sampled_lengths: Sequence[int],
    *,
    maximum: int,
) -> tuple[int, int]:
    """返回首个不超预算的长度与所需尝试次数。"""

    if maximum <= 0:
        raise ValueError("maximum must be positive")
    for trials, length in enumerate(sampled_lengths, start=1):
        if length <= maximum:
            return length, trials
    raise ValueError("no sampled generation fits the budget")


def gpu_hours(num_gpus: int, minutes: float) -> float:
    if num_gpus <= 0 or minutes < 0:
        raise ValueError("invalid compute values")
    return num_gpus * minutes / 60.0


def relative_improvement(new: float, baseline: float) -> float:
    if baseline == 0:
        raise ValueError("baseline must be non-zero")
    return (new - baseline) / baseline


def demo() -> None:
    pool = (
        QuestionTrace("geo-long", "geometry", 6_200),
        QuestionTrace("geo-short", "geometry", 2_100),
        QuestionTrace("nt-long", "number-theory", 5_900),
        QuestionTrace("nt-mid", "number-theory", 3_800),
        QuestionTrace("bio-long", "biology", 5_100),
        QuestionTrace("bio-mid", "biology", 3_300),
        QuestionTrace("easy-7b", "geometry", 4_500, qwen7_correct=True),
        QuestionTrace("easy-32b", "biology", 4_800, qwen32_correct=True),
        QuestionTrace("broken", "number-theory", 7_000, clean_format=False),
    )
    selected = curate_subset(
        pool,
        target_size=4,
        seed=7,
        preselected_ids=("geo-long",),
    )

    labels = completion_only_labels((11, 12, 21, 22, 23), assistant_start=2)
    overlap = has_ngram_overlap(
        "find the number of positive integers n below this exact upper bound",
        "please find the number of positive integers n below this exact upper bound now",
    )

    scripted_reasoning = (
        ReasoningPass(("assume", "answer=2")),
        ReasoningPass(("check", "parity", "contradiction")),
        ReasoningPass(("redo", "derive", "answer=3")),
    )
    natural = budget_force(scripted_reasoning, max_thinking_tokens=20)
    extended = budget_force(
        scripted_reasoning,
        max_thinking_tokens=20,
        ignore_end_times=2,
    )
    shortened = budget_force(
        (ReasoningPass(tuple(f"t{i}" for i in range(10))),),
        max_thinking_tokens=4,
    )

    compute = (2_000.0, 6_100.0, 7_320.0)
    accuracy = (0.30, 0.50, 0.567)
    control = control_rate(
        observed_compute=(1_000, 2_100, 4_500),
        lower_bounds=(0, 0, 0),
        upper_bounds=(1_024, 2_048, 4_096),
    )
    slope = average_pairwise_scaling(compute, accuracy)
    accepted_length, trials = rejection_sample_by_length(
        (8_500, 7_200, 3_900),
        maximum=4_000,
    )

    print("s1 disclosed-mechanism arithmetic:")
    print("  curated ids                    =", tuple(x.qid for x in selected))
    print("  completion-only labels         =", labels)
    print("  8-gram overlap                 =", overlap)
    print("  natural / extended tokens      =", natural.token_count, "/", extended.token_count)
    print("  ignored end markers            =", extended.ignored_end_markers)
    print("  shortened / forced exit        =", shortened.token_count, "/", shortened.forced_exit)
    print(f"  control                         = {control:.1%}")
    print(f"  average pairwise scaling        = {slope:.8f} accuracy/token")
    print(f"  peak performance                = {peak_performance(accuracy):.1%}")
    print("  rejection sample length / tries =", accepted_length, "/", trials)
    print(f"  16 H100 × 26 min                = {gpu_hours(16, 26):.2f} GPU-hours")
    print(f"  AIME 56.7 vs 44.6 relative gain = {relative_improvement(56.7, 44.6):.1%}")

    assert all(passes_quality_and_difficulty(x) for x in selected)
    assert len({x.qid for x in selected}) == 4
    assert labels == (IGNORE_INDEX, IGNORE_INDEX, 21, 22, 23)
    assert overlap
    assert natural.token_count == 2
    assert extended.thinking_tokens == (
        "assume",
        "answer=2",
        "Wait",
        "check",
        "parity",
        "contradiction",
        "Wait",
        "redo",
        "derive",
        "answer=3",
    )
    assert shortened.forced_exit and shortened.token_count == 4
    assert math.isclose(control, 1 / 3)
    assert majority_vote(("A", "B", "A", "C")) == "A"
    assert math.isclose(gpu_hours(16, 26), 104 / 15)


if __name__ == "__main__":
    demo()
