#!/usr/bin/env python3
"""BART 去噪预训练机制的零依赖最小实现。

这不是完整 Transformer，而是把论文中最容易混淆的数据与监督逻辑拆开：

1. token masking、token deletion、text infilling、sentence permutation、
   document rotation 五种噪声；
2. 最终 BART 配方：句子打乱 + 30% token 的 span infilling；
3. decoder teacher forcing：输入右移后的原文，标签仍是完整原文；
4. 自回归 causal mask 与重建负对数似然。

运行：python3 papers/to-2026/code/bart_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence


MASK = "<mask>"
BOS = "<s>"


@dataclass(frozen=True)
class InfillingResult:
    """一次 text infilling 的结果。"""

    corrupted: tuple[str, ...]
    masked_token_count: int
    positive_span_lengths: tuple[int, ...]
    zero_length_span_count: int


@dataclass(frozen=True)
class DenoisingExample:
    """BART 预训练时送入 encoder、decoder 和 loss 的三条序列。"""

    original: tuple[str, ...]
    encoder_input: tuple[str, ...]
    decoder_input: tuple[str, ...]
    labels: tuple[str, ...]


def sample_poisson(lam: float, rng: random.Random) -> int:
    """用 Knuth 算法采样 Poisson(lam)，无需 NumPy。"""

    if lam <= 0:
        raise ValueError("lam must be positive")
    threshold = math.exp(-lam)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def token_masking(
    tokens: Sequence[str], *, probability: float = 0.3, seed: int = 0
) -> list[str]:
    """随机挑选固定数量的位置并替换为 <mask>。"""

    count = _noise_count(len(tokens), probability)
    rng = random.Random(seed)
    positions = set(rng.sample(range(len(tokens)), count))
    return [MASK if index in positions else token for index, token in enumerate(tokens)]


def token_deletion(
    tokens: Sequence[str], *, probability: float = 0.3, seed: int = 0
) -> list[str]:
    """随机删除固定数量的 token；长度变化本身也是模型要恢复的信息。"""

    count = _noise_count(len(tokens), probability)
    rng = random.Random(seed)
    positions = set(rng.sample(range(len(tokens)), count))
    return [token for index, token in enumerate(tokens) if index not in positions]


def sentence_permutation(
    sentences: Sequence[Sequence[str]], *, seed: int = 0
) -> list[str]:
    """随机打乱句子，再展平为文档 token。"""

    order = list(range(len(sentences)))
    random.Random(seed).shuffle(order)
    return [token for index in order for token in sentences[index]]


def document_rotation(tokens: Sequence[str], *, seed: int = 0) -> list[str]:
    """均匀选择起点并旋转文档。"""

    if not tokens:
        return []
    pivot = random.Random(seed).randrange(len(tokens))
    return list(tokens[pivot:]) + list(tokens[:pivot])


def text_infilling(
    tokens: Sequence[str],
    *,
    probability: float = 0.3,
    poisson_lambda: float = 3.0,
    seed: int = 0,
) -> InfillingResult:
    """把 Poisson(3) 长度的 span 各压缩为一个 <mask>。

    论文还允许采到长度 0，此时插入一个 <mask>。这里先采样 span 长度直到
    正长度总和覆盖目标 token 数，再随机分配未遮盖 token 形成互不重叠的 span。
    这是便于阅读和测试的等价教学构造，不复刻 fairseq 的张量化边界处理。
    """

    if not tokens:
        raise ValueError("tokens must not be empty")

    target = _noise_count(len(tokens), probability)
    rng = random.Random(seed)
    positive_lengths: list[int] = []
    zero_count = 0
    covered = 0

    while covered < target:
        length = sample_poisson(poisson_lambda, rng)
        if length == 0:
            zero_count += 1
            continue
        length = min(length, target - covered)
        positive_lengths.append(length)
        covered += length

    # 把 n - target 个可见 token 随机分到 span 前后共 k+1 个 gap。
    visible_count = len(tokens) - target
    gaps = _random_composition(visible_count, len(positive_lengths) + 1, rng)

    corrupted: list[str] = []
    cursor = 0
    for span_index, span_length in enumerate(positive_lengths):
        gap_length = gaps[span_index]
        corrupted.extend(tokens[cursor : cursor + gap_length])
        cursor += gap_length
        corrupted.append(MASK)
        cursor += span_length
    corrupted.extend(tokens[cursor : cursor + gaps[-1]])

    # 0-length span 不消费原 token，只在边界插入一个 mask。
    for _ in range(zero_count):
        boundary = rng.randrange(len(corrupted) + 1)
        corrupted.insert(boundary, MASK)

    assert cursor + gaps[-1] == len(tokens)
    return InfillingResult(
        corrupted=tuple(corrupted),
        masked_token_count=target,
        positive_span_lengths=tuple(positive_lengths),
        zero_length_span_count=zero_count,
    )


def build_bart_example(
    sentences: Sequence[Sequence[str]],
    *,
    noise_probability: float = 0.3,
    poisson_lambda: float = 3.0,
    seed: int = 2019,
) -> DenoisingExample:
    """构造最终论文配方的一个预训练样本。"""

    original = [token for sentence in sentences for token in sentence]
    shuffled = sentence_permutation(sentences, seed=seed)
    infilled = text_infilling(
        shuffled,
        probability=noise_probability,
        poisson_lambda=poisson_lambda,
        seed=seed + 1,
    )
    labels = list(original)
    return DenoisingExample(
        original=tuple(original),
        encoder_input=infilled.corrupted,
        decoder_input=tuple(shift_tokens_right(labels)),
        labels=tuple(labels),
    )


def shift_tokens_right(tokens: Sequence[str], *, bos: str = BOS) -> list[str]:
    """teacher forcing：第 t 个 decoder 输入只能含原文的 < t token。"""

    if not tokens:
        raise ValueError("tokens must not be empty")
    return [bos, *tokens[:-1]]


def causal_attention_mask(length: int) -> list[list[float]]:
    """返回加到 attention logits 上的下三角 mask。"""

    if length <= 0:
        raise ValueError("length must be positive")
    return [
        [0.0 if key_index <= query_index else -math.inf for key_index in range(length)]
        for query_index in range(length)
    ]


def reconstruction_nll(correct_token_probabilities: Sequence[float]) -> float:
    """按 token 平均的重建负对数似然。"""

    if not correct_token_probabilities:
        raise ValueError("probabilities must not be empty")
    if any(not 0.0 < probability <= 1.0 for probability in correct_token_probabilities):
        raise ValueError("each probability must be in (0, 1]")
    return -sum(math.log(probability) for probability in correct_token_probabilities) / len(
        correct_token_probabilities
    )


def _noise_count(length: int, probability: float) -> int:
    if length <= 0:
        raise ValueError("tokens must not be empty")
    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must be in (0, 1]")
    return min(length, max(1, round(length * probability)))


def _random_composition(
    total: int, part_count: int, rng: random.Random
) -> list[int]:
    """把非负整数 total 随机拆成 part_count 份，允许 0。"""

    if part_count <= 0:
        raise ValueError("part_count must be positive")
    if part_count == 1:
        return [total]
    separators = sorted(rng.randrange(total + 1) for _ in range(part_count - 1))
    points = [0, *separators, total]
    return [points[index + 1] - points[index] for index in range(part_count)]


def _demo() -> None:
    sentences = [
        ["BART", "reads", "corrupted", "text", "."],
        ["The", "decoder", "reconstructs", "the", "original", "."],
        ["Pretraining", "supports", "many", "tasks", "."],
    ]
    original = [token for sentence in sentences for token in sentence]

    print("=== Five noising functions ===")
    print("original:   ", " ".join(original))
    print("masking:    ", " ".join(token_masking(original, seed=7)))
    print("deletion:   ", " ".join(token_deletion(original, seed=7)))
    print("permutation:", " ".join(sentence_permutation(sentences, seed=7)))
    print("rotation:   ", " ".join(document_rotation(original, seed=7)))
    infilled = text_infilling(original, seed=7)
    print("infilling:  ", " ".join(infilled.corrupted))
    print("span lengths:", infilled.positive_span_lengths)

    print("\n=== Final BART pre-training recipe ===")
    example = build_bart_example(sentences, seed=7)
    print("encoder input:", " ".join(example.encoder_input))
    print("decoder input:", " ".join(example.decoder_input))
    print("labels:       ", " ".join(example.labels))

    # 最关键的不变量：decoder 看右移原文，而不是右移后的噪声文本。
    assert example.decoder_input[0] == BOS
    assert example.decoder_input[1:] == example.labels[:-1]
    assert example.encoder_input != example.labels
    assert sum(infilled.positive_span_lengths) == infilled.masked_token_count
    zero_length_case = text_infilling(list("abcdefghijklmnopqrst"), seed=3)
    assert zero_length_case.zero_length_span_count == 1
    mask = causal_attention_mask(4)
    assert mask[1][1] == 0.0 and mask[1][2] == -math.inf
    assert reconstruction_nll([0.5, 0.25]) == -math.log(0.5 * 0.25) / 2

    print("causal mask:")
    for row in mask:
        print(" ", ["0" if value == 0.0 else "-inf" for value in row])
    print("all invariants passed")


if __name__ == "__main__":
    _demo()
