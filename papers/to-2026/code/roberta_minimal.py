#!/usr/bin/env python3
"""RoBERTa 训练配方的零依赖最小实现。

这不是一个完整 Transformer，而是把论文中最容易实现错的四件事拆开：

1. 15% MLM 位置与 80% / 10% / 10% 扰动；
2. 每次读取样本时重新生成的动态掩码；
3. 不切断自然句子的 FULL-SENTENCES 打包；
4. 大批量、训练步数与线性 warmup/decay 的预算换算。

运行：python3 papers/to-2026/code/roberta_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Iterable, Sequence


SPECIAL_TOKENS = frozenset({"<s>", "</s>", "<pad>", "<mask>"})


@dataclass(frozen=True)
class MaskedExample:
    """一次 MLM 数据增强的结果；未参与损失的位置以 None 标记。"""

    original: tuple[str, ...]
    corrupted: tuple[str, ...]
    labels: tuple[str | None, ...]
    selected_positions: tuple[int, ...]


@dataclass(frozen=True)
class TrainingBudget:
    """用序列数和最大 token 数比较训练预算。"""

    batch_size: int
    steps: int
    max_sequence_length: int = 512

    @property
    def sequences(self) -> int:
        return self.batch_size * self.steps

    @property
    def max_tokens(self) -> int:
        return self.sequences * self.max_sequence_length


def stable_seed(base_seed: int, sample_id: int, epoch: int) -> int:
    """从样本和 epoch 得到跨进程稳定、可复现的随机种子。"""

    payload = f"{base_seed}:{sample_id}:{epoch}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def corrupt_for_mlm(
    tokens: Sequence[str],
    vocabulary: Sequence[str],
    *,
    seed: int,
    mask_probability: float = 0.15,
) -> MaskedExample:
    """按论文的 15% 与 80/10/10 规则构造一个 MLM 样本。

    为便于教学与测试，这里固定选择 round(候选数 * 15%) 个位置；论文中的
    “均匀选择 15%”在大数据下与之等价。真实训练通常在 collator 中在线完成。
    """

    if not 0.0 < mask_probability <= 1.0:
        raise ValueError("mask_probability must be in (0, 1]")

    candidates = [
        index for index, token in enumerate(tokens) if token not in SPECIAL_TOKENS
    ]
    if not candidates:
        raise ValueError("tokens must contain at least one non-special token")

    replacement_vocabulary = [
        token for token in vocabulary if token not in SPECIAL_TOKENS
    ]
    if not replacement_vocabulary:
        raise ValueError("vocabulary must contain at least one non-special token")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected_count = max(1, round(len(candidates) * mask_probability))
    selected = sorted(candidates[:selected_count])

    corrupted = list(tokens)
    labels: list[str | None] = [None] * len(tokens)
    for index in selected:
        original_token = tokens[index]
        labels[index] = original_token
        draw = rng.random()
        if draw < 0.80:
            corrupted[index] = "<mask>"
        elif draw < 0.90:
            corrupted[index] = rng.choice(replacement_vocabulary)
        # 剩余 10% 保持原 token；标签仍然存在，仍要预测。

    return MaskedExample(
        original=tuple(tokens),
        corrupted=tuple(corrupted),
        labels=tuple(labels),
        selected_positions=tuple(selected),
    )


def dynamic_mask(
    tokens: Sequence[str],
    vocabulary: Sequence[str],
    *,
    sample_id: int,
    epoch: int,
    base_seed: int = 1907,
) -> MaskedExample:
    """RoBERTa 风格：同一样本每次（此处用 epoch 表示）生成新掩码。"""

    return corrupt_for_mlm(
        tokens,
        vocabulary,
        seed=stable_seed(base_seed, sample_id, epoch),
    )


def static_mask(
    tokens: Sequence[str],
    vocabulary: Sequence[str],
    *,
    sample_id: int,
    epoch: int,
    duplicate_count: int = 10,
    base_seed: int = 1907,
) -> MaskedExample:
    """BERT 数据复制近似：预先生成有限个 pattern，再循环复用。"""

    if duplicate_count <= 0:
        raise ValueError("duplicate_count must be positive")
    pattern_id = epoch % duplicate_count
    return corrupt_for_mlm(
        tokens,
        vocabulary,
        seed=stable_seed(base_seed, sample_id, pattern_id),
    )


def pack_full_sentences(
    documents: Sequence[Sequence[Sequence[str]]],
    *,
    max_length: int,
) -> list[list[str]]:
    """把完整自然句连续装箱；允许跨文档，并在文档间加入 </s>。

    每个输出块以 <s> 开头。为突出 FULL-SENTENCES 的关键性质，函数拒绝切开
    超长句子；生产实现通常会先在 tokenizer 层处理这类边界情况。
    """

    if max_length < 3:
        raise ValueError("max_length must be at least 3")

    units: list[tuple[list[str], bool]] = []
    for document_index, document in enumerate(documents):
        for sentence_index, sentence in enumerate(document):
            sentence_tokens = list(sentence)
            if not sentence_tokens:
                continue
            is_new_document = document_index > 0 and sentence_index == 0
            units.append((sentence_tokens, is_new_document))

    packed: list[list[str]] = []
    block = ["<s>"]
    for sentence, is_new_document in units:
        prefix = ["</s>"] if is_new_document else []
        required = len(prefix) + len(sentence)
        if required > max_length - 1:
            raise ValueError("a sentence is too long to preserve as one unit")

        if len(block) + required > max_length:
            packed.append(block)
            block = ["<s>"]
            # 新块已经构成明确边界，无需再放文档分隔符。
            prefix = []

        block.extend(prefix)
        block.extend(sentence)

    if len(block) > 1:
        packed.append(block)
    return packed


def steps_for_equal_sequences(reference: TrainingBudget, new_batch_size: int) -> int:
    """返回看到相同序列数所需的更新步数。"""

    if new_batch_size <= 0:
        raise ValueError("new_batch_size must be positive")
    return math.ceil(reference.sequences / new_batch_size)


def linear_warmup_decay_lr(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
) -> float:
    """RoBERTa 使用的线性 warmup 后线性衰减学习率。"""

    if not 0 <= step <= total_steps:
        raise ValueError("step must be between 0 and total_steps")
    if not 0 < warmup_steps < total_steps:
        raise ValueError("warmup_steps must be inside the training interval")
    if peak_lr <= 0:
        raise ValueError("peak_lr must be positive")

    if step <= warmup_steps:
        return peak_lr * step / warmup_steps
    return peak_lr * (total_steps - step) / (total_steps - warmup_steps)


def visible_labels(labels: Iterable[str | None]) -> str:
    """把只在选中位置存在的 MLM 标签打印成紧凑文本。"""

    return " ".join(token if token is not None else "·" for token in labels)


def run_self_checks() -> None:
    tokens = ["<s>"] + [f"t{i}" for i in range(40)] + ["</s>"]
    vocabulary = [f"v{i}" for i in range(100)]

    example = corrupt_for_mlm(tokens, vocabulary, seed=7)
    assert len(example.selected_positions) == 6  # round(40 * 0.15)
    assert all(example.labels[i] is not None for i in example.selected_positions)
    assert all(
        example.labels[i] is None
        for i in range(len(tokens))
        if i not in example.selected_positions
    )

    # 动态 mask 不被限制为 10 个预存 pattern；静态版本每 10 个 epoch 循环。
    dynamic_patterns = {
        dynamic_mask(tokens, vocabulary, sample_id=3, epoch=epoch).selected_positions
        for epoch in range(12)
    }
    assert len(dynamic_patterns) > 10
    assert (
        static_mask(tokens, vocabulary, sample_id=3, epoch=0)
        == static_mask(tokens, vocabulary, sample_id=3, epoch=10)
    )

    bert_budget = TrainingBudget(batch_size=256, steps=1_000_000)
    assert steps_for_equal_sequences(bert_budget, 2_048) == 125_000
    assert steps_for_equal_sequences(bert_budget, 8_192) == 31_250

    assert linear_warmup_decay_lr(
        0, warmup_steps=30_000, total_steps=500_000, peak_lr=4e-4
    ) == 0.0
    assert linear_warmup_decay_lr(
        30_000, warmup_steps=30_000, total_steps=500_000, peak_lr=4e-4
    ) == 4e-4
    assert linear_warmup_decay_lr(
        500_000, warmup_steps=30_000, total_steps=500_000, peak_lr=4e-4
    ) == 0.0


def main() -> None:
    run_self_checks()

    tokens = (
        "<s> robust pretraining depends on data masking batch size sequence length "
        "and enough optimization steps rather than a new encoder architecture today </s>"
    ).split()
    vocabulary = (
        "model token language corpus dynamic random robust encoder objective "
        "training data masking batch sequence optimization architecture"
    ).split()

    print("Dynamic masking: the same sample receives fresh corruption")
    for epoch in range(3):
        example = dynamic_mask(tokens, vocabulary, sample_id=42, epoch=epoch)
        print(f"epoch {epoch}: {' '.join(example.corrupted)}")
        print(f"labels : {visible_labels(example.labels)}")

    print("\nFULL-SENTENCES packing (max_length=13)")
    documents = [
        [["A", "short", "sentence", "."], ["Another", "one", "."]],
        [["A", "new", "document", "."]],
    ]
    for block in pack_full_sentences(documents, max_length=13):
        print(block)

    bert_budget = TrainingBudget(batch_size=256, steps=1_000_000)
    print("\nEqual sequence budget")
    for batch_size in (256, 2_048, 8_192):
        steps = steps_for_equal_sequences(bert_budget, batch_size)
        print(f"batch={batch_size:>4,} -> steps={steps:>9,}")

    print("\nRoBERTa-large learning-rate schedule")
    for step in (0, 15_000, 30_000, 265_000, 500_000):
        lr = linear_warmup_decay_lr(
            step,
            warmup_steps=30_000,
            total_steps=500_000,
            peak_lr=4e-4,
        )
        print(f"step={step:>7,}: lr={lr:.8f}")


if __name__ == "__main__":
    main()
