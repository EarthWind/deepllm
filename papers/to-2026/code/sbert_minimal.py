#!/usr/bin/env python3
"""Sentence-BERT 核心机制的零依赖最小实现。

这不是完整 BERT，也不下载模型。脚本只把论文最容易混淆的部分拆开：

1. CLS / MEAN / MAX 三种 pooling，其中 MEAN 必须忽略 padding；
2. NLI 分类特征 [u, v, |u-v|]、STS 余弦回归与 triplet loss；
3. 同一个 encoder 被所有句子复用（孪生网络共享参数）；
4. 语料离线编码、查询在线编码、余弦 Top-k 检索；
5. Cross-Encoder 与 Bi-Encoder 在全配对场景中的工作量区别。

运行：python3 papers/to-2026/code/sbert_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Sequence


Vector = list[float]
Matrix = list[Vector]


def cls_pool(token_embeddings: Sequence[Sequence[float]]) -> Vector:
    """返回首 token 表示；要求上游确实把首 token 设为 [CLS]。"""

    matrix = _validated_matrix(token_embeddings)
    return list(matrix[0])


def masked_mean_pool(
    token_embeddings: Sequence[Sequence[float]],
    attention_mask: Sequence[int],
) -> Vector:
    """对有效 token 求均值，padding 的 mask 必须为 0。"""

    matrix = _validated_matrix(token_embeddings)
    mask = _validated_mask(attention_mask, len(matrix))
    valid_count = sum(mask)
    if valid_count == 0:
        raise ValueError("attention_mask must retain at least one token")

    dimension = len(matrix[0])
    return [
        sum(row[column] * keep for row, keep in zip(matrix, mask)) / valid_count
        for column in range(dimension)
    ]


def masked_max_pool(
    token_embeddings: Sequence[Sequence[float]],
    attention_mask: Sequence[int],
) -> Vector:
    """逐维取最大值；padding 位置不能参与最大值竞争。"""

    matrix = _validated_matrix(token_embeddings)
    mask = _validated_mask(attention_mask, len(matrix))
    valid_rows = [row for row, keep in zip(matrix, mask) if keep]
    if not valid_rows:
        raise ValueError("attention_mask must retain at least one token")
    return [max(row[column] for row in valid_rows) for column in range(len(matrix[0]))]


def l2_normalize(vector: Sequence[float]) -> Vector:
    """把向量归一化到单位球；零向量没有可定义的方向。"""

    values = _validated_vector(vector)
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return [value / norm for value in values]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """计算余弦相似度，范围为 [-1, 1]。"""

    left_values, right_values = _validated_pair(left, right)
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    dot_product = sum(a * b for a, b in zip(left_values, right_values))
    return dot_product / (left_norm * right_norm)


def euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """计算论文 triplet objective 使用的欧氏距离。"""

    left_values, right_values = _validated_pair(left, right)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left_values, right_values)))


def nli_classification_features(
    left: Sequence[float], right: Sequence[float]
) -> Vector:
    """构造论文 NLI softmax 头的 [u, v, |u-v|]。"""

    left_values, right_values = _validated_pair(left, right)
    difference = [abs(a - b) for a, b in zip(left_values, right_values)]
    return [*left_values, *right_values, *difference]


def softmax(logits: Sequence[float]) -> Vector:
    """数值稳定的 softmax。"""

    values = _validated_vector(logits)
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def cross_entropy_from_logits(logits: Sequence[float], label: int) -> float:
    """单个 NLI 样本的多类交叉熵。"""

    probabilities = softmax(logits)
    if not 0 <= label < len(probabilities):
        raise ValueError("label is outside the logits range")
    return -math.log(probabilities[label])


def sts_regression_loss(
    left: Sequence[float],
    right: Sequence[float],
    gold_score: float,
    *,
    gold_scale: float = 5.0,
) -> float:
    """STS 余弦回归的 MSE；把论文 0–5 标签归一化到 0–1。

    论文写的是余弦与金标的均方误差。工程实现必须先约定同一量纲；
    Sentence Transformers 常见输入是 [0, 1] 浮点标签，因此这里显式除以 5。
    """

    if gold_scale <= 0.0 or not 0.0 <= gold_score <= gold_scale:
        raise ValueError("gold_score must be within [0, gold_scale]")
    target = gold_score / gold_scale
    prediction = cosine_similarity(left, right)
    return (prediction - target) ** 2


def triplet_loss(
    anchor: Sequence[float],
    positive: Sequence[float],
    negative: Sequence[float],
    *,
    margin: float = 1.0,
) -> float:
    """max(||a-p|| - ||a-n|| + margin, 0)。论文使用 margin=1。"""

    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    positive_distance = euclidean_distance(anchor, positive)
    negative_distance = euclidean_distance(anchor, negative)
    return max(positive_distance - negative_distance + margin, 0.0)


@dataclass(frozen=True)
class SearchHit:
    """一个精确余弦检索结果。"""

    index: int
    text: str
    score: float


def semantic_search(
    query_embedding: Sequence[float],
    corpus: Sequence[str],
    corpus_embeddings: Sequence[Sequence[float]],
    *,
    top_k: int = 3,
) -> list[SearchHit]:
    """对已编码语料做精确 Top-k；大规模语料应换成 ANN 索引。"""

    if len(corpus) != len(corpus_embeddings):
        raise ValueError("corpus and corpus_embeddings must have the same length")
    if not corpus:
        return []
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    hits = [
        SearchHit(index, text, cosine_similarity(query_embedding, embedding))
        for index, (text, embedding) in enumerate(zip(corpus, corpus_embeddings))
    ]
    return sorted(hits, key=lambda hit: (-hit.score, hit.index))[:top_k]


def unordered_pair_count(sentence_count: int) -> int:
    """n 个句子的无序两两组合数 n(n-1)/2。"""

    if sentence_count < 0:
        raise ValueError("sentence_count must be non-negative")
    return sentence_count * (sentence_count - 1) // 2


def exact_similarity_cost(sentence_count: int, dimension: int) -> int:
    """精确全配对向量点积的乘加项数；编码之后仍是 O(n²d)。"""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    return unordered_pair_count(sentence_count) * dimension


class ToySharedEncoder:
    """演示“同一参数、独立编码”的微型 encoder。

    词表向量只为让脚本可离线运行，不是训练好的语言模型。所有句子都经过
    同一个 token_table 与同一个 mean pooling，这正是 shared weights 的含义。
    """

    def __init__(self) -> None:
        # 维度依次粗略表示：宠物、休息、地面物、太空、金融。
        self.token_table: dict[str, tuple[float, ...]] = {
            "cat": (1.0, 0.0, 0.0, 0.0, 0.0),
            "kitten": (0.95, 0.0, 0.0, 0.0, 0.0),
            "feline": (0.9, 0.0, 0.0, 0.0, 0.0),
            "rests": (0.0, 1.0, 0.0, 0.0, 0.0),
            "naps": (0.0, 0.95, 0.0, 0.0, 0.0),
            "sleeps": (0.0, 0.9, 0.0, 0.0, 0.0),
            "mat": (0.0, 0.0, 1.0, 0.0, 0.0),
            "rug": (0.0, 0.0, 0.9, 0.0, 0.0),
            "carpet": (0.0, 0.0, 0.95, 0.0, 0.0),
            "rocket": (0.0, 0.0, 0.0, 1.0, 0.0),
            "space": (0.0, 0.0, 0.0, 0.9, 0.0),
            "launches": (0.0, 0.0, 0.0, 0.8, 0.0),
            "stock": (0.0, 0.0, 0.0, 0.0, 1.0),
            "market": (0.0, 0.0, 0.0, 0.0, 0.9),
            "rises": (0.0, 0.0, 0.0, 0.0, 0.7),
        }
        self.unknown = (0.01, 0.01, 0.01, 0.01, 0.01)

    def encode(self, sentence: str) -> Vector:
        tokens = re.findall(r"[a-z]+", sentence.lower())
        if not tokens:
            raise ValueError("sentence must contain at least one alphabetic token")
        token_embeddings = [self.token_table.get(token, self.unknown) for token in tokens]
        pooled = masked_mean_pool(token_embeddings, [1] * len(token_embeddings))
        return l2_normalize(pooled)

    def encode_many(self, sentences: Iterable[str]) -> Matrix:
        return [self.encode(sentence) for sentence in sentences]


def _validated_vector(vector: Sequence[float]) -> Vector:
    if not vector:
        raise ValueError("vector must not be empty")
    values = [float(value) for value in vector]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("vector values must be finite")
    return values


def _validated_pair(
    left: Sequence[float], right: Sequence[float]
) -> tuple[Vector, Vector]:
    left_values = _validated_vector(left)
    right_values = _validated_vector(right)
    if len(left_values) != len(right_values):
        raise ValueError("vectors must have the same dimension")
    return left_values, right_values


def _validated_matrix(matrix: Sequence[Sequence[float]]) -> Matrix:
    if not matrix:
        raise ValueError("token_embeddings must not be empty")
    rows = [_validated_vector(row) for row in matrix]
    dimension = len(rows[0])
    if any(len(row) != dimension for row in rows):
        raise ValueError("all token embeddings must have the same dimension")
    return rows


def _validated_mask(mask: Sequence[int], expected_length: int) -> list[int]:
    if len(mask) != expected_length:
        raise ValueError("attention_mask length must match token count")
    values = list(mask)
    if any(value not in (0, 1) for value in values):
        raise ValueError("attention_mask values must be 0 or 1")
    return values


def _self_check() -> None:
    # padding 的巨大数值不能污染 mean/max pooling。
    tokens = [[1.0, 3.0], [3.0, 1.0], [999.0, 999.0]]
    assert masked_mean_pool(tokens, [1, 1, 0]) == [2.0, 2.0]
    assert masked_max_pool(tokens, [1, 1, 0]) == [3.0, 3.0]
    assert cls_pool(tokens) == [1.0, 3.0]

    assert nli_classification_features([1.0, 2.0], [4.0, 0.0]) == [
        1.0,
        2.0,
        4.0,
        0.0,
        3.0,
        2.0,
    ]
    assert math.isclose(cross_entropy_from_logits([0.0, 2.0, 0.0], 1), 0.2395447662)
    assert math.isclose(sts_regression_loss([1.0, 0.0], [1.0, 0.0], 5.0), 0.0)
    assert math.isclose(triplet_loss([0.0], [0.2], [2.0], margin=1.0), 0.0)

    assert unordered_pair_count(10_000) == 49_995_000
    assert exact_similarity_cost(10_000, 768) == 38_396_160_000


def main() -> None:
    _self_check()

    corpus = [
        "A cat rests on a mat.",
        "A feline sleeps on a rug.",
        "A rocket launches into space.",
        "The stock market rises.",
    ]
    query = "A kitten naps on a carpet."

    # 孪生网络不是两个模型：语料和查询都调用同一个 encoder 实例。
    encoder = ToySharedEncoder()
    corpus_embeddings = encoder.encode_many(corpus)  # 可离线缓存
    query_embedding = encoder.encode(query)  # 查询到来时只编码一次
    hits = semantic_search(query_embedding, corpus, corpus_embeddings, top_k=3)

    print("Sentence-BERT minimal mechanics: self-check passed")
    print(f"query: {query}")
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. cosine={hit.score:.4f} | {hit.text}")

    sentence_count = 10_000
    pair_count = unordered_pair_count(sentence_count)
    print(f"unordered pairs for {sentence_count:,} sentences: {pair_count:,}")
    print(
        "Cross-Encoder needs one expensive pair encoding per pair; "
        "SBERT needs 10,000 expensive sentence encodings, then cheap vector comparisons."
    )


if __name__ == "__main__":
    main()
