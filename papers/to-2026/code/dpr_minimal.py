#!/usr/bin/env python3
"""DPR（Dense Passage Retrieval）核心机制的零依赖最小实现。

这不是完整 BERT，也不下载 Wikipedia。脚本专门隔离论文中最容易混淆的机制：

1. 问题 Encoder 与段落 Encoder 参数独立，而不是 Sentence-BERT 式共享权重；
2. 用 [CLS] 风格的定长向量和未归一化内积给问题—段落打分；
3. 一个 B×B 相似度矩阵把 batch 内其他正段落复用为负例；
4. 每个问题再加入一个 BM25 hard negative 时，候选扩展到 2B；
5. 段落离线编码、查询在线编码、Maximum Inner Product Search；
6. Top-k retrieval accuracy 检查“前 k 个段落中是否出现答案”。

运行：python3 papers/to-2026/code/dpr_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import string
from typing import Iterable, Sequence


Vector = list[float]
Matrix = list[Vector]


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    """DPR 论文默认相似度：E_Q(q)^T E_P(p)。"""

    left_values, right_values = _validated_pair(left, right)
    return sum(a * b for a, b in zip(left_values, right_values))


def l2_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """欧氏距离；论文消融发现其表现与内积接近。"""

    left_values, right_values = _validated_pair(left, right)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left_values, right_values)))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """余弦仅供消融对照；原论文最终系统不先归一化向量。"""

    left_values, right_values = _validated_pair(left, right)
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return dot_product(left_values, right_values) / (left_norm * right_norm)


def similarity_matrix(
    question_embeddings: Sequence[Sequence[float]],
    passage_embeddings: Sequence[Sequence[float]],
) -> Matrix:
    """计算 QP^T；行是问题，列是候选段落。"""

    questions = _validated_matrix(question_embeddings, "question_embeddings")
    passages = _validated_matrix(passage_embeddings, "passage_embeddings")
    if len(questions[0]) != len(passages[0]):
        raise ValueError("question and passage embeddings need the same dimension")
    return [[dot_product(question, passage) for passage in passages] for question in questions]


def logsumexp(values: Sequence[float]) -> float:
    """数值稳定的 log(sum(exp(values)))。"""

    logits = _validated_vector(values)
    maximum = max(logits)
    return maximum + math.log(sum(math.exp(value - maximum) for value in logits))


@dataclass(frozen=True)
class BatchLoss:
    """一次 in-batch NLL 的可检查结果。"""

    loss: float
    scores: tuple[tuple[float, ...], ...]
    positive_columns: tuple[int, ...]
    candidate_count: int
    negative_pair_count: int


def in_batch_nll(
    question_embeddings: Sequence[Sequence[float]],
    positive_passage_embeddings: Sequence[Sequence[float]],
    hard_negative_embeddings: Sequence[Sequence[float]] | None = None,
) -> BatchLoss:
    """计算 DPR 的 batch 内负例 softmax NLL。

    第 i 个问题与第 i 个 positive passage 对应，因此 B×B 分数矩阵的对角线
    是正例，其余 B(B-1) 项是 batch 内负例。若给出每题一个 hard negative，
    它们被追加为 B 个候选，并同样供 batch 中所有问题比较。
    """

    questions = _validated_matrix(question_embeddings, "question_embeddings")
    positives = _validated_matrix(
        positive_passage_embeddings, "positive_passage_embeddings"
    )
    if len(questions) != len(positives):
        raise ValueError("each question needs exactly one positive passage")

    candidates = list(positives)
    if hard_negative_embeddings is not None:
        hard_negatives = _validated_matrix(
            hard_negative_embeddings, "hard_negative_embeddings"
        )
        if len(hard_negatives) != len(questions):
            raise ValueError("this demo expects one hard negative per question")
        candidates.extend(hard_negatives)

    scores = similarity_matrix(questions, candidates)
    positive_columns = tuple(range(len(questions)))
    per_question_losses = [
        logsumexp(row) - row[positive_column]
        for row, positive_column in zip(scores, positive_columns)
    ]
    total_pairs = len(questions) * len(candidates)
    positive_pairs = len(questions)
    return BatchLoss(
        loss=sum(per_question_losses) / len(per_question_losses),
        scores=tuple(tuple(row) for row in scores),
        positive_columns=positive_columns,
        candidate_count=len(candidates),
        negative_pair_count=total_pairs - positive_pairs,
    )


@dataclass(frozen=True)
class PairCounts:
    """一个 batch 内实际参与 softmax 的配对数量。"""

    batch_size: int
    candidate_passages: int
    total_pairs: int
    positive_pairs: int
    negative_pairs: int


def in_batch_pair_counts(batch_size: int, *, hard_negatives_per_question: int = 0) -> PairCounts:
    """计算 batch 内复用正段落和 hard negatives 后的配对规模。"""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if hard_negatives_per_question < 0:
        raise ValueError("hard_negatives_per_question must be non-negative")
    candidate_passages = batch_size * (1 + hard_negatives_per_question)
    total_pairs = batch_size * candidate_passages
    return PairCounts(
        batch_size=batch_size,
        candidate_passages=candidate_passages,
        total_pairs=total_pairs,
        positive_pairs=batch_size,
        negative_pairs=total_pairs - batch_size,
    )


@dataclass(frozen=True)
class Passage:
    """检索索引中的基本单元。"""

    passage_id: str
    title: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    """一个精确 MIPS 结果。"""

    rank: int
    passage: Passage
    score: float


def maximum_inner_product_search(
    question_embedding: Sequence[float],
    passages: Sequence[Passage],
    passage_embeddings: Sequence[Sequence[float]],
    *,
    top_k: int,
) -> list[SearchHit]:
    """用穷举实现精确 MIPS；大语料应替换为 FAISS 等索引。"""

    if len(passages) != len(passage_embeddings):
        raise ValueError("passages and passage_embeddings must have the same length")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    scored = [
        (dot_product(question_embedding, embedding), index, passage)
        for index, (passage, embedding) in enumerate(zip(passages, passage_embeddings))
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        SearchHit(rank=rank, passage=passage, score=score)
        for rank, (score, _, passage) in enumerate(scored[:top_k], start=1)
    ]


def hybrid_score(bm25_score: float, dpr_score: float, *, weight: float = 1.1) -> float:
    """论文 BM25+DPR 的线性组合：BM25 + lambda * DPR。"""

    if not all(math.isfinite(value) for value in (bm25_score, dpr_score, weight)):
        raise ValueError("scores and weight must be finite")
    return bm25_score + weight * dpr_score


def split_into_passages(
    title: str,
    words: Sequence[str],
    *,
    words_per_passage: int = 100,
) -> list[Passage]:
    """按论文方式切成不重叠、固定词数的 passages。"""

    if words_per_passage <= 0:
        raise ValueError("words_per_passage must be positive")
    if not title.strip():
        raise ValueError("title must not be empty")
    return [
        Passage(
            passage_id=f"{_slug(title)}-{offset // words_per_passage}",
            title=title,
            text=" ".join(words[offset : offset + words_per_passage]),
        )
        for offset in range(0, len(words), words_per_passage)
        if words[offset : offset + words_per_passage]
    ]


def normalize_answer(text: str) -> str:
    """用于教学评价的 SQuAD/DrQA 风格答案规范化。"""

    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def passage_contains_answer(passage: Passage, answers: Sequence[str]) -> bool:
    """检查规范化后的任一答案字符串是否出现在段落中。"""

    haystack = normalize_answer(f"{passage.title} {passage.text}")
    return any(
        normalized and normalized in haystack
        for answer in answers
        if (normalized := normalize_answer(answer))
    )


def top_k_retrieval_accuracy(
    ranked_passages: Sequence[Sequence[Passage]],
    gold_answers: Sequence[Sequence[str]],
    *,
    top_k: int,
) -> float:
    """问题中有多少比例能在前 k 个段落里找到至少一个答案字符串。"""

    if len(ranked_passages) != len(gold_answers):
        raise ValueError("ranked passages and gold answers must align by question")
    if not ranked_passages:
        raise ValueError("at least one question is required")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    hits = sum(
        any(passage_contains_answer(passage, answers) for passage in passages[:top_k])
        for passages, answers in zip(ranked_passages, gold_answers)
    )
    return hits / len(ranked_passages)


class ToyQuestionEncoder:
    """问题侧的独立教学参数；真实 DPR 使用 BERT-base [CLS]。"""

    def __init__(self) -> None:
        # 维度粗略表示：反派、指环王、人物、地理、水域、金融。
        self.parameters: dict[str, tuple[float, ...]] = {
            "bad": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "guy": (0.9, 0.0, 0.0, 0.0, 0.0, 0.0),
            "villain": (1.1, 0.0, 0.0, 0.0, 0.0, 0.0),
            "lord": (0.0, 0.8, 0.0, 0.0, 0.0, 0.0),
            "rings": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            "who": (0.0, 0.0, 0.7, 0.0, 0.0, 0.0),
            "where": (0.0, 0.0, 0.0, 0.8, 0.0, 0.0),
            "water": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            "stock": (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        }

    def encode(self, question: str) -> Vector:
        return _bag_encoder(question, self.parameters)


class ToyPassageEncoder:
    """段落侧的另一套独立参数，刻意与问题词表不同。"""

    def __init__(self) -> None:
        self.parameters: dict[str, tuple[float, ...]] = {
            "sauron": (1.1, 0.5, 0.5, 0.0, 0.0, 0.0),
            "villain": (1.2, 0.0, 0.0, 0.0, 0.0, 0.0),
            "portraying": (0.3, 0.0, 0.8, 0.0, 0.0, 0.0),
            "rings": (0.0, 1.2, 0.0, 0.0, 0.0, 0.0),
            "sea": (0.0, 0.0, 0.0, 0.2, 1.1, 0.0),
            "channel": (0.0, 0.0, 0.0, 0.2, 0.9, 0.0),
            "market": (0.0, 0.0, 0.0, 0.0, 0.0, 1.2),
            "shares": (0.0, 0.0, 0.0, 0.0, 0.0, 0.9),
            "football": (0.0, 0.0, 0.2, 0.0, 0.0, 0.0),
        }

    def encode(self, passage: Passage) -> Vector:
        # 论文把 title 与 text 一起送入 passage encoder。
        return _bag_encoder(f"{passage.title} {passage.text}", self.parameters)

    def encode_many(self, passages: Iterable[Passage]) -> Matrix:
        return [self.encode(passage) for passage in passages]


def _bag_encoder(
    text: str, parameters: dict[str, tuple[float, ...]], *, dimension: int = 6
) -> Vector:
    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens:
        raise ValueError("text needs at least one alphabetic token")
    result = [0.0] * dimension
    for token in tokens:
        for index, value in enumerate(parameters.get(token, ())):
            result[index] += value
    # 极小偏置避免无已知词文本退化成零向量；不做 L2 normalize。
    return [value + 0.01 for value in result]


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "passage"


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
        raise ValueError("vectors need the same dimension")
    return left_values, right_values


def _validated_matrix(matrix: Sequence[Sequence[float]], name: str) -> Matrix:
    if not matrix:
        raise ValueError(f"{name} must not be empty")
    rows = [_validated_vector(row) for row in matrix]
    dimension = len(rows[0])
    if any(len(row) != dimension for row in rows):
        raise ValueError(f"all rows in {name} need the same dimension")
    return rows


def _self_check() -> None:
    assert dot_product([1.0, 2.0], [3.0, 4.0]) == 11.0
    assert math.isclose(l2_distance([0.0, 0.0], [3.0, 4.0]), 5.0)
    assert math.isclose(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)

    questions = [[2.0, 0.0], [0.0, 2.0]]
    positives = [[2.0, 0.0], [0.0, 2.0]]
    easy = in_batch_nll(questions, positives)
    assert easy.positive_columns == (0, 1)
    assert easy.candidate_count == 2
    assert easy.negative_pair_count == 2
    assert easy.loss < 0.1

    paper_batch = in_batch_pair_counts(128, hard_negatives_per_question=1)
    assert paper_batch.candidate_passages == 256
    assert paper_batch.total_pairs == 32_768
    assert paper_batch.negative_pairs == 32_640

    chunks = split_into_passages("Demo", [str(i) for i in range(205)])
    assert [len(chunk.text.split()) for chunk in chunks] == [100, 100, 5]

    passages = [
        Passage("p0", "Middle-earth", "Sala Baker portrayed the villain Sauron in the Rings trilogy."),
        Passage("p1", "Finance", "Market shares rose after the quarterly report."),
    ]
    assert passage_contains_answer(passages[0], ["Sauron"])
    assert math.isclose(top_k_retrieval_accuracy([passages], [["Sauron"]], top_k=1), 1.0)


def main() -> None:
    _self_check()

    passages = [
        Passage(
            "p0",
            "The Lord of the Rings",
            "Sala Baker is best known for portraying the villain Sauron in the Rings trilogy.",
        ),
        Passage(
            "p1",
            "English Channel",
            "The channel is a body of water connecting the southern North Sea to the Atlantic.",
        ),
        Passage(
            "p2",
            "Financial market",
            "Market shares rose after the quarterly report.",
        ),
        Passage(
            "p3",
            "Association football",
            "Football is played between two teams on a rectangular field.",
        ),
    ]
    question = "Who is the bad guy in Lord of the Rings?"

    # DPR 的两塔是两个不同对象、两套参数。
    question_encoder = ToyQuestionEncoder()
    passage_encoder = ToyPassageEncoder()

    # 离线：先编码完整 passage collection。
    passage_embeddings = passage_encoder.encode_many(passages)
    # 在线：每个新问题只编码一次，再做 MIPS。
    question_embedding = question_encoder.encode(question)
    hits = maximum_inner_product_search(
        question_embedding,
        passages,
        passage_embeddings,
        top_k=3,
    )

    print("DPR minimal mechanics: self-check passed")
    print(f"question: {question}")
    for hit in hits:
        answer_flag = passage_contains_answer(hit.passage, ["Sauron"])
        print(
            f"{hit.rank}. dot={hit.score:.4f} answer={str(answer_flag).lower()} "
            f"| {hit.passage.title}"
        )

    counts = in_batch_pair_counts(128, hard_negatives_per_question=1)
    print(
        "batch=128 + one BM25 hard negative/question: "
        f"{counts.candidate_passages} candidate passages, "
        f"{counts.total_pairs:,} scores, {counts.negative_pairs:,} negative pairs"
    )


if __name__ == "__main__":
    main()
