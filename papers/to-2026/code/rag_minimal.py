#!/usr/bin/env python3
"""A dependency-free RAG teaching demo.

This file intentionally separates two ideas:

1. ``MinimalRAG`` demonstrates the engineering pipeline:
   chunk -> retrieve -> build prompt -> generate.
2. ``rag_sequence_probability`` and ``rag_token_probability`` demonstrate the
   latent-document marginalization proposed in the 2020 RAG paper.

The retriever is BM25 and the default "generator" extracts one evidence
sentence, so this script can run without downloading a model or configuring an
API key. Replace ``extractive_generator`` with an LLM call in a real system.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？!?])\s*")


def tokenize(text: str) -> list[str]:
    """Tokenize English by word and Chinese by character for this tiny demo."""

    return [token.lower() for token in TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class Document:
    source: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    source: str
    title: str
    text: str
    chunk_id: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    probability: float


def chunk_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[Chunk]:
    """Split documents into overlapping character windows.

    Production systems usually split with tokenizer-aware lengths and preserve
    headings, tables, code blocks, and parent-document relationships.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for document in documents:
        text = re.sub(r"\s+", " ", document.text).strip()
        for chunk_id, start in enumerate(range(0, len(text), step)):
            content = text[start : start + chunk_size].strip()
            if content:
                chunks.append(
                    Chunk(
                        source=document.source,
                        title=document.title,
                        text=content,
                        chunk_id=chunk_id,
                    )
                )
            if start + chunk_size >= len(text):
                break
    return chunks


class BM25Retriever:
    """Small BM25 implementation used only to make the example runnable."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("chunks cannot be empty")

        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.lengths = [sum(frequencies.values()) for frequencies in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths)

        document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequency.update(frequencies.keys())
        self.document_frequency = document_frequency

    def _idf(self, term: str) -> float:
        document_count = len(self.chunks)
        frequency = self.document_frequency.get(term, 0)
        return math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))

    def _score(self, query_terms: set[str], chunk_index: int) -> float:
        frequencies = self.term_frequencies[chunk_index]
        length = self.lengths[chunk_index]
        length_normalizer = self.k1 * (
            1.0 - self.b + self.b * length / self.average_length
        )

        score = 0.0
        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if term_frequency:
                score += self._idf(term) * (
                    term_frequency * (self.k1 + 1.0)
                    / (term_frequency + length_normalizer)
                )
        return score

    def retrieve(self, query: str, *, top_k: int = 3) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_terms = set(tokenize(query))
        ranked = sorted(
            (
                (self._score(query_terms, index), index)
                for index in range(len(self.chunks))
            ),
            reverse=True,
        )[:top_k]

        # Convert retrieval scores to a top-k probability distribution. This is
        # analogous to p_eta(z | x), although the paper uses DPR inner products.
        max_score = ranked[0][0]
        unnormalized = [math.exp(score - max_score) for score, _ in ranked]
        denominator = sum(unnormalized)

        return [
            RetrievedChunk(
                chunk=self.chunks[index],
                score=score,
                probability=weight / denominator,
            )
            for (score, index), weight in zip(ranked, unnormalized)
        ]


def build_prompt(query: str, evidence: Sequence[RetrievedChunk]) -> str:
    context = "\n\n".join(
        (
            f"[E{index}] {item.chunk.title} "
            f"(source={item.chunk.source}, chunk={item.chunk.chunk_id})\n"
            f"{item.chunk.text}"
        )
        for index, item in enumerate(evidence, start=1)
    )
    return f"""你是一个严格依据证据回答问题的助手。
如果证据不足，请明确回答“现有证据不足”，不要补写常识。
回答中的事实后必须给出 [E1] 形式的引用。

问题：
{query}

证据：
{context}

答案："""


Generator = Callable[[str, str, Sequence[RetrievedChunk]], str]


def extractive_generator(
    query: str,
    _prompt: str,
    evidence: Sequence[RetrievedChunk],
) -> str:
    """Pick the most query-relevant sentence as a deterministic stand-in."""

    query_terms = set(tokenize(query))
    candidates: list[tuple[int, int, str]] = []
    for evidence_index, item in enumerate(evidence, start=1):
        sentences = SENTENCE_BOUNDARY_RE.split(item.chunk.text)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                overlap = len(query_terms & set(tokenize(sentence)))
                candidates.append((overlap, -evidence_index, sentence))

    if not candidates or max(candidates)[0] == 0:
        return "现有证据不足。"

    _, negative_index, sentence = max(candidates)
    return f"{sentence} [E{-negative_index}]"


class MinimalRAG:
    def __init__(self, retriever: BM25Retriever, generator: Generator) -> None:
        self.retriever = retriever
        self.generator = generator

    def answer(
        self,
        query: str,
        *,
        top_k: int = 3,
    ) -> tuple[str, list[RetrievedChunk], str]:
        evidence = self.retriever.retrieve(query, top_k=top_k)
        prompt = build_prompt(query, evidence)
        answer = self.generator(query, prompt, evidence)
        return answer, evidence, prompt


def rag_sequence_probability(
    document_probabilities: Sequence[float],
    token_probabilities_by_document: Sequence[Sequence[float]],
) -> float:
    """Compute sum_z p(z|x) * product_t p(y_t|x,z,y_<t)."""

    return sum(
        document_probability * math.prod(token_probabilities)
        for document_probability, token_probabilities in zip(
            document_probabilities,
            token_probabilities_by_document,
        )
    )


def rag_token_probability(
    document_probabilities: Sequence[float],
    token_probabilities_by_document: Sequence[Sequence[float]],
) -> float:
    """Compute product_t sum_z p(z|x) * p(y_t|x,z,y_<t)."""

    token_count = len(token_probabilities_by_document[0])
    return math.prod(
        sum(
            document_probability
            * token_probabilities_by_document[document_index][token_index]
            for document_index, document_probability in enumerate(
                document_probabilities
            )
        )
        for token_index in range(token_count)
    )


def probability_demo() -> None:
    document_probabilities = [0.7, 0.3]
    token_probabilities = [
        [0.9, 0.2],  # Document 1 strongly supports token 1.
        [0.3, 0.8],  # Document 2 strongly supports token 2.
    ]

    sequence_probability = rag_sequence_probability(
        document_probabilities,
        token_probabilities,
    )
    token_probability = rag_token_probability(
        document_probabilities,
        token_probabilities,
    )

    assert math.isclose(sequence_probability, 0.198)
    assert math.isclose(token_probability, 0.2736)
    print(f"RAG-Sequence probability: {sequence_probability:.4f}")
    print(f"RAG-Token probability:    {token_probability:.4f}")


def pipeline_demo() -> None:
    documents = [
        Document(
            source="rag-paper",
            title="RAG 的两类记忆",
            text=(
                "RAG 把生成模型参数中的知识称为参数化记忆，把可检索的文档索引称为"
                "非参数化记忆。原论文使用 DPR 检索器和 BART 生成器。部署时可以替换"
                "非参数化记忆索引，在不重新训练整个生成模型的情况下更新可检索知识。"
            ),
        ),
        Document(
            source="dpr-notes",
            title="DPR 检索",
            text=(
                "DPR 使用两个 BERT 编码器分别表示问题和文档，再通过向量内积评分。"
                "文档向量可提前计算并写入近似最近邻索引。"
            ),
        ),
        Document(
            source="generation-notes",
            title="RAG-Sequence 与 RAG-Token",
            text=(
                "RAG-Sequence 为整段输出边缘化一次潜在文档。RAG-Token 则在每个"
                "输出 token 的概率上边缘化同一批候选文档；它并不是每生成一个 token"
                "就重新执行一次检索。"
            ),
        ),
    ]

    chunks = chunk_documents(documents, chunk_size=120, overlap=20)
    rag = MinimalRAG(BM25Retriever(chunks), extractive_generator)
    query = "RAG 原论文如何更新外部知识？"
    answer, evidence, prompt = rag.answer(query, top_k=2)

    print("\nRetrieved evidence:")
    for index, item in enumerate(evidence, start=1):
        print(
            f"E{index}: score={item.score:.3f}, "
            f"p={item.probability:.3f}, source={item.chunk.source}"
        )
    print(f"\nAnswer:\n{answer}")
    print(f"\nPrompt preview:\n{prompt[:260]}...")


if __name__ == "__main__":
    probability_demo()
    pipeline_demo()
