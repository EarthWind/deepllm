"""The Pile 核心数据机制的零依赖教学实现。

这不是 825 GiB 语料的下载器，也不是官方复现脚本。它用很小的数据演示：

1. 论文表 1 中 raw size、epochs、effective size 与 mixture weight 的关系；
2. 按目标权重生成可复现的组件配额与交错数据流；
3. 文档 shingle、MinHash 签名与 LSH 候选去重的基本机制；
4. held-out 精确文本移除与下游 13-gram 去污染的区别；
5. tokenizer 可比的 bits per UTF-8 byte（BPB）如何计算；
6. 最终 ``{"text": ..., "meta": ...}`` JSONL 记录与分片。

运行：

    python3 papers/to-2026/code/pile_minimal.py

完整论文工程应使用流式压缩 I/O、外存索引、组件级许可清单和大规模
MinHashLSH；这里刻意只保留可以在一台普通机器上读懂和测试的机制。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
import random
import re
import unicodedata
from typing import Iterable, Iterator, Mapping, Sequence


@dataclass(frozen=True)
class Component:
    """论文表 1 中一个 Pile component 的规模和重采样配置。"""

    name: str
    raw_gib: float
    epochs: float
    paper_weight_pct: float

    @property
    def effective_gib(self) -> float:
        return self.raw_gib * self.epochs


# 数值来自论文 Table 1。表内各数已四舍五入，因此合计允许极小误差。
PILE_COMPONENTS: tuple[Component, ...] = (
    Component("Pile-CC", 227.12, 1.0, 18.11),
    Component("PubMed Central", 90.27, 2.0, 14.40),
    Component("Books3", 100.96, 1.5, 12.07),
    Component("OpenWebText2", 62.77, 2.0, 10.01),
    Component("ArXiv", 56.21, 2.0, 8.96),
    Component("GitHub", 95.16, 1.0, 7.59),
    Component("FreeLaw", 51.15, 1.5, 6.12),
    Component("Stack Exchange", 32.20, 2.0, 5.13),
    Component("USPTO Backgrounds", 22.90, 2.0, 3.65),
    Component("PubMed Abstracts", 19.26, 2.0, 3.07),
    Component("Gutenberg (PG-19)", 10.88, 2.5, 2.17),
    Component("OpenSubtitles", 12.98, 1.5, 1.55),
    Component("Wikipedia (en)", 6.38, 3.0, 1.53),
    Component("DM Mathematics", 7.75, 2.0, 1.24),
    Component("Ubuntu IRC", 5.52, 2.0, 0.88),
    Component("BookCorpus2", 6.30, 1.5, 0.75),
    Component("EuroParl", 4.59, 2.0, 0.73),
    Component("HackerNews", 3.90, 2.0, 0.62),
    Component("YouTube Subtitles", 3.73, 2.0, 0.60),
    Component("PhilPapers", 2.38, 2.0, 0.38),
    Component("NIH ExPorter", 1.89, 2.0, 0.30),
    Component("Enron Emails", 0.88, 2.0, 0.14),
)


@dataclass(frozen=True)
class Document:
    """统一后的最小文档结构；真实组件会保留更多来源 metadata。"""

    text: str
    source: str
    meta: Mapping[str, object] = field(default_factory=dict)

    def as_pile_record(self) -> dict[str, object]:
        metadata = {"pile_set_name": self.source, **self.meta}
        return {"text": self.text, "meta": metadata}


@dataclass(frozen=True)
class MixtureRow:
    name: str
    raw_gib: float
    epochs: float
    effective_gib: float
    derived_weight_pct: float
    paper_weight_pct: float


def mixture_table(
    components: Sequence[Component] = PILE_COMPONENTS,
) -> list[MixtureRow]:
    """从 raw size × epochs 推导 effective size 和字节权重。"""

    total_effective = sum(component.effective_gib for component in components)
    if total_effective <= 0:
        raise ValueError("total effective size must be positive")

    return [
        MixtureRow(
            name=component.name,
            raw_gib=component.raw_gib,
            epochs=component.epochs,
            effective_gib=component.effective_gib,
            derived_weight_pct=100.0
            * component.effective_gib
            / total_effective,
            paper_weight_pct=component.paper_weight_pct,
        )
        for component in components
    ]


def largest_remainder_quotas(
    total_records: int,
    components: Sequence[Component] = PILE_COMPONENTS,
) -> dict[str, int]:
    """把连续 mixture weights 转为总数严格相等的整数配额。

    论文实际按文档数与 epochs 加权抽样，并在大数下逼近期望比例。这个函数
    是便于审计的小规模替代：先取 floor，再把余数给小数部分最大的组件。
    """

    if total_records < 0:
        raise ValueError("total_records must be non-negative")
    total_weight = sum(component.paper_weight_pct for component in components)
    if total_weight <= 0:
        raise ValueError("component weights must be positive")

    exact = {
        component.name: total_records
        * component.paper_weight_pct
        / total_weight
        for component in components
    }
    quotas = {name: math.floor(value) for name, value in exact.items()}
    remaining = total_records - sum(quotas.values())
    order = sorted(exact, key=lambda name: (exact[name] - quotas[name], name), reverse=True)
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


def weighted_document_stream(
    sources: Mapping[str, Sequence[Document]],
    total_records: int,
    *,
    seed: int = 210100027,
) -> list[Document]:
    """按已有组件的论文权重产生一个可复现教学数据流。

    小组件可能需要被多次遍历，所以这里会在每轮重新洗牌后循环读取。真实
    Pile 使用流式构建和 30 个交错输出 piles，不能把这段代码当作规模化实现。
    """

    known = {component.name: component for component in PILE_COMPONENTS}
    unknown = sorted(set(sources) - set(known))
    if unknown:
        raise KeyError(f"unknown components: {unknown}")
    if any(not documents for documents in sources.values()):
        raise ValueError("every selected component must contain at least one document")

    selected = [known[name] for name in sources]
    quotas = largest_remainder_quotas(total_records, selected)
    rng = random.Random(seed)
    output: list[Document] = []

    for component in selected:
        documents = list(sources[component.name])
        remaining = quotas[component.name]
        while remaining:
            rng.shuffle(documents)
            take = min(remaining, len(documents))
            output.extend(documents[:take])
            remaining -= take

    rng.shuffle(output)
    return output


_WHITESPACE = re.compile(r"\s+")
_WORDS = re.compile(r"\w+", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """为教学哈希做保守规范化，不代表官方每个组件的处理规则。"""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE.sub(" ", normalized).strip()


def stable_u64(value: str) -> int:
    """与 Python 进程无关的 64-bit hash。"""

    digest = hashlib.blake2b(
        value.encode("utf-8"),
        digest_size=8,
        person=b"pile-demo",
    ).digest()
    return int.from_bytes(digest, "big")


def exact_document_hash(text: str) -> str:
    """规范化后做 SHA-256；用于 held-out 的精确匹配示例。"""

    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def word_shingles(text: str, size: int = 5) -> set[str]:
    """把文档转为连续 word shingles。论文未规定这里的教学分词细节。"""

    if size <= 0:
        raise ValueError("shingle size must be positive")
    words = _WORDS.findall(normalize_text(text))
    if not words:
        return set()
    if len(words) < size:
        return {" ".join(words)}
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    """两个 shingle 集合的精确 Jaccard 相似度。"""

    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


_MERSENNE_61 = (1 << 61) - 1


def minhash_signature(
    shingles: set[str],
    *,
    num_perm: int = 64,
    seed: int = 2021,
) -> tuple[int, ...]:
    """计算可复现 MinHash 签名；签名位置一致率估计 Jaccard。"""

    if num_perm <= 0:
        raise ValueError("num_perm must be positive")
    if not shingles:
        return tuple([_MERSENNE_61] * num_perm)

    values = [stable_u64(shingle) % _MERSENNE_61 for shingle in shingles]
    rng = random.Random(seed)
    permutations = [
        (
            rng.randrange(1, _MERSENNE_61),
            rng.randrange(0, _MERSENNE_61),
        )
        for _ in range(num_perm)
    ]
    return tuple(
        min((a * value + b) % _MERSENNE_61 for value in values)
        for a, b in permutations
    )


def signature_similarity(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("signatures must have the same positive length")
    return sum(a == b for a, b in zip(left, right)) / len(left)


def lsh_candidate_pairs(
    signatures: Sequence[Sequence[int]],
    *,
    bands: int = 16,
) -> set[tuple[int, int]]:
    """把相同 band 的签名放入同一桶，返回需要精查的候选对。"""

    if not signatures:
        return set()
    width = len(signatures[0])
    if bands <= 0 or width % bands:
        raise ValueError("signature length must be divisible by bands")
    rows = width // bands
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)

    for index, signature in enumerate(signatures):
        if len(signature) != width:
            raise ValueError("all signatures must have equal length")
        for band in range(bands):
            start = band * rows
            key = (band, tuple(signature[start : start + rows]))
            buckets[key].append(index)

    candidates: set[tuple[int, int]] = set()
    for members in buckets.values():
        for right_position, right in enumerate(members):
            for left in members[:right_position]:
                candidates.add((left, right))
    return candidates


def minhash_lsh_deduplicate(
    documents: Sequence[Document],
    *,
    threshold: float = 0.5,
    shingle_size: int = 5,
    num_perm: int = 64,
    bands: int = 16,
) -> tuple[list[Document], list[tuple[int, int, float]]]:
    """用 LSH 缩小候选，再按精确 Jaccard 删除较后的近重复文档。

    论文使用 Datasketch MinHashLSH、10 个 hash 和约 0.5 的 Jaccard 阈值。
    这里用 64 个排列提升小样本稳定性，并保留精确 Jaccard 复查步骤。
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    shingle_sets = [word_shingles(document.text, shingle_size) for document in documents]
    signatures = [
        minhash_signature(shingles, num_perm=num_perm)
        for shingles in shingle_sets
    ]
    candidates = sorted(lsh_candidate_pairs(signatures, bands=bands))

    removed: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for left, right in candidates:
        if left in removed or right in removed:
            continue
        score = jaccard(shingle_sets[left], shingle_sets[right])
        if score >= threshold:
            removed.add(right)
            matches.append((left, right, score))

    kept = [document for index, document in enumerate(documents) if index not in removed]
    return kept, matches


def remove_exact_heldout(
    training: Iterable[Document],
    heldout: Iterable[Document],
) -> tuple[list[Document], list[Document]]:
    """删除与 held-out 文档规范化后完全相同的训练文档。"""

    heldout_hashes = {exact_document_hash(document.text) for document in heldout}
    kept: list[Document] = []
    removed: list[Document] = []
    for document in training:
        target = removed if exact_document_hash(document.text) in heldout_hashes else kept
        target.append(document)
    return kept, removed


def token_ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
    """生成词级 n-grams，用于演示论文评测里的 13-gram 去污染。"""

    if n <= 0:
        raise ValueError("n must be positive")
    tokens = _WORDS.findall(normalize_text(text))
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def remove_ngram_contamination(
    training: Iterable[Document],
    evaluation_texts: Iterable[str],
    *,
    n: int = 13,
) -> tuple[list[Document], list[Document]]:
    """删除与任一评测文本共享至少一个 n-gram 的训练文档。"""

    evaluation_ngrams: set[tuple[str, ...]] = set()
    for text in evaluation_texts:
        evaluation_ngrams.update(token_ngrams(text, n))

    kept: list[Document] = []
    removed: list[Document] = []
    for document in training:
        contaminated = bool(token_ngrams(document.text, n) & evaluation_ngrams)
        (removed if contaminated else kept).append(document)
    return kept, removed


def bits_per_utf8_byte(
    mean_nll_nats: float,
    token_count: int,
    utf8_byte_count: int,
) -> float:
    """把 token 平均负对数似然（nats）换算为 BPB。"""

    if mean_nll_nats < 0:
        raise ValueError("negative log likelihood must be non-negative")
    if token_count <= 0 or utf8_byte_count <= 0:
        raise ValueError("token and byte counts must be positive")
    tokens_per_byte = token_count / utf8_byte_count
    return tokens_per_byte * mean_nll_nats / math.log(2.0)


def split_into_shards(
    documents: Sequence[Document],
    *,
    num_shards: int = 30,
) -> list[list[Document]]:
    """轮询分到若干 shard；官方 30 piles 的外存交错算法更复杂。"""

    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    shards = [[] for _ in range(num_shards)]
    for index, document in enumerate(documents):
        shards[index % num_shards].append(document)
    return shards


def iter_jsonl(documents: Iterable[Document]) -> Iterator[str]:
    """输出与 Pile 顶层 text/meta 结构一致的 JSONL 行。"""

    for document in documents:
        yield json.dumps(document.as_pile_record(), ensure_ascii=False, sort_keys=True)


def self_check() -> None:
    rows = mixture_table()
    raw_total = sum(row.raw_gib for row in rows)
    effective_total = sum(row.effective_gib for row in rows)
    assert math.isclose(raw_total, 825.18, abs_tol=0.02)
    assert math.isclose(effective_total, 1254.20, abs_tol=0.03)
    assert math.isclose(sum(row.paper_weight_pct for row in rows), 100.0, abs_tol=0.02)
    assert max(abs(row.derived_weight_pct - row.paper_weight_pct) for row in rows) < 0.01

    quotas = largest_remainder_quotas(10_000)
    assert sum(quotas.values()) == 10_000
    assert quotas["Pile-CC"] > quotas["Wikipedia (en)"]

    documents = [
        Document(
            "Dense corpora mix research papers books source code and dialogue for language models.",
            "Pile-CC",
        ),
        Document(
            "Dense corpora mix research papers, books, source code and dialogue for language models!",
            "Pile-CC",
        ),
        Document(
            "A completely unrelated legal opinion discusses a narrow question of procedure.",
            "Pile-CC",
        ),
    ]
    deduplicated, matches = minhash_lsh_deduplicate(
        documents,
        threshold=0.7,
        shingle_size=3,
        num_perm=64,
        bands=32,
    )
    assert len(deduplicated) == 2
    assert len(matches) == 1

    heldout = [Document("A held out sentence.  ", "ArXiv")]
    training = [
        Document("a held out sentence.", "Pile-CC"),
        Document("A distinct training sentence.", "Pile-CC"),
    ]
    kept, removed = remove_exact_heldout(training, heldout)
    assert len(kept) == 1 and len(removed) == 1

    benchmark = "one two three four five six seven eight nine ten eleven twelve thirteen"
    contaminated = Document(
        benchmark + " appears verbatim inside this longer training document",
        "OpenWebText2",
    )
    clean = Document("this document shares no sufficiently long benchmark phrase", "Books3")
    kept, removed = remove_ngram_contamination(
        [contaminated, clean],
        [benchmark],
        n=13,
    )
    assert kept == [clean] and removed == [contaminated]

    bpb = bits_per_utf8_byte(1.0, token_count=29335, utf8_byte_count=100000)
    assert math.isclose(bpb, 0.4232, rel_tol=1e-3)

    tiny_sources = {
        "Pile-CC": [Document("web text", "Pile-CC")],
        "Wikipedia (en)": [Document("reference text", "Wikipedia (en)")],
    }
    stream = weighted_document_stream(tiny_sources, 20, seed=7)
    assert len(stream) == 20
    assert sum(len(shard) for shard in split_into_shards(stream, num_shards=3)) == 20
    assert json.loads(next(iter_jsonl(stream)))["meta"]["pile_set_name"] in tiny_sources


def demo() -> None:
    self_check()
    rows = sorted(mixture_table(), key=lambda row: row.paper_weight_pct, reverse=True)
    raw_total = sum(row.raw_gib for row in rows)
    effective_total = sum(row.effective_gib for row in rows)
    quotas = largest_remainder_quotas(10_000)

    print("The Pile minimal mechanics: self-check passed")
    print(
        f"raw={raw_total:.2f} GiB, weighted-cycle={effective_total:.2f} GiB, "
        f"components={len(rows)}"
    )
    print("top mixture components (paper weight -> records in a 10,000-record plan):")
    for row in rows[:5]:
        print(
            f"- {row.name:18s} {row.paper_weight_pct:5.2f}% -> "
            f"{quotas[row.name]:4d}"
        )

    example_nll = 1.0
    example_bpb = bits_per_utf8_byte(
        example_nll,
        token_count=29335,
        utf8_byte_count=100000,
    )
    print(f"NLL={example_nll:.1f} nat/token at 0.29335 token/byte -> BPB={example_bpb:.4f}")
    print("paper dedup reminder: Pile-wide=no; OWT2/Pile-CC MinHashLSH=yes")


if __name__ == "__main__":
    demo()
