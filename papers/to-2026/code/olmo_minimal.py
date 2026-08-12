"""OLMo 开放训练链路的零依赖教学实现。

这不是 OLMo 官方训练器，也不会真的训练 70 亿参数模型。它把论文里最值得
复现和审计的机制压缩成一台普通机器可运行的代码：

1. 由论文 Table 2 重建 Dolma 语料构成；
2. 文档追加 EOS、拼接并切成固定长度因果语言模型样本；
3. 用种子、epoch 和 manifest 重建样本顺序与批次；
4. 复现 warmup、线性衰减和最后 cooldown-to-zero 学习率；
5. 核算 OLMo-7B 的参数量、checkpoint token 位置和碳排；
6. 演示 rank classification 与跨 tokenizer 的 bits-per-byte。

运行：

    python3 papers/to-2026/code/olmo_minimal.py

精确复现实验应使用官方仓库、公开 checkpoint、训练日志和 data-order
artifacts。本文件只负责解释机制，并刻意保持 Python 标准库零依赖。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class DolmaSource:
    """论文 Table 2 中一个 Dolma 来源；tokens 单位为十亿。"""

    name: str
    disk_gb: float
    documents_million: float
    tokens_billion: float


DOLMA_SOURCES: tuple[DolmaSource, ...] = (
    DolmaSource("Common Crawl", 9812.0, 3734.0, 2180.0),
    DolmaSource("GitHub", 1043.0, 210.0, 342.0),
    DolmaSource("Reddit", 339.0, 377.0, 80.0),
    DolmaSource("Semantic Scholar", 268.0, 38.8, 57.0),
    DolmaSource("Project Gutenberg", 20.4, 0.056, 5.2),
    DolmaSource("Wikipedia", 16.2, 6.2, 3.7),
)


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class PackedBlock:
    """可追踪来源的定长 token block。"""

    tokens: tuple[int, ...]
    doc_ids: tuple[str, ...]
    sources: tuple[str, ...]


@dataclass(frozen=True)
class ModelShape:
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    mlp_hidden: int = 11008
    padded_vocab: int = 50304
    tie_embeddings: bool = False


def dolma_table(
    sources: Sequence[DolmaSource] = DOLMA_SOURCES,
) -> list[dict[str, float | str]]:
    """计算完整 Dolma 语料表中的 token 占比。

    注意：这是 Table 2 的完整 corpus 比例，不应直接冒充某个训练 run 的
    精确采样比例；后者应以发布的 data-order artifact 为准。
    """

    total = sum(source.tokens_billion for source in sources)
    if total <= 0:
        raise ValueError("total tokens must be positive")
    return [
        {
            "source": source.name,
            "disk_gb": source.disk_gb,
            "documents_million": source.documents_million,
            "tokens_billion": source.tokens_billion,
            "token_share_pct": 100.0 * source.tokens_billion / total,
        }
        for source in sources
    ]


def pack_documents(
    documents: Iterable[Document],
    *,
    block_size: int = 2048,
    eos_token_id: int = 0,
    drop_remainder: bool = True,
) -> list[PackedBlock]:
    """按 OLMo 论文描述：每篇文档追加 EOS，再连续拼接和切块。

    block 可以跨文档边界；EOS 让模型看见边界。真实流水线在分片与并行读取
    上更复杂，但关键语义就是“文档 -> EOS -> 连续 token stream”。
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    token_stream: list[int] = []
    owner_stream: list[tuple[str, str]] = []
    for document in documents:
        if not document.tokens:
            continue
        values = [*document.tokens, eos_token_id]
        token_stream.extend(values)
        owner_stream.extend([(document.doc_id, document.source)] * len(values))

    if not drop_remainder and token_stream and len(token_stream) % block_size:
        padding = block_size - len(token_stream) % block_size
        token_stream.extend([eos_token_id] * padding)
        owner_stream.extend([("<padding>", "<padding>")] * padding)

    usable = len(token_stream) - len(token_stream) % block_size
    blocks: list[PackedBlock] = []
    for start in range(0, usable, block_size):
        owners = owner_stream[start : start + block_size]
        # dict 保持首次出现次序，便于 manifest 追踪跨文档 block。
        doc_ids = tuple(dict.fromkeys(doc_id for doc_id, _ in owners))
        sources = tuple(dict.fromkeys(source for _, source in owners))
        blocks.append(
            PackedBlock(
                tokens=tuple(token_stream[start : start + block_size]),
                doc_ids=doc_ids,
                sources=sources,
            )
        )
    return blocks


def epoch_order(length: int, *, seed: int, epoch: int = 0) -> list[int]:
    """产生可复现的 epoch 内顺序；不同 epoch 使用独立确定性种子。"""

    if length < 0 or epoch < 0:
        raise ValueError("length and epoch must be non-negative")
    order = list(range(length))
    # 显式构造字符串再哈希，避免 seed + epoch 的简单碰撞模式。
    digest = hashlib.sha256(f"olmo-demo:{seed}:{epoch}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(order)
    return order


def batches_for_epoch(
    blocks: Sequence[PackedBlock],
    *,
    seed: int,
    epoch: int,
    batch_size: int,
) -> list[list[PackedBlock]]:
    """用相同参数即可重建每个 global batch 包含哪些 block。"""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    order = epoch_order(len(blocks), seed=seed, epoch=epoch)
    return [
        [blocks[index] for index in order[start : start + batch_size]]
        for start in range(0, len(order), batch_size)
        if len(order[start : start + batch_size]) == batch_size
    ]


def manifest_hash(
    batches: Sequence[Sequence[PackedBlock]],
    *,
    seed: int,
    epoch: int,
) -> str:
    """为 data order 生成内容可审计的 SHA-256 指纹。"""

    manifest = {
        "seed": seed,
        "epoch": epoch,
        "batches": [
            [
                {
                    "tokens": list(block.tokens),
                    "doc_ids": list(block.doc_ids),
                    "sources": list(block.sources),
                }
                for block in batch
            ]
            for batch in batches
        ],
    }
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def source_share_by_window(
    blocks: Sequence[PackedBlock], *, window_size: int
) -> list[dict[str, float]]:
    """检查连续窗口的来源比例，捕捉 shuffle/组 batch 异常。

    教学 block 只保存来源集合，因此每个 block 在多个来源间均分权重。生产审计
    应按每个 token 的 provenance 计数。
    """

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    windows: list[dict[str, float]] = []
    for start in range(0, len(blocks), window_size):
        group = blocks[start : start + window_size]
        if not group:
            continue
        counts: dict[str, float] = {}
        for block in group:
            weight = 1.0 / len(block.sources)
            for source in block.sources:
                counts[source] = counts.get(source, 0.0) + weight
        windows.append({name: value / len(group) for name, value in counts.items()})
    return windows


def learning_rate(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int = 5000,
    cooldown_steps: int = 1000,
    peak_lr: float = 3e-4,
    floor_ratio: float = 0.1,
) -> float:
    """OLMo-7B：warmup -> 线性降至 0.1×peak -> 最后降至零。

    ``step`` 取 0..total_steps。论文在最后 1000 step 追加 cooldown-to-zero；
    这里让两个线性片段在边界连续。
    """

    if not (0 <= step <= total_steps):
        raise ValueError("step must lie in [0, total_steps]")
    if warmup_steps <= 0 or cooldown_steps <= 0:
        raise ValueError("warmup and cooldown must be positive")
    decay_end = total_steps - cooldown_steps
    if not warmup_steps < decay_end:
        raise ValueError("total_steps is too small")
    if not 0 <= floor_ratio <= 1:
        raise ValueError("floor_ratio must lie in [0, 1]")

    if step <= warmup_steps:
        return peak_lr * step / warmup_steps
    floor_lr = peak_lr * floor_ratio
    if step <= decay_end:
        progress = (step - warmup_steps) / (decay_end - warmup_steps)
        return peak_lr + progress * (floor_lr - peak_lr)
    progress = (step - decay_end) / cooldown_steps
    return floor_lr * (1.0 - progress)


def parameter_ledger(shape: ModelShape = ModelShape()) -> dict[str, int]:
    """核算无 bias、non-parametric LN、SwiGLU 的 decoder 参数量。"""

    embeddings = shape.padded_vocab * shape.d_model
    # Wq, Wk, Wv, Wo；OLMo-7B 使用 full multi-head attention。
    attention_per_layer = 4 * shape.d_model * shape.d_model
    # SwiGLU 有 gate、value、output 三个矩阵。
    swiglu_per_layer = 3 * shape.d_model * shape.mlp_hidden
    blocks = shape.n_layers * (attention_per_layer + swiglu_per_layer)
    lm_head = 0 if shape.tie_embeddings else embeddings
    total = embeddings + blocks + lm_head
    return {
        "token_embeddings": embeddings,
        "attention_per_layer": attention_per_layer,
        "swiglu_per_layer": swiglu_per_layer,
        "all_decoder_blocks": blocks,
        "untied_lm_head": lm_head,
        "total": total,
    }


def checkpoint_ledger(
    *,
    total_steps: int,
    every_steps: int = 1000,
    tokens_per_step: int = 2048 * 2048,
) -> list[dict[str, int]]:
    """列出每个定期 checkpoint 之前约见过多少 token。"""

    if total_steps < 0 or every_steps <= 0 or tokens_per_step <= 0:
        raise ValueError("invalid checkpoint arguments")
    steps = list(range(every_steps, total_steps + 1, every_steps))
    if total_steps and (not steps or steps[-1] != total_steps):
        steps.append(total_steps)
    return [
        {"step": step, "tokens_seen": step * tokens_per_step}
        for step in steps
    ]


def rank_classification_score(
    conditional_logprobs: Sequence[float],
    *,
    normalization: str = "none",
    unconditional_logprobs: Sequence[float] | None = None,
) -> float:
    """Catwalk 风格候选答案打分：none、per_token 或 unconditional。"""

    if not conditional_logprobs:
        raise ValueError("candidate must contain at least one token")
    score = sum(conditional_logprobs)
    if normalization == "none":
        return score
    if normalization == "per_token":
        return score / len(conditional_logprobs)
    if normalization == "unconditional":
        if unconditional_logprobs is None:
            raise ValueError("unconditional logprobs are required")
        return score - sum(unconditional_logprobs)
    raise ValueError(f"unknown normalization: {normalization}")


def bits_per_byte(total_nll_nats: float, utf8_bytes: int) -> float:
    """Paloma 使用的 tokenizer-agnostic BPB：NLL / (bytes × ln 2)。"""

    if total_nll_nats < 0 or utf8_bytes <= 0:
        raise ValueError("NLL must be non-negative and bytes positive")
    return total_nll_nats / (utf8_bytes * math.log(2.0))


def carbon_tonnes(
    *, energy_mwh: float, pue: float, carbon_kg_per_kwh: float
) -> float:
    """运营碳排下界：MWh × 1000 × PUE × kg/kWh / 1000。"""

    if min(energy_mwh, pue, carbon_kg_per_kwh) < 0:
        raise ValueError("carbon inputs must be non-negative")
    return energy_mwh * pue * carbon_kg_per_kwh


def _demo_documents() -> list[Document]:
    return [
        Document("cc-1", "Common Crawl", (11, 12, 13)),
        Document("gh-1", "GitHub", (21, 22)),
        Document("s2-1", "Semantic Scholar", (31, 32, 33, 34)),
        Document("wiki-1", "Wikipedia", (41, 42, 43)),
        Document("reddit-1", "Reddit", (51, 52)),
    ]


def main() -> None:
    table = dolma_table()
    assert math.isclose(sum(row["token_share_pct"] for row in table), 100.0)
    assert math.isclose(sum(row["tokens_billion"] for row in table), 2667.9)

    blocks = pack_documents(
        _demo_documents(), block_size=4, eos_token_id=0, drop_remainder=False
    )
    assert any(0 in block.tokens for block in blocks)
    batches0 = batches_for_epoch(blocks, seed=42, epoch=0, batch_size=2)
    batches0_again = batches_for_epoch(blocks, seed=42, epoch=0, batch_size=2)
    batches1 = batches_for_epoch(blocks, seed=42, epoch=1, batch_size=2)
    hash0 = manifest_hash(batches0, seed=42, epoch=0)
    assert hash0 == manifest_hash(batches0_again, seed=42, epoch=0)
    assert hash0 != manifest_hash(batches1, seed=42, epoch=1)

    ledger = parameter_ledger()
    assert ledger["total"] == 6_888_095_744

    total_steps = 600_000
    decay_end = total_steps - 1000
    schedule = {
        "start": learning_rate(0, total_steps=total_steps),
        "warmup_end": learning_rate(5000, total_steps=total_steps),
        "decay_end": learning_rate(decay_end, total_steps=total_steps),
        "training_end": learning_rate(total_steps, total_steps=total_steps),
    }
    assert math.isclose(schedule["warmup_end"], 3e-4)
    assert math.isclose(schedule["decay_end"], 3e-5)
    assert math.isclose(schedule["training_end"], 0.0)

    # 同一个短答案，sum log-likelihood 与 per-token 会回答不同的问题。
    candidates = {
        "short_sum": rank_classification_score([-0.3], normalization="none"),
        "long_sum": rank_classification_score([-0.2, -0.2], normalization="none"),
        "short_mean": rank_classification_score([-0.3], normalization="per_token"),
        "long_mean": rank_classification_score([-0.2, -0.2], normalization="per_token"),
    }
    assert candidates["short_sum"] > candidates["long_sum"]
    assert candidates["long_mean"] > candidates["short_mean"]

    a100_carbon = carbon_tonnes(
        energy_mwh=104.0, pue=1.1, carbon_kg_per_kwh=0.610
    )
    assert math.isclose(a100_carbon, 69.784)

    checkpoints = checkpoint_ledger(total_steps=3000)
    report = {
        "scope": "pedagogical OLMo-1 open-science pipeline",
        "dolma": {
            "table": table,
            "table_total_tokens_billion": sum(
                source.tokens_billion for source in DOLMA_SOURCES
            ),
        },
        "packing": {
            "blocks": [asdict(block) for block in blocks],
            "epoch0_manifest_sha256": hash0,
            "source_windows": source_share_by_window(blocks, window_size=2),
        },
        "olmo_7b_parameter_ledger": ledger,
        "learning_rate": schedule,
        "checkpoint_demo": checkpoints,
        "rank_classification_demo": candidates,
        "bpb_demo": bits_per_byte(total_nll_nats=69.314718, utf8_bytes=100),
        "a100_operational_carbon_tonnes": a100_carbon,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
