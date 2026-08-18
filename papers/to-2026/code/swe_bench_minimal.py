#!/usr/bin/env python3
"""SWE-bench 的零依赖教学实现。

它演示论文中三个可独立理解的核心部件：

1. 用 issue 文本对仓库文件做 BM25 排序，并在上下文预算内装入文件；
2. 从模型输出中提取、校验并统计 unified diff；
3. 根据 FAIL_TO_PASS / PASS_TO_PASS 测试状态判定补丁结果。

脚本只处理合成字符串和测试状态，不 checkout 仓库、不应用补丁，也不执行任何
不可信代码。真实 SWE-bench 复现应使用官方数据集和隔离的 Docker harness。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*|\d+")
HUNK_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$"
)


def tokenize(text: str) -> list[str]:
    """一个确定、轻量的 code-aware tokenizer；不是论文使用的模型 tokenizer。"""

    result: list[str] = []
    for raw_token in TOKEN_RE.findall(text):
        token = raw_token.lower()
        result.append(token)
        # 同时保留完整标识符与路径片段，使 `src/cache/path.py` 能匹配
        # issue 中的自然语言 `cache path`。
        parts = re.split(r"[./:_-]+", token)
        result.extend(part for part in parts if part and part != token)
    return result


@dataclass(frozen=True)
class RepoFile:
    path: str
    content: str

    @property
    def retrieval_text(self) -> str:
        # 论文基线把路径放在文件内容前；路径中的标识符常常是强定位信号。
        return f"{self.path}\n{self.content}"


@dataclass(frozen=True)
class RankedFile:
    file: RepoFile
    score: float
    token_count: int


def bm25_rank(
    query: str,
    files: Sequence[RepoFile],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[RankedFile]:
    """按 BM25 分数返回仓库文件，公式对应经典 Okapi BM25。"""

    if not files:
        return []

    documents = [tokenize(file.retrieval_text) for file in files]
    term_counts = [Counter(document) for document in documents]
    lengths = [len(document) for document in documents]
    avg_length = sum(lengths) / len(lengths)
    query_terms = Counter(tokenize(query))

    document_frequency: Counter[str] = Counter()
    for counts in term_counts:
        document_frequency.update(counts.keys())

    ranked: list[RankedFile] = []
    n_documents = len(files)
    for file, counts, length in zip(files, term_counts, lengths):
        score = 0.0
        for term, query_frequency in query_terms.items():
            frequency = counts[term]
            if frequency == 0:
                continue
            df = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0 + (n_documents - df + 0.5) / (df + 0.5)
            )
            normalization = frequency + k1 * (
                1.0 - b + b * length / max(avg_length, 1.0)
            )
            score += (
                query_frequency
                * inverse_document_frequency
                * frequency
                * (k1 + 1.0)
                / normalization
            )
        ranked.append(RankedFile(file, score, length))

    return sorted(ranked, key=lambda item: (-item.score, item.file.path))


def pack_context(
    ranked: Iterable[RankedFile], max_tokens: int
) -> tuple[list[RankedFile], int]:
    """按排名贪心装入完整文件；不截断文件，便于解释 token budget 的影响。"""

    selected: list[RankedFile] = []
    used = 0
    for item in ranked:
        if item.score <= 0:
            continue
        if used + item.token_count <= max_tokens:
            selected.append(item)
            used += item.token_count
    return selected, used


def oracle_file_recall(
    retrieved_paths: Iterable[str], gold_edited_paths: Iterable[str]
) -> tuple[float, bool, bool]:
    """返回论文 Table 3 的单实例 Avg / Any / All 三种 recall 信号。"""

    retrieved = set(retrieved_paths)
    gold = set(gold_edited_paths)
    if not gold:
        raise ValueError("gold edited paths must not be empty")
    overlap = retrieved & gold
    return len(overlap) / len(gold), bool(overlap), gold <= retrieved


@dataclass(frozen=True)
class PatchSummary:
    files: tuple[str, ...]
    hunks: int
    added_lines: int
    removed_lines: int


def extract_patch(model_output: str) -> str:
    """从解释性文字或 Markdown fence 中提取第一个 unified diff。"""

    lines = model_output.splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith("diff --git ") or line.startswith("--- a/")
        ),
        None,
    )
    if start is None:
        raise ValueError("model output does not contain a unified diff")

    patch_lines: list[str] = []
    for line in lines[start:]:
        if line.strip() == "```" and patch_lines:
            break
        patch_lines.append(line)
    return "\n".join(patch_lines).rstrip() + "\n"


def _safe_repo_path(header_path: str) -> str | None:
    """把 a/x 或 b/x 归一化为 x，并拒绝越出仓库的路径。"""

    path = header_path.split("\t", 1)[0].strip()
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        path = path[2:]
    if not path or path.startswith("/"):
        raise ValueError(f"unsafe absolute or empty path: {header_path!r}")
    parts = path.split("/")
    if any(part in {"", ".", "..", ".git"} for part in parts):
        raise ValueError(f"unsafe repository path: {header_path!r}")
    return path


def summarize_patch(patch: str) -> PatchSummary:
    """做结构与路径层面的严格校验；它不替代 `git apply --check`。"""

    lines = patch.splitlines()
    files: list[str] = []
    hunks = additions = removals = 0
    index = 0

    while index < len(lines):
        if not lines[index].startswith("--- "):
            index += 1
            continue

        old_path = _safe_repo_path(lines[index][4:])
        if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
            raise ValueError("each --- header must be followed by a +++ header")
        new_path = _safe_repo_path(lines[index + 1][4:])
        effective_path = new_path or old_path
        if effective_path is None:
            raise ValueError("both sides of a file patch cannot be /dev/null")
        files.append(effective_path)
        index += 2

        file_hunks = 0
        while index < len(lines) and not lines[index].startswith("--- "):
            line = lines[index]
            if line.startswith("@@ "):
                if HUNK_RE.match(line) is None:
                    raise ValueError(f"malformed hunk header: {line!r}")
                hunks += 1
                file_hunks += 1
            elif line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                removals += 1
            index += 1
        if file_hunks == 0:
            raise ValueError(f"file patch has no hunks: {effective_path}")

    if not files:
        raise ValueError("patch has no file headers")
    return PatchSummary(tuple(dict.fromkeys(files)), hunks, additions, removals)


class TestStatus(str, Enum):
    PASS = "PASS"
    XFAIL = "XFAIL"
    FAIL = "FAIL"
    ERROR = "ERROR"
    MISSING = "MISSING"


class Outcome(str, Enum):
    PATCH_FAILED = "patch failed"
    RESOLVED = "resolved"
    BREAKING_RESOLVED = "breaking resolved"
    PARTIALLY_RESOLVED = "partially resolved"
    WORK_IN_PROGRESS = "work in progress"
    NO_OP = "no-op"
    REGRESSION = "regression"


@dataclass(frozen=True)
class Evaluation:
    patch_applied: bool
    f2p_passed: int
    f2p_total: int
    p2p_passed: int
    p2p_total: int
    outcome: Outcome

    @property
    def resolved(self) -> bool:
        return self.outcome is Outcome.RESOLVED


def evaluate_tests(
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
    observed: Mapping[str, TestStatus | str],
    *,
    patch_applied: bool = True,
) -> Evaluation:
    """复现论文六类结果；缺失测试按失败处理。"""

    if not fail_to_pass:
        raise ValueError("a SWE-bench instance must contain at least one F2P test")
    if not patch_applied:
        return Evaluation(False, 0, len(fail_to_pass), 0, len(pass_to_pass), Outcome.PATCH_FAILED)

    passing = {TestStatus.PASS.value, TestStatus.XFAIL.value}

    def passed(test: str) -> bool:
        status = observed.get(test, TestStatus.MISSING)
        value = status.value if isinstance(status, TestStatus) else str(status)
        return value in passing

    f2p_passed = sum(passed(test) for test in fail_to_pass)
    p2p_passed = sum(passed(test) for test in pass_to_pass)
    all_f2p = f2p_passed == len(fail_to_pass)
    any_f2p = f2p_passed > 0
    all_p2p = p2p_passed == len(pass_to_pass)

    if all_f2p and all_p2p:
        outcome = Outcome.RESOLVED
    elif all_f2p:
        outcome = Outcome.BREAKING_RESOLVED
    elif any_f2p and all_p2p:
        outcome = Outcome.PARTIALLY_RESOLVED
    elif any_f2p:
        outcome = Outcome.WORK_IN_PROGRESS
    elif all_p2p:
        outcome = Outcome.NO_OP
    else:
        outcome = Outcome.REGRESSION

    return Evaluation(
        True,
        f2p_passed,
        len(fail_to_pass),
        p2p_passed,
        len(pass_to_pass),
        outcome,
    )


def aggregate(evaluations: Sequence[Evaluation]) -> dict[str, float]:
    """计算论文主指标：% Apply 与 % Resolved。"""

    if not evaluations:
        raise ValueError("evaluations must not be empty")
    total = len(evaluations)
    return {
        "% Apply": 100.0 * sum(item.patch_applied for item in evaluations) / total,
        "% Resolved": 100.0 * sum(item.resolved for item in evaluations) / total,
    }


def demo() -> None:
    files = [
        RepoFile(
            "src/cache/path.py",
            "def normalize_cache_path(value): return value.replace('\\\\', '/')",
        ),
        RepoFile(
            "src/http/client.py",
            "def request(url, timeout): return transport.send(url, timeout)",
        ),
        RepoFile(
            "tests/test_cache_path.py",
            "def test_windows_cache_path_is_normalized(): ...",
        ),
        RepoFile("docs/cache.md", "Cache directories may use platform path separators."),
    ]
    issue = "Windows cache path keeps duplicate separators after normalization"
    ranked = bm25_rank(issue, files)
    print("BM25 context")
    for budget in (30, 80):
        selected, used = pack_context(ranked, max_tokens=budget)
        recall, any_gold, all_gold = oracle_file_recall(
            (item.file.path for item in selected),
            {"src/cache/path.py"},
        )
        selected_paths = ", ".join(item.file.path for item in selected)
        print(f"  budget={budget:2}: {selected_paths}")
        print(
            f"             used={used}/{budget}, oracle recall={recall:.0%}, "
            f"any={any_gold}, all={all_gold}"
        )

    model_output = """Here is the patch:
```diff
diff --git a/src/cache/path.py b/src/cache/path.py
--- a/src/cache/path.py
+++ b/src/cache/path.py
@@ -1 +1,4 @@
-def normalize_cache_path(value): return value.replace('\\\\', '/')
+def normalize_cache_path(value):
+    normalized = value.replace('\\\\', '/')
+    while '//' in normalized: normalized = normalized.replace('//', '/')
+    return normalized
```
"""
    summary = summarize_patch(extract_patch(model_output))
    print("\nPatch summary")
    print(
        f"  files={summary.files}, hunks={summary.hunks}, "
        f"+{summary.added_lines}/-{summary.removed_lines}"
    )

    f2p = ["test_duplicate_separator", "test_unc_path"]
    p2p = ["test_posix_path", "test_empty_path"]
    cases = {
        "resolved": {
            **{test: TestStatus.PASS for test in f2p},
            **{test: TestStatus.PASS for test in p2p},
        },
        "breaking": {
            **{test: TestStatus.PASS for test in f2p},
            p2p[0]: TestStatus.FAIL,
            p2p[1]: TestStatus.PASS,
        },
        "partial": {
            f2p[0]: TestStatus.PASS,
            f2p[1]: TestStatus.FAIL,
            **{test: TestStatus.PASS for test in p2p},
        },
        "work-in-progress": {
            f2p[0]: TestStatus.PASS,
            f2p[1]: TestStatus.FAIL,
            p2p[0]: TestStatus.FAIL,
            p2p[1]: TestStatus.PASS,
        },
        "no-op": {
            **{test: TestStatus.FAIL for test in f2p},
            **{test: TestStatus.PASS for test in p2p},
        },
        "regression": {
            **{test: TestStatus.FAIL for test in f2p},
            p2p[0]: TestStatus.FAIL,
            p2p[1]: TestStatus.PASS,
        },
    }
    evaluations = [evaluate_tests(f2p, p2p, statuses) for statuses in cases.values()]
    evaluations.append(evaluate_tests(f2p, p2p, {}, patch_applied=False))

    print("\nEvaluation outcomes")
    for name, evaluation in zip([*cases, "invalid patch"], evaluations):
        print(
            f"  {name:17} -> {evaluation.outcome.value:20} "
            f"F2P={evaluation.f2p_passed}/{evaluation.f2p_total} "
            f"P2P={evaluation.p2p_passed}/{evaluation.p2p_total}"
        )

    metrics = aggregate(evaluations)
    print("\nAggregate")
    print(f"  % Apply={metrics['% Apply']:.2f}")
    print(f"  % Resolved={metrics['% Resolved']:.2f}")

    assert summary.files == ("src/cache/path.py",)
    assert evaluations[0].outcome is Outcome.RESOLVED
    assert evaluations[-1].outcome is Outcome.PATCH_FAILED
    assert metrics == {"% Apply": 600 / 7, "% Resolved": 100 / 7}

    unsafe_patch = """--- a/src/cache/path.py
+++ b/../../outside.py
@@ -1 +1 @@
-old
+new
"""
    try:
        summarize_patch(unsafe_patch)
    except ValueError:
        pass
    else:
        raise AssertionError("path traversal must be rejected")


if __name__ == "__main__":
    demo()
