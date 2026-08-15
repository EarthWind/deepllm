"""WebGPT 的三个最小机制：受限浏览、引用审计、偏好奖励与 best-of-N。

这个脚本完全离线，只使用 Python 标准库。它不是 WebGPT 的复现，而是把论文中
最容易被混为一谈的几层接口拆开：

1. 浏览器只暴露 Search / Click / Find / Quote / Back / End 等文本动作；
2. 引用审计只能检查“引用是否存在、引文是否来自已访问页面”，不能证明来源为真；
3. reward model 学的是答案间的相对偏好，best-of-N 再把推理时计算换成更高分候选。

运行：python papers/to-2026/code/webgpt_browser_reward_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb, exp, log
import re
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Page:
    page_id: int
    title: str
    domain: str
    text: str


@dataclass(frozen=True)
class Reference:
    page_id: int
    title: str
    domain: str
    quote: str


@dataclass
class OfflineWeb:
    """一个确定性的离线网页夹具，避免把演示变成网络爬虫。"""

    pages: dict[int, Page]

    def search(self, query: str) -> list[Page]:
        words = {word.lower() for word in re.findall(r"[a-zA-Z]+", query)}
        ranked: list[tuple[int, Page]] = []
        for page in self.pages.values():
            haystack = f"{page.title} {page.text}".lower()
            score = sum(word in haystack for word in words)
            if score:
                ranked.append((score, page))
        return [page for _, page in sorted(ranked, key=lambda item: (-item[0], item[1].page_id))]


@dataclass
class TextBrowser:
    web: OfflineWeb
    question: str
    max_actions: int = 12
    current: Page | None = None
    search_results: list[Page] = field(default_factory=list)
    history: list[Page] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    actions_used: int = 0
    ended: bool = False

    def act(self, command: str) -> str:
        """执行一个严格的文本动作；无效动作同样消耗预算。"""
        if self.ended:
            raise RuntimeError("trajectory already ended")
        if self.actions_used >= self.max_actions:
            self.ended = True
            raise RuntimeError("action budget exhausted")

        self.actions_used += 1
        command = command.strip()

        if command.startswith("Search "):
            query = command.removeprefix("Search ").strip()
            if not query:
                return "INVALID: empty query"
            self.search_results = self.web.search(query)
            rendered = " | ".join(
                f"[{page.page_id}] {page.title} ({page.domain})"
                for page in self.search_results
            )
            return rendered or "NO RESULTS"

        if command.startswith("Clicked on link "):
            raw_id = command.removeprefix("Clicked on link ").strip()
            if not raw_id.isdigit():
                return "INVALID: link id must be an integer"
            page_id = int(raw_id)
            allowed = {page.page_id for page in self.search_results}
            if page_id not in allowed:
                return "INVALID: link is not in current search results"
            if self.current is not None:
                self.history.append(self.current)
            self.current = self.web.pages[page_id]
            return f"{self.current.title}\n{self.current.text}"

        if command.startswith("Find in page: "):
            needle = command.removeprefix("Find in page: ").strip()
            if self.current is None:
                return "INVALID: no open page"
            index = self.current.text.lower().find(needle.lower())
            return f"MATCH AT {index}" if index >= 0 else "NO MATCH"

        if command.startswith("Quote: "):
            quote = command.removeprefix("Quote: ").strip()
            if self.current is None:
                return "INVALID: no open page"
            if not quote or quote not in self.current.text:
                return "INVALID: quote must be an exact span of the open page"
            self.references.append(
                Reference(
                    page_id=self.current.page_id,
                    title=self.current.title,
                    domain=self.current.domain,
                    quote=quote,
                )
            )
            return f"SAVED REFERENCE [{len(self.references)}]"

        if command == "Back":
            if not self.history:
                return "INVALID: empty history"
            self.current = self.history.pop()
            return f"BACK TO {self.current.title}"

        if command == "End: Answer":
            if not self.references:
                return "INVALID: at least one reference is required"
            self.ended = True
            return "READY TO ANSWER"

        if command in {"End: Nonsense", "End: Controversial"}:
            self.ended = True
            return command.upper()

        return "INVALID: unsupported action"


@dataclass(frozen=True)
class CitationAudit:
    cited: tuple[int, ...]
    missing_claim_lines: tuple[int, ...]
    invalid_ids: tuple[int, ...]


CITATION = re.compile(r"\[(\d+)\]")


def audit_citations(answer: str, references: Sequence[Reference]) -> CitationAudit:
    """做结构审计，而不是事实核验或自然语言蕴含判断。"""
    valid_ids = set(range(1, len(references) + 1))
    cited = tuple(int(raw) for raw in CITATION.findall(answer))
    invalid = tuple(sorted(set(cited) - valid_ids))
    missing: list[int] = []

    for line_number, line in enumerate(answer.splitlines(), start=1):
        line = line.strip()
        if line and not line.startswith("#") and not CITATION.search(line):
            missing.append(line_number)

    return CitationAudit(cited, tuple(missing), invalid)


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    exp_value = exp(value)
    return exp_value / (1.0 + exp_value)


def preference_probability(reward_a: float, reward_b: float) -> float:
    """Bradley–Terry / Elo 风格偏好概率 P(A > B)。"""
    return sigmoid(reward_a - reward_b)


def pairwise_cross_entropy(
    reward_a: float, reward_b: float, target_a: float
) -> float:
    """target_a 为 1/0/0.5，分别表示 A 胜、B 胜、平局。"""
    probability = min(max(preference_probability(reward_a, reward_b), 1e-12), 1 - 1e-12)
    return -(target_a * log(probability) + (1 - target_a) * log(1 - probability))


@dataclass(frozen=True)
class Candidate:
    answer: str
    train_reward: float
    validation_reward: float


def best_of_n(candidates: Sequence[Candidate], n: int) -> Candidate:
    """从前 n 个完整轨迹中返回训练 reward model 评分最高者。"""
    if not 1 <= n <= len(candidates):
        raise ValueError("n must satisfy 1 <= n <= len(candidates)")
    return max(candidates[:n], key=lambda candidate: candidate.train_reward)


def expected_validation_reward(candidates: Iterable[Candidate], n: int) -> float:
    """WebGPT 附录 I 的组合数加权估计。

    先按训练 RM 分数从小到大排序。第 i 个样本恰好成为大小为 n 的子集最大值时，
    其权重为 C(i-1, n-1) / C(N, n)。独立 validation RM 只负责估计质量。
    """
    ordered = sorted(candidates, key=lambda candidate: candidate.train_reward)
    total = len(ordered)
    if not 1 <= n <= total:
        raise ValueError("n must satisfy 1 <= n <= number of candidates")

    denominator = comb(total, n)
    return sum(
        comb(index - 1, n - 1) / denominator * candidate.validation_reward
        for index, candidate in enumerate(ordered, start=1)
        if index >= n
    )


def browser_demo() -> tuple[TextBrowser, str]:
    pages = {
        1: Page(
            1,
            "Mission measurement note",
            "science.example",
            "The mission measured the planet with two independent instruments. "
            "The reported radius was inferred from repeated observations.",
        ),
        2: Page(
            2,
            "Anonymous spectacular claim",
            "rumor.example",
            "A viral post calls the planet the largest object ever observed.",
        ),
        3: Page(
            3,
            "Instrument calibration archive",
            "archive.example",
            "Calibration records explain the uncertainty range of the measurement.",
        ),
    }
    browser = TextBrowser(OfflineWeb(pages), "How was the planet radius estimated?")
    commands = [
        "Search planet mission measurement radius",
        "Clicked on link 1",
        "Find in page: independent instruments",
        "Quote: The mission measured the planet with two independent instruments.",
        "End: Answer",
    ]
    for command in commands:
        print(f"> {command}\n  {browser.act(command)}")

    answer = (
        "The estimate combined repeated observations from two independent instruments [1].\n"
        "Its exact uncertainty is not established by the saved excerpt [1]."
    )
    return browser, answer


def reward_demo() -> list[Candidate]:
    # 最后一个候选被 train RM 高估，模拟“搜索找到了 RM 漏洞”。
    return [
        Candidate("concise but incomplete", 0.10, 0.20),
        Candidate("supported answer", 0.62, 0.68),
        Candidate("well-supported synthesis", 0.81, 0.83),
        Candidate("confident citation cherry-picking", 0.97, 0.12),
    ]


def main() -> None:
    print("== 1. Offline browser trajectory ==")
    browser, answer = browser_demo()
    print("\nSaved references:")
    for index, reference in enumerate(browser.references, start=1):
        print(f"[{index}] {reference.title} — {reference.domain}: {reference.quote}")

    print("\n== 2. Citation structure audit ==")
    audit = audit_citations(answer, browser.references)
    print(audit)
    print("Note: passing this audit does NOT prove that the source or answer is true.")

    print("\n== 3. Pairwise reward model ==")
    probability = preference_probability(1.0, 0.0)
    print(f"P(A preferred | r_A-r_B=1) = {probability:.3f}")
    print(f"loss when A wins = {pairwise_cross_entropy(1.0, 0.0, 1.0):.3f}")
    print(f"loss on a tie    = {pairwise_cross_entropy(1.0, 0.0, 0.5):.3f}")

    print("\n== 4. Best-of-N and an independent validation RM ==")
    candidates = reward_demo()
    print("prefix selection is illustrative; expectation averages all size-n subsets")
    for n in (1, 2, 4):
        selected = best_of_n(candidates, n)
        expected = expected_validation_reward(candidates, n)
        print(
            f"n={n}: prefix_selected={selected.answer!r}, "
            f"train={selected.train_reward:.2f}, validation={selected.validation_reward:.2f}, "
            f"expected_validation_over_subsets={expected:.3f}"
        )


if __name__ == "__main__":
    main()
