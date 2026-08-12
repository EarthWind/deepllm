"""Foundation Models Report（2021）的零依赖系统风险教学实现。

这份报告提出的是一种技术—社会系统分析框架，而不是新的神经网络层。因此本文件
不伪造一个“Foundation Model 架构”，而把报告中最容易在工程里落空的概念写成
可执行对象：

1. foundation model -> adaptation -> deployment 的工件血缘；
2. homogenization 带来的共同模式（common-mode）相关失效；
3. intrinsic property、adaptation 与 deployment context 如何共同形成外部伤害；
4. 为什么平均 benchmark 分数会掩盖最差群体 / 最差切片；
5. 预训练的一次性成本要复用多少次，才可能被更便宜的适配摊薄。

运行：

    python3 papers/to-2026/code/foundation_models_report_demo.py

仅使用 Python 标准库。输出用于解释概念，不是安全认证、法律判断或真实碳核算。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Artifact:
    """生态链上的一个可审计工件，而不只是一个模型文件。"""

    name: str
    kind: str
    owner: str
    version: str
    inputs: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


class LineageGraph:
    """最小血缘图：检查依赖是否存在，并回答影响范围问题。"""

    def __init__(self, artifacts: Iterable[Artifact]) -> None:
        items = list(artifacts)
        self.artifacts = {item.name: item for item in items}
        if len(self.artifacts) != len(items):
            raise ValueError("artifact names must be unique")
        missing = {
            dependency
            for item in items
            for dependency in item.inputs
            if dependency not in self.artifacts
        }
        if missing:
            raise ValueError(f"unknown lineage inputs: {sorted(missing)}")
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError(f"lineage cycle detected at {name!r}")
            if name in visited:
                return
            visiting.add(name)
            for dependency in self.artifacts[name].inputs:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        for name in self.artifacts:
            visit(name)

    def descendants(self, source: str) -> list[str]:
        """返回 source 变化后所有可能需要重测的下游工件。"""

        if source not in self.artifacts:
            raise KeyError(source)
        affected: set[str] = set()
        frontier = [source]
        while frontier:
            current = frontier.pop()
            direct = [
                item.name
                for item in self.artifacts.values()
                if current in item.inputs and item.name not in affected
            ]
            affected.update(direct)
            frontier.extend(direct)
        affected.discard(source)
        return sorted(affected)

    def missing_evidence(self, required: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
        """按工件类型检查最小证据包；真实系统应使用正式 schema。"""

        gaps: dict[str, list[str]] = {}
        for item in self.artifacts.values():
            expected = required.get(item.kind, ())
            missing = sorted(set(expected) - set(item.evidence))
            if missing:
                gaps[item.name] = missing
        return gaps


@dataclass(frozen=True)
class CommonModeResult:
    systems: int
    core_failure_probability: float
    local_failure_probability: float
    marginal_failure_probability: float
    expected_failed_systems: float
    pairwise_failure_correlation: float
    probability_at_least_threshold: float
    independent_probability_at_least_threshold: float


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _binomial_tail(n: int, p: float, threshold: int) -> float:
    """P[X >= threshold], X ~ Binomial(n, p)，适合教学用的小 n。"""

    if threshold <= 0:
        return 1.0
    if threshold > n:
        return 0.0
    return sum(
        math.comb(n, k) * p**k * (1.0 - p) ** (n - k)
        for k in range(threshold, n + 1)
    )


def common_mode_analysis(
    systems: int,
    core_failure_probability: float,
    local_failure_probability: float,
    *,
    catastrophic_threshold: int | None = None,
) -> CommonModeResult:
    """比较共享底座与同边际失败率的独立系统。

    模型：所有系统共享 Bernoulli 变量 C；系统 i 另有独立局部变量 L_i。
    F_i = C OR L_i。若 C 发生，全部下游同时失败；否则只剩局部失败。

    对照组保留相同的单系统边际失败率 q，但令每个系统相互独立。这样比较的
    不是“共享模型是否让单个系统更差”，而是“共享依赖怎样改变尾部风险”。
    """

    if systems <= 0:
        raise ValueError("systems must be positive")
    _validate_probability(core_failure_probability, "core_failure_probability")
    _validate_probability(local_failure_probability, "local_failure_probability")
    threshold = catastrophic_threshold or max(1, math.ceil(0.8 * systems))
    if not 1 <= threshold <= systems:
        raise ValueError("catastrophic_threshold must be within [1, systems]")

    p_core = core_failure_probability
    p_local = local_failure_probability
    marginal = p_core + (1.0 - p_core) * p_local
    both_fail = p_core + (1.0 - p_core) * p_local**2
    variance = marginal * (1.0 - marginal)
    correlation = 0.0 if variance == 0.0 else (both_fail - marginal**2) / variance

    # 共享底座：C 发生时必然越过阈值；否则由局部二项分布决定。
    shared_tail = p_core + (1.0 - p_core) * _binomial_tail(
        systems, p_local, threshold
    )
    independent_tail = _binomial_tail(systems, marginal, threshold)
    return CommonModeResult(
        systems=systems,
        core_failure_probability=p_core,
        local_failure_probability=p_local,
        marginal_failure_probability=marginal,
        expected_failed_systems=systems * marginal,
        pairwise_failure_correlation=correlation,
        probability_at_least_threshold=shared_tail,
        independent_probability_at_least_threshold=independent_tail,
    )


@dataclass(frozen=True)
class RiskPath:
    """从底座内在缺陷到具体部署伤害的一条简化路径。"""

    name: str
    intrinsic_prevalence: float
    adaptation_retention: float
    exposure_probability: float
    impact: float
    control_effectiveness: float = 0.0

    def residual_risk(self) -> float:
        """演示性乘法模型，不声称真实社会伤害可以完全标量化。"""

        for field_name, value in (
            ("intrinsic_prevalence", self.intrinsic_prevalence),
            ("adaptation_retention", self.adaptation_retention),
            ("exposure_probability", self.exposure_probability),
            ("impact", self.impact),
            ("control_effectiveness", self.control_effectiveness),
        ):
            _validate_probability(value, field_name)
        return (
            self.intrinsic_prevalence
            * self.adaptation_retention
            * self.exposure_probability
            * self.impact
            * (1.0 - self.control_effectiveness)
        )

    def sensitivity(self) -> dict[str, float]:
        """各项提高 1 个百分点时风险的局部变化，用来找干预位置。"""

        values = {
            "intrinsic_prevalence": self.intrinsic_prevalence,
            "adaptation_retention": self.adaptation_retention,
            "exposure_probability": self.exposure_probability,
            "impact": self.impact,
            "remaining_after_control": 1.0 - self.control_effectiveness,
        }
        sensitivity: dict[str, float] = {}
        for key in values:
            product = 1.0
            for other_key, value in values.items():
                if other_key != key:
                    product *= value
            sensitivity[key] = product
        return sensitivity


@dataclass(frozen=True)
class SliceScore:
    name: str
    score: float
    weight: float
    minimum: float

    def __post_init__(self) -> None:
        _validate_probability(self.score, f"{self.name}.score")
        _validate_probability(self.minimum, f"{self.name}.minimum")
        if self.weight < 0:
            raise ValueError("slice weight must be non-negative")


@dataclass(frozen=True)
class EvaluationResult:
    weighted_average: float
    worst_slice_score: float
    failed_slices: tuple[str, ...]
    release_allowed: bool


def evaluate_slices(slices: Sequence[SliceScore]) -> EvaluationResult:
    """同时报告平均值、最差切片与硬门槛，避免一个总分决定发布。"""

    if not slices:
        raise ValueError("at least one slice is required")
    total_weight = sum(item.weight for item in slices)
    if total_weight <= 0:
        raise ValueError("slice weights must sum to a positive value")
    average = sum(item.score * item.weight for item in slices) / total_weight
    failures = tuple(sorted(item.name for item in slices if item.score < item.minimum))
    return EvaluationResult(
        weighted_average=average,
        worst_slice_score=min(item.score for item in slices),
        failed_slices=failures,
        release_allowed=not failures,
    )


@dataclass(frozen=True)
class AmortizationResult:
    break_even_tasks: int | None
    foundation_cost_at_break_even: float | None
    scratch_cost_at_break_even: float | None


def adaptation_break_even(
    pretraining_cost: float,
    adaptation_cost_per_task: float,
    scratch_cost_per_task: float,
) -> AmortizationResult:
    """求 P + nA <= nS 的最小整数 n。

    cost 可以是 kWh、kgCO2e、GPU-hour 或货币，但三项必须使用同一单位，且
    必须预先确定系统边界。若 A >= S，预训练成本永远无法靠该项复用摊薄。
    """

    if min(pretraining_cost, adaptation_cost_per_task, scratch_cost_per_task) < 0:
        raise ValueError("costs must be non-negative")
    saving = scratch_cost_per_task - adaptation_cost_per_task
    if saving <= 0:
        return AmortizationResult(None, None, None)
    tasks = max(1, math.ceil(pretraining_cost / saving))
    foundation = pretraining_cost + tasks * adaptation_cost_per_task
    scratch = tasks * scratch_cost_per_task
    # 浮点数接近整数边界时，ceil 可能受表示误差影响；用循环校正。
    while tasks > 1 and pretraining_cost + (tasks - 1) * adaptation_cost_per_task <= (
        tasks - 1
    ) * scratch_cost_per_task:
        tasks -= 1
    while foundation > scratch:
        tasks += 1
        foundation = pretraining_cost + tasks * adaptation_cost_per_task
        scratch = tasks * scratch_cost_per_task
    return AmortizationResult(tasks, foundation, scratch)


def _demo_lineage() -> dict[str, object]:
    artifacts = (
        Artifact("raw_web", "dataset", "data-team", "2021-08", evidence=("license",)),
        Artifact(
            "curated_mix",
            "dataset",
            "data-team",
            "v3",
            inputs=("raw_web",),
            evidence=("license", "provenance", "pii-audit"),
        ),
        Artifact(
            "fm_checkpoint",
            "foundation-model",
            "model-provider",
            "v1",
            inputs=("curated_mix",),
            evidence=("model-card", "capability-eval"),
        ),
        Artifact(
            "medical_adapter",
            "adapted-system",
            "health-team",
            "v7",
            inputs=("fm_checkpoint",),
            evidence=("domain-eval", "human-review"),
        ),
        Artifact(
            "triage_product",
            "deployment",
            "hospital",
            "2026.08",
            inputs=("medical_adapter",),
            evidence=("monitoring", "incident-plan"),
        ),
    )
    graph = LineageGraph(artifacts)
    required = {
        "dataset": ("license", "provenance", "pii-audit"),
        "foundation-model": ("model-card", "capability-eval", "risk-eval"),
        "adapted-system": ("domain-eval", "human-review"),
        "deployment": ("monitoring", "incident-plan", "appeal-channel"),
    }
    return {
        "affected_if_foundation_changes": graph.descendants("fm_checkpoint"),
        "evidence_gaps": graph.missing_evidence(required),
    }


def main() -> None:
    common_mode = common_mode_analysis(
        systems=20,
        core_failure_probability=0.02,
        local_failure_probability=0.03,
        catastrophic_threshold=16,
    )
    risk_path = RiskPath(
        name="under-representation -> domain error -> denied service",
        intrinsic_prevalence=0.30,
        adaptation_retention=0.70,
        exposure_probability=0.40,
        impact=0.90,
        control_effectiveness=0.50,
    )
    evaluation = evaluate_slices(
        (
            SliceScore("common-language", score=0.94, weight=0.80, minimum=0.85),
            SliceScore("low-resource-language", score=0.61, weight=0.05, minimum=0.80),
            SliceScore("long-context", score=0.82, weight=0.10, minimum=0.78),
            SliceScore("adversarial", score=0.58, weight=0.05, minimum=0.75),
        )
    )
    amortization = adaptation_break_even(
        pretraining_cost=1_000.0,
        adaptation_cost_per_task=5.0,
        scratch_cost_per_task=80.0,
    )
    report = {
        "lineage": _demo_lineage(),
        "homogenization": asdict(common_mode),
        "risk_path": {
            "residual_risk": risk_path.residual_risk(),
            "local_sensitivity": risk_path.sensitivity(),
        },
        "evaluation": asdict(evaluation),
        "amortization": asdict(amortization),
    }

    # 这些断言既是回归检查，也把报告最重要的判断写成机器可读规则。
    assert common_mode.probability_at_least_threshold > (
        common_mode.independent_probability_at_least_threshold
    )
    assert evaluation.weighted_average > 0.85
    assert not evaluation.release_allowed  # 高平均分不能覆盖切片门槛失败。
    assert amortization.break_even_tasks == 14
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
