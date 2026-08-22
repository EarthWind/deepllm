"""Teaching utilities inspired by *On the Dangers of Stochastic Parrots*.

This is not an implementation of a language model.  It is a small, dependency-
free audit notebook in script form: document a dataset, measure representation
gaps, estimate training emissions, and separate measured facts from assumptions.
Run with ``python stochastic_parrots_audit.py --test``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import argparse
import json
import math
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class DatasetCard:
    name: str
    source: str
    collection_date: str
    languages: tuple[str, ...]
    license: str
    consent_known: bool
    pii_removed: bool
    documented_limitations: tuple[str, ...]


@dataclass(frozen=True)
class EnergyEstimate:
    gpu_hours: float
    power_kw: float
    pue: float
    carbon_intensity_kg_per_kwh: float

    @property
    def energy_kwh(self) -> float:
        return self.gpu_hours * self.power_kw * self.pue

    @property
    def emissions_kg_co2e(self) -> float:
        return self.energy_kwh * self.carbon_intensity_kg_per_kwh


def representation_rates(
    counts: Mapping[str, int], *, denominator: int | None = None
) -> dict[str, float]:
    """Return percentages and make the denominator explicit.

    ``counts`` should be measured from a documented sample.  The function does
    not claim that a text share equals a population share; it only exposes the
    arithmetic that an audit should make visible.
    """
    total = sum(counts.values()) if denominator is None else denominator
    if total <= 0:
        raise ValueError("denominator must be positive")
    if any(v < 0 for v in counts.values()):
        raise ValueError("counts cannot be negative")
    return {group: 100.0 * n / total for group, n in counts.items()}


def normalized_gap(observed: Mapping[str, float], reference: Mapping[str, float]) -> dict[str, float]:
    """Compute observed/reference ratios, leaving missing reference groups visible."""
    groups = set(observed) | set(reference)
    result: dict[str, float] = {}
    for group in sorted(groups):
        ref = reference.get(group, 0.0)
        result[group] = math.inf if ref == 0 and observed.get(group, 0.0) > 0 else (
            0.0 if ref == 0 else observed.get(group, 0.0) / ref
        )
    return result


def estimate_emissions(estimate: EnergyEstimate) -> dict[str, float]:
    """Return auditable energy/carbon figures; inputs remain assumptions."""
    if estimate.gpu_hours < 0 or estimate.power_kw < 0:
        raise ValueError("hours and power must be non-negative")
    if estimate.pue < 1:
        raise ValueError("PUE should be >= 1")
    if estimate.carbon_intensity_kg_per_kwh < 0:
        raise ValueError("carbon intensity must be non-negative")
    return {
        "gpu_hours": estimate.gpu_hours,
        "power_kw": estimate.power_kw,
        "pue": estimate.pue,
        "energy_kwh": estimate.energy_kwh,
        "carbon_intensity_kg_per_kwh": estimate.carbon_intensity_kg_per_kwh,
        "emissions_kg_co2e": estimate.emissions_kg_co2e,
    }


def contamination_rate(records: Iterable[tuple[str, bool]]) -> float:
    """Fraction of records flagged by a reproducible overlap/PII checker."""
    rows = list(records)
    if not rows:
        raise ValueError("at least one record is required")
    return sum(flagged for _, flagged in rows) / len(rows)


def risk_register() -> list[dict[str, str]]:
    """A compact pre-development checklist, not a risk score."""
    return [
        {"risk": "数据来源与同意不清", "evidence": "来源、许可、抓取日期", "action": "暂停扩展并补齐 dataset card"},
        {"risk": "代表性缺口", "evidence": "语言/群体分布与目标人群对照", "action": "分层评测、补采样、限制使用范围"},
        {"risk": "身份与隐私泄露", "evidence": "PII/近重复/成员推断审计", "action": "去标识、删除、访问控制"},
        {"risk": "部署后伤害", "evidence": "领域红队与受影响群体反馈", "action": "人工复核、申诉通道、可撤销发布"},
        {"risk": "训练成本与机会成本", "evidence": "GPU 小时、PUE、碳强度、复现实验价值", "action": "先做目标-成本评估"},
    ]


def demo() -> None:
    card = DatasetCard(
        name="example-corpus",
        source="公开网页（示例）",
        collection_date="2021-01",
        languages=("zh", "en"),
        license="mixed / to be verified",
        consent_known=False,
        pii_removed=False,
        documented_limitations=("长尾语言不足", "来源质量异质", "近重复未完全审计"),
    )
    observed = representation_rates({"en": 920, "zh": 60, "other": 20})
    reference = {"en": 80.0, "zh": 15.0, "other": 5.0}
    energy = estimate_emissions(EnergyEstimate(1_200, 0.45, 1.2, 0.38))
    print("dataset card:", json.dumps(asdict(card), ensure_ascii=False, indent=2))
    print("observed share (%):", observed)
    print("observed/reference ratio:", normalized_gap(observed, reference))
    print("emissions estimate:", energy)
    print("risk register rows:", len(risk_register()))


def run_tests() -> None:
    assert representation_rates({"a": 3, "b": 1}) == {"a": 75.0, "b": 25.0}
    assert representation_rates({"a": 3}, denominator=4)["a"] == 75.0
    assert normalized_gap({"a": 20}, {"a": 10})["a"] == 2.0
    e = EnergyEstimate(10, 0.5, 1.2, 0.4)
    assert estimate_emissions(e)["energy_kwh"] == 6.0
    assert round(contamination_rate([("x", True), ("y", False)]), 3) == 0.5
    try:
        representation_rates({"a": 0})
    except ValueError:
        pass
    else:
        raise AssertionError("zero denominator should fail")
    print("all tests passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    run_tests() if args.test else demo()
