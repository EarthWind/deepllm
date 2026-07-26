#!/usr/bin/env python3
"""Minimal, dependency-free scaling-law fitting and visualization demo.

The default data are synthetic observations generated from the dataset-size
law in Kaplan et al. (2020), with small deterministic perturbations.  To fit
your own sweep, pass a CSV file containing ``scale,loss`` columns:

    python3 scaling_laws_demo.py --csv sweep.csv --floor 0

The script writes two SVG files used by the accompanying blog post.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Observation:
    """One controlled-sweep measurement."""

    scale: float
    loss: float


@dataclass(frozen=True)
class PowerLawFit:
    """Parameters of L(X) = floor + amplitude * X**(-alpha)."""

    floor: float
    amplitude: float
    alpha: float
    r_squared: float

    def predict(self, scale: float) -> float:
        if scale <= 0:
            raise ValueError("scale must be positive")
        return self.floor + self.amplitude * scale ** (-self.alpha)

    @property
    def characteristic_scale(self) -> float:
        """Return X_c when the fit is written as (X_c / X)**alpha."""

        return self.amplitude ** (1.0 / self.alpha)


def fit_power_law(
    observations: Iterable[Observation],
    *,
    floor: float = 0.0,
) -> PowerLawFit:
    """Fit a power law by ordinary least squares in log-log space.

    ``floor`` must be fixed before fitting.  Jointly estimating the floor and
    exponent is often ill-conditioned over a narrow scale range; use a
    physically motivated floor or compare several candidates explicitly.
    """

    points = sorted(observations, key=lambda item: item.scale)
    if len(points) < 3:
        raise ValueError("at least three observations are required")
    if any(point.scale <= 0 for point in points):
        raise ValueError("all scale values must be positive")
    if any(point.loss <= floor for point in points):
        raise ValueError("every loss must be greater than floor")

    xs = [math.log(point.scale) for point in points]
    ys = [math.log(point.loss - floor) for point in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)

    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("scale values must not all be equal")

    slope = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    ) / denominator
    intercept = y_mean - slope * x_mean
    alpha = -slope
    if alpha <= 0:
        raise ValueError("fitted exponent is not positive")

    residual = sum(
        (y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)
    )
    total = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 - residual / total if total else 1.0

    return PowerLawFit(
        floor=floor,
        amplitude=math.exp(intercept),
        alpha=alpha,
        r_squared=r_squared,
    )


def estimate_training_flops(params: float, tokens: float) -> float:
    """Kaplan-style dense Transformer training estimate C ~= 6 * N * D."""

    if params <= 0 or tokens <= 0:
        raise ValueError("params and tokens must be positive")
    return 6.0 * params * tokens


def kaplan_allocation(compute_multiplier: float) -> dict[str, float]:
    """Scale the compute-efficient allocation using the paper's exponents."""

    if compute_multiplier <= 0:
        raise ValueError("compute_multiplier must be positive")
    return {
        "model": compute_multiplier**0.73,
        "batch": compute_multiplier**0.24,
        "steps": compute_multiplier**0.03,
        "processed_tokens": compute_multiplier**0.27,
    }


def load_csv(path: Path) -> list[Observation]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"scale", "loss"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("CSV must contain scale,loss columns")
        return [
            Observation(scale=float(row["scale"]), loss=float(row["loss"]))
            for row in reader
        ]


def build_demo_observations() -> list[Observation]:
    """Create repeatable pseudo-measurements near alpha_D = 0.095."""

    start, stop, count = 2.2e7, 2.3e10, 12
    perturbations = (
        1.008,
        0.994,
        1.004,
        0.997,
        1.006,
        0.993,
        1.002,
        0.996,
        1.005,
        0.995,
        1.003,
        0.999,
    )
    observations: list[Observation] = []
    for index in range(count):
        fraction = index / (count - 1)
        tokens = start * (stop / start) ** fraction
        ideal_loss = (5.4e13 / tokens) ** 0.095
        observations.append(
            Observation(tokens, ideal_loss * perturbations[index])
        )
    return observations


def _log_position(value: float, low: float, high: float) -> float:
    return (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))


def _polyline(points: Sequence[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def write_fit_svg(
    observations: Sequence[Observation],
    fit: PowerLawFit,
    output_path: Path,
) -> None:
    """Render the fitted log-log curve without plotting dependencies."""

    width, height = 1400, 760
    left, top, plot_width, plot_height = 100, 150, 850, 500
    x_low = min(point.scale for point in observations) / 1.15
    x_high = max(point.scale for point in observations) * 1.15
    observed_losses = [point.loss for point in observations]
    predicted_losses = [fit.predict(point.scale) for point in observations]
    y_low = min(observed_losses + predicted_losses) * 0.90
    y_high = max(observed_losses + predicted_losses) * 1.10

    def map_x(value: float) -> float:
        return left + _log_position(value, x_low, x_high) * plot_width

    def map_y(value: float) -> float:
        return top + (1.0 - _log_position(value, y_low, y_high)) * plot_height

    curve = []
    samples = 160
    for index in range(samples):
        fraction = index / (samples - 1)
        scale = x_low * (x_high / x_low) ** fraction
        curve.append((map_x(scale), map_y(fit.predict(scale))))

    x_ticks = (1e8, 1e9, 1e10)
    y_ticks = (2.0, 2.5, 3.0, 4.0)
    ten_x_improvement = 1.0 - 10.0 ** (-fit.alpha)
    circles = "\n".join(
        (
            f'<circle cx="{map_x(point.scale):.1f}" '
            f'cy="{map_y(point.loss):.1f}" r="7" '
            'fill="#f59e0b" stroke="#fef3c7" stroke-width="2"/>'
        )
        for point in observations
    )
    vertical_grid = "\n".join(
        (
            f'<line x1="{map_x(tick):.1f}" y1="{top}" '
            f'x2="{map_x(tick):.1f}" y2="{top + plot_height}" '
            'class="grid"/>'
        )
        for tick in x_ticks
        if x_low <= tick <= x_high
    )
    horizontal_grid = "\n".join(
        (
            f'<line x1="{left}" y1="{map_y(tick):.1f}" '
            f'x2="{left + plot_width}" y2="{map_y(tick):.1f}" '
            'class="grid"/>'
        )
        for tick in y_ticks
        if y_low <= tick <= y_high
    )
    x_labels = "\n".join(
        (
            f'<text x="{map_x(tick):.1f}" y="{top + plot_height + 34}" '
            f'class="tick" text-anchor="middle">10^{int(math.log10(tick))}</text>'
        )
        for tick in x_ticks
        if x_low <= tick <= x_high
    )
    y_labels = "\n".join(
        (
            f'<text x="{left - 18}" y="{map_y(tick) + 6:.1f}" '
            f'class="tick" text-anchor="end">{tick:g}</text>'
        )
        for tick in y_ticks
        if y_low <= tick <= y_high
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">在双对数坐标中拟合数据规模扩展律</title>
  <desc id="desc">橙色点是带轻微扰动的合成实验数据，蓝色直线是依赖零第三方库的 Python 最小二乘拟合结果。</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#071426"/>
      <stop offset="1" stop-color="#102951"/>
    </linearGradient>
    <linearGradient id="line" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#38bdf8"/>
      <stop offset="1" stop-color="#818cf8"/>
    </linearGradient>
    <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="5" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <style>
      .title {{ font: 700 34px "Noto Sans CJK SC","Microsoft YaHei",sans-serif; fill: #f8fafc; }}
      .subtitle {{ font: 400 17px "Noto Sans CJK SC","Microsoft YaHei",sans-serif; fill: #cbd5e1; }}
      .section {{ font: 700 22px "Noto Sans CJK SC","Microsoft YaHei",sans-serif; fill: #e2e8f0; }}
      .label {{ font: 600 18px "Noto Sans CJK SC","Microsoft YaHei",sans-serif; fill: #f8fafc; }}
      .small {{ font: 400 16px "Noto Sans CJK SC","Microsoft YaHei",sans-serif; fill: #cbd5e1; }}
      .mono {{ font: 500 17px "JetBrains Mono","Noto Sans Mono CJK SC",monospace; fill: #bae6fd; }}
      .tick {{ font: 400 15px "JetBrains Mono","Noto Sans Mono CJK SC",monospace; fill: #94a3b8; }}
      .grid {{ stroke: #334155; stroke-width: 1.2; stroke-dasharray: 5 7; }}
    </style>
  </defs>
  <rect width="{width}" height="{height}" rx="28" fill="url(#bg)"/>
  <text x="70" y="66" class="title">为什么幂律在 log-log 图上接近直线？</text>
  <text x="70" y="98" class="subtitle">示例使用论文的数据受限形式 L(D) = (D_c / D)^α_D；点为可复现的合成观测，不是论文原始数据。</text>
  <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" rx="16" fill="#0b1930" stroke="#334155"/>
  {vertical_grid}
  {horizontal_grid}
  <polyline points="{_polyline(curve)}" fill="none" stroke="url(#line)" stroke-width="5" filter="url(#glow)"/>
  {circles}
  {x_labels}
  {y_labels}
  <text x="{left + plot_width / 2}" y="{height - 42}" class="label" text-anchor="middle">训练数据量 D（tokens，对数刻度）</text>
  <text x="34" y="{top + plot_height / 2}" class="label" text-anchor="middle" transform="rotate(-90 34 {top + plot_height / 2})">验证损失 L（对数刻度）</text>

  <rect x="1000" y="150" width="330" height="154" rx="18" fill="#112542" stroke="#38bdf8" stroke-width="2"/>
  <text x="1030" y="190" class="section">拟合结果</text>
  <text x="1030" y="230" class="mono">α = {fit.alpha:.4f}</text>
  <text x="1030" y="262" class="mono">R² = {fit.r_squared:.4f}</text>
  <text x="1030" y="288" class="small">目标值：论文 α_D ≈ 0.095</text>

  <rect x="1000" y="326" width="330" height="154" rx="18" fill="#112542" stroke="#f59e0b" stroke-width="2"/>
  <text x="1030" y="366" class="section">读懂指数</text>
  <text x="1030" y="407" class="mono">D × 10</text>
  <text x="1030" y="440" class="label">拟合项下降 {ten_x_improvement:.1%}</text>
  <text x="1030" y="466" class="small">幂指数很小，边际收益递减很慢</text>

  <rect x="1000" y="502" width="330" height="148" rx="18" fill="#112542" stroke="#818cf8" stroke-width="2"/>
  <text x="1030" y="542" class="section">拟合边界</text>
  <text x="1030" y="579" class="small">先保证模型规模、算力不构成瓶颈；</text>
  <text x="1030" y="608" class="small">不要把受限区间混进单变量拟合。</text>
  <text x="1030" y="637" class="small">外推必须报告区间与误差。</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def write_allocation_svg(output_path: Path) -> None:
    """Render the exponent allocation behind the paper's Figure 3 intuition."""

    compute_exponent = 9.0
    model_exponent = compute_exponent * 0.73
    batch_exponent = compute_exponent * 0.24
    step_exponent = compute_exponent * 0.03
    bar_x, bar_y, bar_width, bar_height = 100, 255, 1200, 110
    unit = bar_width / compute_exponent
    model_width = model_exponent * unit
    batch_width = batch_exponent * unit
    step_width = step_exponent * unit
    allocation = kaplan_allocation(1e9)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="720" viewBox="0 0 1400 720" role="img" aria-labelledby="title desc">
  <title id="title">Kaplan 扩展律的固定算力分配</title>
  <desc id="desc">算力增加十亿倍时，论文拟合建议模型规模增加约 6.57 个数量级，batch 增加 2.16 个数量级，串行训练步数只增加 0.27 个数量级。</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#071426"/>
      <stop offset="1" stop-color="#102951"/>
    </linearGradient>
    <linearGradient id="model" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#6366f1"/>
      <stop offset="1" stop-color="#8b5cf6"/>
    </linearGradient>
    <linearGradient id="batch" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#0ea5e9"/>
      <stop offset="1" stop-color="#38bdf8"/>
    </linearGradient>
    <style>
      .title {{ font: 700 34px "Noto Sans CJK SC","Microsoft YaHei",sans-serif; fill: #f8fafc; }}
      .subtitle {{ font: 400 17px "Noto Sans CJK SC","Microsoft YaHei",sans-serif; fill: #cbd5e1; }}
      .section {{ font: 700 22px "Noto Sans CJK SC","Microsoft YaHei",sans-serif; fill: #e2e8f0; }}
      .label {{ font: 600 18px "Noto Sans CJK SC","Microsoft YaHei",sans-serif; fill: #f8fafc; }}
      .small {{ font: 400 16px "Noto Sans CJK SC","Microsoft YaHei",sans-serif; fill: #cbd5e1; }}
      .mono {{ font: 500 17px "JetBrains Mono","Noto Sans Mono CJK SC",monospace; fill: #e0f2fe; }}
      .tick {{ font: 400 14px "JetBrains Mono","Noto Sans Mono CJK SC",monospace; fill: #94a3b8; }}
    </style>
  </defs>
  <rect width="1400" height="720" rx="28" fill="url(#bg)"/>
  <text x="70" y="66" class="title">固定预算怎么分？先在指数空间里看</text>
  <text x="70" y="98" class="subtitle">示意 C 放大 10⁹ 倍。因为 C ≈ 6NBS，log C 的增长可拆成 log N、log B 与 log S。</text>

  <text x="{bar_x}" y="205" class="section">新增 9 个数量级的训练算力</text>
  <rect x="{bar_x}" y="{bar_y}" width="{bar_width}" height="{bar_height}" rx="18" fill="#0b1930" stroke="#475569" stroke-width="2"/>
  <path d="M{bar_x + 18} {bar_y} H{bar_x + model_width} V{bar_y + bar_height} H{bar_x + 18} Q{bar_x} {bar_y + bar_height} {bar_x} {bar_y + bar_height - 18} V{bar_y + 18} Q{bar_x} {bar_y} {bar_x + 18} {bar_y}" fill="url(#model)"/>
  <rect x="{bar_x + model_width}" y="{bar_y}" width="{batch_width}" height="{bar_height}" fill="url(#batch)"/>
  <path d="M{bar_x + model_width + batch_width} {bar_y} H{bar_x + bar_width - 18} Q{bar_x + bar_width} {bar_y} {bar_x + bar_width} {bar_y + 18} V{bar_y + bar_height - 18} Q{bar_x + bar_width} {bar_y + bar_height} {bar_x + bar_width - 18} {bar_y + bar_height} H{bar_x + model_width + batch_width} Z" fill="#f59e0b"/>
  <text x="{bar_x + model_width / 2}" y="{bar_y + 48}" class="label" text-anchor="middle">模型规模 N</text>
  <text x="{bar_x + model_width / 2}" y="{bar_y + 80}" class="mono" text-anchor="middle">+6.57 orders</text>
  <text x="{bar_x + model_width + batch_width / 2}" y="{bar_y + 48}" class="label" text-anchor="middle">Batch B</text>
  <text x="{bar_x + model_width + batch_width / 2}" y="{bar_y + 80}" class="mono" text-anchor="middle">+2.16</text>
  <path d="M{bar_x + model_width + batch_width + step_width / 2} {bar_y + bar_height + 8} V{bar_y + bar_height + 50} H{bar_x + bar_width - 90}" fill="none" stroke="#f59e0b" stroke-width="2"/>
  <text x="{bar_x + bar_width - 82}" y="{bar_y + bar_height + 56}" class="label">串行步数 S：+0.27</text>

  <g>
    <rect x="100" y="465" width="360" height="154" rx="18" fill="#112542" stroke="#8b5cf6" stroke-width="2"/>
    <text x="130" y="505" class="section">模型规模</text>
    <text x="130" y="548" class="mono">N × {allocation["model"]:.2e}</text>
    <text x="130" y="582" class="small">N ∝ C^0.73</text>

    <rect x="520" y="465" width="360" height="154" rx="18" fill="#112542" stroke="#38bdf8" stroke-width="2"/>
    <text x="550" y="505" class="section">已处理 token</text>
    <text x="550" y="548" class="mono">D_processed × {allocation["processed_tokens"]:.0f}</text>
    <text x="550" y="582" class="small">B × S ∝ C^0.27</text>

    <rect x="940" y="465" width="360" height="154" rx="18" fill="#112542" stroke="#f59e0b" stroke-width="2"/>
    <text x="970" y="505" class="section">串行训练时间</text>
    <text x="970" y="548" class="mono">S × {allocation["steps"]:.2f}</text>
    <text x="970" y="582" class="small">S ∝ C^0.03，几乎不增长</text>
  </g>
  <text x="700" y="680" class="small" text-anchor="middle">这是 Kaplan 2020 在其数据、架构和早停定义下的经验外推；不是跨时代不变的训练配方。</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parents[1] / "images"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        help="optional CSV file with scale,loss columns",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=0.0,
        help="fixed irreducible-loss floor used by the fit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="directory for generated SVG files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observations = (
        load_csv(args.csv) if args.csv else build_demo_observations()
    )
    fit = fit_power_law(observations, floor=args.floor)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fit_path = args.output_dir / "scaling-laws-fit.svg"
    allocation_path = args.output_dir / "scaling-laws-compute-allocation.svg"
    write_fit_svg(observations, fit, fit_path)
    write_allocation_svg(allocation_path)

    example_flops = estimate_training_flops(params=1e9, tokens=20e9)
    print(f"alpha={fit.alpha:.6f}")
    print(f"R^2={fit.r_squared:.6f}")
    print(f"characteristic_scale={fit.characteristic_scale:.6e}")
    print(f"example_training_flops={example_flops:.6e}")
    print(f"wrote {fit_path}")
    print(f"wrote {allocation_path}")


if __name__ == "__main__":
    main()
