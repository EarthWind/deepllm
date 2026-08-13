"""ZeRO (2019/2020) 核心机制的零依赖教学实现。

这不是 DeepSpeed 的替代训练器，也不会启动多进程、CUDA 或 NCCL。它用
Python 列表把论文最重要的等价关系做成可执行断言：

1. 重算 mixed-precision Adam 的 16 bytes/parameter 显存账本；
2. 展示 ZeRO-DP 三个累积阶段怎样分片 optimizer、gradient、parameter；
3. 用 reduce-scatter gradient -> owner update -> all-gather parameter 模拟
   ZeRO-1/2 的一次 Adam step，并与普通数据并行逐项比较；
4. 展示 ZeRO-3 按 layer 临时 materialize 参数的生命周期；
5. 重算论文采用的简化通信量：DP/Stage 1/2 为 2*P，Stage 3 为 3*P；
6. 区分 ZeRO-DP model states 与 ZeRO-R residual states。

运行：

    python3 papers/to-2026/code/zero_minimal.py
    python3 papers/to-2026/code/zero_minimal.py --parameters 7500000000 --dp 64
    python3 papers/to-2026/code/zero_minimal.py --json

真实训练请使用官方 DeepSpeed、PyTorch distributed 与 NCCL。这里刻意保持
标准库零依赖，以便在没有 GPU 的机器上逐行核对数学和所有权变化。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from typing import Iterable, Sequence


Vector = list[float]


def split_evenly(values: Sequence[float], parts: int) -> list[Vector]:
    """把一维张量切成等长 shards；教学代码要求严格可整除。"""

    if parts <= 0:
        raise ValueError("parts must be positive")
    if len(values) % parts:
        raise ValueError("vector length must be divisible by parts")
    width = len(values) // parts
    return [list(values[i * width : (i + 1) * width]) for i in range(parts)]


def all_gather(shards: Sequence[Sequence[float]]) -> Vector:
    """模拟把每个 rank 拥有的不同 shard 拼成完整张量。"""

    return [value for shard in shards for value in shard]


def average_vectors(vectors: Sequence[Sequence[float]]) -> Vector:
    if not vectors:
        raise ValueError("at least one vector is required")
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise ValueError("all vectors must have the same length")
    return [sum(vector[i] for vector in vectors) / len(vectors) for i in range(width)]


def reduce_scatter_mean(
    local_gradients: Sequence[Sequence[float]],
) -> list[Vector]:
    """先对 DP gradients 求平均，再让 rank i 只保留第 i 个 shard。"""

    return split_evenly(average_vectors(local_gradients), len(local_gradients))


@dataclass(frozen=True)
class MemoryLedger:
    stage: int
    parameter_bytes: float
    gradient_bytes: float
    optimizer_bytes: float

    @property
    def total_bytes(self) -> float:
        return self.parameter_bytes + self.gradient_bytes + self.optimizer_bytes


def memory_ledger(
    parameters: int,
    dp_degree: int,
    stage: int,
    *,
    optimizer_multiplier: int = 12,
) -> MemoryLedger:
    """重算论文 Section 5 的 model-state 显存公式。

    optimizer_multiplier=12 对应 mixed-precision Adam 的 FP32 master weights、
    first moment 和 second moment，各 4 bytes/parameter。
    """

    if parameters <= 0 or dp_degree <= 0:
        raise ValueError("parameters and dp_degree must be positive")
    if stage not in (0, 1, 2, 3):
        raise ValueError("stage must be 0, 1, 2, or 3")

    parameter_divisor = dp_degree if stage >= 3 else 1
    gradient_divisor = dp_degree if stage >= 2 else 1
    optimizer_divisor = dp_degree if stage >= 1 else 1
    return MemoryLedger(
        stage=stage,
        parameter_bytes=2.0 * parameters / parameter_divisor,
        gradient_bytes=2.0 * parameters / gradient_divisor,
        optimizer_bytes=optimizer_multiplier * parameters / optimizer_divisor,
    )


def paper_communication_elements(parameters: int, stage: int) -> int:
    """论文的带宽主导简化模型，不包含 (N-1)/N 等 ring 有限规模修正。

    baseline / Stage 1 / Stage 2:
        gradient reduce-scatter P + parameter all-gather P = 2P

    Stage 3:
        forward parameter gather P + backward parameter gather P
        + gradient reduce-scatter P = 3P
    """

    if parameters <= 0:
        raise ValueError("parameters must be positive")
    if stage not in (0, 1, 2, 3):
        raise ValueError("stage must be 0, 1, 2, or 3")
    return parameters * (3 if stage == 3 else 2)


@dataclass(frozen=True)
class AdamState:
    parameters: Vector
    first_moment: Vector
    second_moment: Vector


def adam_update(
    parameters: Sequence[float],
    gradients: Sequence[float],
    first_moment: Sequence[float],
    second_moment: Sequence[float],
    *,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> AdamState:
    if not (
        len(parameters)
        == len(gradients)
        == len(first_moment)
        == len(second_moment)
    ):
        raise ValueError("Adam tensors must have equal lengths")

    new_first = [
        beta1 * old + (1.0 - beta1) * gradient
        for old, gradient in zip(first_moment, gradients)
    ]
    new_second = [
        beta2 * old + (1.0 - beta2) * gradient * gradient
        for old, gradient in zip(second_moment, gradients)
    ]
    # 为了突出分片等价性，这里省略与 shard ownership 无关的 bias correction。
    new_parameters = [
        value - learning_rate * moment / (math.sqrt(variance) + epsilon)
        for value, moment, variance in zip(parameters, new_first, new_second)
    ]
    return AdamState(new_parameters, new_first, new_second)


@dataclass(frozen=True)
class ZeROStepTrace:
    parameters: Vector
    parameter_shards: tuple[Vector, ...]
    gradient_shards: tuple[Vector, ...]
    first_moment_shards: tuple[Vector, ...]
    second_moment_shards: tuple[Vector, ...]
    collectives: tuple[str, ...]


def zero_sharded_adam_step(
    replicated_parameters: Sequence[float],
    local_gradients: Sequence[Sequence[float]],
    first_moment_shards: Sequence[Sequence[float]],
    second_moment_shards: Sequence[Sequence[float]],
    *,
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> ZeROStepTrace:
    """模拟 ZeRO owner-computes optimizer step。

    每个 rank 从自己的 microbatch 得到完整 local gradient。reduce-scatter 后，
    rank i 只拿到全局平均 gradient 的第 i 个 shard，也只更新对应 parameter、
    FP32 master/Adam state shard。最后 all-gather 更新后的低精度参数供下一步用。

    这条数据流与 ZeRO-2 的持久状态最接近；ZeRO-1 的最终 gradient shard 也可
    如此产生，但在 backward 期间仍保留完整 local gradient，故显存公式不同。
    """

    world_size = len(local_gradients)
    if world_size == 0:
        raise ValueError("at least one rank is required")
    parameter_shards = split_evenly(replicated_parameters, world_size)
    gradient_shards = reduce_scatter_mean(local_gradients)
    if not (
        len(first_moment_shards)
        == len(second_moment_shards)
        == world_size
    ):
        raise ValueError("one optimizer-state shard is required per rank")

    updated = [
        adam_update(
            parameter_shard,
            gradient_shard,
            first_shard,
            second_shard,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )
        for parameter_shard, gradient_shard, first_shard, second_shard in zip(
            parameter_shards,
            gradient_shards,
            first_moment_shards,
            second_moment_shards,
        )
    ]
    updated_parameter_shards = tuple(state.parameters for state in updated)
    return ZeROStepTrace(
        parameters=all_gather(updated_parameter_shards),
        parameter_shards=updated_parameter_shards,
        gradient_shards=tuple(gradient_shards),
        first_moment_shards=tuple(state.first_moment for state in updated),
        second_moment_shards=tuple(state.second_moment for state in updated),
        collectives=("REDUCE_SCATTER gradients", "ALL_GATHER updated parameters"),
    )


@dataclass(frozen=True)
class LayerMaterialization:
    phase: str
    layer: str
    persistent_owned_elements: int
    transient_full_layer_elements: int
    action: str


def zero3_layer_schedule(
    layer_sizes: Sequence[tuple[str, int]], dp_degree: int
) -> list[LayerMaterialization]:
    """展示 Stage 3 的 temporal liveness，而非实现真实 prefetch engine。"""

    if dp_degree <= 0:
        raise ValueError("dp_degree must be positive")
    if any(size <= 0 or size % dp_degree for _, size in layer_sizes):
        raise ValueError("each layer size must be positive and divisible by dp_degree")

    events: list[LayerMaterialization] = []
    for phase, layers in (
        ("forward", layer_sizes),
        ("backward", tuple(reversed(layer_sizes))),
    ):
        for name, size in layers:
            events.append(
                LayerMaterialization(
                    phase=phase,
                    layer=name,
                    persistent_owned_elements=size // dp_degree,
                    transient_full_layer_elements=size,
                    action="gather -> compute -> discard non-owned shards",
                )
            )
    return events


def vectors_close(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        for a, b in zip(left, right)
    )


def gibibytes(byte_count: float) -> float:
    return byte_count / 2**30


def decimal_gigabytes(byte_count: float) -> float:
    return byte_count / 1e9


def ledger_rows(parameters: int, dp_degree: int) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    baseline = memory_ledger(parameters, dp_degree, 0).total_bytes
    for stage in range(4):
        ledger = memory_ledger(parameters, dp_degree, stage)
        rows.append(
            {
                "stage": stage,
                "parameter_GB": round(decimal_gigabytes(ledger.parameter_bytes), 3),
                "gradient_GB": round(decimal_gigabytes(ledger.gradient_bytes), 3),
                "optimizer_GB": round(decimal_gigabytes(ledger.optimizer_bytes), 3),
                "total_GB": round(decimal_gigabytes(ledger.total_bytes), 3),
                "reduction_x": round(baseline / ledger.total_bytes, 3),
                "communication_P": paper_communication_elements(parameters, stage)
                / parameters,
            }
        )
    return rows


def assert_paper_examples() -> None:
    seven_point_five_b = 7_500_000_000
    rows = ledger_rows(seven_point_five_b, 64)
    assert math.isclose(rows[0]["total_GB"], 120.0)
    assert math.isclose(rows[1]["total_GB"], 31.406)
    assert math.isclose(rows[2]["total_GB"], 16.641)
    assert math.isclose(rows[3]["total_GB"], 1.875)

    one_trillion = memory_ledger(1_000_000_000_000, 1024, 3)
    assert math.isclose(decimal_gigabytes(one_trillion.total_bytes), 15.625)


def assert_sharded_adam_equivalence() -> ZeROStepTrace:
    parameters = [0.12, -0.25, 0.40, 0.31, -0.08, 0.77, -0.52, 0.09]
    local_gradients = [
        [0.10, -0.20, 0.05, 0.30, -0.12, 0.08, 0.20, -0.04],
        [0.06, -0.10, 0.09, 0.26, -0.08, 0.12, 0.16, -0.02],
        [0.14, -0.24, 0.03, 0.22, -0.10, 0.04, 0.18, -0.06],
        [0.10, -0.18, 0.07, 0.34, -0.14, 0.16, 0.22, -0.08],
    ]
    world_size = len(local_gradients)
    shard_width = len(parameters) // world_size
    zero_state = [[0.0] * shard_width for _ in range(world_size)]

    dense = adam_update(
        parameters,
        average_vectors(local_gradients),
        [0.0] * len(parameters),
        [0.0] * len(parameters),
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    sharded = zero_sharded_adam_step(
        parameters,
        local_gradients,
        zero_state,
        zero_state,
    )

    assert vectors_close(dense.parameters, sharded.parameters)
    assert vectors_close(dense.first_moment, all_gather(sharded.first_moment_shards))
    assert vectors_close(dense.second_moment, all_gather(sharded.second_moment_shards))
    return sharded


def print_table(rows: Iterable[dict[str, float | int]]) -> None:
    print("stage | params GB | grads GB | optimizer GB | total GB | save | comm")
    print("------|-----------|----------|--------------|----------|------|-----")
    for row in rows:
        print(
            f"{row['stage']:>5} | {row['parameter_GB']:>9.3f} | "
            f"{row['gradient_GB']:>8.3f} | {row['optimizer_GB']:>12.3f} | "
            f"{row['total_GB']:>8.3f} | {row['reduction_x']:>4.1f}x | "
            f"{row['communication_P']:.0f}P"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=int, default=7_500_000_000)
    parser.add_argument("--dp", type=int, default=64)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    assert_paper_examples()
    trace = assert_sharded_adam_equivalence()
    rows = ledger_rows(args.parameters, args.dp)
    schedule = zero3_layer_schedule(
        (("embedding", 1_024), ("transformer_0", 4_096), ("lm_head", 1_024)),
        dp_degree=4,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "parameters": args.parameters,
                    "dp_degree": args.dp,
                    "memory": rows,
                    "adam_equivalence": {
                        "passed": True,
                        "collectives": trace.collectives,
                        "gradient_shards": trace.gradient_shards,
                        "updated_parameters": trace.parameters,
                    },
                    "zero3_schedule": [asdict(event) for event in schedule],
                    "paper_checks": {
                        "7.5B_dp64_stage_totals_GB": [
                            row["total_GB"]
                            for row in ledger_rows(7_500_000_000, 64)
                        ],
                        "1T_dp1024_stage3_GB": decimal_gigabytes(
                            memory_ledger(1_000_000_000_000, 1024, 3).total_bytes
                        ),
                        "activation_checkpoint_33GB_mp16_GB": 33 / 16,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(f"ZeRO model-state ledger: P={args.parameters:,}, DP={args.dp}")
    print_table(rows)
    print()
    print("dense Adam == sharded owner-computes Adam: PASS")
    print("collectives:", " -> ".join(trace.collectives))
    print("gradient ownership:", trace.gradient_shards)
    print()
    print("ZeRO-3 temporal schedule (parameters are full only around one layer):")
    for event in schedule:
        print(
            f"  {event.phase:8s} {event.layer:13s} "
            f"owned={event.persistent_owned_elements:4d}, "
            f"live={event.transient_full_layer_elements:4d}: {event.action}"
        )
    print()
    print("paper examples: PASS")
    print("  7.5B, DP=64 totals: 120.000 / 31.406 / 16.641 / 1.875 GB")
    print("  1T, DP=1024, Stage 3: 15.625 GB of model states per GPU")
    print("  100B activation checkpoints, MP=16: 33 GB -> about 2.06 GB")
    print("note: activation, temporary buffers and fragmentation are not in 16P")


if __name__ == "__main__":
    main()
