"""Megatron-LM (2019) 张量并行核心机制的零依赖教学实现。

这不是 NVIDIA Megatron-LM 的替代训练器，也不启动多进程/NCCL。它在一台
普通机器上用 Python 列表模拟论文中最关键的分布式等价性：

1. MLP 第一层按输出列切分、第二层按输入行切分；
2. 非线性 GeLU 留在本地，两次 GEMM 中间不通信；
3. forward 在第二层后 all-reduce，backward 在输入梯度上 all-reduce；
4. 词表并行交叉熵不 all-gather 巨大的完整 logits；
5. tensor-parallel 与 data-parallel 通信组正交组合；
6. 重算论文中的参数量、词表 padding、弱扩展效率与通信量。

运行：

    python3 papers/to-2026/code/megatron_lm_minimal.py

真实训练应使用官方 Megatron-LM/Megatron-Core、PyTorch distributed 与 NCCL；
这里刻意保持 Python 标准库零依赖，以便逐行核对矩阵和通信语义。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Callable, Sequence


Matrix = list[list[float]]


def shape(matrix: Matrix) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be non-empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix must be rectangular")
    return len(matrix), width


def transpose(matrix: Matrix) -> Matrix:
    rows, cols = shape(matrix)
    return [[matrix[row][col] for row in range(rows)] for col in range(cols)]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    left_rows, inner = shape(left)
    right_rows, right_cols = shape(right)
    if inner != right_rows:
        raise ValueError(f"matmul mismatch: {shape(left)} @ {shape(right)}")
    return [
        [
            sum(left[row][k] * right[k][col] for k in range(inner))
            for col in range(right_cols)
        ]
        for row in range(left_rows)
    ]


def elementwise(
    matrix: Matrix, function: Callable[[float], float]
) -> Matrix:
    return [[function(value) for value in row] for row in matrix]


def hadamard(left: Matrix, right: Matrix) -> Matrix:
    if shape(left) != shape(right):
        raise ValueError("hadamard operands must have equal shapes")
    return [
        [a * b for a, b in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def add_matrices(matrices: Sequence[Matrix]) -> Matrix:
    """模拟 SUM all-reduce 的数学结果。"""

    if not matrices:
        raise ValueError("at least one matrix is required")
    rows, cols = shape(matrices[0])
    if any(shape(matrix) != (rows, cols) for matrix in matrices):
        raise ValueError("all-reduce inputs must have equal shapes")
    return [
        [sum(matrix[row][col] for matrix in matrices) for col in range(cols)]
        for row in range(rows)
    ]


def split_columns(matrix: Matrix, parts: int) -> list[Matrix]:
    rows, cols = shape(matrix)
    if parts <= 0 or cols % parts:
        raise ValueError("column count must be divisible by parts")
    width = cols // parts
    return [
        [row[rank * width : (rank + 1) * width] for row in matrix]
        for rank in range(parts)
    ]


def split_rows(matrix: Matrix, parts: int) -> list[Matrix]:
    rows, _ = shape(matrix)
    if parts <= 0 or rows % parts:
        raise ValueError("row count must be divisible by parts")
    height = rows // parts
    return [matrix[rank * height : (rank + 1) * height] for rank in range(parts)]


def concatenate_columns(matrices: Sequence[Matrix]) -> Matrix:
    if not matrices:
        raise ValueError("at least one matrix is required")
    rows = shape(matrices[0])[0]
    if any(shape(matrix)[0] != rows for matrix in matrices):
        raise ValueError("all matrices must have equal row counts")
    return [
        [value for matrix in matrices for value in matrix[row]]
        for row in range(rows)
    ]


def matrices_close(
    left: Matrix, right: Matrix, *, tolerance: float = 1e-9
) -> bool:
    return shape(left) == shape(right) and all(
        math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)
        for left_row, right_row in zip(left, right)
        for a, b in zip(left_row, right_row)
    )


def gelu(value: float) -> float:
    """精确 GeLU：x Φ(x)。"""

    return 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))


def gelu_derivative(value: float) -> float:
    normal_pdf = math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)
    normal_cdf = 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
    return normal_cdf + value * normal_pdf


@dataclass(frozen=True)
class MLPForward:
    output: Matrix
    pre_activation: Matrix
    hidden: Matrix


@dataclass(frozen=True)
class MLPBackward:
    input_gradient: Matrix
    first_weight_gradient: Matrix
    second_weight_gradient: Matrix


def dense_mlp_forward(inputs: Matrix, first: Matrix, second: Matrix) -> MLPForward:
    pre_activation = matmul(inputs, first)
    hidden = elementwise(pre_activation, gelu)
    return MLPForward(matmul(hidden, second), pre_activation, hidden)


def dense_mlp_backward(
    inputs: Matrix,
    first: Matrix,
    second: Matrix,
    cache: MLPForward,
    output_gradient: Matrix,
) -> MLPBackward:
    second_gradient = matmul(transpose(cache.hidden), output_gradient)
    hidden_gradient = matmul(output_gradient, transpose(second))
    activation_gradient = hadamard(
        hidden_gradient, elementwise(cache.pre_activation, gelu_derivative)
    )
    first_gradient = matmul(transpose(inputs), activation_gradient)
    input_gradient = matmul(activation_gradient, transpose(first))
    return MLPBackward(input_gradient, first_gradient, second_gradient)


@dataclass(frozen=True)
class TensorParallelTrace:
    output: Matrix
    local_pre_activations: tuple[Matrix, ...]
    local_hidden: tuple[Matrix, ...]
    local_partial_outputs: tuple[Matrix, ...]
    forward_collectives: int


def tensor_parallel_mlp_forward(
    inputs: Matrix, first: Matrix, second: Matrix, *, world_size: int
) -> TensorParallelTrace:
    """Megatron MLP：column-parallel -> local GeLU -> row-parallel。

    ``inputs`` 在 TP ranks 上复制。A=[A1,...,Ap] 按列切；B 以相同中间
    维度按行切。每个 rank 直接计算 GeLU(XAi)Bi，最后 SUM all-reduce。
    """

    first_shards = split_columns(first, world_size)
    second_shards = split_rows(second, world_size)
    local_pre = tuple(matmul(inputs, shard) for shard in first_shards)
    local_hidden = tuple(elementwise(values, gelu) for values in local_pre)
    partials = tuple(
        matmul(hidden, shard)
        for hidden, shard in zip(local_hidden, second_shards)
    )
    return TensorParallelTrace(
        output=add_matrices(partials),
        local_pre_activations=local_pre,
        local_hidden=local_hidden,
        local_partial_outputs=partials,
        forward_collectives=1,
    )


def tensor_parallel_mlp_backward(
    inputs: Matrix,
    first: Matrix,
    second: Matrix,
    trace: TensorParallelTrace,
    output_gradient: Matrix,
) -> MLPBackward:
    """模拟 row/column shards 的 backward 与输入梯度 all-reduce。"""

    world_size = len(trace.local_hidden)
    first_shards = split_columns(first, world_size)
    second_shards = split_rows(second, world_size)
    local_first_gradients: list[Matrix] = []
    local_second_gradients: list[Matrix] = []
    local_input_gradients: list[Matrix] = []

    for first_shard, second_shard, pre, hidden in zip(
        first_shards,
        second_shards,
        trace.local_pre_activations,
        trace.local_hidden,
    ):
        local_second_gradients.append(matmul(transpose(hidden), output_gradient))
        hidden_gradient = matmul(output_gradient, transpose(second_shard))
        activation_gradient = hadamard(
            hidden_gradient, elementwise(pre, gelu_derivative)
        )
        local_first_gradients.append(matmul(transpose(inputs), activation_gradient))
        local_input_gradients.append(
            matmul(activation_gradient, transpose(first_shard))
        )

    return MLPBackward(
        input_gradient=add_matrices(local_input_gradients),
        first_weight_gradient=concatenate_columns(local_first_gradients),
        second_weight_gradient=[
            row for shard in local_second_gradients for row in shard
        ],
    )


def dense_cross_entropy(logits: Matrix, targets: Sequence[int]) -> list[float]:
    rows, vocabulary = shape(logits)
    if rows != len(targets):
        raise ValueError("one target is required per logits row")
    losses = []
    for row, target in zip(logits, targets):
        if not 0 <= target < vocabulary:
            raise ValueError("target is outside the vocabulary")
        maximum = max(row)
        log_sum_exp = maximum + math.log(sum(math.exp(x - maximum) for x in row))
        losses.append(log_sum_exp - row[target])
    return losses


@dataclass(frozen=True)
class VocabParallelTrace:
    losses: tuple[float, ...]
    global_maxima: tuple[float, ...]
    global_exp_sums: tuple[float, ...]
    target_logits: tuple[float, ...]
    gathered_logits_elements: int
    fused_loss_elements: int


def vocab_parallel_cross_entropy(
    logits_shards: Sequence[Matrix], targets: Sequence[int]
) -> VocabParallelTrace:
    """在分片 logits 上计算精确交叉熵，不构造完整 [tokens, vocab]。

    通信语义对应历史实现：MAX all-reduce、目标 logit SUM all-reduce、
    exp-sum SUM all-reduce；每次只传每 token 标量，而非完整 vocabulary。
    """

    if not logits_shards:
        raise ValueError("at least one logits shard is required")
    tokens, shard_width = shape(logits_shards[0])
    if any(shape(shard) != (tokens, shard_width) for shard in logits_shards):
        raise ValueError("all logits shards must have equal shapes")
    if len(targets) != tokens:
        raise ValueError("one target is required per token")

    world_size = len(logits_shards)
    vocabulary = shard_width * world_size
    global_maxima: list[float] = []
    global_sums: list[float] = []
    target_logits: list[float] = []
    losses: list[float] = []

    for token_index, target in enumerate(targets):
        if not 0 <= target < vocabulary:
            raise ValueError("target is outside the sharded vocabulary")
        maximum = max(
            max(shard[token_index]) for shard in logits_shards
        )
        exp_sum = sum(
            math.exp(value - maximum)
            for shard in logits_shards
            for value in shard[token_index]
        )
        owner = target // shard_width
        local_target = target % shard_width
        target_logit = logits_shards[owner][token_index][local_target]
        global_maxima.append(maximum)
        global_sums.append(exp_sum)
        target_logits.append(target_logit)
        losses.append(maximum + math.log(exp_sum) - target_logit)

    return VocabParallelTrace(
        losses=tuple(losses),
        global_maxima=tuple(global_maxima),
        global_exp_sums=tuple(global_sums),
        target_logits=tuple(target_logits),
        gathered_logits_elements=tokens * vocabulary,
        # 三个标量 all-reduce；论文只强调从 b*s*v 降到 b*s，常数因子另计。
        fused_loss_elements=3 * tokens,
    )


@dataclass(frozen=True)
class ParallelGroups:
    tensor_parallel: tuple[tuple[int, ...], ...]
    data_parallel: tuple[tuple[int, ...], ...]


def build_parallel_groups(world_size: int, tensor_parallel_size: int) -> ParallelGroups:
    """构造论文 Appendix B 的正交通信组。"""

    if world_size <= 0 or tensor_parallel_size <= 0:
        raise ValueError("parallel sizes must be positive")
    if world_size % tensor_parallel_size:
        raise ValueError("world size must be divisible by tensor parallel size")
    data_parallel_size = world_size // tensor_parallel_size
    tp_groups = tuple(
        tuple(range(start, start + tensor_parallel_size))
        for start in range(0, world_size, tensor_parallel_size)
    )
    dp_groups = tuple(
        tuple(tp_rank + replica * tensor_parallel_size for replica in range(data_parallel_size))
        for tp_rank in range(tensor_parallel_size)
    )
    return ParallelGroups(tp_groups, dp_groups)


def padded_vocabulary_size(
    vocabulary_size: int, *, tensor_parallel_size: int, tensor_core_multiple: int = 128
) -> int:
    """让每 rank 的 vocab shard 都是 Tensor Core 友好的整数倍。"""

    if min(vocabulary_size, tensor_parallel_size, tensor_core_multiple) <= 0:
        raise ValueError("vocabulary arguments must be positive")
    multiple = tensor_parallel_size * tensor_core_multiple
    return math.ceil(vocabulary_size / multiple) * multiple


def approximate_gpt_parameters(
    *, layers: int, hidden_size: int, vocabulary_size: int, sequence_length: int
) -> int:
    """论文年代 GPT-2 的常用近似：12 L H² + vocab/position embeddings。

    忽略 bias 与 LayerNorm 的低阶项；适合解释为何 72×3072 接近 8.3B。
    """

    if min(layers, hidden_size, vocabulary_size, sequence_length) <= 0:
        raise ValueError("model dimensions must be positive")
    transformer_blocks = 12 * layers * hidden_size * hidden_size
    embeddings = (vocabulary_size + sequence_length) * hidden_size
    return transformer_blocks + embeddings


def model_state_gib(
    parameters: int,
    *,
    tensor_parallel_size: int = 1,
    bytes_per_parameter: int = 16,
) -> float:
    """粗估 mixed-precision Adam model states，不含 activations/buffers。

    16 bytes/parameter 近似包含 FP16 参数与梯度、FP32 master 参数和两个
    FP32 Adam moments。论文没有用这个公式报告显存，它只是容量教学账本。
    """

    if min(parameters, tensor_parallel_size, bytes_per_parameter) <= 0:
        raise ValueError("memory arguments must be positive")
    return parameters * bytes_per_parameter / tensor_parallel_size / 2**30


def all_reduce_ring_bytes_per_rank(
    elements: int, *, world_size: int, bytes_per_element: int = 2
) -> float:
    """ring all-reduce 的理想 send volume：2(p-1)/p × payload。

    这是用于理解通信量的模型，不是论文的实际 NCCL 拓扑测量。
    """

    if elements < 0 or world_size <= 0 or bytes_per_element <= 0:
        raise ValueError("invalid communication arguments")
    return 2 * (world_size - 1) / world_size * elements * bytes_per_element


def scaling_efficiency(
    *, sustained_petaflops: float, baseline_teraflops: float, gpu_count: int
) -> float:
    ideal_petaflops = baseline_teraflops * gpu_count / 1000.0
    return 100.0 * sustained_petaflops / ideal_petaflops


def _demo_weights() -> tuple[Matrix, Matrix, Matrix, Matrix]:
    inputs = [[0.2, -0.4, 0.7, 1.0], [-0.3, 0.8, 0.5, -0.6]]
    first = [
        [0.10, -0.20, 0.30, 0.40, -0.10, 0.05, 0.20, -0.30],
        [0.40, 0.10, -0.50, 0.20, 0.30, -0.25, 0.15, 0.35],
        [-0.20, 0.50, 0.10, -0.40, 0.20, 0.45, -0.35, 0.10],
        [0.30, -0.10, 0.25, 0.15, -0.45, 0.20, 0.40, -0.05],
    ]
    second = [
        [0.20, -0.10, 0.30, 0.05],
        [-0.30, 0.40, 0.10, -0.20],
        [0.15, 0.25, -0.35, 0.30],
        [0.50, -0.20, 0.05, 0.10],
        [-0.10, 0.35, 0.20, -0.45],
        [0.25, 0.15, -0.10, 0.40],
        [-0.40, 0.05, 0.45, 0.20],
        [0.30, -0.35, 0.15, 0.25],
    ]
    output_gradient = [[0.3, -0.2, 0.1, 0.4], [-0.1, 0.5, -0.3, 0.2]]
    return inputs, first, second, output_gradient


def main() -> None:
    inputs, first, second, output_gradient = _demo_weights()
    dense = dense_mlp_forward(inputs, first, second)
    parallel = tensor_parallel_mlp_forward(
        inputs, first, second, world_size=2
    )
    assert matrices_close(dense.output, parallel.output)

    dense_backward = dense_mlp_backward(
        inputs, first, second, dense, output_gradient
    )
    parallel_backward = tensor_parallel_mlp_backward(
        inputs, first, second, parallel, output_gradient
    )
    assert matrices_close(
        dense_backward.input_gradient, parallel_backward.input_gradient
    )
    assert matrices_close(
        dense_backward.first_weight_gradient,
        parallel_backward.first_weight_gradient,
    )
    assert matrices_close(
        dense_backward.second_weight_gradient,
        parallel_backward.second_weight_gradient,
    )

    logits = [
        [2.0, -1.0, 0.5, 1.2, -0.3, 0.8, 1.7, -0.9],
        [-0.5, 0.1, 1.4, 0.7, 2.2, -1.3, 0.4, 1.0],
    ]
    targets = [6, 2]
    logits_shards = split_columns(logits, 2)
    parallel_ce = vocab_parallel_cross_entropy(logits_shards, targets)
    dense_ce = dense_cross_entropy(logits, targets)
    assert all(
        math.isclose(a, b, rel_tol=1e-12)
        for a, b in zip(dense_ce, parallel_ce.losses)
    )

    groups = build_parallel_groups(16, 4)
    assert groups.tensor_parallel[0] == (0, 1, 2, 3)
    assert groups.data_parallel[0] == (0, 4, 8, 12)

    padded_vocab = padded_vocabulary_size(50_257, tensor_parallel_size=8)
    assert padded_vocab == 51_200
    parameters = approximate_gpt_parameters(
        layers=72,
        hidden_size=3072,
        vocabulary_size=padded_vocab,
        sequence_length=1024,
    )
    efficiency = scaling_efficiency(
        sustained_petaflops=15.1,
        baseline_teraflops=39.0,
        gpu_count=512,
    )
    assert math.isclose(efficiency, 75.62, rel_tol=1e-3)

    activation_elements = 8 * 1024 * 3072
    report = {
        "scope": "Megatron-LM 2019 intra-layer tensor parallelism",
        "mlp_equivalence": {
            "dense_output": dense.output,
            "tensor_parallel_output": parallel.output,
            "forward_all_reduces_for_one_mlp": parallel.forward_collectives,
            "backward_input_gradient_matches": matrices_close(
                dense_backward.input_gradient,
                parallel_backward.input_gradient,
            ),
        },
        "vocab_parallel_cross_entropy": {
            "dense_losses": dense_ce,
            "parallel_losses": parallel_ce.losses,
            "naive_all_gather_elements": parallel_ce.gathered_logits_elements,
            "fused_collective_scalar_elements": parallel_ce.fused_loss_elements,
        },
        "parallel_groups_16_gpus_tp4": asdict(groups),
        "paper_8_3b_configuration": {
            "padded_vocabulary": padded_vocab,
            "approximate_parameters": parameters,
            "mixed_precision_adam_states_gib_total": model_state_gib(parameters),
            "mixed_precision_adam_states_gib_per_tp_rank": model_state_gib(
                parameters, tensor_parallel_size=8
            ),
            "reported_512_gpu_efficiency_pct_recomputed": efficiency,
        },
        "communication_model": {
            "activation_shape": [8, 1024, 3072],
            "one_bf16_ring_all_reduce_mib_per_rank_tp8":
                all_reduce_ring_bytes_per_rank(
                    activation_elements, world_size=8
                )
                / 2**20,
            "collectives_per_transformer_layer_train_step": 4,
            "note": "ideal ring traffic model; not a measured paper result",
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
