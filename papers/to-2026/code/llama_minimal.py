"""A dependency-free, auditable miniature of the original 2023 LLaMA.

This is educational scalar Python, not a training implementation or a fast
inference engine.  It mirrors the disclosed LLaMA 1 architecture closely:

* decoder-only causal language modeling;
* sequential pre-RMSNorm residual blocks;
* bias-free multi-head self-attention (MHA, not later GQA);
* RoPE on queries and keys;
* SwiGLU with the paper's roughly (2/3) * 4d intermediate width;
* a separate, untied output projection; and
* an incremental KV cache that stores rotated keys and unrotated values.

The implementation uses Python lists so it runs without NumPy or PyTorch and
so every operation is visible.  Use it only with tiny dimensions.

Run:
    python3 papers/to-2026/code/llama_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Iterable


Vector = list[float]
Matrix = list[Vector]
Heads = list[Vector]
SequenceStates = list[Vector]


def _dot(left: Iterable[float], right: Iterable[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _add(left: Vector, right: Vector) -> Vector:
    if len(left) != len(right):
        raise ValueError("vector widths differ")
    return [a + b for a, b in zip(left, right)]


def _linear(x: Vector, weight: Matrix) -> Vector:
    """Apply a bias-free linear layer whose weight is [out, in]."""

    if not weight or len(x) != len(weight[0]):
        raise ValueError("linear input width does not match weight")
    return [_dot(row, x) for row in weight]


def _softmax(scores: Vector) -> Vector:
    if not scores:
        raise ValueError("softmax needs at least one score")
    row_max = max(scores)
    weights = [math.exp(score - row_max) for score in scores]
    denominator = sum(weights)
    return [weight / denominator for weight in weights]


def _silu(x: float) -> float:
    # Stable x * sigmoid(x), avoiding exp overflow for very negative x.
    if x >= 0.0:
        return x / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return x * exp_x / (1.0 + exp_x)


def _random_matrix(
    rows: int,
    cols: int,
    rng: random.Random,
    *,
    scale: float | None = None,
) -> Matrix:
    bound = (1.0 / math.sqrt(cols)) if scale is None else scale
    return [
        [rng.uniform(-bound, bound) for _ in range(cols)] for _ in range(rows)
    ]


def _split_heads(x: Vector, n_heads: int) -> Heads:
    if len(x) % n_heads:
        raise ValueError("model dimension must be divisible by n_heads")
    head_dim = len(x) // n_heads
    return [x[h * head_dim : (h + 1) * head_dim] for h in range(n_heads)]


def _merge_heads(heads: Heads) -> Vector:
    return [value for head in heads for value in head]


def _max_abs_difference(left: SequenceStates, right: SequenceStates) -> float:
    if len(left) != len(right):
        raise ValueError("sequence lengths differ")
    if not left:
        return 0.0
    return max(
        abs(a - b)
        for left_row, right_row in zip(left, right)
        for a, b in zip(left_row, right_row)
    )


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int = 32
    dim: int = 16
    n_layers: int = 2
    n_heads: int = 4
    multiple_of: int = 8
    norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    seed: int = 2023

    def __post_init__(self) -> None:
        positive_ints = {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "multiple_of": self.multiple_of,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for adjacent-pair RoPE")
        if self.norm_eps <= 0.0 or self.rope_theta <= 0.0:
            raise ValueError("norm_eps and rope_theta must be positive")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def hidden_dim(self) -> int:
        # Official v1 code starts with 4d, multiplies by 2/3 because SwiGLU
        # has three matrices, then rounds upward for hardware-friendly shapes.
        raw = int(2 * (4 * self.dim) / 3)
        return self.multiple_of * (
            (raw + self.multiple_of - 1) // self.multiple_of
        )


class RMSNorm:
    """RMSNorm: scale by root-mean-square, without subtracting the mean."""

    def __init__(self, dim: int, eps: float) -> None:
        self.weight = [1.0 for _ in range(dim)]
        self.eps = eps

    def __call__(self, x: SequenceStates) -> SequenceStates:
        output: SequenceStates = []
        for token in x:
            if len(token) != len(self.weight):
                raise ValueError("RMSNorm input width does not match weight")
            mean_square = sum(value * value for value in token) / len(token)
            inverse_rms = 1.0 / math.sqrt(mean_square + self.eps)
            output.append(
                [
                    value * inverse_rms * scale
                    for value, scale in zip(token, self.weight)
                ]
            )
        return output


def apply_rope(head: Vector, position: int, theta: float) -> Vector:
    """Rotate adjacent coordinate pairs using the token's absolute position."""

    if len(head) % 2:
        raise ValueError("RoPE head dimension must be even")
    rotated: Vector = []
    for pair_start in range(0, len(head), 2):
        inverse_frequency = theta ** (-pair_start / len(head))
        angle = position * inverse_frequency
        cosine, sine = math.cos(angle), math.sin(angle)
        real, imaginary = head[pair_start], head[pair_start + 1]
        rotated.extend(
            [
                real * cosine - imaginary * sine,
                real * sine + imaginary * cosine,
            ]
        )
    return rotated


@dataclass
class KVCache:
    """One layer's cache: token -> head -> head_dim."""

    keys: list[Heads] = field(default_factory=list)
    values: list[Heads] = field(default_factory=list)

    def __len__(self) -> int:
        if len(self.keys) != len(self.values):
            raise RuntimeError("key/value cache lengths diverged")
        return len(self.keys)


class MultiHeadAttention:
    """Original LLaMA 1 MHA with optional incremental KV caching."""

    def __init__(self, config: LlamaConfig, rng: random.Random) -> None:
        self.config = config
        self.wq = _random_matrix(config.dim, config.dim, rng)
        self.wk = _random_matrix(config.dim, config.dim, rng)
        self.wv = _random_matrix(config.dim, config.dim, rng)
        self.wo = _random_matrix(config.dim, config.dim, rng)

    def __call__(
        self,
        x: SequenceStates,
        *,
        start_pos: int,
        cache: KVCache | None,
    ) -> SequenceStates:
        if not x:
            raise ValueError("attention input cannot be empty")
        if start_pos < 0:
            raise ValueError("start_pos must be non-negative")
        if cache is None and start_pos != 0:
            raise ValueError("nonzero start_pos requires the preceding KV cache")
        if cache is not None and len(cache) != start_pos:
            raise ValueError(
                f"cache contains {len(cache)} tokens, expected start_pos={start_pos}"
            )

        query_heads: list[Heads] = []
        new_key_heads: list[Heads] = []
        new_value_heads: list[Heads] = []

        for offset, token in enumerate(x):
            position = start_pos + offset
            queries = _split_heads(_linear(token, self.wq), self.config.n_heads)
            keys = _split_heads(_linear(token, self.wk), self.config.n_heads)
            values = _split_heads(_linear(token, self.wv), self.config.n_heads)
            query_heads.append(
                [
                    apply_rope(head, position, self.config.rope_theta)
                    for head in queries
                ]
            )
            new_key_heads.append(
                [
                    apply_rope(head, position, self.config.rope_theta)
                    for head in keys
                ]
            )
            new_value_heads.append(values)

        if cache is not None:
            # K is cached after RoPE because historical absolute positions do
            # not change.  V is not rotated by RoPE.
            cache.keys.extend(new_key_heads)
            cache.values.extend(new_value_heads)
            all_keys, all_values = cache.keys, cache.values
        else:
            all_keys, all_values = new_key_heads, new_value_heads

        scale = 1.0 / math.sqrt(self.config.head_dim)
        outputs: SequenceStates = []
        for offset, heads_for_query in enumerate(query_heads):
            absolute_position = start_pos + offset
            valid_length = absolute_position + 1
            output_heads: Heads = []

            for head_index, query in enumerate(heads_for_query):
                scores = [
                    scale * _dot(query, all_keys[key_pos][head_index])
                    for key_pos in range(valid_length)
                ]
                probabilities = _softmax(scores)
                attended = [0.0 for _ in range(self.config.head_dim)]
                for key_pos, probability in enumerate(probabilities):
                    value = all_values[key_pos][head_index]
                    for index in range(self.config.head_dim):
                        attended[index] += probability * value[index]
                output_heads.append(attended)

            outputs.append(_linear(_merge_heads(output_heads), self.wo))
        return outputs


class SwiGLUFeedForward:
    """W2(SiLU(W1 x) elementwise-multiply W3 x)."""

    def __init__(self, config: LlamaConfig, rng: random.Random) -> None:
        self.w1 = _random_matrix(config.hidden_dim, config.dim, rng)
        self.w3 = _random_matrix(config.hidden_dim, config.dim, rng)
        self.w2 = _random_matrix(config.dim, config.hidden_dim, rng)

    def __call__(self, x: SequenceStates) -> SequenceStates:
        output: SequenceStates = []
        for token in x:
            gate = _linear(token, self.w1)
            value = _linear(token, self.w3)
            hidden = [_silu(a) * b for a, b in zip(gate, value)]
            output.append(_linear(hidden, self.w2))
        return output


class TransformerBlock:
    """Sequential pre-norm residual block used by LLaMA 1."""

    def __init__(self, config: LlamaConfig, rng: random.Random) -> None:
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = MultiHeadAttention(config, rng)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.feed_forward = SwiGLUFeedForward(config, rng)

    def __call__(
        self,
        x: SequenceStates,
        *,
        start_pos: int,
        cache: KVCache | None,
    ) -> SequenceStates:
        attention_output = self.attention(
            self.attention_norm(x), start_pos=start_pos, cache=cache
        )
        hidden = [_add(residual, update) for residual, update in zip(x, attention_output)]
        ffn_output = self.feed_forward(self.ffn_norm(hidden))
        return [
            _add(residual, update) for residual, update in zip(hidden, ffn_output)
        ]


class TinyLlama:
    """Tiny forward-only LLaMA with separate embedding and LM-head weights."""

    def __init__(self, config: LlamaConfig) -> None:
        self.config = config
        rng = random.Random(config.seed)
        self.tok_embeddings = _random_matrix(
            config.vocab_size, config.dim, rng, scale=0.15
        )
        self.layers = [
            TransformerBlock(config, rng) for _ in range(config.n_layers)
        ]
        self.norm = RMSNorm(config.dim, config.norm_eps)
        # LLaMA 1's original inference code uses a distinct output matrix.
        self.output = _random_matrix(
            config.vocab_size, config.dim, rng, scale=0.15
        )

    def new_caches(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    def __call__(
        self,
        tokens: list[int],
        *,
        start_pos: int = 0,
        caches: list[KVCache] | None = None,
    ) -> SequenceStates:
        if not tokens:
            raise ValueError("tokens cannot be empty")
        if any(token < 0 or token >= self.config.vocab_size for token in tokens):
            raise ValueError("token id outside vocabulary")
        if caches is not None and len(caches) != len(self.layers):
            raise ValueError("one KV cache is required per transformer layer")

        hidden = [self.tok_embeddings[token][:] for token in tokens]
        for layer_index, layer in enumerate(self.layers):
            cache = None if caches is None else caches[layer_index]
            hidden = layer(hidden, start_pos=start_pos, cache=cache)
        hidden = self.norm(hidden)
        return [_linear(token, self.output) for token in hidden]


def next_token_cross_entropy(logits: SequenceStates, tokens: list[int]) -> float:
    """Teacher-forced causal LM loss: position t predicts tokens[t + 1]."""

    if len(logits) != len(tokens) or len(tokens) < 2:
        raise ValueError("need matching logits/tokens with at least two positions")
    total = 0.0
    for position, target in enumerate(tokens[1:]):
        row = logits[position]
        row_max = max(row)
        log_denominator = row_max + math.log(
            sum(math.exp(value - row_max) for value in row)
        )
        total += log_denominator - row[target]
    return total / (len(tokens) - 1)


def greedy_generate(
    model: TinyLlama,
    prompt: list[int],
    *,
    max_new_tokens: int,
) -> list[int]:
    """Prefill once, then decode one token at a time with KV caches."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if max_new_tokens == 0:
        return prompt[:]

    caches = model.new_caches()
    logits = model(prompt, start_pos=0, caches=caches)
    result = prompt[:]

    for step in range(max_new_tokens):
        next_token = max(range(model.config.vocab_size), key=logits[-1].__getitem__)
        result.append(next_token)
        if step + 1 < max_new_tokens:
            # The sampled token has not been put into the cache yet.  Its
            # absolute position is exactly the current cache length.
            logits = model([next_token], start_pos=len(caches[0]), caches=caches)
    return result


def self_test() -> None:
    config = LlamaConfig(
        vocab_size=23,
        dim=12,
        n_layers=2,
        n_heads=3,
        multiple_of=4,
        seed=7,
    )
    model = TinyLlama(config)
    tokens = [1, 5, 3, 7, 2, 9]

    # 1. Full causal forward is the reference.
    full_logits = model(tokens)

    # 2. Token-at-a-time decode with rotated K / plain V caches must match.
    token_caches = model.new_caches()
    incremental_logits: SequenceStates = []
    for position, token in enumerate(tokens):
        step_logits = model([token], start_pos=position, caches=token_caches)
        incremental_logits.extend(step_logits)
    incremental_error = _max_abs_difference(full_logits, incremental_logits)

    # 3. Chunked prefill exercises causal masking inside a multi-token chunk.
    chunk_caches = model.new_caches()
    first_chunk = model(tokens[:3], start_pos=0, caches=chunk_caches)
    second_chunk = model(tokens[3:], start_pos=3, caches=chunk_caches)
    chunk_error = _max_abs_difference(full_logits, first_chunk + second_chunk)

    # 4. Changing a future token cannot affect any earlier logits.
    modified = tokens[:]
    modified[-1] = 4
    modified_logits = model(modified)
    causal_error = _max_abs_difference(full_logits[:-1], modified_logits[:-1])

    # 5. RoPE is a rotation and therefore preserves a head's Euclidean norm.
    head = [0.2, -0.4, 0.7, 0.1]
    rotated = apply_rope(head, position=37, theta=config.rope_theta)
    rope_norm_error = abs(_dot(head, head) - _dot(rotated, rotated))

    # 6. Loss and generation smoke tests cover the top-level interfaces.
    loss = next_token_cross_entropy(full_logits, tokens)
    generated = greedy_generate(model, tokens[:3], max_new_tokens=3)

    assert config.hidden_dim == 32
    assert model.tok_embeddings is not model.output
    assert incremental_error < 1e-12, incremental_error
    assert chunk_error < 1e-12, chunk_error
    assert causal_error < 1e-12, causal_error
    assert rope_norm_error < 1e-12, rope_norm_error
    assert math.isfinite(loss) and loss > 0.0
    assert len(generated) == 6
    assert all(len(cache) == len(tokens) for cache in token_caches)

    print("LLaMA educational reference: all checks passed")
    print(f"  hidden dimension:             {config.hidden_dim}")
    print(f"  full vs token-cache error:    {incremental_error:.3e}")
    print(f"  full vs chunked-cache error:  {chunk_error:.3e}")
    print(f"  causal invariance error:      {causal_error:.3e}")
    print(f"  RoPE norm error:              {rope_norm_error:.3e}")
    print(f"  next-token loss:              {loss:.6f}")
    print(f"  greedy token ids:             {generated}")


if __name__ == "__main__":
    self_test()
