"""ELMo's core representation mechanics, implemented with no dependencies.

This is an educational, auditable companion to Peters et al. (2018).  It does
not reproduce the 93.6M-parameter pretrained biLM.  Instead, it isolates the
parts that define ELMo:

* a character-CNN style max-over-time feature extractor;
* alignment and concatenation of forward/backward language-model states;
* the three exported biLM representation levels;
* task-specific, softmax-normalized scalar mixing with a learned scale gamma;
* forward/backward next-token targets for the coupled biLM objective.

Run:
    python3 elmo_minimal.py
"""

from __future__ import annotations

from math import exp, isclose, log, sqrt
from typing import Sequence


Vector = list[float]
SequenceMatrix = list[Vector]
LayerStack = list[SequenceMatrix]


def _validate_vectors(vectors: Sequence[Sequence[float]], name: str) -> int:
    """Validate a non-empty rectangular collection and return its width."""
    if not vectors:
        raise ValueError(f"{name} must not be empty")
    width = len(vectors[0])
    if width == 0:
        raise ValueError(f"{name} vectors must have positive width")
    if any(len(vector) != width for vector in vectors):
        raise ValueError(f"{name} vectors must all have the same width")
    return width


def softmax(logits: Sequence[float]) -> Vector:
    """Numerically stable softmax used for ELMo's scalar mixture weights."""
    if not logits:
        raise ValueError("logits must not be empty")
    maximum = max(logits)
    numerators = [exp(value - maximum) for value in logits]
    denominator = sum(numerators)
    return [value / denominator for value in numerators]


def layer_norm(vector: Sequence[float], eps: float = 1e-12) -> Vector:
    """Normalize one vector without trainable affine parameters.

    The paper notes that normalizing each biLM layer before scalar mixing can
    help because the layers have different activation distributions.
    """
    if not vector:
        raise ValueError("vector must not be empty")
    mean = sum(vector) / len(vector)
    variance = sum((value - mean) ** 2 for value in vector) / len(vector)
    inverse_std = 1.0 / sqrt(variance + eps)
    return [(value - mean) * inverse_std for value in vector]


def scalar_mix(
    tensors: Sequence[Sequence[Sequence[float]]],
    logits: Sequence[float],
    gamma: float = 1.0,
    normalize: bool = False,
) -> SequenceMatrix:
    """Mix a stack shaped [layers, time, width] into [time, width].

    ELMo computes ``gamma * sum_j softmax(logits)[j] * h_j``.  ``logits`` and
    ``gamma`` are task parameters; the pretrained biLM can remain frozen.
    """
    if len(tensors) != len(logits):
        raise ValueError("one scalar logit is required for every layer")
    if not tensors:
        raise ValueError("at least one layer is required")

    time_steps = len(tensors[0])
    if time_steps == 0:
        raise ValueError("layers must contain at least one token")
    width = _validate_vectors(tensors[0], "layer 0")
    for layer_index, layer in enumerate(tensors[1:], start=1):
        if len(layer) != time_steps:
            raise ValueError("all layers must have the same sequence length")
        if _validate_vectors(layer, f"layer {layer_index}") != width:
            raise ValueError("all layers must have the same vector width")

    weights = softmax(logits)
    mixed: SequenceMatrix = []
    for time_index in range(time_steps):
        vectors = [list(layer[time_index]) for layer in tensors]
        if normalize:
            vectors = [layer_norm(vector) for vector in vectors]
        mixed.append(
            [
                gamma
                * sum(weight * vector[dimension] for weight, vector in zip(weights, vectors))
                for dimension in range(width)
            ]
        )
    return mixed


def align_backward_states(
    states_in_reverse_processing_order: Sequence[Sequence[float]],
) -> SequenceMatrix:
    """Return backward states in the original left-to-right token order.

    A backward LM consumes ``t_N, ..., t_1``.  Downstream token ``t_k`` must be
    paired with the state that summarizes ``t_{k+1}, ..., t_N`` at that same
    original position, so implementations reverse the time axis back before
    concatenating it with the forward state.
    """
    _validate_vectors(states_in_reverse_processing_order, "backward states")
    return [list(vector) for vector in reversed(states_in_reverse_processing_order)]


def concatenate_directions(
    forward_states: Sequence[Sequence[float]],
    backward_states_in_token_order: Sequence[Sequence[float]],
) -> SequenceMatrix:
    """Concatenate direction-specific states at matching token positions."""
    if len(forward_states) != len(backward_states_in_token_order):
        raise ValueError("forward and backward sequences must have equal length")
    forward_width = _validate_vectors(forward_states, "forward states")
    backward_width = _validate_vectors(
        backward_states_in_token_order,
        "backward states",
    )
    if forward_width != backward_width:
        raise ValueError("both directions must have the same projected width")
    return [
        list(forward) + list(backward)
        for forward, backward in zip(forward_states, backward_states_in_token_order)
    ]


def build_elmo_layers(
    token_representations: Sequence[Sequence[float]],
    forward_layers: Sequence[Sequence[Sequence[float]]],
    backward_layers_in_token_order: Sequence[Sequence[Sequence[float]]],
) -> LayerStack:
    """Create ELMo's exported stack ``[h_0, h_1, ..., h_L]``.

    The character-CNN token representation has width ``d``.  Every biLSTM
    level concatenates two ``d``-wide projected directions, so ``h_0`` repeats
    the token vector as ``[x_k; x_k]`` to give all exported levels width ``2d``.
    """
    token_width = _validate_vectors(token_representations, "token representations")
    if len(forward_layers) != len(backward_layers_in_token_order):
        raise ValueError("forward and backward stacks must have equal depth")
    if not forward_layers:
        raise ValueError("at least one biLSTM layer is required")

    num_tokens = len(token_representations)
    exported: LayerStack = [
        [list(token) + list(token) for token in token_representations]
    ]
    for layer_index, (forward, backward) in enumerate(
        zip(forward_layers, backward_layers_in_token_order),
        start=1,
    ):
        if len(forward) != num_tokens or len(backward) != num_tokens:
            raise ValueError(f"biLSTM layer {layer_index} has the wrong sequence length")
        if _validate_vectors(forward, "forward layer") != token_width:
            raise ValueError("forward projected width must match token width")
        if _validate_vectors(backward, "backward layer") != token_width:
            raise ValueError("backward projected width must match token width")
        exported.append(concatenate_directions(forward, backward))
    return exported


def valid_conv_max(
    character_vectors: Sequence[Sequence[float]],
    kernel: Sequence[Sequence[float]],
    bias: float = 0.0,
) -> float:
    """Apply one 1-D character filter and max-pool over valid positions.

    This is one scalar channel of the original character CNN.  A real ELMo
    token encoder uses 2048 filters spanning several character n-gram widths,
    then two highway layers and a projection to 512 dimensions.
    """
    char_width = _validate_vectors(character_vectors, "character vectors")
    kernel_width = _validate_vectors(kernel, "kernel")
    if char_width != kernel_width:
        raise ValueError("kernel and character embedding widths must match")
    if len(kernel) > len(character_vectors):
        raise ValueError("kernel cannot be wider than the token")

    responses = []
    for start in range(len(character_vectors) - len(kernel) + 1):
        value = bias
        for offset, kernel_row in enumerate(kernel):
            char_row = character_vectors[start + offset]
            value += sum(char * weight for char, weight in zip(char_row, kernel_row))
        responses.append(max(0.0, value))  # ReLU, then max-over-time pooling.
    return max(responses)


def character_cnn(
    character_vectors: Sequence[Sequence[float]],
    filters: Sequence[tuple[Sequence[Sequence[float]], float]],
) -> Vector:
    """Return one max-pooled feature per supplied character filter."""
    if not filters:
        raise ValueError("at least one character filter is required")
    return [
        valid_conv_max(character_vectors, kernel, bias)
        for kernel, bias in filters
    ]


def highway(
    vector: Sequence[float],
    transformed: Sequence[float],
    gate_logits: Sequence[float],
) -> Vector:
    """Combine transformed and carry paths for one highway layer.

    ``transformed`` would normally be ``g(W_H x + b_H)`` and ``gate_logits``
    would be ``W_T x + b_T``.  They are explicit here so the gating equation is
    easy to audit without adding a matrix library.
    """
    if not (len(vector) == len(transformed) == len(gate_logits)):
        raise ValueError("all highway inputs must have the same width")
    gates = [1.0 / (1.0 + exp(-value)) for value in gate_logits]
    return [
        gate * changed + (1.0 - gate) * original
        for original, changed, gate in zip(vector, transformed, gates)
    ]


def bilm_targets(tokens: Sequence[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return forward and backward prediction pairs for one sentence."""
    if len(tokens) < 2:
        raise ValueError("at least two tokens are needed for an LM objective")
    forward = list(zip(tokens[:-1], tokens[1:]))
    backward = list(zip(tokens[1:], tokens[:-1]))
    return forward, backward


def coupled_bilm_nll(
    forward_target_probabilities: Sequence[float],
    backward_target_probabilities: Sequence[float],
) -> float:
    """Mean negative log-likelihood across both LM directions."""
    probabilities = list(forward_target_probabilities) + list(
        backward_target_probabilities
    )
    if not probabilities:
        raise ValueError("at least one target probability is required")
    if any(not 0.0 < probability <= 1.0 for probability in probabilities):
        raise ValueError("probabilities must lie in (0, 1]")
    return -sum(log(probability) for probability in probabilities) / len(probabilities)


def _self_check() -> None:
    weights = softmax([0.0, 0.0, 0.0])
    assert all(isclose(weight, 1 / 3) for weight in weights)

    reversed_states = [[30.0], [20.0], [10.0]]
    assert align_backward_states(reversed_states) == [[10.0], [20.0], [30.0]]

    tokens = [[1.0, 0.0], [0.0, 1.0]]
    forward = [
        [[1.0, 1.0], [2.0, 2.0]],
        [[3.0, 3.0], [4.0, 4.0]],
    ]
    backward = [
        [[5.0, 5.0], [6.0, 6.0]],
        [[7.0, 7.0], [8.0, 8.0]],
    ]
    layers = build_elmo_layers(tokens, forward, backward)
    assert len(layers) == 3
    assert len(layers[0]) == 2
    assert len(layers[0][0]) == 4
    assert layers[0][0] == [1.0, 0.0, 1.0, 0.0]
    assert layers[2][1] == [4.0, 4.0, 8.0, 8.0]

    mixed = scalar_mix(layers, [0.0, 0.0, 0.0], gamma=3.0)
    assert all(
        isclose(actual, expected)
        for actual, expected in zip(mixed[0], [5.0, 4.0, 13.0, 12.0])
    )

    characters = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    filters = [
        ([[1.0, 1.0]], 0.0),
        ([[1.0, 0.0], [0.0, 1.0]], 0.0),
    ]
    assert character_cnn(characters, filters) == [2.0, 2.0]

    forward_targets, backward_targets = bilm_targets(["the", "bank", "closed"])
    assert forward_targets == [("the", "bank"), ("bank", "closed")]
    assert backward_targets == [("bank", "the"), ("closed", "bank")]
    assert isclose(coupled_bilm_nll([0.5], [0.25]), -log(0.125) / 2)


def main() -> None:
    """Print an inspectable example of ELMo's exported representation stack."""
    _self_check()

    tokens = ["the", "bank", "closed"]
    forward_targets, backward_targets = bilm_targets(tokens)
    print("Coupled biLM supervision:")
    print(f"  forward : {forward_targets}")
    print(f"  backward: {backward_targets}")

    token_representations = [[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]]
    forward_layers = [
        [[0.2, 0.8], [0.4, 0.9], [0.7, 0.3]],
        [[0.1, 0.5], [0.9, 0.2], [0.8, 0.1]],
    ]
    backward_layers = [
        [[0.6, 0.2], [0.3, 0.7], [0.5, 0.5]],
        [[0.4, 0.3], [0.2, 0.9], [0.1, 0.8]],
    ]
    layers = build_elmo_layers(
        token_representations,
        forward_layers,
        backward_layers,
    )

    logits = [-0.5, 1.2, 0.1]
    gamma = 1.3
    weights = softmax(logits)
    contextual = scalar_mix(layers, logits, gamma)

    print("\nExported ELMo stack:")
    print(f"  shape: layers={len(layers)}, tokens={len(layers[0])}, width={len(layers[0][0])}")
    print("  levels: character token layer + biLSTM layer 1 + biLSTM layer 2")
    print(f"  task softmax weights: {[round(weight, 4) for weight in weights]}")
    print(f"  task gamma: {gamma}")
    print(f"  contextual vector for 'bank': {[round(v, 4) for v in contextual[1]]}")
    print("\nAll self-checks passed.")


if __name__ == "__main__":
    main()
