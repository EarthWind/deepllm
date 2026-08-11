"""ULMFiT's model-agnostic fine-tuning mechanics, with no dependencies.

This is not a reproduction of the pretrained AWD-LSTM checkpoint.  It isolates
the parts introduced or made central by ULMFiT:

* discriminative (layer-wise) learning rates;
* the slanted triangular learning-rate schedule (STLR);
* gradual unfreezing;
* concat pooling for document classification;
* document chunking in the style of BPTT for Text Classification (BPT3C).

Run:
    python3 ulmfit_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Iterable, Sequence


def discriminative_learning_rates(
    num_groups: int,
    top_learning_rate: float,
    layer_decay: float = 2.6,
) -> list[float]:
    """Return bottom-to-top learning rates for ``num_groups`` layer groups.

    ULMFiT chooses the top layer's rate first and divides by 2.6 for every
    lower layer.  The last returned value is therefore ``top_learning_rate``.
    """
    if num_groups < 1:
        raise ValueError("num_groups must be positive")
    if top_learning_rate <= 0:
        raise ValueError("top_learning_rate must be positive")
    if layer_decay <= 1:
        raise ValueError("layer_decay must be greater than 1")

    return [
        top_learning_rate / layer_decay ** (num_groups - 1 - group)
        for group in range(num_groups)
    ]


def slanted_triangular_learning_rate(
    step: int,
    total_steps: int,
    max_learning_rate: float,
    cut_fraction: float = 0.1,
    ratio: float = 32.0,
) -> float:
    """Compute the paper's STLR value at a zero-based optimizer step.

    The schedule linearly warms up for ``cut_fraction`` of all updates and
    spends the remaining updates on a long linear decay.  Both endpoints are
    ``max_learning_rate / ratio`` and the turning point reaches the maximum.
    """
    if total_steps < 2:
        raise ValueError("total_steps must be at least 2")
    if not 0 <= step < total_steps:
        raise ValueError("step must satisfy 0 <= step < total_steps")
    if max_learning_rate <= 0:
        raise ValueError("max_learning_rate must be positive")
    if not 0 < cut_fraction < 1:
        raise ValueError("cut_fraction must be between 0 and 1")
    if ratio <= 1:
        raise ValueError("ratio must be greater than 1")

    # The paper defines cut = floor(T * cut_frac).  Clamping keeps this small
    # demonstrator well-defined when T is tiny.
    cut = min(max(int(total_steps * cut_fraction), 1), total_steps - 1)
    if step < cut:
        progress = step / cut
    else:
        progress = 1.0 - (step - cut) / (cut * (1 / cut_fraction - 1))

    return max_learning_rate * (1 + progress * (ratio - 1)) / ratio


@dataclass(frozen=True)
class UnfreezingStage:
    """One classifier fine-tuning stage."""

    name: str
    trainable_groups: tuple[str, ...]
    epochs: int


def gradual_unfreezing_stages(
    encoder_groups: Sequence[str],
    classifier_group: str = "classifier",
    final_epochs: int = 2,
) -> list[UnfreezingStage]:
    """Build the common ULMFiT head-to-bottom unfreezing plan.

    ``encoder_groups`` must be ordered from the lowest/general layer to the
    highest/task-nearest layer.  The freshly initialized classifier is trained
    first.  We then add one lower encoder group per stage, while keeping all
    previously unfrozen groups trainable.  The fully unfrozen final stage may
    run for more than one epoch, as in the paper.
    """
    if not encoder_groups:
        raise ValueError("at least one encoder group is required")
    if final_epochs < 1:
        raise ValueError("final_epochs must be positive")

    stages = [
        UnfreezingStage(
            name="train classifier head",
            trainable_groups=(classifier_group,),
            epochs=1,
        )
    ]
    for depth in range(1, len(encoder_groups) + 1):
        trainable = tuple(encoder_groups[-depth:]) + (classifier_group,)
        fully_unfrozen = depth == len(encoder_groups)
        stages.append(
            UnfreezingStage(
                name=(
                    "fine-tune all layers"
                    if fully_unfrozen
                    else f"unfreeze top {depth} encoder group(s)"
                ),
                trainable_groups=trainable,
                epochs=final_epochs if fully_unfrozen else 1,
            )
        )
    return stages


def concat_pool(hidden_states: Sequence[Sequence[float]]) -> list[float]:
    """Return [last hidden; temporal max; temporal mean].

    ``hidden_states`` has shape [time, hidden_size].  The result has shape
    [3 * hidden_size], matching Equation (4) in the paper.
    """
    if not hidden_states:
        raise ValueError("hidden_states must contain at least one time step")
    hidden_size = len(hidden_states[0])
    if hidden_size == 0:
        raise ValueError("hidden_size must be positive")
    if any(len(state) != hidden_size for state in hidden_states):
        raise ValueError("all hidden states must have the same width")

    last = list(hidden_states[-1])
    maximum = [max(state[i] for state in hidden_states) for i in range(hidden_size)]
    mean = [
        sum(state[i] for state in hidden_states) / len(hidden_states)
        for i in range(hidden_size)
    ]
    return last + maximum + mean


def bpt3c_chunks(tokens: Sequence[int], chunk_length: int) -> list[list[int]]:
    """Split one document into ordered chunks without losing token order.

    A real BPT3C loop also carries the recurrent state across these chunks and
    accumulates their hidden states for concat pooling.  This helper makes the
    segmentation part explicit; it deliberately does not pretend that chunks
    are independent training examples.
    """
    if chunk_length < 1:
        raise ValueError("chunk_length must be positive")
    return [
        list(tokens[start : start + chunk_length])
        for start in range(0, len(tokens), chunk_length)
    ]


def layer_group_learning_rates(
    group_names: Iterable[str],
    top_learning_rate: float,
    layer_decay: float = 2.6,
) -> dict[str, float]:
    """Attach discriminative learning rates to bottom-to-top group names."""
    names = list(group_names)
    rates = discriminative_learning_rates(
        len(names),
        top_learning_rate,
        layer_decay,
    )
    return dict(zip(names, rates))


def _self_check() -> None:
    rates = discriminative_learning_rates(4, 0.01)
    assert len(rates) == 4
    assert isclose(rates[-1], 0.01)
    assert all(isclose(rates[i + 1] / rates[i], 2.6) for i in range(3))

    total_steps = 100
    schedule = [
        slanted_triangular_learning_rate(step, total_steps, 0.01)
        for step in range(total_steps)
    ]
    assert schedule.index(max(schedule)) == 10
    assert isclose(schedule[0], 0.01 / 32)
    assert schedule[-1] < schedule[10]

    pooled = concat_pool([[1.0, 5.0], [3.0, 2.0], [2.0, 8.0]])
    assert pooled == [2.0, 8.0, 3.0, 8.0, 2.0, 5.0]

    chunks = bpt3c_chunks(list(range(10)), 4)
    assert chunks == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]

    stages = gradual_unfreezing_stages(
        ["embedding", "lstm_1", "lstm_2", "lstm_3"]
    )
    assert stages[0].trainable_groups == ("classifier",)
    assert stages[-1].trainable_groups == (
        "embedding",
        "lstm_1",
        "lstm_2",
        "lstm_3",
        "classifier",
    )


def main() -> None:
    """Print a small, auditable ULMFiT training plan."""
    _self_check()

    groups = ["embedding", "lstm_1", "lstm_2", "lstm_3", "classifier"]
    print("Discriminative learning rates (bottom -> top):")
    for name, rate in layer_group_learning_rates(groups, 0.01).items():
        print(f"  {name:>10}: {rate:.8f}")

    print("\nSTLR checkpoints (T=100, cut_frac=0.1, ratio=32):")
    for step in (0, 5, 10, 50, 99):
        rate = slanted_triangular_learning_rate(step, 100, 0.01)
        print(f"  step {step:>2}: {rate:.8f}")

    print("\nGradual unfreezing:")
    encoder = ["embedding", "lstm_1", "lstm_2", "lstm_3"]
    for stage, spec in enumerate(gradual_unfreezing_stages(encoder), start=1):
        names = ", ".join(spec.trainable_groups)
        print(f"  stage {stage}: {spec.name}; train [{names}]")

    print("\nAll self-checks passed.")


if __name__ == "__main__":
    main()
