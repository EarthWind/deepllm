"""Dependency-free teaching utilities for Whisper's audio front-end.

This is not a Whisper checkpoint or a speech recognizer.  It implements the
signal-processing ideas needed to understand the interface: 16 kHz framing,
STFT magnitudes, a compact mel-like filterbank, 30-second padding/chunking, and
a temperature fallback policy for decoding.  Run ``--test`` for checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
from typing import Sequence


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def hann(n: int) -> list[float]:
    if n < 2:
        raise ValueError("window length must be >= 2")
    return [0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1)) for i in range(n)]


def stft_power(samples: Sequence[float], sample_rate: int = 16_000, frame_ms: float = 25.0, hop_ms: float = 10.0) -> list[list[float]]:
    """Return a small one-sided power spectrogram using a direct DFT."""
    frame = max(2, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    if len(samples) < frame:
        samples = list(samples) + [0.0] * (frame - len(samples))
    window = hann(frame)
    bins = frame // 2 + 1
    result: list[list[float]] = []
    for start in range(0, len(samples) - frame + 1, hop):
        frame_values = [samples[start + i] * window[i] for i in range(frame)]
        row = []
        for k in range(bins):
            real = sum(frame_values[n] * math.cos(2 * math.pi * k * n / frame) for n in range(frame))
            imag = -sum(frame_values[n] * math.sin(2 * math.pi * k * n / frame) for n in range(frame))
            row.append((real * real + imag * imag) / frame)
        result.append(row)
    return result


def mel_filterbank(sample_rate: int, fft_size: int, mel_bins: int = 80, f_min: float = 0.0, f_max: float | None = None) -> list[list[float]]:
    """Triangular mel filters, shape [mel_bins, fft_size//2+1]."""
    f_max = f_max or sample_rate / 2.0
    low, high = hz_to_mel(f_min), hz_to_mel(f_max)
    points = [mel_to_hz(low + (high - low) * i / (mel_bins + 1)) for i in range(mel_bins + 2)]
    bins = [min(fft_size // 2, max(0, int((fft_size + 1) * f / sample_rate))) for f in points]
    bank: list[list[float]] = []
    for m in range(1, mel_bins + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        row = [0.0] * (fft_size // 2 + 1)
        for k in range(left, max(left + 1, center)):
            row[k] = (k - left) / max(1, center - left)
        for k in range(center, max(center + 1, right)):
            if k < len(row):
                row[k] = (right - k) / max(1, right - center)
        bank.append(row)
    return bank


def log_mel_spectrogram(samples: Sequence[float], sample_rate: int = 16_000, mel_bins: int = 80) -> list[list[float]]:
    """Compute log-mel frames; output is [frames][mel_bins]."""
    frame = int(sample_rate * 25.0 / 1000.0)
    power = stft_power(samples, sample_rate)
    bank = mel_filterbank(sample_rate, frame, mel_bins)
    return [[math.log(max(1e-10, sum(w * p for w, p in zip(filt, row)))) for filt in bank] for row in power]


@dataclass(frozen=True)
class AudioChunk:
    start_seconds: float
    end_seconds: float
    samples: tuple[float, ...]


def chunk_audio(samples: Sequence[float], sample_rate: int = 16_000, seconds: float = 30.0) -> list[AudioChunk]:
    """Split audio into Whisper-like fixed windows, zero-padding the final chunk."""
    size = int(sample_rate * seconds)
    if size <= 0:
        raise ValueError("seconds must be positive")
    if not samples:
        return [AudioChunk(0.0, seconds, tuple(0.0 for _ in range(size)))]
    chunks: list[AudioChunk] = []
    for start in range(0, len(samples), size):
        part = list(samples[start:start + size])
        part.extend([0.0] * (size - len(part)))
        chunks.append(AudioChunk(start / sample_rate, (start + size) / sample_rate, tuple(part)))
    return chunks


def temperature_fallback(avg_logprob: float, compression_ratio: float, *, temperatures: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), logprob_floor: float = -1.0, compression_ceiling: float = 2.4) -> float:
    """Choose the first decoding temperature that passes simple quality gates."""
    for temperature in temperatures:
        if avg_logprob >= logprob_floor and compression_ratio <= compression_ceiling:
            return temperature
        # A failed greedy decode gets a hotter retry; caller recomputes metrics.
        avg_logprob += 0.15
        compression_ratio -= 0.1
    return temperatures[-1]


def demo() -> None:
    sample_rate = 16_000
    samples = [0.2 * math.sin(2 * math.pi * 440 * i / sample_rate) for i in range(sample_rate)]
    chunks = chunk_audio(samples, sample_rate)
    mel = log_mel_spectrogram(samples, sample_rate, mel_bins=8)
    print("chunks:", len(chunks), "chunk samples:", len(chunks[0].samples))
    print("log-mel shape:", len(mel), "frames x", len(mel[0]), "bins")
    print("fallback temperature:", temperature_fallback(-1.4, 3.1))


def run_tests() -> None:
    sr = 16_000
    samples = [0.0] * (30 * sr + 123)
    chunks = chunk_audio(samples, sr)
    assert len(chunks) == 2 and len(chunks[0].samples) == 30 * sr
    mel = log_mel_spectrogram([0.1] * 4000, sr, mel_bins=16)
    assert mel and len(mel[0]) == 16
    assert len(mel_filterbank(sr, 400, 16)) == 16
    assert temperature_fallback(-2.0, 3.0) > 0.0
    try:
        chunk_audio([], seconds=0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero seconds should fail")
    print("all tests passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    run_tests() if args.test else demo()
