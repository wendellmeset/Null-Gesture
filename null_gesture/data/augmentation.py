"""Synthetic data augmentation for IMU gesture windows.

Turns 2 real recordings into hundreds of realistic variants via:
  - Gaussian noise injection (scaled per-channel)
  - Time warping (random smooth distortion of the time axis)
  - Magnitude scaling (simulate faster/slower gestures)
  - Per-channel bias drift (simulate sensor offset)
  - Time shifting (simulate earlier/later onset)
  - Channel masking (simulate brief sensor dropouts)
"""

from __future__ import annotations

import numpy as np


def _smooth_warp(length: int, sigma: float = 0.1, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a smooth random warp curve for time-axis distortion."""
    if rng is None:
        rng = np.random.default_rng()
    raw = rng.normal(0, 1, length)
    # Smooth with a boxcar of width 5
    kernel = np.ones(5) / 5.0
    raw = np.convolve(raw, kernel, mode="same")
    if raw.std() < 1e-9:
        return np.arange(length, dtype=np.float64)
    raw = raw / raw.std() * sigma
    warp = np.cumsum(1.0 + raw)
    # Normalise to [0, length-1]
    warp = (warp - warp[0]) / (warp[-1] - warp[0]) * (length - 1)
    return warp


def time_warp(signal: np.ndarray, sigma: float = 0.1, rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply smooth random time warping to a 2D signal.

    Args:
        signal: (timesteps, channels)
        sigma: warp strength (0.05=mild, 0.20=aggressive)
        rng: optional numpy Generator

    Returns:
        warped signal with same shape
    """
    T, C = signal.shape
    warp = _smooth_warp(T, sigma, rng)
    out = np.zeros_like(signal)
    src = np.arange(T, dtype=np.float64)
    for c in range(C):
        out[:, c] = np.interp(src, warp, signal[:, c])
    return out


def augment(
    sample: np.ndarray,
    n_variants: int = 200,
    noise_range: tuple[float, float] = (0.005, 0.06),
    warp_prob: float = 0.6,
    warp_sigma_range: tuple[float, float] = (0.03, 0.15),
    scale_range: tuple[float, float] = (0.70, 1.40),
    bias_range: tuple[float, float] = (0.0, 0.04),
    shift_range: tuple[int, int] = (-10, 10),
    mask_prob: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    """Generate synthetic gesture samples from one real recording.

    Args:
        sample: (timesteps, channels) — a single real gesture window.
        n_variants: how many synthetic copies to produce.
        noise_range: (min, max) fraction of per-channel std to add as Gaussian noise.
        warp_prob: probability of applying time warping to each variant.
        warp_sigma_range: (min, max) warp strength.
        scale_range: (min, max) uniform magnitude scaling factor.
        bias_range: (min, max) fraction of std for random per-channel bias.
        shift_range: (min, max) timesteps to randomly shift.
        mask_prob: probability of zeroing a random channel for a short segment.
        seed: reproducibility.

    Returns:
        (n_variants, timesteps, channels) augmented array including the original.
    """
    rng = np.random.default_rng(seed)
    T, C = sample.shape
    ch_std = np.std(sample, axis=0, keepdims=True)  # (1, C)
    # Guard against zero-variance channels
    ch_std = np.where(ch_std < 1e-9, 1e-9, ch_std)

    variants = np.empty((n_variants, T, C), dtype=np.float32)
    variants[0] = sample.astype(np.float32)  # always include the original

    for i in range(1, n_variants):
        v = sample.copy().astype(np.float32)

        # 1. Gaussian noise (scaled to per-channel std)
        noise_frac = rng.uniform(*noise_range)
        v += rng.normal(0, 1, (T, C)).astype(np.float32) * ch_std * noise_frac

        # 2. Time warping
        if rng.random() < warp_prob:
            sigma = rng.uniform(*warp_sigma_range)
            v = time_warp(v, sigma=sigma, rng=rng).astype(np.float32)

        # 3. Magnitude scaling
        scale = rng.uniform(*scale_range)
        v *= scale

        # 4. Per-channel bias drift
        bias_frac = rng.uniform(*bias_range)
        v += rng.normal(0, 1, (1, C)).astype(np.float32) * ch_std * bias_frac

        # 5. Time shifting (roll + pad with edge values)
        shift = int(rng.integers(*shift_range, endpoint=True))
        if shift != 0:
            v = np.roll(v, shift, axis=0)
            if shift > 0:
                v[:shift] = v[shift : shift + 1]
            else:
                v[shift:] = v[shift - 1 : shift]

        # 6. Random channel dropout (briefly zero a channel)
        if rng.random() < mask_prob:
            ch = int(rng.integers(0, C))
            # Zero out a random 5-20% contiguous segment of that channel
            seg_len = int(rng.integers(T // 20, T // 5))
            seg_start = int(rng.integers(0, T - seg_len))
            v[seg_start : seg_start + seg_len, ch] = 0.0

        variants[i] = v

    return variants


def augment_dataset(
    samples: dict[str, list[np.ndarray]],
    n_per_sample: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment a full dataset of gesture recordings.

    Args:
        samples: {gesture_name: [sample_array, ...]} where each array is (T, 6)
        n_per_sample: variants to generate per real recording.
        seed: base seed for reproducibility.

    Returns:
        X: (total_variants, timesteps, channels)
        y: (total_variants,) integer class labels
    """
    all_x: list[np.ndarray] = []
    all_y: list[int] = []

    gesture_list = list(samples.keys())  # preserve insertion order
    label_map = {g: i for i, g in enumerate(gesture_list)}

    for g_idx, gesture in enumerate(gesture_list):
        for s_idx, sample in enumerate(samples[gesture]):
            sub_seed = seed + g_idx * 1000 + s_idx
            variants = augment(sample, n_variants=n_per_sample, seed=sub_seed)
            all_x.append(variants)
            all_y.append(np.full(n_per_sample, label_map[gesture], dtype=np.int64))

    X = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y
