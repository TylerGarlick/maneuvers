"""Catalog of maneuvers and simple signal generators for each maneuver type.

This module provides:
- MANEUVERS: a list of maneuver names (approx 30)
- synthesize_maneuver: create accel/gyro arrays for a specified maneuver

The signals are intentionally simple: parametric combinations of
sinusoids, ramps, bumps and noise chosen to resemble the qualitative
patterns from the MIT maneuvers gallery for demo and unit tests.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

# Approximate catalog of ~30 maneuvers (names adapted for internal use)
MANEUVERS = [
    "left_roll",
    "right_roll",
    "left_bank",
    "right_bank",
    "pitch_up",
    "pitch_down",
    "climb",
    "descent",
    "loop",
    "barrel_roll",
    "aileron_roll",
    "spiral_climb",
    "spiral_descent",
    "corkscrew",
    "s_turn_left",
    "s_turn_right",
    "yaw_left",
    "yaw_right",
    "stall",
    "spin",
    "hammerhead",
    "immelmann",
    "split_s",
    "pull_up",
    "push_over",
    "sideslip",
    "skid",
    "surge_forward",
    "heave_up",
    "none",
]


def _bump(length: int, shape: str = "hann") -> np.ndarray:
    if shape == "hann":
        return np.hanning(length)
    if shape == "gauss":
        x = np.linspace(-2, 2, length)
        return np.exp(-x * x)
    if shape == "pulse":
        out = np.zeros(length)
        out[length // 4 : 3 * length // 4] = 1.0
        return out
    return np.ones(length)


def synthesize_maneuver(
    name: str, length_s: float, fs: int, rng: np.random.Generator | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesize a maneuver signal for the given `name`.

    Returns:
        accel: (N, 3) acceleration in m/s^2
        gyro: (N, 3) angular rates in rad/s
    """
    if rng is None:
        rng = np.random.default_rng()
    N = int(length_s * fs)
    t = np.linspace(0, length_s, N, endpoint=False)

    # base noise
    accel = 0.03 * rng.standard_normal((N, 3))
    gyro = 0.005 * rng.standard_normal((N, 3))

    amp = 0.6 + 0.4 * rng.random()
    freq = 1.0 + 2.0 * rng.random()

    b = _bump(N, shape="hann")

    if name == "none":
        return accel, gyro

    # Rolls: rotation about forward axis -> gyro x and lateral accel
    if name in ("left_roll", "right_roll", "aileron_roll", "barrel_roll"):
        sign = -1 if "left" in name else 1
        gyro[:, 0] += sign * (0.6 * b * (1.0 + 0.5 * np.sin(2 * np.pi * freq * t)))
        accel[:, 1] += 0.4 * sign * b
        if name == "barrel_roll":
            gyro[:, 2] += 0.2 * b * np.sin(4 * np.pi * freq * t)

    # Banks: sustained lateral accel, small gyro
    if name in ("left_bank", "right_bank", "banked_turn_left", "banked_turn_right"):
        sign = -1 if "left" in name else 1
        accel[:, 1] += sign * (0.5 * b)
        gyro[:, 2] += sign * 0.1 * b

    # Pitch maneuvers: rotation about lateral axis -> gyro y and z accel changes
    if name in ("pitch_up", "pitch_down", "pull_up", "push_over"):
        sign = 1 if name in ("pitch_up", "pull_up") else -1
        gyro[:, 1] += sign * (0.5 * b * (1.0 + 0.4 * np.cos(2 * np.pi * freq * t)))
        accel[:, 2] += -sign * 0.6 * b

    if name in ("climb", "descent", "heave_up"):
        sign = 1 if name == "climb" or name == "heave_up" else -1
        accel[:, 2] += sign * 0.6 * b
        gyro[:, 1] += 0.15 * b

    if name == "loop":
        gyro[:, 1] += 1.0 * b * np.cos(2 * np.pi * freq * t)
        accel[:, 2] += -0.8 * b * np.cos(2 * np.pi * freq * t)

    if name in ("spiral_climb", "spiral_descent"):
        sign = 1 if "climb" in name else -1
        gyro[:, 0] += 0.6 * b * np.sin(2 * np.pi * freq * t)
        gyro[:, 2] += 0.4 * b
        accel[:, 2] += sign * 0.3 * b

    if name == "corkscrew":
        gyro[:, 0] += 0.6 * b * np.sin(4 * np.pi * freq * t)
        gyro[:, 2] += 0.6 * b * np.cos(2 * np.pi * freq * t)
        accel[:, 0] += 0.2 * b * np.sin(2 * np.pi * freq * t)

    if name in ("s_turn_left", "s_turn_right"):
        sign = -1 if "left" in name else 1
        accel[:, 1] += sign * 0.5 * (np.sin(2 * np.pi * freq * t) * b)
        gyro[:, 2] += sign * 0.25 * b

    if name in ("yaw_left", "yaw_right", "skid", "sideslip"):
        sign = -1 if "left" in name or name == "skid" else 1
        gyro[:, 2] += sign * 0.5 * b * np.sin(2 * np.pi * freq * t)
        accel[:, 0] += 0.2 * b * np.cos(2 * np.pi * freq * t)

    if name in ("stall", "spin"):
        # noisy high-rate oscillation
        gyro[:, :] += 0.8 * b[:, None] * (
            0.5 * np.sin(6 * np.pi * freq * t)[:, None] + 0.2 * rng.standard_normal((N, 3))
        )
        accel[:, :] += 0.6 * b[:, None] * (0.2 * np.cos(2 * np.pi * freq * t)[:, None])

    if name in ("hammerhead", "immelmann", "split_s"):
        # composite maneuvers: strong pitch with some yaw
        gyro[:, 1] += 0.9 * b * np.sin(2 * np.pi * freq * t)
        gyro[:, 2] += 0.4 * b * np.sin(1.5 * 2 * np.pi * freq * t)
        accel[:, 2] += -0.7 * b

    if name in ("surge_forward",):
        accel[:, 0] += 0.7 * b

    # small randomization to avoid identical deterministic patterns
    accel += 0.02 * rng.standard_normal(accel.shape)
    gyro += 0.01 * rng.standard_normal(gyro.shape)

    return accel, gyro
