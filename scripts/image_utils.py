from __future__ import annotations

import numpy as np


_CHANNEL_INDEX = {
    "B": 0,
    "G": 1,
    "R": 2,
}

_PATTERNS = {
    "RGGB": (("R", "G"), ("G", "B")),
    "GRBG": (("G", "R"), ("B", "G")),
    "BGGR": (("B", "G"), ("G", "R")),
    "GBRG": (("G", "B"), ("R", "G")),
}


def make_bayer_mosaic(image: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Create a single-channel Bayer mosaic from a BGR image."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a 3-channel BGR image")
    if pattern not in _PATTERNS:
        raise ValueError(f"unsupported Bayer pattern: {pattern}")

    mosaic = np.empty(image.shape[:2], dtype=image.dtype)
    pattern_tile = _PATTERNS[pattern]

    for y_mod in (0, 1):
        for x_mod in (0, 1):
            color = pattern_tile[y_mod][x_mod]
            channel = _CHANNEL_INDEX[color]
            mosaic[y_mod::2, x_mod::2] = image[y_mod::2, x_mod::2, channel]

    return mosaic


def diff_stats(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    """Return simple absolute-difference statistics."""

    if np.issubdtype(actual.dtype, np.floating) or np.issubdtype(expected.dtype, np.floating):
        diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        max_abs_diff = float(diff.max(initial=0.0))
    else:
        diff = np.abs(actual.astype(np.int64) - expected.astype(np.int64))
        max_abs_diff = int(diff.max(initial=0))

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "different_values": int(np.count_nonzero(diff)),
    }
