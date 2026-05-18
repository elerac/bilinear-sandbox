from __future__ import annotations

import numpy as np


RGGB = "RGGB"
GRBG = "GRBG"
BGGR = "BGGR"
GBRG = "GBRG"


_PATTERN_POSITIONS = {
    RGGB: {"R": (0, 0), "G1": (0, 1), "G2": (1, 0), "B": (1, 1)},
    GRBG: {"G1": (0, 0), "R": (0, 1), "B": (1, 0), "G2": (1, 1)},
    BGGR: {"B": (0, 0), "G1": (0, 1), "G2": (1, 0), "R": (1, 1)},
    GBRG: {"G1": (0, 0), "B": (0, 1), "R": (1, 0), "G2": (1, 1)},
}


def _demosaic_bgr(bayer: np.ndarray, pattern: str) -> np.ndarray:
    output = np.zeros((*bayer.shape, 3), dtype=bayer.dtype)
    _demosaic_bgr_into(bayer, output, pattern)
    return output


def _demosaic_bgr_into(bayer: np.ndarray, output: np.ndarray, pattern: str) -> None:
    height, width = bayer.shape
    output[...] = 0
    if height < 3 or width < 3:
        return

    if np.issubdtype(bayer.dtype, np.floating):
        work = bayer
        avg2 = _float_avg2
        avg4 = _float_avg4
    else:
        work = bayer.astype(np.int64)
        avg2 = _int_avg2
        avg4 = _int_avg4

    blue = np.zeros((height, width), dtype=work.dtype)
    green = np.zeros((height, width), dtype=work.dtype)
    red = np.zeros((height, width), dtype=work.dtype)
    positions = _PATTERN_POSITIONS[pattern]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            parity = (y % 2, x % 2)

            if parity == positions["R"]:
                red[y, x] = work[y, x]
                green[y, x] = avg4(
                    work[y - 1, x],
                    work[y + 1, x],
                    work[y, x - 1],
                    work[y, x + 1],
                )
                blue[y, x] = avg4(
                    work[y - 1, x - 1],
                    work[y - 1, x + 1],
                    work[y + 1, x - 1],
                    work[y + 1, x + 1],
                )
            elif parity == positions["B"]:
                blue[y, x] = work[y, x]
                green[y, x] = avg4(
                    work[y - 1, x],
                    work[y + 1, x],
                    work[y, x - 1],
                    work[y, x + 1],
                )
                red[y, x] = avg4(
                    work[y - 1, x - 1],
                    work[y - 1, x + 1],
                    work[y + 1, x - 1],
                    work[y + 1, x + 1],
                )
            else:
                green[y, x] = work[y, x]
                if _horizontal_neighbors_are_red(y, x, positions):
                    red[y, x] = avg2(work[y, x - 1], work[y, x + 1])
                    blue[y, x] = avg2(work[y - 1, x], work[y + 1, x])
                else:
                    blue[y, x] = avg2(work[y, x - 1], work[y, x + 1])
                    red[y, x] = avg2(work[y - 1, x], work[y + 1, x])

    for channel in (blue, green, red):
        _copy_inner_border(channel)

    output[..., 0] = blue.astype(bayer.dtype, copy=False)
    output[..., 1] = green.astype(bayer.dtype, copy=False)
    output[..., 2] = red.astype(bayer.dtype, copy=False)


def _int_avg2(a: int, b: int) -> int:
    return (a + b + 1) // 2


def _int_avg4(a: int, b: int, c: int, d: int) -> int:
    return (a + b + c + d + 2) // 4


def _float_avg2(a: np.floating, b: np.floating) -> np.floating:
    return (a + b) / 2


def _float_avg4(a: np.floating, b: np.floating, c: np.floating, d: np.floating) -> np.floating:
    return (a + b + c + d) / 4


def _horizontal_neighbors_are_red(y: int, x: int, positions: dict[str, tuple[int, int]]) -> bool:
    return (y % 2, (x - 1) % 2) == positions["R"] or (y % 2, (x + 1) % 2) == positions["R"]


def _copy_inner_border(channel: np.ndarray) -> None:
    channel[0, :] = channel[1, :]
    channel[-1, :] = channel[-2, :]
    channel[:, 0] = channel[:, 1]
    channel[:, -1] = channel[:, -2]
