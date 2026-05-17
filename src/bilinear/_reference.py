from __future__ import annotations

import cv2
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

_PATTERN_CODES = {
    RGGB: cv2.COLOR_BayerRGGB2BGR,
    GRBG: cv2.COLOR_BayerGRBG2BGR,
    BGGR: cv2.COLOR_BayerBGGR2BGR,
    GBRG: cv2.COLOR_BayerGBRG2BGR,
}


def _demosaic_bgr(bayer: np.ndarray, pattern: str) -> np.ndarray:
    height, width = bayer.shape
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

    if height < 3 or width < 3:
        return np.stack((blue, green, red), axis=2).astype(bayer.dtype)

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

    return np.stack((blue, green, red), axis=2).astype(bayer.dtype)


def _demosaic_bgr_fast(bayer: np.ndarray, pattern: str) -> np.ndarray:
    height, width = bayer.shape
    if height < 3 or width < 3:
        return np.zeros((height, width, 3), dtype=bayer.dtype)

    if bayer.dtype in (np.uint8, np.uint16):
        return cv2.demosaicing(bayer, _PATTERN_CODES[pattern])

    return _filter2d_demosaic_bgr(bayer, pattern)


def _filter2d_demosaic_bgr(bayer: np.ndarray, pattern: str) -> np.ndarray:
    height, width = bayer.shape
    output = np.zeros((height, width, 3), dtype=bayer.dtype)

    kernel_dtype = bayer.dtype
    cross_kernel = np.array(
        [[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]],
        dtype=kernel_dtype,
    )
    diagonal_kernel = np.array(
        [[0.25, 0.0, 0.25], [0.0, 0.0, 0.0], [0.25, 0.0, 0.25]],
        dtype=kernel_dtype,
    )
    horizontal_kernel = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 0.0]],
        dtype=kernel_dtype,
    )
    vertical_kernel = np.array(
        [[0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
        dtype=kernel_dtype,
    )

    cross = cv2.filter2D(bayer, -1, cross_kernel, borderType=cv2.BORDER_CONSTANT)
    diagonal = cv2.filter2D(bayer, -1, diagonal_kernel, borderType=cv2.BORDER_CONSTANT)
    horizontal = cv2.filter2D(bayer, -1, horizontal_kernel, borderType=cv2.BORDER_CONSTANT)
    vertical = cv2.filter2D(bayer, -1, vertical_kernel, borderType=cv2.BORDER_CONSTANT)

    positions = _PATTERN_POSITIONS[pattern]
    red_mask = _parity_mask(bayer.shape, positions["R"])
    blue_mask = _parity_mask(bayer.shape, positions["B"])
    green1_mask = _parity_mask(bayer.shape, positions["G1"])
    green2_mask = _parity_mask(bayer.shape, positions["G2"])

    blue = output[..., 0]
    green = output[..., 1]
    red = output[..., 2]

    red[red_mask] = bayer[red_mask]
    green[red_mask] = cross[red_mask]
    blue[red_mask] = diagonal[red_mask]

    blue[blue_mask] = bayer[blue_mask]
    green[blue_mask] = cross[blue_mask]
    red[blue_mask] = diagonal[blue_mask]

    for green_mask, green_position in ((green1_mask, positions["G1"]), (green2_mask, positions["G2"])):
        green[green_mask] = bayer[green_mask]
        if _horizontal_neighbors_are_red(green_position[0], green_position[1], positions):
            red[green_mask] = horizontal[green_mask]
            blue[green_mask] = vertical[green_mask]
        else:
            blue[green_mask] = horizontal[green_mask]
            red[green_mask] = vertical[green_mask]

    for channel in (blue, green, red):
        _copy_inner_border(channel)

    return output


def _parity_mask(shape: tuple[int, int], parity: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[parity[0] :: 2, parity[1] :: 2] = True
    return mask


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
