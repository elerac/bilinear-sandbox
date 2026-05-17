from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from bilinear import demosaicing
from image_utils import diff_stats, make_bayer_mosaic


_BGR_CODES = {
    "RGGB": cv2.COLOR_BayerRGGB2BGR,
    "GRBG": cv2.COLOR_BayerGRBG2BGR,
    "BGGR": cv2.COLOR_BayerBGGR2BGR,
    "GBRG": cv2.COLOR_BayerGBRG2BGR,
}

_PATTERN_POSITIONS = {
    "RGGB": {"R": (0, 0), "G1": (0, 1), "G2": (1, 0), "B": (1, 1)},
    "GRBG": {"G1": (0, 0), "R": (0, 1), "B": (1, 0), "G2": (1, 1)},
    "BGGR": {"B": (0, 0), "G1": (0, 1), "G2": (1, 0), "R": (1, 1)},
    "GBRG": {"G1": (0, 0), "B": (0, 1), "R": (1, 0), "G2": (1, 1)},
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark bilinear.demosaicing against OpenCV.")
    parser.add_argument("--image", type=Path, default=Path("curtains.jpg"))
    parser.add_argument("--pattern", choices=sorted(_BGR_CODES), default="RGGB")
    parser.add_argument("--dtype", choices=["uint8", "uint16", "float32", "float64", "both"], default="both")
    parser.add_argument(
        "--include-reference",
        "--reference",
        action="store_true",
        dest="include_reference",
        help="also benchmark the slow pure Python reference implementation",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.scale <= 0:
        raise ValueError("--scale must be greater than 0")

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {args.image}")
    if args.scale != 1.0:
        image = cv2.resize(image, None, fx=args.scale, fy=args.scale)

    code = _BGR_CODES[args.pattern]
    uint8_mosaic = make_bayer_mosaic(image, pattern=args.pattern)
    mosaics = {
        "uint8": uint8_mosaic,
        "uint16": uint8_mosaic.astype(np.uint16) * 257,
        "float32": uint8_mosaic.astype(np.float32) / np.float32(255),
        "float64": uint8_mosaic.astype(np.float64) / np.float64(255),
    }
    dtype_names = ["uint8", "uint16", "float32", "float64"] if args.dtype == "both" else [args.dtype]

    print(f"image: {args.image}")
    print(f"scale: {args.scale}")
    print(f"pattern: {args.pattern}")
    print(f"iterations: {args.iterations}, warmup: {args.warmup}")

    for index, dtype_name in enumerate(dtype_names):
        if index:
            print()
        run_benchmark_section(
            mosaic=mosaics[dtype_name],
            pattern=args.pattern,
            code=code,
            warmup=args.warmup,
            iterations=args.iterations,
            include_reference=args.include_reference,
        )

    return 0


def run_benchmark_section(
    mosaic: np.ndarray,
    pattern: str,
    code: int,
    warmup: int,
    iterations: int,
    include_reference: bool,
) -> None:
    fast_result, fast_times = benchmark(
        lambda: demosaicing(mosaic, code, fast=True),
        warmup=warmup,
        iterations=iterations,
    )
    slow_result, slow_times = benchmark(
        lambda: demosaicing(mosaic, code, fast=False),
        warmup=warmup,
        iterations=iterations,
    )

    reference_result: np.ndarray | None = None
    reference_times: list[float] | None = None
    if include_reference:
        from bilinear._python import demosaicing as python_demosaicing

        reference_result, reference_times = benchmark(
            lambda: python_demosaicing._demosaic_bgr(mosaic, pattern),
            warmup=warmup,
            iterations=iterations,
        )
    fast_vs_slow_stats = diff_stats(fast_result, slow_result)
    reference_stats = diff_stats(fast_result, reference_result) if reference_result is not None else None

    if not np.issubdtype(mosaic.dtype, np.floating):
        opencv_label = "opencv"
        opencv_result, opencv_times = benchmark(
            lambda: cv2.demosaicing(mosaic, code),
            warmup=warmup,
            iterations=iterations,
        )
    else:
        opencv_label = "opencv filter2D"
        opencv_result, opencv_times = benchmark(
            lambda: opencv_filter2d_demosaic_bgr(mosaic, pattern),
            warmup=warmup,
            iterations=iterations,
        )

    print(f"dtype: {mosaic.dtype}")
    print(f"bayer shape: {mosaic.shape}, dtype: {mosaic.dtype}")
    print_timing("bilinear cpu fast", fast_times)
    print_timing("bilinear cpu slow", slow_times)
    if reference_times is not None:
        print_timing("bilinear reference", reference_times)
    print_timing(opencv_label, opencv_times)
    print(f"fast vs slow speedup: {statistics.median(slow_times) / statistics.median(fast_times):.2f}x")
    if reference_times is not None:
        print(f"fast vs reference speedup: {statistics.median(reference_times) / statistics.median(fast_times):.2f}x")
    print(f"fast/{opencv_label} speed ratio: {statistics.median(fast_times) / statistics.median(opencv_times):.2f}x")
    print_diff_stats("cpu fast vs slow", fast_vs_slow_stats)
    print_diff_stats(f"cpu fast vs {opencv_label}", diff_stats(fast_result, opencv_result))
    print_diff_stats(f"cpu slow vs {opencv_label}", diff_stats(slow_result, opencv_result))
    if reference_stats is not None:
        print_diff_stats("cpu fast vs reference", reference_stats)


def opencv_filter2d_demosaic_bgr(bayer: np.ndarray, pattern: str) -> np.ndarray:
    if bayer.dtype not in (np.float32, np.float64):
        raise ValueError("filter2D baseline expects float32 or float64 input")

    height, width = bayer.shape
    output = np.zeros((height, width, 3), dtype=bayer.dtype)
    if height < 3 or width < 3:
        return output

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
    red_mask = parity_mask(bayer.shape, positions["R"])
    blue_mask = parity_mask(bayer.shape, positions["B"])
    green1_mask = parity_mask(bayer.shape, positions["G1"])
    green2_mask = parity_mask(bayer.shape, positions["G2"])

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
        if horizontal_neighbors_are_red(green_position, positions):
            red[green_mask] = horizontal[green_mask]
            blue[green_mask] = vertical[green_mask]
        else:
            blue[green_mask] = horizontal[green_mask]
            red[green_mask] = vertical[green_mask]

    for channel in (blue, green, red):
        copy_inner_border(channel)

    return output


def parity_mask(shape: tuple[int, int], parity: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[parity[0] :: 2, parity[1] :: 2] = True
    return mask


def horizontal_neighbors_are_red(
    parity: tuple[int, int],
    positions: dict[str, tuple[int, int]],
) -> bool:
    y, x = parity
    return (y, (x - 1) % 2) == positions["R"] or (y, (x + 1) % 2) == positions["R"]


def copy_inner_border(channel: np.ndarray) -> None:
    channel[0, :] = channel[1, :]
    channel[-1, :] = channel[-2, :]
    channel[:, 0] = channel[:, 1]
    channel[:, -1] = channel[:, -2]


def benchmark(function: Callable[[], np.ndarray], warmup: int, iterations: int) -> tuple[np.ndarray, list[float]]:
    for _ in range(warmup):
        function()

    result: np.ndarray | None = None
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - start)

    if result is None:
        raise ValueError("iterations must be at least 1")
    return result, times


def print_timing(name: str, times: list[float]) -> None:
    print(
        f"{name}: "
        f"min {min(times) * 1000:.2f} ms, "
        f"median {statistics.median(times) * 1000:.2f} ms, "
        f"mean {statistics.mean(times) * 1000:.2f} ms"
    )


def print_diff_stats(name: str, stats: dict[str, float | int]) -> None:
    print(
        f"{name}: "
        f"max abs diff {stats['max_abs_diff']}, "
        f"mean abs diff {stats['mean_abs_diff']:.6f}, "
        f"different values {stats['different_values']}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
