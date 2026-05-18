from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from fastimg import demosaicing
from image_utils import diff_stats, make_bayer_mosaic


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare fastimg.demosaicing with OpenCV.")
    parser.add_argument("--image", type=Path, default=Path("curtains.jpg"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {args.image}")

    code = cv2.COLOR_BayerRGGB2BGR
    mosaic = make_bayer_mosaic(image, pattern="RGGB")

    actual = demosaicing(mosaic, code)
    expected = cv2.demosaicing(mosaic, code)
    stats = diff_stats(actual, expected)

    print(f"image: {args.image}")
    print(f"bayer shape: {mosaic.shape}, dtype: {mosaic.dtype}")
    print(f"result shape: {actual.shape}, dtype: {actual.dtype}")
    print(f"max abs diff: {stats['max_abs_diff']}")
    print(f"mean abs diff: {stats['mean_abs_diff']:.6f}")
    print(f"different values: {stats['different_values']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / "curtains_bayer.png"), mosaic)
    cv2.imwrite(str(args.output_dir / "curtains_fastimg.png"), actual)
    cv2.imwrite(str(args.output_dir / "curtains_opencv.png"), expected)
    cv2.imwrite(str(args.output_dir / "curtains_absdiff.png"), cv2.absdiff(actual, expected))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
