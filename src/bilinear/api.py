from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._backend import impl


RGGB = "RGGB"
GRBG = "GRBG"
BGGR = "BGGR"
GBRG = "GBRG"


# OpenCV Bayer bilinear conversion code values. OpenCV has many aliases with
# shared integer values, so this maps the behavior of the integer code itself.
_CODE_PATTERNS = {
    46: RGGB,
    47: GRBG,
    48: BGGR,
    49: GBRG,
}


def demosaicing(
    src: NDArray[np.generic],
    code: int,
    fast: bool = True,
) -> NDArray[np.generic]:
    """Convert a Bayer image to BGR color with bilinear interpolation.

    The public interface intentionally mirrors the simple OpenCV call shape:
    ``cv2.demosaicing(src, code)``. Only 3-channel bilinear Bayer conversion
    codes are supported. By default, the native backend uses the optimized C++
    kernel. Pass ``fast=False`` to use the simpler baseline CPU implementation.
    """

    bayer = np.asarray(src)
    if bayer.ndim != 2:
        raise ValueError("src must be a 2D Bayer image")
    if bayer.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raise ValueError("src must have dtype uint8, uint16, float32, or float64")
    if code not in _CODE_PATTERNS:
        raise ValueError(f"unsupported Bayer bilinear conversion code: {code}")

    bayer = np.ascontiguousarray(bayer)
    pattern = _CODE_PATTERNS[code]

    if fast:
        return impl._demosaic_bgr_fast(bayer, pattern)
    return impl._demosaic_bgr(bayer, pattern)
