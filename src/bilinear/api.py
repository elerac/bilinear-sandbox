from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._backend import native
from ._python import demosaicing as _python_demosaicing


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
) -> NDArray[np.generic]:
    """Convert a Bayer image to BGR color with bilinear interpolation.

    The public interface intentionally mirrors the simple OpenCV call shape:
    ``cv2.demosaicing(src, code)``. Only 3-channel bilinear Bayer conversion
    codes are supported.
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
    output = np.empty((*bayer.shape, 3), dtype=bayer.dtype)

    if bayer.shape[0] < 3 or bayer.shape[1] < 3:
        output.fill(0)
        return output

    mod = native()
    if mod is None:
        _python_demosaicing._demosaic_bgr_into(bayer, output, pattern)
        return output

    mod._demosaic_bgr_into(bayer, output, pattern)

    return output
