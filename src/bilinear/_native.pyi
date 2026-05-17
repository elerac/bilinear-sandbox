from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _demosaic_bgr_into(
    bayer: NDArray[np.uint8] | NDArray[np.uint16] | NDArray[np.float32] | NDArray[np.float64],
    output: NDArray[np.uint8] | NDArray[np.uint16] | NDArray[np.float32] | NDArray[np.float64],
    pattern: str,
) -> None: ...
