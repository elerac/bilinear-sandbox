from __future__ import annotations

import cv2
import numpy as np
import pytest

from fastimg import demosaicing


COLOR_CODES = [
    cv2.COLOR_BayerRGGB2BGR,
    cv2.COLOR_BayerGRBG2BGR,
    cv2.COLOR_BayerBGGR2BGR,
    cv2.COLOR_BayerGBRG2BGR,
]


@pytest.mark.parametrize("dtype,max_value", [(np.uint8, 256), (np.uint16, 65536)])
@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 3), (5, 7), (20, 21)])
@pytest.mark.parametrize("code", COLOR_CODES)
def test_color_demosaicing_matches_opencv(
    dtype: type[np.generic],
    max_value: int,
    shape: tuple[int, int],
    code: int,
) -> None:
    src = random_bayer(shape, dtype, max_value)

    actual = demosaicing(src, code)
    expected = cv2.demosaicing(src, code)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype,max_value", [(np.uint8, 255), (np.uint16, 65535)])
@pytest.mark.parametrize("shape", [(3, 3), (4, 6), (7, 9), (20, 21)])
@pytest.mark.parametrize("code", COLOR_CODES)
def test_color_demosaicing_edge_cases(
    dtype: type[np.generic],
    max_value: int,
    shape: tuple[int, int],
    code: int,
) -> None:
    cases = [
        np.zeros(shape, dtype=dtype),
        np.full(shape, max_value, dtype=dtype),
        np.arange(np.prod(shape), dtype=dtype).reshape(shape),
    ]

    for src in cases:
        actual = demosaicing(src, code)
        expected = cv2.demosaicing(src, code)
        np.testing.assert_array_equal(actual, expected)


def test_rejects_removed_fast_keyword() -> None:
    src = random_bayer((20, 21), np.uint8, 256)

    with pytest.raises(TypeError, match="fast"):
        demosaicing(src, cv2.COLOR_BayerRGGB2BGR, fast=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_demosaicing_uses_true_averages(dtype: type[np.floating]) -> None:
    src = np.array(
        [
            [0.0, 1.5, 2.25, 3.75, 4.5],
            [5.25, 6.5, 7.75, 8.125, 9.5],
            [10.0, 11.25, 12.5, 13.75, 14.0],
            [15.5, 16.25, 17.5, 18.75, 19.25],
            [20.0, 21.5, 22.25, 23.5, 24.75],
        ],
        dtype=dtype,
    )

    actual = demosaicing(src, cv2.COLOR_BayerRGGB2BGR)

    assert actual.dtype == dtype
    assert_float_array_matches(
        actual[2, 2],
        np.array(
            [
                _float_avg4(src[1, 1], src[1, 3], src[3, 1], src[3, 3]),
                _float_avg4(src[1, 2], src[3, 2], src[2, 1], src[2, 3]),
                src[2, 2],
            ],
            dtype=dtype,
        ),
    )
    assert_float_array_matches(
        actual[1, 1],
        np.array(
            [
                src[1, 1],
                _float_avg4(src[0, 1], src[2, 1], src[1, 0], src[1, 2]),
                _float_avg4(src[0, 0], src[0, 2], src[2, 0], src[2, 2]),
            ],
            dtype=dtype,
        ),
    )
    assert_float_array_matches(
        actual[2, 1],
        np.array(
            [
                _float_avg2(src[1, 1], src[3, 1]),
                src[2, 1],
                _float_avg2(src[2, 0], src[2, 2]),
            ],
            dtype=dtype,
        ),
    )
    assert_float_array_matches(
        actual[1, 2],
        np.array(
            [
                _float_avg2(src[1, 1], src[1, 3]),
                src[1, 2],
                _float_avg2(src[0, 2], src[2, 2]),
            ],
            dtype=dtype,
        ),
    )
    assert_float_array_matches(actual[0, 2], actual[1, 2])
    assert_float_array_matches(actual[4, 2], actual[3, 2])
    assert_float_array_matches(actual[2, 0], actual[2, 1])
    assert_float_array_matches(actual[2, 4], actual[2, 3])


@pytest.mark.parametrize(
    "code",
    [
        cv2.COLOR_BayerRGGB2BGR_EA,
        cv2.COLOR_BayerRGGB2GRAY,
        cv2.COLOR_BayerRGGB2BGRA,
        cv2.COLOR_BayerRGGB2RGBA,
    ],
)
def test_rejects_unsupported_code(code: int) -> None:
    src = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="unsupported"):
        demosaicing(src, code)


def test_rejects_non_bayer_image() -> None:
    src = np.zeros((5, 5, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="2D"):
        demosaicing(src, cv2.COLOR_BayerRGGB2BGR)


def test_rejects_unsupported_dtype() -> None:
    src = np.zeros((5, 5), dtype=np.float16)

    with pytest.raises(ValueError, match="uint8, uint16, float32, or float64"):
        demosaicing(src, cv2.COLOR_BayerRGGB2BGR)


def random_bayer(shape: tuple[int, int], dtype: type[np.generic], max_value: int) -> np.ndarray:
    rng = np.random.default_rng(seed=sum(shape) + max_value)
    return rng.integers(0, max_value, size=shape, dtype=dtype)


def _float_avg2(a: np.floating, b: np.floating) -> np.floating:
    return (a + b) / 2


def _float_avg4(a: np.floating, b: np.floating, c: np.floating, d: np.floating) -> np.floating:
    return (a + b + c + d) / 4


def assert_float_array_matches(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(actual, expected, rtol=0, atol=np.finfo(actual.dtype).eps * 64)
