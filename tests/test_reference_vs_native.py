from __future__ import annotations

import importlib
import os

import cv2
import numpy as np
import pytest


COLOR_CODES = [
    cv2.COLOR_BayerRGGB2BGR,
    cv2.COLOR_BayerGRBG2BGR,
    cv2.COLOR_BayerBGGR2BGR,
    cv2.COLOR_BayerGBRG2BGR,
]


@pytest.fixture(autouse=True)
def restore_backend_after_test() -> None:
    original_backend = os.environ.get("BILINEAR_BACKEND")
    yield
    if original_backend is None:
        os.environ.pop("BILINEAR_BACKEND", None)
    else:
        os.environ["BILINEAR_BACKEND"] = original_backend
    reload_api_modules()


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("shape", [(0, 0), (0, 5), (5, 0), (1, 1), (2, 5), (5, 2), (7, 9)])
@pytest.mark.parametrize("code", COLOR_CODES)
@pytest.mark.parametrize("fast", [False, True])
def test_native_matches_reference_for_shapes(dtype: type[np.generic], shape: tuple[int, int], code: int, fast: bool) -> None:
    src = random_bayer(shape, dtype)

    expected = run_with_backend("python", src, code, fast=fast)
    actual = run_with_backend("native", src, code, fast=fast)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.flags.c_contiguous
    assert_demosaicing_matches(actual, expected, fast=fast)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("layout", ["contiguous", "sliced", "transposed"])
@pytest.mark.parametrize("code", COLOR_CODES)
@pytest.mark.parametrize("fast", [False, True])
def test_native_matches_reference_for_layouts(dtype: type[np.generic], layout: str, code: int, fast: bool) -> None:
    src = layout_case(layout, dtype)
    before = src.copy()

    expected = run_with_backend("python", src, code, fast=fast)
    actual = run_with_backend("native", src, code, fast=fast)

    np.testing.assert_array_equal(src, before)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.flags.c_contiguous
    assert_demosaicing_matches(actual, expected, fast=fast)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_python_fast_accepts_supported_dtypes(dtype: type[np.generic]) -> None:
    src = random_bayer((7, 9), dtype)
    actual = run_with_backend("python", src, cv2.COLOR_BayerRGGB2BGR, fast=True)
    if np.issubdtype(dtype, np.floating):
        expected = run_with_backend("python", src, cv2.COLOR_BayerRGGB2BGR, fast=False)
    else:
        expected = cv2.demosaicing(np.ascontiguousarray(src), cv2.COLOR_BayerRGGB2BGR)

    assert actual.shape == (*src.shape, 3)
    assert actual.dtype == src.dtype
    assert actual.flags.c_contiguous
    assert_demosaicing_matches(actual, expected, fast=True)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("shape", [(0, 0), (0, 5), (5, 0), (1, 1), (2, 5), (5, 2)])
def test_python_fast_returns_zeros_for_small_images(dtype: type[np.generic], shape: tuple[int, int]) -> None:
    src = random_bayer(shape, dtype)

    actual = run_with_backend("python", src, cv2.COLOR_BayerRGGB2BGR, fast=True)

    assert actual.shape == (*shape, 3)
    assert actual.dtype == src.dtype
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, np.zeros((*shape, 3), dtype=dtype))


@pytest.mark.parametrize("backend", ["python", "native", "auto"])
def test_backend_modes(backend: str) -> None:
    src = random_bayer((8, 9), np.uint16)

    expected = run_with_backend("python", src, cv2.COLOR_BayerGBRG2BGR)
    actual = run_with_backend(backend, src, cv2.COLOR_BayerGBRG2BGR)

    np.testing.assert_array_equal(actual, expected)


def run_with_backend(backend: str, src: np.ndarray, code: int, fast: bool = True) -> np.ndarray:
    os.environ["BILINEAR_BACKEND"] = backend
    api = reload_api_modules()
    return api.demosaicing(src, code, fast=fast)


def assert_demosaicing_matches(actual: np.ndarray, expected: np.ndarray, fast: bool) -> None:
    if fast and np.issubdtype(actual.dtype, np.floating):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=np.finfo(actual.dtype).eps * 64)
    else:
        np.testing.assert_array_equal(actual, expected)


def reload_api_modules():
    import bilinear
    import bilinear._backend
    import bilinear.api

    importlib.reload(bilinear._backend)
    api = importlib.reload(bilinear.api)
    bilinear.demosaicing = api.demosaicing
    return api


def layout_case(layout: str, dtype: type[np.generic]) -> np.ndarray:
    base = random_bayer((10, 12), dtype)
    if layout == "contiguous":
        return base.copy()
    if layout == "sliced":
        return base[1:9, ::2]
    if layout == "transposed":
        return base.T
    raise AssertionError(f"unknown layout: {layout}")


def random_bayer(shape: tuple[int, int], dtype: type[np.generic]) -> np.ndarray:
    rng = np.random.default_rng(seed=sum(shape) + np.dtype(dtype).itemsize)
    if np.issubdtype(dtype, np.floating):
        return rng.random(size=shape, dtype=dtype) * np.array(16.0, dtype=dtype)
    return rng.integers(0, np.iinfo(dtype).max + 1, size=shape, dtype=dtype)
