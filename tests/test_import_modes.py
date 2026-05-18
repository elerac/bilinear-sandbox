from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def restore_backend_after_test() -> None:
    original_backend = os.environ.get("FASTIMG_BACKEND")
    original_old_backend = os.environ.get("BILINEAR_BACKEND")
    yield
    if original_backend is None:
        os.environ.pop("FASTIMG_BACKEND", None)
    else:
        os.environ["FASTIMG_BACKEND"] = original_backend
    if original_old_backend is None:
        os.environ.pop("BILINEAR_BACKEND", None)
    else:
        os.environ["BILINEAR_BACKEND"] = original_old_backend
    reload_api_modules()


def test_default_backend_is_auto_preferring_native_when_available() -> None:
    os.environ.pop("FASTIMG_BACKEND", None)

    api = reload_api_modules()

    import fastimg

    assert fastimg.backend() == "native"
    assert fastimg.native_error() is None
    assert api.demosaicing(np.zeros((3, 3), dtype=np.uint8), cv2.COLOR_BayerRGGB2BGR).shape == (3, 3, 3)


def test_can_force_python_backend() -> None:
    os.environ["FASTIMG_BACKEND"] = "python"

    reload_api_modules()

    import fastimg
    import fastimg._backend as backend

    assert fastimg.backend() == "python"
    assert backend.native() is None
    assert fastimg.native_error() is None


def test_can_force_native_backend() -> None:
    os.environ["FASTIMG_BACKEND"] = "native"

    reload_api_modules()

    import fastimg
    import fastimg._backend as backend

    assert fastimg.backend() == "native"
    assert backend.native() is not None
    assert fastimg.native_error() is None


def test_demosaicing_module_import_keeps_package_api() -> None:
    import fastimg
    from fastimg.demosaicing import demosaicing as module_demosaicing

    from fastimg import demosaicing as package_demosaicing

    assert callable(module_demosaicing)
    assert callable(package_demosaicing)
    assert not isinstance(package_demosaicing, ModuleType)
    assert not isinstance(fastimg.demosaicing, ModuleType)

    src = np.zeros((3, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        module_demosaicing(src, cv2.COLOR_BayerRGGB2BGR),
        package_demosaicing(src, cv2.COLOR_BayerRGGB2BGR),
    )


def test_invalid_backend_value_is_rejected() -> None:
    os.environ["FASTIMG_BACKEND"] = "invalid"

    import fastimg._backend as backend

    with pytest.raises(RuntimeError, match="FASTIMG_BACKEND"):
        importlib.reload(backend)


def test_old_backend_env_var_is_ignored() -> None:
    os.environ.pop("FASTIMG_BACKEND", None)
    os.environ["BILINEAR_BACKEND"] = "python"

    reload_api_modules()

    import fastimg

    assert fastimg.backend() == "native"


def test_native_backend_unavailable_message() -> None:
    import fastimg
    import fastimg._backend as backend

    with without_native_module(fastimg):
        os.environ["FASTIMG_BACKEND"] = "native"

        with pytest.raises(RuntimeError, match="native extension could not be imported"):
            importlib.reload(backend)


def test_auto_backend_warns_and_falls_back_to_python_when_native_unavailable() -> None:
    import fastimg

    with without_native_module(fastimg):
        os.environ["FASTIMG_BACKEND"] = "auto"
        with pytest.warns(RuntimeWarning, match="using the pure Python backend"):
            api = reload_api_modules()

        import fastimg

        assert fastimg.backend() == "python"
        assert fastimg.native_error() is not None
        actual = api.demosaicing(np.zeros((3, 3), dtype=np.uint8), cv2.COLOR_BayerRGGB2BGR)

    expected = run_python(np.zeros((3, 3), dtype=np.uint8), cv2.COLOR_BayerRGGB2BGR)
    np.testing.assert_array_equal(actual, expected)


def test_pure_backend_import_smoke() -> None:
    env = os.environ.copy()
    env["FASTIMG_BACKEND"] = "python"

    proc = subprocess.run(
        [sys.executable, "-c", "import fastimg; assert fastimg.backend() == 'python'"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0


def test_old_bilinear_import_is_not_supported() -> None:
    sys.modules.pop("bilinear", None)
    assert importlib.util.find_spec("bilinear") is None


class without_native_module:
    def __init__(self, package: ModuleType) -> None:
        self.package = package
        self.module: ModuleType | None = None
        self.had_attr = False
        self.attr: Any = None

    def __enter__(self) -> None:
        self.module = sys.modules.pop("fastimg._native", None)
        self.had_attr = hasattr(self.package, "_native")
        if self.had_attr:
            self.attr = getattr(self.package, "_native")
            delattr(self.package, "_native")
        sys.modules["fastimg._native"] = None  # type: ignore[assignment]

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        sys.modules.pop("fastimg._native", None)
        if self.module is not None:
            sys.modules["fastimg._native"] = self.module
        if self.had_attr:
            setattr(self.package, "_native", self.attr)


def run_python(src: np.ndarray, code: int) -> np.ndarray:
    os.environ["FASTIMG_BACKEND"] = "python"
    api = reload_api_modules()
    return api.demosaicing(src, code)


def reload_api_modules():
    import fastimg
    import fastimg._backend
    import fastimg.api

    importlib.reload(fastimg._backend)
    api = importlib.reload(fastimg.api)
    demosaicing_module = importlib.import_module("fastimg.demosaicing")
    importlib.reload(demosaicing_module)
    importlib.reload(fastimg)
    return api
