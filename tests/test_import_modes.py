from __future__ import annotations

import importlib
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
    original_backend = os.environ.get("BILINEAR_BACKEND")
    yield
    if original_backend is None:
        os.environ.pop("BILINEAR_BACKEND", None)
    else:
        os.environ["BILINEAR_BACKEND"] = original_backend
    reload_api_modules()


def test_default_backend_is_auto_preferring_native_when_available() -> None:
    os.environ.pop("BILINEAR_BACKEND", None)

    api = reload_api_modules()

    import bilinear

    assert bilinear.backend() == "native"
    assert bilinear.native_error() is None
    assert api.demosaicing(np.zeros((3, 3), dtype=np.uint8), cv2.COLOR_BayerRGGB2BGR).shape == (3, 3, 3)


def test_can_force_python_backend() -> None:
    os.environ["BILINEAR_BACKEND"] = "python"

    reload_api_modules()

    import bilinear
    import bilinear._backend as backend

    assert bilinear.backend() == "python"
    assert backend.native() is None
    assert bilinear.native_error() is None


def test_can_force_native_backend() -> None:
    os.environ["BILINEAR_BACKEND"] = "native"

    reload_api_modules()

    import bilinear
    import bilinear._backend as backend

    assert bilinear.backend() == "native"
    assert backend.native() is not None
    assert bilinear.native_error() is None


def test_legacy_module_import_keeps_package_api() -> None:
    import bilinear
    from bilinear.demosaicing import demosaicing as legacy_demosaicing

    from bilinear import demosaicing as package_demosaicing

    assert callable(legacy_demosaicing)
    assert callable(package_demosaicing)
    assert not isinstance(package_demosaicing, ModuleType)
    assert not isinstance(bilinear.demosaicing, ModuleType)

    src = np.zeros((3, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        legacy_demosaicing(src, cv2.COLOR_BayerRGGB2BGR),
        package_demosaicing(src, cv2.COLOR_BayerRGGB2BGR),
    )


def test_invalid_backend_value_is_rejected() -> None:
    os.environ["BILINEAR_BACKEND"] = "invalid"

    import bilinear._backend as backend

    with pytest.raises(RuntimeError, match="BILINEAR_BACKEND"):
        importlib.reload(backend)


def test_native_backend_unavailable_message() -> None:
    import bilinear
    import bilinear._backend as backend

    with without_native_module(bilinear):
        os.environ["BILINEAR_BACKEND"] = "native"

        with pytest.raises(RuntimeError, match="native extension could not be imported"):
            importlib.reload(backend)


def test_auto_backend_warns_and_falls_back_to_python_when_native_unavailable() -> None:
    import bilinear

    with without_native_module(bilinear):
        os.environ["BILINEAR_BACKEND"] = "auto"
        with pytest.warns(RuntimeWarning, match="using the pure Python backend"):
            api = reload_api_modules()

        import bilinear

        assert bilinear.backend() == "python"
        assert bilinear.native_error() is not None
        actual = api.demosaicing(np.zeros((3, 3), dtype=np.uint8), cv2.COLOR_BayerRGGB2BGR)

    expected = run_python(np.zeros((3, 3), dtype=np.uint8), cv2.COLOR_BayerRGGB2BGR)
    np.testing.assert_array_equal(actual, expected)


def test_pure_backend_import_smoke() -> None:
    env = os.environ.copy()
    env["BILINEAR_BACKEND"] = "python"

    proc = subprocess.run(
        [sys.executable, "-c", "import bilinear; assert bilinear.backend() == 'python'"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0


class without_native_module:
    def __init__(self, package: ModuleType) -> None:
        self.package = package
        self.module: ModuleType | None = None
        self.had_attr = False
        self.attr: Any = None

    def __enter__(self) -> None:
        self.module = sys.modules.pop("bilinear._native", None)
        self.had_attr = hasattr(self.package, "_native")
        if self.had_attr:
            self.attr = getattr(self.package, "_native")
            delattr(self.package, "_native")
        sys.modules["bilinear._native"] = None  # type: ignore[assignment]

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        sys.modules.pop("bilinear._native", None)
        if self.module is not None:
            sys.modules["bilinear._native"] = self.module
        if self.had_attr:
            setattr(self.package, "_native", self.attr)


def run_python(src: np.ndarray, code: int) -> np.ndarray:
    os.environ["BILINEAR_BACKEND"] = "python"
    api = reload_api_modules()
    return api.demosaicing(src, code)


def reload_api_modules():
    import bilinear
    import bilinear._backend
    import bilinear.api

    importlib.reload(bilinear._backend)
    api = importlib.reload(bilinear.api)
    demosaicing_module = importlib.import_module("bilinear.demosaicing")
    importlib.reload(demosaicing_module)
    importlib.reload(bilinear)
    return api
