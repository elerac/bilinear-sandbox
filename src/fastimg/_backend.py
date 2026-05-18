from __future__ import annotations

import importlib
import os
import warnings
from types import ModuleType


_VALID_BACKENDS = {"auto", "native", "python"}
_ENV_NAME = "FASTIMG_BACKEND"

_requested_backend = os.environ.get(_ENV_NAME, "auto").lower()

if _requested_backend not in _VALID_BACKENDS:
    raise RuntimeError(f"{_ENV_NAME} must be one of: {', '.join(sorted(_VALID_BACKENDS))}")

_native_module: ModuleType | None = None
_native_error: Exception | None = None

if _requested_backend != "python":
    try:
        _native_module = importlib.import_module("._native", __package__)
    except Exception as exc:
        _native_error = exc

        if _requested_backend == "native":
            raise RuntimeError(
                f"{_ENV_NAME}=native was requested, but the native extension could not be imported."
            ) from exc

        warnings.warn(
            "fastimg native extension is not available; using the pure Python backend. "
            "The package will work, but demosaicing operations will be slower.",
            RuntimeWarning,
            stacklevel=2,
        )

_BACKEND_NAME = "native" if _native_module is not None else "python"


def backend() -> str:
    """Return the active backend: 'native' or 'python'."""

    return _BACKEND_NAME


def native() -> ModuleType | None:
    """Return the native module, or None when using the Python backend."""

    return _native_module


def native_error() -> Exception | None:
    """Return the exception that prevented native import, if any."""

    return _native_error


__all__ = ["backend", "native", "native_error"]
