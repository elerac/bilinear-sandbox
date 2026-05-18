# Modern repository guide: nanobind accelerator with pure Python fallback

This is a guide for structuring a new Python package that uses nanobind to optionally accelerate processing routines with native C++ code. This guide uses `fastimg` as a placeholder name for the package, but the same structure applies to any package with a similar design. Adjust the names and details as needed for your project.

## 1. Core design

Use **one public Python package** with two private backends:

```text
public API
  ↓
backend selector
  ↓
native backend if available
  ↓
pure Python backend otherwise
```

The native backend is an accelerator, not a separate package. The pure Python backend is part of the installed package, so it can serve three roles:

1. runtime fallback when the native extension is unavailable;
2. correctness reference for tests;
3. debug/reference backend during development.

Nanobind’s packaging guide uses the pattern of a Python package importing a private compiled extension installed inside the package directory, and it recommends `scikit-build-core` for the CMake-based build path. `scikit-build-core` also supports `src/<package_name>` package discovery and can retry a failed native build as a pure Python build using an `if.failed` override. ([Nanobind][1])

The final runtime behavior should be:

```text
FASTIMG_BACKEND=auto      # default: use native if available, otherwise Python
FASTIMG_BACKEND=native    # require native; fail if unavailable
FASTIMG_BACKEND=python    # force pure Python backend
```

This gives you a safe default while still letting users or CI require native performance explicitly.

---

## 2. Repository layout

```text
fastimg/
├── pyproject.toml
├── CMakeLists.txt
├── uv.lock
├── .python-version
├── README.md
├── LICENSE
├── .gitignore
│
├── src/
│   └── fastimg/
│       ├── __init__.py
│       ├── py.typed
│       ├── _backend.py
│       ├── _native.pyi
│       │
│       ├── filters.py
│       ├── transforms.py
│       ├── color.py
│       │
│       └── _python/
│           ├── __init__.py
│           ├── filters.py
│           ├── transforms.py
│           └── color.py
│
├── cpp/
│   ├── include/
│   │   └── fastimg/
│   │       ├── common.hpp
│   │       ├── filters.hpp
│   │       ├── transforms.hpp
│   │       └── color.hpp
│   │
│   ├── src/
│   │   ├── common.cpp
│   │   ├── filters.cpp
│   │   ├── transforms.cpp
│   │   └── color.cpp
│   │
│   └── bindings/
│       └── _native.cpp
│
├── tests/
│   ├── conftest.py
│   ├── test_backend_selection.py
│   ├── test_filters_parity.py
│   ├── test_transforms_parity.py
│   ├── test_color_parity.py
│   └── test_api_errors.py
│
├── benchmarks/
│   ├── bench_filters.py
│   └── bench_transforms.py
│
└── .github/
    └── workflows/
        ├── ci.yml
        └── release.yml
```

The key design rule is:

```text
src/fastimg/*.py
    public API and input validation

src/fastimg/_python/*.py
    pure Python backend, installed with the package

src/fastimg/_native.*
    private compiled extension

cpp/src/*.cpp
    pure C++ kernels

cpp/bindings/_native.cpp
    nanobind glue only
```

Do not put the Python reference implementation under `tests/reference/` if you want runtime fallback. Put it under `src/fastimg/_python/` so it is included in wheels and sdists.

---

## 3. `pyproject.toml`

Use `scikit-build-core` as the build backend. `uv` still manages your virtual environment, lockfile, dependencies, commands, builds, and publishing; the build backend is what knows how to drive CMake and nanobind. `uv` project mode uses `pyproject.toml`, creates a `.venv`, maintains `uv.lock`, runs commands through the locked environment, and can build source and wheel distributions with `uv build`. ([Astral Docs][2])

```toml
[build-system]
requires = [
  "scikit-build-core>=0.10",
  "nanobind>=1.3.2"
]
build-backend = "scikit_build_core.build"

[project]
name = "fastimg"
version = "0.1.0"
description = "Fast image processing routines with optional nanobind accelerators"
readme = "README.md"
requires-python = ">=3.11"
license = "BSD-3-Clause"
license-files = ["LICENSE"]
authors = [
  { name = "Your Name", email = "you@example.com" }
]
dependencies = [
  "numpy>=2.0"
]

[dependency-groups]
dev = [
  "hypothesis>=6",
  "mypy>=1.8",
  "pytest>=8",
  "pytest-cov>=5",
  "ruff>=0.8"
]
bench = [
  "pytest-benchmark>=4"
]

[tool.scikit-build]
minimum-version = "build-system.requires"
wheel.packages = ["src/fastimg"]
build-dir = "build/{wheel_tag}"

sdist.include = [
  "CMakeLists.txt",
  "cpp/**"
]

[tool.scikit-build.messages]
after-success = "fastimg wheel built successfully."

[[tool.scikit-build.overrides]]
if.failed = true
wheel.cmake = false
wheel.py-api = "py3"
messages.after-success = """
Native extension build failed, so fastimg was built as a pure Python wheel.
The package will work, but image operations will be slower.
"""

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 240
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true
files = ["src/fastimg"]

[tool.cibuildwheel]
build = "cp311-* cp312-* cp313-* cp314-*"
archs = ["auto64"]
build-frontend = "uv"
build-verbosity = 1
test-requires = [
  "pytest",
  "numpy",
  "hypothesis"
]
test-command = "python -c \"import os; os.environ['FASTIMG_BACKEND']='native'; import fastimg; assert fastimg.backend() == 'native'\" && python -m pytest {project}/tests -q"
```

A few details matter here.

`scikit-build-core>=0.10` is important because the `if.failed` override was added for cases like pure Python fallback after native build failure. `wheel.cmake = false` disables the CMake build, and CMake-less builds target `purelib` by default. ([scikit-build-core][3])

`license = "BSD-3-Clause"` and `license-files = ["LICENSE"]` follow the modern Python packaging metadata format, where `license` is an SPDX expression and `license-files` lists files to include in distributions. ([Python Packaging][4])

The `cibuildwheel` test command intentionally requires the native backend. This prevents your release workflow from accidentally accepting a pure Python fallback wheel when you intended to build a native wheel. `cibuildwheel` supports `uv` as a build frontend, though its docs note that GraalPy currently has a compatibility issue with `uv`; the config above only targets CPython wheels. ([cibuildwheel][5])

Python `>=3.11` is a reasonable current floor for a new native-extension project. As of May 2026, Python 3.10 is security-only and reaches end of life in October 2026, while Python 3.11+ remains supported. ([Python Developer's Guide][6])

---

## 4. `CMakeLists.txt`

Keep the CMake file focused on the native extension.

```cmake
cmake_minimum_required(VERSION 3.18...3.30)

project(fastimg LANGUAGES CXX)

if (NOT SKBUILD)
  message(WARNING "This project is intended to be built through scikit-build-core.")
endif()

find_package(
  Python 3.11
  REQUIRED
  COMPONENTS Interpreter Development.Module
)

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  _native
  cpp/bindings/_native.cpp
  cpp/src/common.cpp
  cpp/src/filters.cpp
  cpp/src/transforms.cpp
  cpp/src/color.cpp
)

target_include_directories(_native PRIVATE cpp/include)
target_compile_features(_native PRIVATE cxx_std_17)

install(TARGETS _native LIBRARY DESTINATION fastimg)
```

Nanobind’s CMake API exists to handle Python extension-module details across platforms, including extension suffixes, compiler/linker flags, and linking to nanobind internals. ([Nanobind][7])

For a first release, I would avoid splitting into many native modules. Start with one private extension:

```text
fastimg._native
```

Then expose everything through public Python wrappers.

---

## 5. Backend selector

Create `src/fastimg/_backend.py`.

```python
# src/fastimg/_backend.py

from __future__ import annotations

import importlib
import os
import warnings
from types import ModuleType

_VALID_BACKENDS = {"auto", "native", "python"}
_ENV_NAME = "FASTIMG_BACKEND"

_requested_backend = os.environ.get(_ENV_NAME, "auto").lower()

if _requested_backend not in _VALID_BACKENDS:
    raise RuntimeError(
        f"{_ENV_NAME} must be one of: "
        + ", ".join(sorted(_VALID_BACKENDS))
    )

_native_module: ModuleType | None = None
_native_error: Exception | None = None

if _requested_backend != "python":
    try:
        _native_module = importlib.import_module("._native", __package__)
    except Exception as exc:
        _native_error = exc

        if _requested_backend == "native":
            raise RuntimeError(
                f"{_ENV_NAME}=native was requested, but the native extension "
                "could not be imported."
            ) from exc

        warnings.warn(
            "fastimg native extension is not available; using the pure Python "
            "backend. The package will work, but image operations will be slower.",
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
```

This module reads `FASTIMG_BACKEND` once, at import time. Users who want to force a backend should set the environment variable before importing `fastimg`.

---

## 6. Public package API

Create `src/fastimg/__init__.py`.

```python
# src/fastimg/__init__.py

from __future__ import annotations

from ._backend import backend, native_error
from .filters import box_blur

__all__ = [
    "backend",
    "native_error",
    "box_blur",
]
```

The user-facing API is always `fastimg.box_blur`, never `fastimg._native.box_blur_u8_3d_into`.

---

## 7. Public wrapper around both backends

Create `src/fastimg/filters.py`.

```python
# src/fastimg/filters.py

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._backend import native
from ._python.filters import box_blur as _box_blur_python


def box_blur(image: ArrayLike, radius: int) -> NDArray[np.uint8]:
    """Apply a box blur to a uint8 grayscale or channel-last image.

    Parameters
    ----------
    image:
        2D grayscale image or 3D channel-last image.
    radius:
        Non-negative blur radius.

    Returns
    -------
    numpy.ndarray
        A new uint8 array with the same shape as the input.
    """
    arr = np.asarray(image)

    if arr.dtype != np.uint8:
        raise TypeError("box_blur expects uint8 input")

    if arr.ndim not in (2, 3):
        raise ValueError("box_blur expects a 2D or 3D channel-last image")

    if radius < 0:
        raise ValueError("radius must be non-negative")

    # Define this policy explicitly. The native backend accepts only
    # contiguous arrays; the public API accepts non-contiguous views.
    src = np.ascontiguousarray(arr)

    mod = native()

    if mod is None:
        return _box_blur_python(src, int(radius))

    dst = np.empty_like(src)

    if src.ndim == 2:
        mod.box_blur_u8_2d_into(src, dst, int(radius))
    else:
        mod.box_blur_u8_3d_into(src, dst, int(radius))

    return dst
```

The public wrapper owns all Python-facing policy:

```text
- accepted dtype
- accepted dimensionality
- non-contiguous input behavior
- output allocation
- error messages
- backend dispatch
```

The C++ function should not be responsible for converting arbitrary Python inputs into valid image arrays.

---

## 8. Pure Python backend

Create `src/fastimg/_python/filters.py`.

```python
# src/fastimg/_python/filters.py

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def box_blur(image: NDArray[np.uint8], radius: int) -> NDArray[np.uint8]:
    """Pure Python/NumPy reference implementation of box_blur.

    This implementation is intentionally simple. It is used both as the
    runtime fallback and as the correctness reference for native tests.
    """
    if image.dtype != np.uint8:
        raise TypeError("box_blur expects uint8 input")

    if image.ndim not in (2, 3):
        raise ValueError("box_blur expects a 2D or 3D channel-last image")

    if radius < 0:
        raise ValueError("radius must be non-negative")

    if radius == 0:
        return image.copy()

    height, width = image.shape[:2]
    out = np.empty_like(image)

    if image.ndim == 2:
        for y in range(height):
            y0 = max(0, y - radius)
            y1 = min(height, y + radius + 1)

            for x in range(width):
                x0 = max(0, x - radius)
                x1 = min(width, x + radius + 1)

                region = image[y0:y1, x0:x1]
                value = region.sum(dtype=np.uint64) // region.size
                out[y, x] = np.uint8(value)

        return out

    channels = image.shape[2]

    for y in range(height):
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)

        for x in range(width):
            x0 = max(0, x - radius)
            x1 = min(width, x + radius + 1)

            region = image[y0:y1, x0:x1, :]
            count = region.shape[0] * region.shape[1]
            sums = region.reshape(-1, channels).sum(axis=0, dtype=np.uint64)
            out[y, x, :] = (sums // count).astype(np.uint8)

    return out
```

Keep this implementation boring and obvious. Do not over-optimize the fallback unless you want the pure Python backend to become a serious second implementation. The main purpose is correctness, portability, and resilience.

---

## 9. Nanobind binding layer

Create `cpp/bindings/_native.cpp`.

```cpp
// cpp/bindings/_native.cpp

#include <cstdint>
#include <limits>
#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "fastimg/filters.hpp"

namespace nb = nanobind;

using U8Image2D = nb::ndarray<
    const uint8_t,
    nb::ndim<2>,
    nb::c_contig,
    nb::device::cpu
>;

using U8Out2D = nb::ndarray<
    uint8_t,
    nb::ndim<2>,
    nb::c_contig,
    nb::device::cpu
>;

using U8Image3D = nb::ndarray<
    const uint8_t,
    nb::ndim<3>,
    nb::c_contig,
    nb::device::cpu
>;

using U8Out3D = nb::ndarray<
    uint8_t,
    nb::ndim<3>,
    nb::c_contig,
    nb::device::cpu
>;


static int checked_dim(size_t value, const char* name) {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " is too large");
    }
    return static_cast<int>(value);
}


void box_blur_u8_2d_into(U8Image2D src, U8Out2D dst, int radius) {
    if (radius < 0) {
        throw std::runtime_error("radius must be non-negative");
    }

    if (src.shape(0) != dst.shape(0) || src.shape(1) != dst.shape(1)) {
        throw std::runtime_error("src and dst must have identical shape");
    }

    const int height = checked_dim(src.shape(0), "height");
    const int width = checked_dim(src.shape(1), "width");

    const uint8_t* src_ptr = src.data();
    uint8_t* dst_ptr = dst.data();

    {
        nb::gil_scoped_release release;
        fastimg::box_blur_u8_2d_into(src_ptr, dst_ptr, height, width, radius);
    }
}


void box_blur_u8_3d_into(U8Image3D src, U8Out3D dst, int radius) {
    if (radius < 0) {
        throw std::runtime_error("radius must be non-negative");
    }

    if (
        src.shape(0) != dst.shape(0) ||
        src.shape(1) != dst.shape(1) ||
        src.shape(2) != dst.shape(2)
    ) {
        throw std::runtime_error("src and dst must have identical shape");
    }

    const int height = checked_dim(src.shape(0), "height");
    const int width = checked_dim(src.shape(1), "width");
    const int channels = checked_dim(src.shape(2), "channels");

    const uint8_t* src_ptr = src.data();
    uint8_t* dst_ptr = dst.data();

    {
        nb::gil_scoped_release release;
        fastimg::box_blur_u8_3d_into(
            src_ptr,
            dst_ptr,
            height,
            width,
            channels,
            radius
        );
    }
}


NB_MODULE(_native, m) {
    m.doc() = "Private native kernels for fastimg";

    m.def(
        "box_blur_u8_2d_into",
        &box_blur_u8_2d_into,
        nb::arg("src").noconvert(),
        nb::arg("dst").noconvert(),
        nb::arg("radius")
    );

    m.def(
        "box_blur_u8_3d_into",
        &box_blur_u8_3d_into,
        nb::arg("src").noconvert(),
        nb::arg("dst").noconvert(),
        nb::arg("radius")
    );
}
```

Nanobind’s `nb::ndarray` lets you constrain dtype, dimensionality, device, and memory layout. That is useful for image processing because it lets the Python wrapper normalize layout before the native function receives the array. Nanobind also documents `.noconvert()` for disabling implicit conversions and `gil_scoped_release` for releasing the interpreter lock during expensive native work. ([Nanobind][8])

Notice the GIL pattern:

```text
1. inspect shapes while holding the GIL
2. extract raw pointers and dimensions
3. release the GIL
4. call pure C++ kernel
5. reacquire automatically when leaving scope
```

That keeps Python/nanobind object handling outside the released-GIL region.

---

## 10. Pure C++ kernel

Create `cpp/include/fastimg/filters.hpp`.

```cpp
// cpp/include/fastimg/filters.hpp

#pragma once

#include <cstdint>

namespace fastimg {

void box_blur_u8_2d_into(
    const uint8_t* src,
    uint8_t* dst,
    int height,
    int width,
    int radius
);

void box_blur_u8_3d_into(
    const uint8_t* src,
    uint8_t* dst,
    int height,
    int width,
    int channels,
    int radius
);

}  // namespace fastimg
```

Create `cpp/src/filters.cpp`.

```cpp
// cpp/src/filters.cpp

#include "fastimg/filters.hpp"

#include <algorithm>
#include <cstdint>

namespace fastimg {

void box_blur_u8_2d_into(
    const uint8_t* src,
    uint8_t* dst,
    int height,
    int width,
    int radius
) {
    for (int y = 0; y < height; ++y) {
        const int y0 = std::max(0, y - radius);
        const int y1 = std::min(height, y + radius + 1);

        for (int x = 0; x < width; ++x) {
            const int x0 = std::max(0, x - radius);
            const int x1 = std::min(width, x + radius + 1);

            uint64_t sum = 0;
            uint64_t count = 0;

            for (int yy = y0; yy < y1; ++yy) {
                for (int xx = x0; xx < x1; ++xx) {
                    sum += src[yy * width + xx];
                    ++count;
                }
            }

            dst[y * width + x] = static_cast<uint8_t>(sum / count);
        }
    }
}


void box_blur_u8_3d_into(
    const uint8_t* src,
    uint8_t* dst,
    int height,
    int width,
    int channels,
    int radius
) {
    for (int y = 0; y < height; ++y) {
        const int y0 = std::max(0, y - radius);
        const int y1 = std::min(height, y + radius + 1);

        for (int x = 0; x < width; ++x) {
            const int x0 = std::max(0, x - radius);
            const int x1 = std::min(width, x + radius + 1);

            for (int c = 0; c < channels; ++c) {
                uint64_t sum = 0;
                uint64_t count = 0;

                for (int yy = y0; yy < y1; ++yy) {
                    for (int xx = x0; xx < x1; ++xx) {
                        const int idx = (yy * width + xx) * channels + c;
                        sum += src[idx];
                        ++count;
                    }
                }

                const int out_idx = (y * width + x) * channels + c;
                dst[out_idx] = static_cast<uint8_t>(sum / count);
            }
        }
    }
}

}  // namespace fastimg
```

This kernel is intentionally simple. Once the package structure and tests are stable, you can replace this with separable filters, integral images, SIMD, tiling, OpenMP, or specialized dtype dispatch without changing the public API.

---

## 11. Optional stub file

Create `src/fastimg/_native.pyi`.

```python
# src/fastimg/_native.pyi

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def box_blur_u8_2d_into(
    src: NDArray[np.uint8],
    dst: NDArray[np.uint8],
    radius: int,
) -> None: ...


def box_blur_u8_3d_into(
    src: NDArray[np.uint8],
    dst: NDArray[np.uint8],
    radius: int,
) -> None: ...
```

Also add:

```text
src/fastimg/py.typed
```

The file can be empty. It tells type checkers that your package is typed.

---

## 12. Tests

### Backend selection tests

Create `tests/test_backend_selection.py`.

```python
# tests/test_backend_selection.py

from __future__ import annotations

import os
import subprocess
import sys

import fastimg


def test_backend_name_is_valid() -> None:
    assert fastimg.backend() in {"native", "python"}


def test_can_force_python_backend() -> None:
    env = os.environ.copy()
    env["FASTIMG_BACKEND"] = "python"

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import fastimg; print(fastimg.backend())",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert proc.stdout.strip() == "python"


def test_invalid_backend_fails() -> None:
    env = os.environ.copy()
    env["FASTIMG_BACKEND"] = "bad-backend"

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import fastimg",
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
```

### Parity tests

Create `tests/test_filters_parity.py`.

```python
# tests/test_filters_parity.py

from __future__ import annotations

import numpy as np
import pytest

import fastimg
from fastimg._python.filters import box_blur as box_blur_python


@pytest.mark.parametrize(
    "shape",
    [
        (0, 0),
        (1, 1),
        (1, 8),
        (8, 1),
        (8, 8),
        (16, 17, 1),
        (16, 17, 3),
        (16, 17, 4),
    ],
)
@pytest.mark.parametrize("radius", [0, 1, 2, 5])
def test_box_blur_matches_python_backend(shape: tuple[int, ...], radius: int) -> None:
    rng = np.random.default_rng(123)
    image = rng.integers(0, 256, size=shape, dtype=np.uint8)

    actual = fastimg.box_blur(image, radius)
    expected = box_blur_python(np.ascontiguousarray(image), radius)

    np.testing.assert_array_equal(actual, expected, strict=True)


def test_box_blur_accepts_non_contiguous_input() -> None:
    rng = np.random.default_rng(123)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

    view = image[:, ::2, :]

    actual = fastimg.box_blur(view, 2)
    expected = box_blur_python(np.ascontiguousarray(view), 2)

    np.testing.assert_array_equal(actual, expected, strict=True)


def test_box_blur_rejects_wrong_dtype() -> None:
    image = np.zeros((8, 8, 3), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        fastimg.box_blur(image, 1)


def test_box_blur_rejects_negative_radius() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="non-negative"):
        fastimg.box_blur(image, -1)
```

Run the same parity tests under both backends:

```bash
FASTIMG_BACKEND=python uv run pytest -q
FASTIMG_BACKEND=native uv run pytest -q
```

The Python run proves the fallback backend and public API work. The native run proves that the C++ implementation matches the Python backend.

---

## 13. Local development with `uv`

Initialize and manage the project with `uv`:

```bash
uv init fastimg
cd fastimg

uv python pin 3.12

uv add "numpy>=2.0"
uv add --dev pytest hypothesis pytest-cov ruff mypy
uv add --group bench pytest-benchmark

uv sync
```

Then run normal checks:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src/fastimg
```

Build a source distribution and wheel:

```bash
uv build --no-sources
```

`uv build --no-sources` is useful before publishing because it verifies that the package builds without relying on local source overrides from `tool.uv.sources`. ([Astral Docs][9])

To explicitly build a pure Python wheel:

```bash
uv build --wheel --no-sources -Cwheel.cmake=false -Cwheel.py-api=py3
```

`uv` supports passing PEP 517 build backend config settings with `--config-setting`, `--config-settings`, or `-C`. ([Astral Docs][10])

For faster native iteration, use an editable install with rebuild behavior:

```bash
uv pip install --no-build-isolation -Ceditable.rebuild=true -ve .
uv run pytest tests/test_filters_parity.py -q
```

Nanobind’s packaging guide documents editable installs with `-Ceditable.rebuild=true` for iterative extension development. ([Nanobind][1])

---

## 14. Backend policy for users

Document this clearly in `README.md`.

```python
import fastimg

print(fastimg.backend())
```

Expected values:

```text
native
python
```

Backend selection:

```bash
# Default: native if available, otherwise Python
python -c "import fastimg; print(fastimg.backend())"

# Require native backend
FASTIMG_BACKEND=native python -c "import fastimg; print(fastimg.backend())"

# Force pure Python backend
FASTIMG_BACKEND=python python -c "import fastimg; print(fastimg.backend())"
```

On Windows PowerShell:

```powershell
$env:FASTIMG_BACKEND = "native"
python -c "import fastimg; print(fastimg.backend())"
```

The rule should be:

```text
auto:
    fallback is allowed, but warn

native:
    fallback is not allowed

python:
    native import is skipped
```

Do not fallback from native execution errors. For example, if `_native` imports successfully but `box_blur_u8_3d_into` raises because of a shape mismatch, let the exception propagate. Fallback should handle missing native availability, not hide native correctness bugs.

---

## 15. CI workflow

Create `.github/workflows/ci.yml`.

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  native-tests:
    name: Native / ${{ matrix.os }} / Python ${{ matrix.python }}
    runs-on: ${{ matrix.os }}

    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ["3.11", "3.12", "3.13", "3.14"]

    steps:
      - uses: actions/checkout@v6

      - uses: actions/setup-python@v6
        with:
          python-version: ${{ matrix.python }}

      - uses: astral-sh/setup-uv@v8
        with:
          enable-cache: true

      - run: uv sync --frozen --group dev

      - run: uv run ruff check .

      - run: uv run mypy src/fastimg

      - run: uv run pytest -q
        env:
          FASTIMG_BACKEND: native

  python-fallback-tests:
    name: Pure Python fallback
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v6

      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"

      - uses: astral-sh/setup-uv@v8
        with:
          enable-cache: true

      - name: Install dependencies without installing project
        run: uv sync --frozen --group dev --no-install-project

      - name: Build pure Python wheel
        run: uv build --wheel --no-sources -Cwheel.cmake=false -Cwheel.py-api=py3

      - name: Install pure Python wheel
        run: uv pip install --reinstall dist/*.whl

      - name: Test pure Python backend
        run: uv run pytest -q
        env:
          FASTIMG_BACKEND: python
```

The `uv` GitHub Actions guide recommends `astral-sh/setup-uv`; it can install `uv`, add it to `PATH`, and cache it. The same guide shows current `actions/checkout@v6` and `actions/setup-python@v6` usage. ([Astral Docs][11])

The pure Python job builds the pure wheel explicitly. That catches packaging mistakes such as forgetting to include `src/fastimg/_python/`.

---

## 16. Release workflow

Create `.github/workflows/release.yml`.

```yaml
name: Release

on:
  push:
    tags:
      - "v*"

jobs:
  sdist:
    name: Build sdist
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"

      - uses: astral-sh/setup-uv@v8
        with:
          enable-cache: true

      - run: uv build --sdist --no-sources

      - uses: actions/upload-artifact@v6
        with:
          name: sdist
          path: dist/*.tar.gz

  pure-wheel:
    name: Build pure Python wheel
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v6

      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"

      - uses: astral-sh/setup-uv@v8
        with:
          enable-cache: true

      - run: uv build --wheel --no-sources -Cwheel.cmake=false -Cwheel.py-api=py3

      - name: Test pure Python wheel
        run: |
          uv venv .wheel-test
          uv pip install --python .wheel-test/bin/python dist/*.whl
          .wheel-test/bin/python -c "import fastimg; assert fastimg.backend() == 'python'"

      - uses: actions/upload-artifact@v6
        with:
          name: pure-wheel
          path: dist/*.whl

  native-wheels:
    name: Build native wheels / ${{ matrix.os }}
    runs-on: ${{ matrix.os }}

    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]

    steps:
      - uses: actions/checkout@v6

      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"

      - name: Install cibuildwheel
        run: python -m pip install cibuildwheel

      - name: Build wheels
        run: python -m cibuildwheel --output-dir wheelhouse

      - uses: actions/upload-artifact@v6
        with:
          name: native-wheels-${{ matrix.os }}
          path: wheelhouse/*.whl

  publish:
    name: Publish to PyPI
    needs: [sdist, pure-wheel, native-wheels]
    runs-on: ubuntu-latest
    environment: pypi

    permissions:
      contents: read
      id-token: write

    steps:
      - uses: actions/download-artifact@v6
        with:
          path: dist
          merge-multiple: true

      - uses: pypa/gh-action-pypi-publish@release/v1
```

Use PyPI Trusted Publishing rather than storing a PyPI API token in GitHub Secrets. PyPI’s documentation recommends the PyPA publish action and requires `id-token: write` for Trusted Publishing. ([PyPI Docs][12])

This release workflow publishes:

```text
fastimg-0.1.0.tar.gz
fastimg-0.1.0-py3-none-any.whl
fastimg-0.1.0-cp311-...manylinux....whl
fastimg-0.1.0-cp312-...manylinux....whl
fastimg-0.1.0-cp313-...manylinux....whl
fastimg-0.1.0-cp314-...manylinux....whl
fastimg-0.1.0-cp311-...macosx....whl
fastimg-0.1.0-cp311-...win_amd64.whl
...
```

The pure Python wheel gives unsupported platforms a working fallback. The native wheels give supported platforms speed.

---

## 17. Optional stable ABI mode

If you can require Python `>=3.12`, consider using Python’s stable ABI to reduce the wheel matrix.

In `pyproject.toml`:

```toml
[project]
requires-python = ">=3.12"

[tool.scikit-build]
wheel.py-api = "cp312"
```

In `CMakeLists.txt`:

```cmake
nanobind_add_module(
  _native
  STABLE_ABI
  cpp/bindings/_native.cpp
  cpp/src/common.cpp
  cpp/src/filters.cpp
  cpp/src/transforms.cpp
  cpp/src/color.cpp
)
```

Nanobind’s packaging docs show `wheel.py-api = "cp312"` together with `STABLE_ABI` for CPython 3.12+ stable ABI wheels. ([Nanobind][1])

For your first release, I would start without stable ABI unless reducing the wheel matrix is more important than keeping the build path simple.

---

## 18. Benchmarks

Keep benchmarks separate from tests.

```python
# benchmarks/bench_filters.py

from __future__ import annotations

import numpy as np

import fastimg
from fastimg._python.filters import box_blur as box_blur_python


def test_box_blur_native_benchmark(benchmark) -> None:
    image = np.random.default_rng(123).integers(
        0,
        256,
        size=(1024, 1024, 3),
        dtype=np.uint8,
    )

    benchmark(fastimg.box_blur, image, 3)


def test_box_blur_python_benchmark(benchmark) -> None:
    image = np.random.default_rng(123).integers(
        0,
        256,
        size=(128, 128, 3),
        dtype=np.uint8,
    )

    benchmark(box_blur_python, image, 3)
```

Run:

```bash
FASTIMG_BACKEND=native uv run pytest benchmarks --benchmark-only
FASTIMG_BACKEND=python uv run pytest benchmarks --benchmark-only
```

Avoid hard performance assertions in normal CI at first. GitHub-hosted runners are noisy, and benchmark regressions need a more controlled setup.

---

## 19. `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# uv
.venv/

# build outputs
build/
dist/
wheelhouse/
*.egg-info/

# native extension artifacts
*.so
*.pyd
*.dll
*.dylib

# OS/editor
.DS_Store
.vscode/
.idea/
```

Commit:

```text
pyproject.toml
uv.lock
.python-version
CMakeLists.txt
README.md
LICENSE
src/
cpp/
tests/
benchmarks/
.github/workflows/
```

Do not commit:

```text
.venv/
build/
dist/
wheelhouse/
compiled extension binaries
```

`uv.lock` should be checked in for reproducible development and CI. `uv` documents `uv.lock` as a cross-platform lockfile that should be committed to version control. ([Astral Docs][2])

---

## 20. README checklist

Your `README.md` should include:

```text
- What the package does
- Installation instructions
- Backend behavior
- How to force native or Python backend
- Supported Python versions
- Supported image dtypes/layouts
- Performance note
- Small examples
- How to report native build failures
```

Example backend section:

````markdown
## Backends

fastimg uses a native nanobind backend when available. If the native
extension is unavailable, it falls back to a pure Python backend.

```bash
python -c "import fastimg; print(fastimg.backend())"
````

Force native mode:

```bash
FASTIMG_BACKEND=native python -c "import fastimg"
```

Force Python fallback mode:

```bash
FASTIMG_BACKEND=python python -c "import fastimg"
```

The pure Python backend is correct but slower.

````

---

## 21. Best-practice rules for this architecture

Use these rules consistently as the project grows.

```text
1. Public functions live in src/fastimg/*.py.

2. Pure Python fallback lives in src/fastimg/_python/*.py.

3. Native extension is private: src/fastimg/_native.*.

4. Native code writes into Python-allocated output arrays.

5. Python wrappers validate dtype, shape, layout, and arguments.

6. Native bindings validate only low-level invariants.

7. C++ kernels know nothing about Python or nanobind.

8. Tests compare public API output against _python backend output.

9. CI runs both FASTIMG_BACKEND=native and FASTIMG_BACKEND=python.

10. Release CI publishes native wheels plus one py3-none-any fallback wheel.
````

The most important practical point is this:

```text
Fallback is for availability.
Tests are for correctness.
Native mode is for performance.
```

Do not let fallback hide native correctness bugs. If the native module imports but gives wrong output, the parity tests should catch it. If the native module cannot be imported, fallback is acceptable in `auto` mode but not in `native` mode.

---

## 22. Final architecture

Your package should end up working like this:

```text
fastimg.box_blur(image, radius)
    ↓
src/fastimg/filters.py
    - validate dtype
    - validate dimensions
    - normalize memory layout
    - allocate output if native
    - dispatch backend
    ↓
src/fastimg/_backend.py
    - select native or Python backend
    ↓
src/fastimg/_native.* or src/fastimg/_python/filters.py
    ↓
cpp/bindings/_native.cpp
    - nanobind ndarray constraints
    - shape checks
    - GIL release
    ↓
cpp/src/filters.cpp
    - pure C++ kernel
```

This structure gives you:

```text
- fast native execution on supported platforms
- working pure Python fallback on unsupported platforms
- one clean public API
- a correctness oracle for native tests
- reproducible uv-managed development
- robust GitHub CI
- PyPI release path with native wheels and fallback wheel
```

[1]: https://nanobind.readthedocs.io/en/latest/packaging.html "Packaging - nanobind documentation"
[2]: https://docs.astral.sh/uv/guides/projects/ "Working on projects | uv"
[3]: https://scikit-build-core.readthedocs.io/en/latest/configuration/overrides.html "Overrides - scikit-build-core 0.12.3.dev25 documentation"
[4]: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/ "Writing your pyproject.toml - Python Packaging User Guide"
[5]: https://cibuildwheel.pypa.io/en/stable/options/ "Options - cibuildwheel"
[6]: https://devguide.python.org/versions/ "Status of Python versions"
[7]: https://nanobind.readthedocs.io/en/latest/api_cmake.html?utm_source=chatgpt.com "CMake API Reference - nanobind documentation"
[8]: https://nanobind.readthedocs.io/en/latest/ndarray.html?utm_source=chatgpt.com "The nb::ndarray<..> class - nanobind documentation"
[9]: https://docs.astral.sh/uv/guides/package/ "Building and publishing a package | uv"
[10]: https://docs.astral.sh/uv/reference/cli/ "Commands | uv"
[11]: https://docs.astral.sh/uv/guides/integration/github/ "Using uv in GitHub Actions | uv"
[12]: https://docs.pypi.org/trusted-publishers/using-a-publisher/ "Publishing with a Trusted Publisher - PyPI Docs"
