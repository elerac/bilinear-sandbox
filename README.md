# bilinear

`bilinear` is a small Python package for bilinear Bayer demosaicing. It mirrors
the simple OpenCV call shape:

```python
from bilinear import demosaicing

bgr = demosaicing(bayer, code)
```

The package keeps one public API and two internal implementations:

- a pure Python reference backend used as the correctness spec
- a native nanobind backend used by default for the hot demosaicing kernel

## Supported API

```python
demosaicing(src, code, fast=True)
```

`src` must be a 2D Bayer image with dtype `numpy.uint8`, `numpy.uint16`,
`numpy.float32`, or `numpy.float64`.
The result is a new C-contiguous BGR image with shape `(height, width, 3)` and
the same dtype as the input. Integer inputs use OpenCV-compatible rounded
averages; floating-point inputs use true real-valued averages with no clamping.

The native backend uses the optimized C++ kernel by default; pass `fast=False`
to use the simpler baseline C++ implementation.

Supported OpenCV-compatible bilinear BGR conversion code values:

| Code | Pattern |
| ---: | :------ |
| `46` | `RGGB` |
| `47` | `GRBG` |
| `48` | `BGGR` |
| `49` | `GBRG` |

Unsupported dimensions, dtypes, and conversion codes raise `ValueError`.

## Backend Selection

Backend selection is controlled by `BILINEAR_BACKEND`:

| Value | Behavior |
| :---- | :------- |
| `native` | Use the nanobind/C++ backend. This is the default. |
| `python` | Use the pure Python reference backend. |
| `auto` | Try native first, then fall back to Python if native is unavailable. |

Examples:

```bash
BILINEAR_BACKEND=native uv run python scripts/benchmark.py
BILINEAR_BACKEND=python uv run pytest
```

The default `native` mode intentionally does not silently fall back. If the
extension is unavailable, set `BILINEAR_BACKEND=python` or build/sync the
project environment.

## Development

This project uses `uv`, `scikit-build-core`, CMake, and nanobind.

Set up the environment and build the editable package:

```bash
uv sync
```

Run tests:

```bash
uv run --locked pytest
BILINEAR_BACKEND=python uv run --locked pytest
BILINEAR_BACKEND=native uv run --locked pytest
BILINEAR_BACKEND=auto uv run --locked pytest
```

Build a local wheel:

```bash
uv build --wheel
```

Run the sample benchmark against OpenCV:

```bash
uv run --locked python scripts/benchmark.py --iterations 10 --warmup 2
```

The benchmark's default `--dtype both` runs `uint8`, `uint16`, `float32`, and
`float64`. It uses `cv2.demosaicing` for integer inputs and a `cv2.filter2D`
bilinear baseline for floating-point inputs, because OpenCV's Bayer demosaicing
accepts only 8-bit and 16-bit integer input.

## Project Layout

```text
src/bilinear/
  api.py          public validation and dispatch
  _backend.py     BILINEAR_BACKEND selection
  _reference.py   pure Python reference implementation
  _native.pyi     internal native stub
  demosaicing.py  compatibility re-export

cpp/
  bindings.cpp
  kernels/demosaicing.cpp
  kernels/demosaicing_fast.cpp
  kernels/demosaicing.hpp
  kernels/demosaicing_fast.hpp

tests/
  test_demosaicing.py
  test_reference_vs_native.py
  test_import_modes.py
```

The Python reference implementation defines correctness. The native backend is
an implementation detail and must produce identical output for the supported
public API.
