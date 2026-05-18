# fastimg

`fastimg` is a small Python package for bilinear Bayer demosaicing. It mirrors
the simple OpenCV call shape:

```python
from fastimg import demosaicing

bgr = demosaicing(bayer, code)
```

The package keeps one public API and two internal implementations:

- a pure Python/NumPy backend used as the correctness spec and fallback
- a native nanobind backend used as an accelerator when available

## Supported API

```python
demosaicing(src, code)
```

`src` must be a 2D Bayer image with dtype `numpy.uint8`, `numpy.uint16`,
`numpy.float32`, or `numpy.float64`.
The result is a new C-contiguous BGR image with shape `(height, width, 3)` and
the same dtype as the input. Integer inputs use OpenCV-compatible rounded
averages; floating-point inputs use true real-valued averages with no clamping.

The native backend uses the optimized C++ kernel when available.

Supported OpenCV-compatible bilinear BGR conversion code values:

| Code | Pattern |
| ---: | :------ |
| `46` | `RGGB` |
| `47` | `GRBG` |
| `48` | `BGGR` |
| `49` | `GBRG` |

Unsupported dimensions, dtypes, and conversion codes raise `ValueError`.

## Backend Selection

Backend selection is controlled by `FASTIMG_BACKEND`:

| Value | Behavior |
| :---- | :------- |
| `auto` | Use native when available, otherwise warn and fall back to Python. This is the default. |
| `native` | Require the nanobind/C++ backend; fail if unavailable. |
| `python` | Use the pure Python/NumPy backend. |

Examples:

```bash
python -c "import fastimg; print(fastimg.backend())"
FASTIMG_BACKEND=native uv run python scripts/benchmark.py
FASTIMG_BACKEND=python uv run pytest
```

Use `FASTIMG_BACKEND=native` in CI or benchmarking when native performance is
required. The default `auto` mode keeps the package importable on platforms
where the extension is unavailable.

## Development

This project uses `uv`, `scikit-build-core`, CMake, and nanobind.

Set up the environment and build the editable package:

```bash
uv sync
```

Run tests:

```bash
uv run --locked pytest
FASTIMG_BACKEND=python uv run --locked pytest
FASTIMG_BACKEND=native uv run --locked pytest
FASTIMG_BACKEND=auto uv run --locked pytest
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
accepts only 8-bit and 16-bit integer input. OpenCV is a development dependency
for tests and scripts, not a runtime dependency of `fastimg`.

## Project Layout

```text
src/fastimg/
  api.py          public validation and dispatch
  _backend.py     FASTIMG_BACKEND selection
  _native.pyi     internal native stub
  _python/        pure Python fallback implementation
  demosaicing.py  compatibility re-export

cpp/
  bindings/_native.cpp
  include/fastimg/demosaicing.hpp
  src/demosaicing.cpp

tests/
  test_demosaicing.py
  test_reference_vs_native.py
  test_import_modes.py
```

The Python reference implementation defines correctness. The native backend is
an implementation detail and must produce identical output for the supported
public API.
