from __future__ import annotations

import os


_BACKEND = os.environ.get("BILINEAR_BACKEND", "native")

if _BACKEND == "python":
    from . import _reference as impl

elif _BACKEND == "native":
    try:
        from . import _native as impl
    except ImportError as e:
        raise ImportError(
            "Native backend is unavailable. "
            "Use BILINEAR_BACKEND=python to run the reference implementation."
        ) from e

elif _BACKEND == "auto":
    try:
        from . import _native as impl
    except ImportError:
        from . import _reference as impl

else:
    raise ValueError("BILINEAR_BACKEND must be 'native', 'python', or 'auto'.")


__all__ = ["impl"]
