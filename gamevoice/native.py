"""Native library load order.

Measured on Windows 11 with winrt-runtime 3.2.1 and onnxruntime 1.29.0:

    import winrt.windows.media.ocr   # fine
    import onnxruntime               # ImportError: DLL initialization routine failed

    import onnxruntime               # fine
    import winrt.windows.media.ocr   # fine

Loading the Windows Runtime first leaves onnxruntime unable to initialise its
own DLL, so the failure looks like "Piper is not installed" long after the real
cause. The reverse order is stable.

GameVoice needs both - the recogniser is WinRT, the voices are onnxruntime - so
this module makes the working order explicit and idempotent, and ``ocr`` calls
it before it touches WinRT. That puts the guarantee in the one place that could
otherwise poison the process.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_lock = threading.Lock()
_done = False
_onnx_available: bool | None = None


def ensure_load_order() -> bool:
    """Import onnxruntime before anything imports WinRT. Safe to call often.

    Returns True if onnxruntime is present. A False result is not fatal: the
    app falls back to the SAPI voices, which do not use onnxruntime at all.
    """
    global _done, _onnx_available

    if _done:
        return bool(_onnx_available)

    with _lock:
        if _done:
            return bool(_onnx_available)
        try:
            import onnxruntime  # noqa: F401

            _onnx_available = True
            log.debug("onnxruntime preloaded")
        except ImportError as exc:
            _onnx_available = False
            log.warning("onnxruntime unavailable, Piper voices disabled: %s", exc)
        except Exception as exc:
            _onnx_available = False
            log.error("onnxruntime failed to initialise: %s", exc)
        _done = True

    return bool(_onnx_available)


def onnx_available() -> bool:
    return ensure_load_order()
