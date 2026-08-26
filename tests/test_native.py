"""Regression test for the WinRT / onnxruntime load-order bug.

Importing the Windows Runtime before onnxruntime leaves onnxruntime unable to
initialise, which disabled every Piper voice and reported it as "Piper is not
installed". ``gamevoice.native`` fixes the order; this proves it stays fixed.

Run in a subprocess because import order cannot be undone inside a process that
has already loaded both.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def run_python(source: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.integration
def test_bad_order_still_breaks_without_the_guard():
    """The bug is real. If this stops failing, the guard is no longer needed."""
    result = run_python(
        """
        import winrt.windows.media.ocr  # noqa: F401
        try:
            import onnxruntime  # noqa: F401
        except Exception as exc:
            print("BROKEN", type(exc).__name__)
        else:
            print("FINE")
        """
    )
    assert result.returncode == 0, result.stderr
    if "FINE" in result.stdout:
        pytest.skip("environment no longer exhibits the load-order conflict")
    assert "BROKEN" in result.stdout


@pytest.mark.integration
def test_importing_ocr_first_leaves_piper_usable():
    """The order the app actually uses: OCR module first, voices afterwards."""
    result = run_python(
        """
        import gamevoice.ocr  # noqa: F401  - imports WinRT internally
        from piper import PiperVoice  # noqa: F401
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "OK" in result.stdout


@pytest.mark.integration
def test_engine_reports_piper_as_available():
    result = run_python(
        """
        import gamevoice.ocr  # noqa: F401
        from gamevoice.native import onnx_available
        print("AVAILABLE" if onnx_available() else "MISSING")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "AVAILABLE" in result.stdout
