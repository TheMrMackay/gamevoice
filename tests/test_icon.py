"""The application icon.

Small sizes carry deliberately simplified artwork, so the test that matters is
that the .ico actually contains those renders rather than downscales of the
256 - that regression is invisible until someone squints at a tray icon.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from gamevoice.config import ASSETS_DIR

ICON = ASSETS_DIR / "icon.ico"
ACTIVE_ICON = ASSETS_DIR / "icon-active.ico"
EXPECTED_SIZES = {16, 20, 24, 32, 40, 48, 64, 128, 256}


@pytest.fixture(scope="module")
def icon():
    if not ICON.is_file():
        pytest.skip("run tools/make_icon.py to build the icon")
    return Image.open(ICON)


class TestIconFile:
    def test_both_colourways_exist(self):
        assert ICON.is_file(), "idle icon missing"
        assert ACTIVE_ICON.is_file(), "listening icon missing"

    def test_every_size_is_present(self, icon):
        assert {w for w, _ in icon.ico.sizes()} >= EXPECTED_SIZES

    def test_small_sizes_are_distinct_artwork(self, icon):
        """Not just the 256 shrunk - the small renders use fewer, thicker bars."""
        downscaled = icon.ico.getimage((256, 256)).convert("RGBA").resize(
            (16, 16), Image.LANCZOS
        )
        embedded = icon.ico.getimage((16, 16)).convert("RGBA")
        difference = np.abs(
            np.array(embedded, dtype=int) - np.array(downscaled, dtype=int)
        ).mean()
        assert difference > 3.0, "the 16px frame looks like a downscaled 256"

    def test_the_two_colourways_differ(self):
        idle = Image.open(ICON).ico.getimage((64, 64)).convert("RGB")
        active = Image.open(ACTIVE_ICON).ico.getimage((64, 64)).convert("RGB")
        difference = np.abs(
            np.array(idle, dtype=int) - np.array(active, dtype=int)
        ).mean()
        assert difference > 10.0, "idle and listening icons look the same"

    def test_corners_are_transparent(self, icon):
        """A rounded icon must not sit on an opaque square."""
        largest = icon.ico.getimage((256, 256)).convert("RGBA")
        alpha = np.array(largest)[:, :, 3]
        assert alpha[0, 0] < 32 and alpha[0, -1] < 32
        assert alpha[-1, 0] < 32 and alpha[-1, -1] < 32

    def test_the_middle_is_opaque(self, icon):
        largest = icon.ico.getimage((256, 256)).convert("RGBA")
        alpha = np.array(largest)[:, :, 3]
        assert alpha[128, 128] == 255


class TestGenerator:
    def test_render_produces_the_requested_size(self):
        import sys

        sys.path.insert(0, str(ASSETS_DIR.parent / "tools"))
        from make_icon import render

        for size in (16, 32, 256):
            image = render(size)
            assert image.size == (size, size)
            assert image.mode == "RGBA"

    def test_an_unknown_theme_is_rejected(self):
        import sys

        sys.path.insert(0, str(ASSETS_DIR.parent / "tools"))
        from make_icon import render

        with pytest.raises(KeyError):
            render(32, theme="chartreuse")


@pytest.mark.integration
class TestQtLoadsIt:
    def test_qicon_reads_the_file(self):
        pytest.importorskip("PySide6")
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        icon = QIcon(str(ICON))
        assert not icon.isNull()
        assert icon.availableSizes(), "Qt found no sizes in the icon"
        assert not icon.pixmap(16, 16).isNull()
