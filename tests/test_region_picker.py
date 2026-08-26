"""The region picker must report real screen pixels, not Qt's scaled ones.

On a display at 125% Qt calls a 2560x1440 screen 2048x1152. A rectangle drawn
in Qt coordinates and handed to the screen grabber therefore captured the wrong
area at four fifths of the resolution. The picker now tracks the drag with
GetCursorPos and uses Qt only for drawing.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gamevoice.capture import ScreenGrabber, set_dpi_aware  # noqa: E402
from gamevoice.config import Region  # noqa: E402
from gamevoice.gui.region_picker import RegionPicker, cursor_position  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    set_dpi_aware()
    app = QApplication.instance() or QApplication([])
    yield app


class TestCursorPosition:
    def test_returns_a_pair_of_ints(self):
        x, y = cursor_position()
        assert isinstance(x, int) and isinstance(y, int)

    @pytest.mark.integration
    def test_cursor_lands_inside_the_physical_desktop(self):
        set_dpi_aware()
        virtual = ScreenGrabber().monitors()[0]
        x, y = cursor_position()
        assert virtual["left"] <= x <= virtual["left"] + virtual["width"]
        assert virtual["top"] <= y <= virtual["top"] + virtual["height"]


class TestRegionFromDrag:
    def test_region_comes_from_the_pixel_coordinates(self, qt_app):
        picker = RegionPicker()
        picker._origin_px = (100, 200)
        picker._current_px = (700, 350)
        assert picker._to_region() == Region(left=100, top=200, width=600, height=150)

    def test_a_backwards_drag_is_normalised(self, qt_app):
        picker = RegionPicker()
        picker._origin_px = (700, 350)
        picker._current_px = (100, 200)
        assert picker._to_region() == Region(left=100, top=200, width=600, height=150)

    def test_a_tiny_drag_is_rejected(self, qt_app):
        picker = RegionPicker()
        picker._origin_px = (100, 100)
        picker._current_px = (105, 104)
        assert picker._to_region() is None

    def test_an_initial_region_round_trips(self, qt_app):
        initial = Region(left=40, top=60, width=800, height=220)
        picker = RegionPicker(initial)
        assert picker._to_region() == initial

    def test_negative_coordinates_survive(self, qt_app):
        """A left-hand monitor sits at negative x; that must not be clamped."""
        picker = RegionPicker()
        picker._origin_px = (-2000, 100)
        picker._current_px = (-1400, 300)
        region = picker._to_region()
        assert region == Region(left=-2000, top=100, width=600, height=200)


@pytest.mark.integration
class TestScalingMismatchIsReal:
    def test_qt_and_windows_disagree_when_scaling_is_on(self, qt_app):
        """Documents why the picker cannot use Qt coordinates.

        Skips on a 100% display, where the two happen to agree.
        """
        screen = QGuiApplication.primaryScreen()
        if screen.devicePixelRatio() == 1.0:
            pytest.skip("display is at 100%; no mismatch to demonstrate")

        physical = ScreenGrabber().monitor_region(1)
        logical = screen.geometry()
        assert physical.width != logical.width(), (
            "Qt and Windows report the same width, so the picker's pixel "
            "tracking would be unnecessary here"
        )
        assert physical.width == pytest.approx(
            logical.width() * screen.devicePixelRatio(), rel=0.02
        )
