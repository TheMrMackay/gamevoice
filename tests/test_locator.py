"""The dialogue locator, without needing a screen or the OCR engine.

The locator's risky parts are its decisions, not its pixels: which lines count
as dialogue, when a scan is allowed to happen, and when tracking switches to a
new box. Those are all tested here against a stub recogniser. The end-to-end
question - does a real box at a real position get found - lives in
test_pipeline.py with the real engine.
"""
from __future__ import annotations

import re
import time

import numpy as np

from gamevoice.capture import WindowInfo
from gamevoice.config import CaptureSettings, DetectSettings, Region
from gamevoice.locator import DialogueLocator, _cluster, _filter_lines
from gamevoice.ocr import OcrLine, OcrOutput

WINDOW = WindowInfo(1, "Game", "game.exe", Region(0, 0, 1280, 720))


def noise_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    rng = np.random.default_rng()
    return rng.integers(0, 255, (height, width, 4), dtype=np.uint8)


class StubOcr:
    """Returns one fixed reading, counting how often it was asked."""

    def __init__(self, output: OcrOutput) -> None:
        self.output = output
        self.calls = 0

    def recognize(self, frame) -> OcrOutput:
        self.calls += 1
        return self.output


class FreshNoiseGrabber:
    """A new noisy frame every grab, so the change gate always passes.

    One generator, advanced per grab: reseeding per call would hand back the
    identical frame each time and gate every scan after the first.
    """

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self._width = width
        self._height = height
        self._rng = np.random.default_rng(0)

    def grab(self, region):
        return self._rng.integers(
            0, 255, (self._height, self._width, 4), dtype=np.uint8
        )


def make_locator(output: OcrOutput, **capture_overrides) -> tuple[DialogueLocator, StubOcr]:
    capture = CaptureSettings(**capture_overrides)
    ocr = StubOcr(output)
    return DialogueLocator(capture, DetectSettings(), ocr, FreshNoiseGrabber()), ocr


def lines_at(*specs: tuple[str, float, float]) -> OcrOutput:
    """(text, left, top) triplets at a fixed 30 px glyph height."""
    built = tuple(
        OcrLine(text=text, top=top, left=left, height=30.0)
        for text, left, top in specs
    )
    return OcrOutput("\n".join(line.text for line in built), built)


def box_of(locator: DialogueLocator) -> Region:
    tracked = locator.tracked
    assert tracked is not None
    return tracked


class TestFiltering:
    def test_a_dialogue_line_is_kept(self):
        output = lines_at(("We should not have come this way.", 100.0, 500.0))
        kept = _filter_lines(output, scale=1.0, ignore=())
        assert len(kept) == 1

    def test_hud_furniture_is_dropped(self):
        output = lines_at(("OBJECTIVES", 40.0, 30.0), ("LOADING", 600.0, 30.0))
        assert _filter_lines(output, scale=1.0, ignore=()) == []

    def test_tiny_and_huge_glyphs_are_dropped(self):
        tiny = OcrLine("a whisper of text", top=100.0, left=100.0, height=2.0)
        huge = OcrLine("A SHOUT", top=300.0, left=100.0, height=200.0)
        kept = _filter_lines(OcrOutput("x", (tiny, huge)), scale=1.0, ignore=())
        assert kept == []

    def test_ignore_patterns_are_applied(self):
        output = lines_at(("Press E to continue", 100.0, 500.0))
        ignore = (re.compile(r"press .* to ", re.IGNORECASE),)
        assert _filter_lines(output, scale=1.0, ignore=ignore) == []


class TestClustering:
    def test_a_paragraph_is_one_cluster(self):
        output = lines_at(
            ("The bridge will not hold much longer.", 100.0, 500.0),
            ("We have to get the villagers across.", 100.0, 540.0),
        )
        kept = _filter_lines(output, scale=1.0, ignore=())
        clusters = _cluster(kept)
        assert len(clusters) == 1
        assert clusters[0][2] == 2

    def test_two_distant_boxes_are_two_clusters(self):
        output = lines_at(
            ("Far above the clouds", 100.0, 40.0),
            ("Something far below", 100.0, 640.0),
        )
        clusters = _cluster(_filter_lines(output, scale=1.0, ignore=()))
        assert len(clusters) == 2


class TestScanning:
    def test_a_box_away_from_the_bottom_is_tracked(self):
        locator, _ = make_locator(
            lines_at(("Elena: We ride at dawn.", 300.0, 60.0))
        )
        assert locator.state == "searching"
        assert locator.scan(WINDOW, time.time()) is not None
        assert locator.state == "tracking"
        box = box_of(locator)
        assert box.top < 200  # nowhere near the bottom band

    def test_the_tracked_box_is_window_relative(self):
        locator, _ = make_locator(lines_at(("Elena: We ride at dawn.", 300.0, 60.0)))
        locator.scan(WINDOW, time.time())
        box = box_of(locator)

        moved = WindowInfo(1, "Game", "game.exe", Region(300, 250, 1280, 720))
        screen = locator.region_for(moved)
        assert screen is not None
        assert screen.left == moved.rect.left + box.left
        assert screen.top == moved.rect.top + box.top

    def test_hud_only_scenes_track_nothing(self):
        locator, ocr = make_locator(lines_at(("OBJECTIVES", 40.0, 30.0)))
        assert locator.scan(WINDOW, time.time()) is None
        assert locator.tracked is None
        assert ocr.calls == 1

    def test_an_oversized_block_is_rejected(self):
        # 1500 chars at 30 px estimates a box wider than the window itself,
        # far past the 60% area cap.
        wide = lines_at(("x" * 1500, 20.0, 300.0))
        locator, _ = make_locator(wide)
        assert locator.scan(WINDOW, time.time()) is None

    def test_off_state_never_scans(self):
        locator, ocr = make_locator(
            lines_at(("Elena: We ride at dawn.", 300.0, 60.0)), auto_region=False
        )
        assert locator.state == "off"
        assert not locator.scan_due(time.time())
        assert ocr.calls == 0


class TestGating:
    def test_scan_interval_bounds_repeats(self):
        locator, _ = make_locator(lines_at(("Elena: We ride at dawn.", 300.0, 60.0)))
        now = 1000.0
        assert locator.scan_due(now)
        locator.scan(WINDOW, now)
        assert not locator.scan_due(now + 0.5)
        assert locator.scan_due(now + 3.5)

    def test_tracking_waits_longer_between_scans(self):
        locator, _ = make_locator(lines_at(("Elena: We ride at dawn.", 300.0, 60.0)))
        now = 1000.0
        locator.scan(WINDOW, now)
        # Just found it, and the fast path is reading: the relocation scan is
        # the slower hold interval, not the search one.
        locator.report_fast(True, now)
        assert not locator.scan_due(now + 1.5)
        assert locator.scan_due(now + 3.5)

    def test_silence_on_the_fast_path_forces_an_early_scan(self):
        locator, _ = make_locator(lines_at(("Elena: We ride at dawn.", 300.0, 60.0)))
        now = 1000.0
        locator.scan(WINDOW, now)
        # No report_fast: the fast path has read nothing since the box was
        # found, so waiting the full hold interval would be wrong.
        assert locator.scan_due(now + 2.5)


class TestSwitching:
    def test_a_one_frame_popup_does_not_steal_the_tracker(self):
        first = lines_at(("Elena: We ride at dawn.", 300.0, 60.0))
        second = lines_at(("Marcus: Then we wait for dusk.", 300.0, 600.0))
        locator, _ = make_locator(first)
        now = 1000.0
        locator.scan(WINDOW, now)
        locator.report_fast(True, now)  # the fast path is reading box one

        # Both scans stay inside lost_seconds of the last fast-path hit, so
        # only the two-scan rule can justify a switch.
        locator._ocr.output = second
        locator.scan(WINDOW, now + 2.0)
        assert box_of(locator).top < 200  # still the first box

        locator._ocr.output = second
        locator.scan(WINDOW, now + 3.5)
        assert box_of(locator).top > 400  # same newcomer twice: switch

    def test_a_silent_fast_path_switches_immediately(self):
        first = lines_at(("Elena: We ride at dawn.", 300.0, 60.0))
        second = lines_at(("Marcus: Then we wait for dusk.", 300.0, 600.0))
        locator, _ = make_locator(first)
        now = 1000.0
        locator.scan(WINDOW, now)
        # No report_fast at all: box one is gone as far as anyone knows.
        locator._ocr.output = second
        locator.scan(WINDOW, now + 10.0)
        assert box_of(locator).top > 400

    def test_a_grown_box_is_absorbed_not_switched(self):
        first = lines_at(("Elena: We ride at dawn.", 300.0, 60.0))
        grown = lines_at(
            ("Elena: We ride at dawn.", 300.0, 60.0),
            ("The horses are ready.", 300.0, 100.0),
        )
        locator, _ = make_locator(first)
        now = 1000.0
        locator.scan(WINDOW, now)
        before = box_of(locator)

        locator._ocr.output = grown
        locator.scan(WINDOW, now + 5.0)
        after = box_of(locator)
        assert after.top <= before.top
        assert after.height > before.height

    def test_reset_forgets_the_box(self):
        locator, _ = make_locator(lines_at(("Elena: We ride at dawn.", 300.0, 60.0)))
        locator.scan(WINDOW, time.time())
        locator.reset()
        assert locator.tracked is None
        assert locator.state == "searching"
