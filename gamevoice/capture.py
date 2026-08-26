"""Screen capture and foreground-window tracking.

The universal way to read text out of *any* game is to look at the pixels, so
this module is deliberately game-agnostic: it grabs a rectangle and hands back
an array. Which rectangle is decided here too, because the useful default -
"the bottom band of whatever window is in front" - needs the window geometry.

Known limit: a game running in *exclusive* fullscreen may hand back black
frames. Borderless windowed works. ``looks_blank`` exists so the app can say
that plainly instead of appearing to work while reading nothing.
"""
from __future__ import annotations

import ctypes
import logging
import threading
from ctypes import wintypes
from dataclasses import dataclass

import numpy as np

from .config import CaptureSettings, Region

log = logging.getLogger(__name__)

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_MAX_PATH = 32768


@dataclass(frozen=True)
class WindowInfo:
    handle: int
    title: str
    exe: str
    rect: Region

    @property
    def is_usable(self) -> bool:
        return self.handle != 0 and self.rect.is_valid()


def _process_exe(pid: int) -> str:
    handle = _kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(_MAX_PATH)
        buffer = ctypes.create_unicode_buffer(_MAX_PATH)
        if _kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return buffer.value.rsplit("\\", 1)[-1].lower()
    finally:
        _kernel32.CloseHandle(handle)
    return ""


def foreground_window() -> WindowInfo:
    """The window the user is actually looking at, with its screen rectangle."""
    handle = _user32.GetForegroundWindow()
    if not handle:
        return WindowInfo(0, "", "", Region())

    length = _user32.GetWindowTextLengthW(handle)
    title_buf = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(handle, title_buf, length + 1)

    pid = wintypes.DWORD()
    _user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))

    rect = wintypes.RECT()
    if not _user32.GetWindowRect(handle, ctypes.byref(rect)):
        return WindowInfo(handle, title_buf.value, _process_exe(pid.value), Region())

    region = Region(
        left=rect.left,
        top=rect.top,
        width=rect.right - rect.left,
        height=rect.bottom - rect.top,
    )
    return WindowInfo(handle, title_buf.value, _process_exe(pid.value), region)


def set_dpi_aware() -> None:
    """Report true pixels on a scaled display.

    Without this the captured rectangle is silently offset on any monitor at
    125% or above, which reads as "OCR sees nothing" rather than as a DPI bug.
    """
    try:
        # Per-monitor v2; correct on mixed-DPI multi-monitor setups.
        _user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            _user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            log.warning("could not set DPI awareness; capture may be offset")


class ScreenGrabber:
    """Thread-confined mss wrapper.

    mss handles are not safe to share between threads, so one grabber belongs
    to one thread and creates its handle lazily on first use there.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def _sct(self):
        sct = getattr(self._local, "sct", None)
        if sct is None:
            import mss  # imported lazily so the GUI can start without it

            # mss.mss() is deprecated in favour of mss.MSS(); keep the old name
            # as a fallback so an older mss still works.
            factory = getattr(mss, "MSS", None) or mss.mss
            sct = factory()
            self._local.sct = sct
        return sct

    def monitors(self) -> list[dict[str, int]]:
        return list(self._sct().monitors)

    def monitor_region(self, index: int) -> Region:
        monitors = self.monitors()
        if not 0 <= index < len(monitors):
            index = 1 if len(monitors) > 1 else 0
        mon = monitors[index]
        return Region(mon["left"], mon["top"], mon["width"], mon["height"])

    def grab(self, region: Region) -> np.ndarray | None:
        """BGRA pixels for a rectangle, or None if the grab failed."""
        if not region.is_valid():
            return None
        try:
            shot = self._sct().grab(region.as_mss())
        except Exception as exc:  # mss raises its own error types
            log.error("screen grab failed for %s: %s", region, exc)
            return None
        frame = np.frombuffer(shot.rgb, dtype=np.uint8)
        # mss gives RGB in .rgb; reshape and append opaque alpha for WinRT.
        frame = frame.reshape(shot.height, shot.width, 3)
        alpha = np.full((shot.height, shot.width, 1), 255, dtype=np.uint8)
        return np.concatenate([frame[:, :, ::-1], alpha], axis=2)

    def close(self) -> None:
        sct = getattr(self._local, "sct", None)
        if sct is not None:
            try:
                sct.close()
            except Exception:  # nothing useful to do on a failed close
                pass
            self._local.sct = None


def bottom_band(area: Region, fraction: float) -> Region:
    """The lower slice of a rectangle, where subtitles usually sit."""
    fraction = min(max(fraction, 0.05), 1.0)
    height = max(int(area.height * fraction), 24)
    return Region(area.left, area.top + area.height - height, area.width, height)


def resolve_regions(
    settings: CaptureSettings,
    grabber: ScreenGrabber,
    window: WindowInfo | None = None,
) -> tuple[Region, Region | None]:
    """Decide what to capture this tick.

    Priority: an explicit region the user picked, else the bottom band of the
    foreground window, else the bottom band of the configured monitor. The
    window fallback is what makes an unprofiled game work on first launch.
    """
    speaker = settings.speaker_region if (
        settings.speaker_region and settings.speaker_region.is_valid()
    ) else None

    if settings.text_region and settings.text_region.is_valid():
        return settings.text_region, speaker

    if window is not None and window.is_usable:
        return bottom_band(window.rect, settings.bottom_band), speaker

    return bottom_band(grabber.monitor_region(settings.monitor), settings.bottom_band), speaker


class ChangeDetector:
    """Decides whether a frame differs enough from the last to be worth reading.

    Grabbing pixels is cheap; recognising them is not. Polling on a timer spends
    that cost on every tick whether or not anything moved, and still reacts late
    when something does. Comparing a downsampled copy first lets the reader poll
    faster *and* work less: it recognises the moment the box changes, and does
    nothing at all while it sits still.

    The comparison is deliberately coarse. A game's background is rarely
    perfectly static - grain, dithering and slow gradients all shift a little -
    so a pixel has to move by more than ``tolerance`` before it counts, and
    enough pixels have to move before the frame does.
    """

    def __init__(
        self,
        threshold: float = 0.004,
        tolerance: int = 10,
        grid: int = 96,
        max_idle: float = 8.0,
    ) -> None:
        self._threshold = max(0.0, min(threshold, 1.0))
        self._tolerance = max(0, min(tolerance, 255))
        self._grid = max(8, grid)
        self._max_idle = max(0.0, max_idle)
        self._previous: np.ndarray | None = None
        self._source_shape: tuple[int, int] | None = None
        self._last_change = 0.0
        self.last_ratio = 0.0

    def _signature(self, frame: np.ndarray) -> np.ndarray:
        """A small greyscale thumbnail, cheap to build and cheap to compare.

        Sampled at fixed positions rather than by a stride: a stride of
        ``size // grid`` collapses very different region sizes onto the same
        thumbnail shape, which made a resized capture area compare as unchanged.
        """
        height, width = frame.shape[0], frame.shape[1]
        rows = np.linspace(0, height - 1, min(self._grid, height)).astype(np.intp)
        cols = np.linspace(0, width - 1, min(self._grid, width)).astype(np.intp)
        sample = frame[np.ix_(rows, cols)][:, :, :3]
        # Mean of the channels is close enough to luminance for a change test
        # and avoids the multiply-and-sum of a weighted conversion.
        return sample.mean(axis=2).astype(np.int16)

    def reset(self) -> None:
        self._previous = None
        self._source_shape = None
        self._last_change = 0.0
        self.last_ratio = 0.0

    def changed(self, frame: np.ndarray, now: float) -> bool:
        if frame is None or frame.size == 0:
            return False

        shape = (int(frame.shape[0]), int(frame.shape[1]))
        signature = self._signature(frame)
        previous = self._previous

        # A different capture size means a different picture, whatever the
        # thumbnails happen to look like.
        if (
            previous is None
            or self._source_shape != shape
            or previous.shape != signature.shape
        ):
            self._previous = signature
            self._source_shape = shape
            self._last_change = now
            self.last_ratio = 1.0
            return True

        moved = np.abs(signature - previous) > self._tolerance
        ratio = float(moved.mean())
        self.last_ratio = ratio

        if ratio >= self._threshold:
            self._previous = signature
            self._last_change = now
            return True

        # A safety net: if nothing has been read for a long while, read anyway.
        # Covers a change too subtle for the threshold, and a first frame that
        # arrived before the text finished drawing.
        if self._max_idle and now - self._last_change >= self._max_idle:
            self._previous = signature
            self._last_change = now
            return True

        return False


def looks_blank(frame: np.ndarray, tolerance: int = 6) -> bool:
    """True when a frame carries no detail worth running OCR over.

    Catches both an exclusive-fullscreen black grab and a genuinely empty
    subtitle area, which lets the caller skip the OCR call entirely.
    """
    if frame is None or frame.size == 0:
        return True
    sample = frame[::4, ::4, :3]
    return bool(sample.max() - sample.min() <= tolerance)
