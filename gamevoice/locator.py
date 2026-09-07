"""Finding the dialogue box wherever a game draws it.

Games are under no obligation to put speech at the bottom of the screen, and
portrait visual novels put it half way up the left edge. Rather than guess a
position, this module asks the recogniser itself: every so often it reads the
whole foreground window at reduced size, and keeps the one cluster of text
that behaves like dialogue. Between scans the engine reads only that box, so
steady-state cost is what it always was.

The scan is bounded by time (``scan_interval``) and gated by a change detector
of its own, so a static menu scene costs almost nothing and a churning 3D
scene costs at most one reduced-size reading per interval.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .capture import ChangeDetector, ScreenGrabber, WindowInfo, looks_blank
from .config import CaptureSettings, DetectSettings, Region
from .dialogue import _NOT_SPEAKERS, _PAGE_MARK
from .ocr import OcrOutput, preprocess

log = logging.getLogger(__name__)

# Glyph heights (in window pixels) a line of dialogue can plausibly have.
_MIN_GLYPH = 6.0
_MAX_GLYPH = 90.0
# Subtitle-sized text scores above UI furniture of the same length.
_SCORE_HEIGHT_BAND = (14.0, 48.0)
# Two lines belong to one paragraph when the gap between them is under this
# fraction of the median glyph height.
_LINE_GAP = 0.8
# How much of the tracked box a new candidate must share to count as "same".
_ABSORB_IOU = 0.4


@dataclass(frozen=True)
class Candidate:
    """One cluster of text found by a scan, in window-relative pixels."""

    box: Region
    text: str
    line_count: int
    score: float


def _has_content(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9]", text))


def _is_furniture(text: str) -> bool:
    """True for HUD labels and interface captions, which are not dialogue."""
    if _PAGE_MARK.match(text):
        return True
    words = [w.lower().strip(".:,") for w in text.split()]
    return bool(words) and all(w in _NOT_SPEAKERS for w in words)


def _line_box(line, scale: float) -> tuple[float, float, float, float]:
    """(left, top, right, bottom) in window pixels.

    OcrLine carries no width, so one is estimated from the text length: an
    average glyph runs a little over half its height wide. The estimate only
    has to be good enough to pad a capture box, not to draw a highlight.
    """
    height = line.height / scale
    width = len(line.text) * height * 0.55
    left = line.left / scale
    top = line.top / scale
    return left, top, left + width, top + height


def _filter_lines(
    output: OcrOutput,
    scale: float,
    ignore: tuple[re.Pattern, ...],
) -> list[tuple[float, float, float, float, str]]:
    """Keep the lines that could be dialogue, with boxes in window pixels."""
    kept = []
    for line in output.lines:
        text = line.text.strip()
        if not _has_content(text) or _is_furniture(text):
            continue
        if any(pattern.search(text) for pattern in ignore):
            continue
        height = line.height / scale
        if not _MIN_GLYPH <= height <= _MAX_GLYPH:
            continue
        left, top, right, bottom = _line_box(line, scale)
        kept.append((left, top, right, bottom, text))
    return kept


def _cluster(
    lines: list[tuple[float, float, float, float, str]],
) -> list[tuple[float, float, float, float, list[str]]]:
    """Group lines into paragraph-like clusters.

    Lines join a cluster when they overlap it horizontally and sit within a
    fraction of a line height of it vertically - the geometry of a block of
    subtitles. Clusters that ran into each other are merged afterwards.
    """
    if not lines:
        return []
    median_h = sorted(b[3] - b[1] for b in lines)[len(lines) // 2]
    gap = median_h * _LINE_GAP

    clusters: list[list[float]] = []  # left, top, right, bottom
    texts: list[list[str]] = []
    for left, top, right, bottom, text in sorted(lines, key=lambda b: b[1]):
        for index, (cl, ct, cr, cb) in enumerate(clusters):
            if top <= cb + gap and bottom >= ct - gap and left <= cr + gap and right >= cl - gap:
                clusters[index] = [min(cl, left), min(ct, top), max(cr, right), max(cb, bottom)]
                texts[index].append(text)
                break
        else:
            clusters.append([left, top, right, bottom])
            texts.append([text])

    merged = list(zip(clusters, texts))
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if _overlaps(merged[i][0], merged[j][0]):
                    merged[i] = (
                        [
                            min(merged[i][0][0], merged[j][0][0]),
                            min(merged[i][0][1], merged[j][0][1]),
                            max(merged[i][0][2], merged[j][0][2]),
                            max(merged[i][0][3], merged[j][0][3]),
                        ],
                        merged[i][1] + merged[j][1],
                    )
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return [(box, texts, len(texts)) for box, texts in merged]


def _overlaps(a: list[float], b: list[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _iou(a: Region, b: Region) -> float:
    left = max(a.left, b.left)
    top = max(a.top, b.top)
    right = min(a.left + a.width, b.left + b.width)
    bottom = min(a.top + a.height, b.top + b.height)
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _as_region(box: list[float]) -> Region:
    return Region(
        left=int(box[0]),
        top=int(box[1]),
        width=max(1, int(box[2] - box[0])),
        height=max(1, int(box[3] - box[1])),
    )


def _pad(box: Region, cluster_h: float, line_count: int, window: WindowInfo) -> Region:
    """Grow a found box so descenders and the next typed character survive."""
    median_h = cluster_h / max(1, line_count)
    pad = max(12.0, median_h * 1.5)
    left = max(0, int(box.left - pad))
    top = max(0, int(box.top - pad))
    right = min(window.rect.width, int(box.left + box.width + pad))
    bottom = min(window.rect.height, int(box.top + box.height + pad))
    return Region(left, top, max(1, right - left), max(1, bottom - top))


class DialogueLocator:
    """Scans the whole window for dialogue, then tracks the box it finds.

    All tracked geometry is window-relative, so a window the user drags
    mid-sentence keeps being read at the same spot, and a move between
    monitors with different scaling stays correct.
    """

    def __init__(
        self,
        capture: CaptureSettings,
        detect: DetectSettings,
        ocr,
        grabber: ScreenGrabber,
        remembered: Region | None = None,
    ) -> None:
        self._capture = capture
        self._ocr = ocr
        self._grabber = grabber
        self._ignore = self._compile(detect.ignore_patterns)
        self._max_chars = detect.max_chars
        self._changes = ChangeDetector(
            threshold=capture.change_threshold,
            tolerance=capture.change_tolerance,
        )
        # Where the box was last session, window-relative. Tracking starts
        # from it immediately; the first scan confirms it or moves on.
        self._tracked: Region | None = (
            remembered if (remembered is not None and remembered.is_valid()) else None
        )
        # The engine sets this to persist each box the locator settles on.
        self.on_track = None
        self._pending: Region | None = None
        self._last_scan = 0.0
        self._last_hit = 0.0
        self._scanning = False

    @staticmethod
    def _compile(patterns: list[str]) -> tuple[re.Pattern, ...]:
        compiled = []
        for pattern in patterns:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                log.warning("ignoring bad ignore pattern %r: %s", pattern, exc)
        return tuple(compiled)

    @property
    def state(self) -> str:
        if not self._capture.auto_region:
            return "off"
        return "tracking" if self._tracked is not None else "searching"

    @property
    def tracked(self) -> Region | None:
        """The tracked box, window-relative. None while nothing is found."""
        return self._tracked

    def reset(self) -> None:
        """Forget everything - new profile, or the user resumed the app."""
        self._tracked = None
        self._pending = None
        self._changes.reset()
        self._last_scan = 0.0
        self._last_hit = 0.0

    def report_fast(self, found_text: bool, now: float | None = None) -> None:
        """The engine says whether the fast path read anything this tick."""
        if found_text:
            self._last_hit = now if now is not None else time.monotonic()

    def region_for(self, window: WindowInfo | None) -> Region | None:
        """The screen rectangle to read this tick, or None for the old path."""
        if not self._capture.auto_region:
            return None
        if self._tracked is None or window is None or not window.is_usable:
            return None
        if (
            self._tracked.left + self._tracked.width > window.rect.width
            or self._tracked.top + self._tracked.height > window.rect.height
        ):
            # A remembered box from a different window size no longer means
            # anything here; searching again re-learns it within a second.
            log.info("tracked box does not fit the window; searching again")
            self._tracked = None
            return None
        return Region(
            left=window.rect.left + self._tracked.left,
            top=window.rect.top + self._tracked.top,
            width=self._tracked.width,
            height=self._tracked.height,
        )

    def scan_due(self, now: float) -> bool:
        if not self._capture.auto_region or self._scanning:
            return False
        interval = (
            self._capture.hold_scan_interval
            if self._tracked is not None
            else self._capture.scan_interval
        )
        if now - self._last_scan >= interval:
            return True
        # Nothing read for a while means the box probably moved or closed.
        lost = self._capture.lost_seconds
        return bool(
            lost > 0
            and self._tracked is not None
            and now - self._last_hit >= lost
            and now - self._last_scan >= min(interval, lost / 2)
        )

    def scan(self, window: WindowInfo, now: float) -> Region | None:
        """One reduced-size read of the whole window. Returns the tracked box."""
        if not window.is_usable:
            return None
        self._scanning = True
        try:
            return self._scan_locked(window, now)
        finally:
            self._scanning = False

    def _scan_locked(self, window: WindowInfo, now: float) -> Region | None:
        self._last_scan = now
        frame = self._grabber.grab(window.rect)
        if frame is None or looks_blank(frame):
            return self._tracked

        reduced, scale = _downscale(frame, self._capture.scan_width)
        if reduced is None:
            return self._tracked
        # A static scene re-reads for nothing; the gate is opportunistic, the
        # interval above is the real cost bound.
        if not self._changes.changed(reduced, now):
            return self._tracked

        prepared = preprocess(reduced, upscale=1.0, contrast=self._capture.contrast,
                              grayscale=self._capture.grayscale, invert=self._capture.invert)
        output = self._ocr.recognize(prepared)
        best = self._choose(self._candidates(output, scale, window))
        if best is not None:
            self._adopt(best, now)
        elif self._tracked is not None and now - self._last_hit >= self._capture.lost_seconds:
            log.info("dialogue box no longer found; searching again")
            self._tracked = None
            self._pending = None
        return self._tracked

    def _candidates(
        self, output: OcrOutput, scale: float, window: WindowInfo
    ) -> list[Candidate]:
        window_area = max(1, window.rect.width * window.rect.height)
        found: list[Candidate] = []
        for box, texts, count in _cluster(_filter_lines(output, scale, self._ignore)):
            # The size test runs on the raw cluster: padding clamps to the
            # window, which would otherwise disguise a wall of text as a
            # full-width strip and let it through.
            raw = _as_region(box)
            if raw.width * raw.height > self._capture.max_region_fraction * window_area:
                continue
            text = "\n".join(texts)
            if len(text) > self._max_chars:
                continue
            region = _pad(raw, box[3] - box[1], count, window)
            found.append(Candidate(region, text, count, _score(region, text, window, count)))
        return found

    def _choose(self, candidates: list[Candidate]) -> Candidate | None:
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.score)

    def _set_tracked(self, box: Region, why: str) -> None:
        self._tracked = box
        if self.on_track is not None:
            try:
                self.on_track(box)
            except Exception as exc:  # persistence must never break reading
                log.warning("on_track callback failed: %s", exc)

    def _adopt(self, best: Candidate, now: float) -> None:
        tracked = self._tracked
        if tracked is None:
            log.info("dialogue found at (%d, %d) %dx%d",
                     best.box.left, best.box.top, best.box.width, best.box.height)
            self._set_tracked(best.box, "found")
            self._changes.reset()
            return

        if _iou(tracked, best.box) >= _ABSORB_IOU or _overlaps(
            [tracked.left, tracked.top, tracked.left + tracked.width,
             tracked.top + tracked.height],
            [best.box.left, best.box.top, best.box.left + best.box.width,
             best.box.top + best.box.height],
        ):
            # Same box, grown or shrunk - typewriter text does this constantly.
            if best.box != tracked:
                self._set_tracked(best.box, "grown")
                self._changes.reset()
            self._pending = None
            return

        # A box somewhere else. If the fast path has been silent, the old one
        # is gone: move now. Otherwise require the same newcomer twice, so a
        # one-frame popup cannot steal the tracker.
        if now - self._last_hit >= self._capture.lost_seconds:
            self._switch_to(best)
        elif self._pending is not None and self._pending == best.box:
            self._switch_to(best)
        else:
            self._pending = best.box

    def _switch_to(self, best: Candidate) -> None:
        log.info("dialogue moved to (%d, %d) %dx%d",
                 best.box.left, best.box.top, best.box.width, best.box.height)
        self._set_tracked(best.box, "moved")
        self._pending = None
        self._changes.reset()


def _downscale(frame: np.ndarray, target_width: int) -> tuple[np.ndarray | None, float]:
    """Shrink a full-window grab to scanner size. Returns (frame, scale)."""
    height, width = frame.shape[0], frame.shape[1]
    if width <= 0 or target_width <= 0:
        return None, 1.0
    if width <= target_width:
        return frame, 1.0
    scale = width / target_width
    target_height = max(1, int(height / scale))
    image = Image.fromarray(frame[:, :, :3][:, :, ::-1])
    reduced = image.resize((target_width, target_height), Image.BILINEAR)
    rgb = np.array(reduced)
    alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
    return np.concatenate([rgb[:, :, ::-1], alpha], axis=2), scale


def _score(box: Region, text: str, window: WindowInfo, line_count: int) -> float:
    """Dialogue-shaped text outranks UI furniture of the same length.

    No positional hard-rejection: a HUD and a subtitle differ by what they
    say and how big it is, not reliably by where they sit. Bottom-most is a
    small tie-break only - dialogue is more often low than a HUD is rare low.
    """
    score = float(len(text))
    median_h = box.height / max(1, line_count)
    if _SCORE_HEIGHT_BAND[0] <= median_h <= _SCORE_HEIGHT_BAND[1]:
        score += 30.0
    if window.rect.height > 0:
        centre = (box.top + box.height / 2) / window.rect.height
        score += centre * 10.0
    return score
