"""Text recognition using the OCR engine built into Windows.

Windows.Media.Ocr is the right tool here: it is already on the machine, needs
no model download, and reads a cropped subtitle band in a few milliseconds -
fast enough to poll while a game is running without competing for the GPU.

The WinRT API is asynchronous, so this module owns a private event loop and
presents a plain blocking ``recognize`` to the rest of the app.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .native import ensure_load_order

log = logging.getLogger(__name__)

# Must run before this module touches WinRT: loading the Windows Runtime first
# breaks onnxruntime's own DLL initialisation, which would disable every Piper
# voice with an error that points at the wrong thing. See native.py.
ensure_load_order()

# Windows refuses images above this on either axis.
MAX_DIMENSION = 10000


# Windows' recogniser reads text of roughly this height most reliably. Well
# below it, accuracy falls off sharply; well above it costs time for nothing.
TARGET_GLYPH_HEIGHT = 42.0


@dataclass(frozen=True)
class OcrLine:
    text: str
    top: float
    left: float
    height: float


def clipped_edges(output: "OcrOutput", height: int, margin: float = 6.0) -> list[str]:
    """Which edges of the capture have text pressed against them.

    Text touching the top or bottom edge almost always means the passage
    continues outside the captured area, which reads as "it stopped in the
    middle" - the same symptom as truncation, but a different cause and a
    different fix. Naming it lets the user widen the region instead of hunting
    for a bug in the reader.
    """
    if not output.lines or height <= 0:
        return []

    edges = []
    if any(line.top <= margin for line in output.lines):
        edges.append("top")
    if any(line.top + line.height >= height - margin for line in output.lines):
        edges.append("bottom")
    return edges


def clamp_scale(width: int, height: int, scale: float) -> float:
    """The largest requested scale that keeps the image inside the OCR limit."""
    if width <= 0 or height <= 0:
        return 1.0
    ceiling = min(MAX_DIMENSION / width, MAX_DIMENSION / height)
    return max(1.0, min(scale, ceiling))


class AdaptiveUpscale:
    """Upscales the capture until the text is large enough to read well.

    A fixed multiplier is wrong in both directions: it wastes time on a game
    with large subtitles and still under-reads one with small ones. This
    measures the glyph height the recogniser actually reported and converges on
    whatever scale puts it near ``target``, so "upscale if needed" means what it
    says. Damped, because reacting fully to one frame oscillates when a line is
    still being typed out.
    """

    def __init__(
        self,
        initial: float = 2.0,
        target: float = TARGET_GLYPH_HEIGHT,
        minimum: float = 1.0,
        maximum: float = 6.0,
        damping: float = 0.5,
        tolerance: float = 0.15,
    ) -> None:
        self._minimum = max(1.0, minimum)
        self._maximum = max(self._minimum, maximum)
        self._scale = min(max(initial, self._minimum), self._maximum)
        self._target = target
        self._damping = min(max(damping, 0.05), 1.0)
        self._tolerance = max(tolerance, 0.0)
        self.samples = 0

    @property
    def scale(self) -> float:
        return self._scale

    def reset(self, initial: float | None = None) -> None:
        if initial is not None:
            self._scale = min(max(initial, self._minimum), self._maximum)
        self.samples = 0

    def observe(self, lines: tuple[OcrLine, ...], applied_scale: float) -> None:
        """Learn from one reading. ``applied_scale`` is what produced it."""
        heights = [line.height for line in lines if line.height > 1.0]
        if not heights or applied_scale <= 0:
            return

        heights.sort()
        measured = heights[len(heights) // 2]
        source_height = measured / applied_scale
        if source_height <= 0.5:
            return

        wanted = min(max(self._target / source_height, self._minimum), self._maximum)
        self.samples += 1

        drift = abs(wanted - self._scale) / max(self._scale, 0.01)
        if drift < self._tolerance:
            return
        self._scale += (wanted - self._scale) * self._damping
        log.debug(
            "upscale %.2f -> %.2f (text %.1f px, target %.0f)",
            wanted, self._scale, source_height, self._target,
        )


@dataclass(frozen=True)
class OcrOutput:
    text: str
    lines: tuple[OcrLine, ...]

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class OcrUnavailable(RuntimeError):
    """Raised when Windows has no usable OCR engine installed."""


def available_languages() -> list[str]:
    try:
        from winrt.windows.media.ocr import OcrEngine

        return [lang.language_tag for lang in OcrEngine.available_recognizer_languages]
    except Exception as exc:
        log.error("could not enumerate OCR languages: %s", exc)
        return []


def preprocess(
    frame: np.ndarray,
    upscale: float = 2.0,
    contrast: float = 1.6,
    grayscale: bool = True,
    invert: bool = False,
) -> np.ndarray:
    """Make small, low-contrast subtitle text easier for the recogniser.

    Upscaling is the single biggest win - game subtitles are often rendered at
    a size the engine treats as marginal, and it reads a 2x image far more
    reliably than a sharpened one.
    """
    if frame is None or frame.size == 0:
        return frame

    image = Image.fromarray(frame[:, :, :3][:, :, ::-1])  # BGRA -> RGB

    if grayscale:
        image = image.convert("L")

    if contrast and contrast != 1.0:
        from PIL import ImageEnhance

        image = ImageEnhance.Contrast(image).enhance(contrast)

    if invert:
        from PIL import ImageOps

        image = ImageOps.invert(image.convert("L"))

    if upscale and upscale != 1.0:
        width = min(int(image.width * upscale), MAX_DIMENSION)
        height = min(int(image.height * upscale), MAX_DIMENSION)
        if width > 0 and height > 0:
            image = image.resize((width, height), Image.LANCZOS)

    rgb = np.array(image.convert("RGB"))
    alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
    return np.concatenate([rgb[:, :, ::-1], alpha], axis=2)


class WindowsOcr:
    """Blocking wrapper over the asynchronous WinRT recogniser."""

    def __init__(self, language: str = "en-US") -> None:
        self._language = language
        self._lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        try:
            from winrt.windows.globalization import Language
            from winrt.windows.media.ocr import OcrEngine

            engine = OcrEngine.try_create_from_language(Language(language))
            if engine is None:
                installed = ", ".join(available_languages()) or "none"
                raise OcrUnavailable(
                    f"Windows has no OCR pack for '{language}'. Installed: {installed}. "
                    "Add one under Settings > Time & language > Language & region."
                )
            self._engine = engine
        except OcrUnavailable:
            raise
        except Exception as exc:
            raise OcrUnavailable(
                f"Could not start the Windows OCR engine: {exc}"
            ) from exc

    @property
    def language(self) -> str:
        return self._language

    def _to_bitmap(self, frame: np.ndarray):
        from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
        from winrt.windows.storage.streams import Buffer

        height, width = frame.shape[0], frame.shape[1]
        raw = np.ascontiguousarray(frame).tobytes()

        buffer = Buffer(len(raw))
        # The buffer protocol exposes `length` bytes, not capacity, so length
        # has to be set before the memoryview is taken or the copy fails.
        buffer.length = len(raw)
        memoryview(buffer)[:] = raw
        return SoftwareBitmap.create_copy_from_buffer(
            buffer, BitmapPixelFormat.BGRA8, width, height
        )

    def recognize(self, frame: np.ndarray) -> OcrOutput:
        """Read a BGRA frame. Returns empty output rather than raising."""
        if frame is None or frame.size == 0:
            return OcrOutput("", ())

        height, width = frame.shape[0], frame.shape[1]
        if width > MAX_DIMENSION or height > MAX_DIMENSION:
            log.warning("frame %dx%d exceeds the OCR limit, skipping", width, height)
            return OcrOutput("", ())

        try:
            with self._lock:
                bitmap = self._to_bitmap(frame)
                result = self._loop.run_until_complete(
                    self._engine.recognize_async(bitmap)
                )
        except Exception as exc:
            log.error("OCR failed on a %dx%d frame: %s", width, height, exc)
            return OcrOutput("", ())

        lines = []
        for line in result.lines:
            words = list(line.words)
            if not words:
                continue
            top = min(w.bounding_rect.y for w in words)
            left = min(w.bounding_rect.x for w in words)
            tall = max(w.bounding_rect.height for w in words)
            lines.append(OcrLine(line.text, float(top), float(left), float(tall)))

        return OcrOutput("\n".join(item.text for item in lines), tuple(lines))

    def close(self) -> None:
        try:
            self._loop.close()
        except Exception:
            pass


def create_engine(language: str = "en-US") -> WindowsOcr:
    """Build a recogniser, falling back to any installed language.

    A machine with only, say, en-GB installed should still work rather than
    refusing to start over an exact tag mismatch.
    """
    try:
        return WindowsOcr(language)
    except OcrUnavailable:
        installed = available_languages()
        for candidate in installed:
            if candidate.split("-")[0] == language.split("-")[0]:
                log.warning("falling back to OCR language %s", candidate)
                return WindowsOcr(candidate)
        if installed:
            log.warning("falling back to OCR language %s", installed[0])
            return WindowsOcr(installed[0])
        raise
