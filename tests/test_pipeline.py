"""End-to-end checks against the real Windows OCR engine and real Piper voices.

These render a synthetic dialogue box rather than needing a game running, so
they can run in CI on any Windows box with an OCR language pack installed.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from gamevoice.capture import looks_blank
from gamevoice.dialogue import parse
from gamevoice.ocr import OcrUnavailable, create_engine, preprocess
from gamevoice.voices import VoiceCatalog, VoiceRouter

pytestmark = pytest.mark.integration


def _font(size: int):
    for candidate in ("georgia.ttf", "segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_dialogue(
    lines: list[str],
    width: int = 900,
    size: int = 30,
    canvas: tuple[int, int] | None = None,
    at: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """A dark subtitle box with pale text, the shape most games use.

    ``canvas`` and ``at`` place the box somewhere on a larger frame, which is
    how the locator tests exercise dialogue away from the bottom band.
    """
    box_height = 40 + len(lines) * (size + 14)
    image = Image.new("RGB", canvas or (width, box_height), (10, 10, 14))
    draw = ImageDraw.Draw(image)
    draw.rectangle([at[0], at[1], at[0] + width, at[1] + box_height], fill=(14, 14, 22))
    font = _font(size)
    for index, text in enumerate(lines):
        draw.text(
            (at[0] + 24, at[1] + 20 + index * (size + 14)),
            text,
            fill=(236, 232, 216),
            font=font,
        )
    rgb = np.array(image)
    alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
    return np.concatenate([rgb[:, :, ::-1], alpha], axis=2)


@pytest.fixture(scope="module")
def ocr():
    try:
        return create_engine("en-US")
    except OcrUnavailable as exc:
        pytest.skip(f"no Windows OCR engine: {exc}")


@pytest.fixture(scope="module")
def catalog():
    found = VoiceCatalog()
    if found.is_empty:
        pytest.skip("no Piper voices installed; run: gamevoice download --starter")
    return found


class TestOcrChain:
    def test_reads_a_synthetic_dialogue_box(self, ocr):
        frame = render_dialogue(["Elena: The bridge will not hold much longer."])
        result = ocr.recognize(preprocess(frame))
        assert "bridge" in result.text.lower()

    def test_attributes_the_speaker(self, ocr):
        frame = render_dialogue(["Elena: The bridge will not hold much longer."])
        utterance = parse(ocr.recognize(preprocess(frame)))
        assert utterance is not None
        assert utterance.speaker == "Elena"
        assert "bridge" in utterance.text.lower()
        assert "elena" not in utterance.text.lower()

    def test_reads_a_name_box_layout(self, ocr):
        frame = render_dialogue(["Marcus", "We were followed out of the city."])
        utterance = parse(ocr.recognize(preprocess(frame)))
        assert utterance is not None
        assert utterance.speaker == "Marcus"
        assert "followed" in utterance.text.lower()

    def test_blank_frame_is_detected_without_ocr(self):
        flat = np.zeros((120, 400, 4), dtype=np.uint8)
        assert looks_blank(flat)

    def test_a_rendered_frame_is_not_blank(self):
        assert not looks_blank(render_dialogue(["Something is written here."]))


class TestLocatorChain:
    def test_finds_dialogue_outside_the_bottom_band(self, ocr):
        from gamevoice.capture import WindowInfo
        from gamevoice.config import CaptureSettings, DetectSettings, Region
        from gamevoice.locator import DialogueLocator

        frame = render_dialogue(
            ["Elena: The bridge will not hold much longer."],
            canvas=(1280, 720),
            at=(200, 40),
        )

        class StubGrabber:
            def grab(self, region):
                return frame

        window = WindowInfo(1, "Game", "game.exe", Region(0, 0, 1280, 720))
        locator = DialogueLocator(CaptureSettings(), DetectSettings(), ocr, StubGrabber())
        assert locator.scan(window, time.time()) is not None
        box = locator.tracked
        assert box is not None
        # The box sits in the top quarter; the old bottom band never saw it.
        assert box.top < 360
        assert box.top < 200
        assert locator.region_for(window) is not None


class TestVoiceChain:
    def test_two_characters_get_different_voices(self, catalog):
        router = VoiceRouter(catalog, game_id="pipeline-test")
        male = router.resolve("Sir Roland")
        female = router.resolve("Lady Isolde")
        assert male.voice.key != female.voice.key
        assert male.voice.gender == "male"
        assert female.voice.gender == "female"

    def test_synthesis_produces_audio(self, catalog):
        from gamevoice.tts import PiperEngine, SynthesisError

        try:
            engine = PiperEngine()
        except SynthesisError as exc:
            pytest.skip(str(exc))

        router = VoiceRouter(catalog, game_id="pipeline-test")
        clip = engine.synthesize(
            "The bridge will not hold much longer.", router.resolve("Elena")
        )
        assert clip is not None
        assert clip.audio.dtype == np.int16
        assert 0.5 < clip.duration < 12.0
        assert np.abs(clip.audio).max() > 500  # not silence

    def test_the_same_line_in_two_voices_differs(self, catalog):
        from gamevoice.tts import PiperEngine, SynthesisError

        try:
            engine = PiperEngine()
        except SynthesisError as exc:
            pytest.skip(str(exc))

        router = VoiceRouter(catalog, game_id="pipeline-test")
        line = "We should not have come this way."
        first = engine.synthesize(line, router.resolve("Sir Roland"))
        second = engine.synthesize(line, router.resolve("Lady Isolde"))
        assert first is not None and second is not None
        shortest = min(first.audio.size, second.audio.size)
        difference = np.abs(
            first.audio[:shortest].astype(np.int32)
            - second.audio[:shortest].astype(np.int32)
        ).mean()
        assert difference > 100
