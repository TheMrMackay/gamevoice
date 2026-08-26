"""Capture scaling and playback sequencing.

Two properties are asserted here because both were reported as defects:
text must be upscaled enough to read, and one spoken line must never be cut
off or overlaid by the next.
"""
from __future__ import annotations

import numpy as np
import pytest

from gamevoice.audio import AudioPlayer, resample
from gamevoice.ocr import (
    MAX_DIMENSION,
    AdaptiveUpscale,
    OcrLine,
    OcrOutput,
    clamp_scale,
    clipped_edges,
)


class TestClippedEdges:
    """Text pressed against an edge means the region is cutting the passage."""

    def output(self, *tops: tuple[float, float]) -> OcrOutput:
        lines = tuple(
            OcrLine(text="x", top=top, left=0.0, height=height) for top, height in tops
        )
        return OcrOutput("\n".join(l.text for l in lines), lines)

    def test_text_clear_of_both_edges_is_fine(self):
        assert clipped_edges(self.output((40.0, 30.0)), height=200) == []

    def test_text_at_the_top_is_reported(self):
        assert "top" in clipped_edges(self.output((1.0, 30.0)), height=200)

    def test_text_at_the_bottom_is_reported(self):
        assert "bottom" in clipped_edges(self.output((170.0, 30.0)), height=200)

    def test_both_edges_can_be_reported(self):
        found = clipped_edges(self.output((0.0, 30.0), (172.0, 28.0)), height=200)
        assert found == ["top", "bottom"]

    def test_no_lines_reports_nothing(self):
        assert clipped_edges(OcrOutput("", ()), height=200) == []

    def test_zero_height_is_handled(self):
        assert clipped_edges(self.output((0.0, 10.0)), height=0) == []


def lines(height: float, count: int = 3) -> tuple[OcrLine, ...]:
    return tuple(
        OcrLine(text="x", top=float(i * height), left=0.0, height=height)
        for i in range(count)
    )


class TestClampScale:
    def test_scale_is_kept_within_the_ocr_limit(self):
        # 2560 wide at 6x would be 15360, past what the recogniser accepts.
        scale = clamp_scale(2560, 400, 6.0)
        assert 2560 * scale <= MAX_DIMENSION

    def test_a_modest_scale_is_untouched(self):
        assert clamp_scale(800, 200, 2.5) == 2.5

    def test_never_downscales_below_one(self):
        assert clamp_scale(MAX_DIMENSION * 2, 100, 3.0) == 1.0

    def test_zero_size_is_handled(self):
        assert clamp_scale(0, 0, 3.0) == 1.0


class TestAdaptiveUpscale:
    def test_small_text_pushes_the_scale_up(self):
        scaler = AdaptiveUpscale(initial=1.0)
        before = scaler.scale
        # 10 px glyphs measured at 1x are far below the target.
        scaler.observe(lines(10.0), applied_scale=1.0)
        assert scaler.scale > before

    def test_large_text_pulls_the_scale_down(self):
        scaler = AdaptiveUpscale(initial=4.0)
        before = scaler.scale
        # 400 px measured at 4x means 100 px of source text: already plenty.
        scaler.observe(lines(400.0), applied_scale=4.0)
        assert scaler.scale < before

    @pytest.mark.parametrize("source_height", [8.0, 12.0, 20.0, 33.0])
    def test_it_lands_on_a_readable_glyph_height(self, source_height):
        """What matters is the height text ends up at, not the scale factor.

        The scaler stops adjusting inside a deadband, so it settles near the
        42 px target rather than exactly on it. Anything in this band reads
        well; below roughly 16 px accuracy falls away sharply.
        """
        scaler = AdaptiveUpscale(initial=1.0)
        for _ in range(30):
            applied = scaler.scale
            scaler.observe(lines(source_height * applied), applied)

        final_height = source_height * scaler.scale
        assert 32.0 <= final_height <= 55.0, f"settled at {final_height:.1f} px"

    def test_text_already_the_right_size_is_left_alone(self):
        scaler = AdaptiveUpscale(initial=2.0)
        for _ in range(10):
            scaler.observe(lines(42.0 * scaler.scale / 2.0 * 2.0 / 2.0), 2.0)
        assert 1.0 <= scaler.scale <= 6.0

    def test_the_maximum_is_respected(self):
        scaler = AdaptiveUpscale(initial=1.0, maximum=3.0)
        for _ in range(40):
            scaler.observe(lines(2.0 * scaler.scale), scaler.scale)
        assert scaler.scale <= 3.0

    def test_the_minimum_is_respected(self):
        scaler = AdaptiveUpscale(initial=4.0, minimum=1.5)
        for _ in range(40):
            scaler.observe(lines(900.0 * scaler.scale), scaler.scale)
        assert scaler.scale >= 1.5

    def test_a_fixed_scaler_never_moves(self):
        scaler = AdaptiveUpscale(initial=2.0, minimum=2.0, maximum=2.0)
        scaler.observe(lines(4.0), 2.0)
        assert scaler.scale == 2.0

    def test_empty_output_is_ignored(self):
        scaler = AdaptiveUpscale(initial=2.0)
        scaler.observe((), 2.0)
        assert scaler.scale == 2.0 and scaler.samples == 0

    def test_degenerate_applied_scale_is_ignored(self):
        scaler = AdaptiveUpscale(initial=2.0)
        scaler.observe(lines(20.0), applied_scale=0.0)
        assert scaler.scale == 2.0


class TestPlaybackSequencing:
    """The player is driven directly; no audio device is opened."""

    def drain(self, player: AudioPlayer, frames: int, blocks: int) -> np.ndarray:
        collected = []
        for _ in range(blocks):
            out = np.zeros((frames, 1), dtype=np.float32)
            player._callback(out, frames, None, None)
            collected.append(out.reshape(-1).copy())
        return np.concatenate(collected)

    def test_two_lines_play_one_after_the_other(self):
        player = AudioPlayer()
        first = np.full(100, 0.5, dtype=np.float32)
        second = np.full(100, -0.5, dtype=np.float32)
        player.enqueue(first, player.sample_rate)
        player.enqueue(second, player.sample_rate)

        output = self.drain(player, frames=50, blocks=6)
        # First clip in full, then the second in full, then silence.
        assert np.allclose(output[:100], 0.5 * player._gain)
        assert np.allclose(output[100:200], -0.5 * player._gain)
        assert np.allclose(output[200:], 0.0)

    def test_no_sample_of_one_line_lands_on_another(self):
        """Overlap would show up as a value that is neither clip's level."""
        player = AudioPlayer()
        player.set_volume(1.0)
        player.enqueue(np.full(80, 1.0, dtype=np.float32), player.sample_rate)
        player.enqueue(np.full(80, -1.0, dtype=np.float32), player.sample_rate)

        output = self.drain(player, frames=32, blocks=8)
        distinct = set(np.unique(np.round(output, 4)))
        assert distinct <= {1.0, -1.0, 0.0}, f"mixed samples found: {distinct}"

    def test_a_line_is_not_truncated_by_the_next_arriving(self):
        player = AudioPlayer()
        player.set_volume(1.0)
        player.enqueue(np.full(120, 1.0, dtype=np.float32), player.sample_rate)
        # A second line arrives while the first is still playing.
        out = np.zeros((40, 1), dtype=np.float32)
        player._callback(out, 40, None, None)
        player.enqueue(np.full(60, -1.0, dtype=np.float32), player.sample_rate)

        rest = self.drain(player, frames=40, blocks=6)
        played_first = 40 + int(np.sum(np.isclose(rest, 1.0)))
        assert played_first == 120, "the first line was cut short"
        assert int(np.sum(np.isclose(rest, -1.0))) == 60

    def test_stop_during_playback_clears_without_error(self):
        player = AudioPlayer()
        player.enqueue(np.full(500, 0.4, dtype=np.float32), player.sample_rate)
        out = np.zeros((64, 1), dtype=np.float32)
        player._callback(out, 64, None, None)
        player.stop()

        after = self.drain(player, frames=64, blocks=2)
        assert np.allclose(after, 0.0)
        assert not player.is_busy

    def test_int16_input_is_scaled_to_float(self):
        player = AudioPlayer()
        player.set_volume(1.0)
        player.enqueue(np.full(64, 16384, dtype=np.int16), player.sample_rate)
        output = self.drain(player, frames=64, blocks=1)
        assert np.allclose(output, 0.5, atol=0.01)

    def test_empty_clip_is_ignored(self):
        player = AudioPlayer()
        player.enqueue(np.zeros(0, dtype=np.int16), player.sample_rate)
        assert not player.is_busy


class TestResample:
    def test_rate_change_scales_the_length(self):
        audio = np.zeros(1000, dtype=np.float32)
        assert resample(audio, 22050, 44100).size == pytest.approx(2000, abs=2)

    def test_same_rate_is_a_passthrough(self):
        audio = np.arange(10, dtype=np.float32)
        assert resample(audio, 22050, 22050) is audio
