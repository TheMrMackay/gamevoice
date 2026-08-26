"""Skipping lines, pixel-change triggering, and profile management."""
from __future__ import annotations

import json

import numpy as np
import pytest

from gamevoice.audio import AudioPlayer
from gamevoice.capture import ChangeDetector
from gamevoice.config import (
    Profile,
    delete_profile,
    is_shipped,
    list_profiles,
    load_profile_file,
    save_profile,
)


def frame(value: int, width: int = 200, height: int = 60) -> np.ndarray:
    return np.full((height, width, 4), value, dtype=np.uint8)


class TestSkipLine:
    """A long line is several clips sharing a group; skip drops all of them."""

    def drain(self, player: AudioPlayer, frames: int, blocks: int) -> np.ndarray:
        collected = []
        for _ in range(blocks):
            out = np.zeros((frames, 1), dtype=np.float32)
            player._callback(out, frames, None, None)
            collected.append(out.reshape(-1).copy())
        return np.concatenate(collected)

    def player_with_two_lines(self) -> AudioPlayer:
        player = AudioPlayer()
        player.set_volume(1.0)
        # Line 1: three pieces. Line 2: one piece.
        for _ in range(3):
            player.enqueue(np.full(100, 1.0, dtype=np.float32), player.sample_rate, group=1)
        player.enqueue(np.full(100, -1.0, dtype=np.float32), player.sample_rate, group=2)
        return player

    def test_skip_drops_every_piece_of_the_current_line(self):
        player = self.player_with_two_lines()
        player._callback(np.zeros((40, 1), dtype=np.float32), 40, None, None)

        assert player.skip_group() is True
        rest = self.drain(player, frames=50, blocks=6)
        assert not np.any(np.isclose(rest, 1.0)), "part of the skipped line still played"

    def test_the_next_line_still_plays_after_a_skip(self):
        player = self.player_with_two_lines()
        player._callback(np.zeros((40, 1), dtype=np.float32), 40, None, None)
        player.skip_group()

        rest = self.drain(player, frames=50, blocks=6)
        assert int(np.sum(np.isclose(rest, -1.0))) == 100

    def test_skip_differs_from_stop(self):
        """stop() clears everything; skip_group() only abandons one line."""
        player = self.player_with_two_lines()
        player._callback(np.zeros((40, 1), dtype=np.float32), 40, None, None)
        player.stop()
        assert not player.is_busy

        player = self.player_with_two_lines()
        player._callback(np.zeros((40, 1), dtype=np.float32), 40, None, None)
        player.skip_group()
        assert player.is_busy, "skip should leave the queued line alone"

    def test_skipping_with_nothing_playing_reports_false(self):
        assert AudioPlayer().skip_group() is False

    def test_an_explicit_group_can_be_skipped(self):
        player = self.player_with_two_lines()
        player._callback(np.zeros((40, 1), dtype=np.float32), 40, None, None)
        assert player.skip_group(2) is True

        rest = self.drain(player, frames=50, blocks=8)
        assert not np.any(np.isclose(rest, -1.0)), "line 2 should have been dropped"
        assert np.any(np.isclose(rest, 1.0)), "line 1 should still be playing"


class TestChangeDetector:
    def test_the_first_frame_always_counts_as_a_change(self):
        assert ChangeDetector().changed(frame(40), now=0.0) is True

    def test_an_identical_frame_is_not_a_change(self):
        detector = ChangeDetector()
        detector.changed(frame(40), now=0.0)
        assert detector.changed(frame(40), now=0.1) is False

    def test_a_repainted_frame_is_a_change(self):
        detector = ChangeDetector()
        detector.changed(frame(40), now=0.0)
        assert detector.changed(frame(200), now=0.1) is True

    def test_faint_noise_is_ignored(self):
        """Grain and dithering must not trigger a re-read on every frame."""
        detector = ChangeDetector(tolerance=10)
        base = frame(120)
        detector.changed(base, now=0.0)
        noisy = base.astype(np.int16) + 4
        assert detector.changed(noisy.astype(np.uint8), now=0.1) is False

    def test_a_small_patch_of_new_text_is_detected(self):
        detector = ChangeDetector(threshold=0.004)
        base = frame(20, width=400, height=200)
        detector.changed(base, now=0.0)

        with_text = base.copy()
        with_text[90:110, 20:380, :3] = 240  # a line of bright text
        assert detector.changed(with_text, now=0.1) is True

    def test_it_reads_again_after_the_idle_timeout(self):
        detector = ChangeDetector(max_idle=5.0)
        detector.changed(frame(60), now=0.0)
        assert detector.changed(frame(60), now=1.0) is False
        assert detector.changed(frame(60), now=6.0) is True

    def test_the_idle_timeout_can_be_disabled(self):
        detector = ChangeDetector(max_idle=0.0)
        detector.changed(frame(60), now=0.0)
        assert detector.changed(frame(60), now=10_000.0) is False

    def test_reset_forgets_the_previous_frame(self):
        detector = ChangeDetector()
        detector.changed(frame(60), now=0.0)
        detector.reset()
        assert detector.changed(frame(60), now=0.1) is True

    def test_a_resized_region_is_handled(self):
        detector = ChangeDetector()
        detector.changed(frame(60, width=200), now=0.0)
        assert detector.changed(frame(60, width=900), now=0.1) is True

    def test_an_empty_frame_is_not_a_change(self):
        assert ChangeDetector().changed(np.zeros((0, 0, 4), np.uint8), now=0.0) is False


class TestProfileManagement:
    @pytest.fixture(autouse=True)
    def isolated_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GAMEVOICE_HOME", str(tmp_path))
        yield tmp_path

    def test_a_user_profile_can_be_deleted(self, isolated_home):
        profile = Profile(name="my-game", title="My Game")
        path = save_profile(profile)
        assert path.is_file()

        assert delete_profile(profile) == "deleted"
        assert not path.is_file()

    def test_deleting_twice_is_not_an_error(self):
        profile = Profile(name="my-game")
        save_profile(profile)
        assert delete_profile(profile) == "deleted"
        assert delete_profile(profile) == "unchanged"

    def test_a_shipped_profile_resets_rather_than_deletes(self):
        """Removing the user copy restores the built-in one, not nothing."""
        shipped = list_profiles()["default"]
        assert is_shipped(shipped)

        shipped.capture.fps = 3.0
        save_profile(shipped)
        assert delete_profile(shipped) == "reset"
        assert "default" in list_profiles(), "the built-in profile vanished"

    def test_a_user_copy_shadows_the_shipped_profile(self):
        shipped = list_profiles()["default"]
        shipped.capture.bottom_band = 0.75
        save_profile(shipped)
        assert list_profiles()["default"].capture.bottom_band == 0.75

        delete_profile(shipped)
        assert list_profiles()["default"].capture.bottom_band != 0.75

    def test_renaming_writes_a_new_file(self):
        profile = Profile(name="before", title="Before")
        old_path = save_profile(profile)

        profile.name = "after"
        new_path = save_profile(profile)
        assert new_path != old_path
        assert load_profile_file(new_path).name == "after"

    def test_edited_match_rules_survive_a_round_trip(self):
        profile = Profile(
            name="matched",
            match_exe=["thegame.exe", "launcher.exe"],
            match_title=["The Game"],
        )
        restored = load_profile_file(save_profile(profile))
        assert restored.match_exe == ["thegame.exe", "launcher.exe"]
        assert restored.match_title == ["The Game"]

    def test_new_capture_fields_round_trip(self):
        profile = Profile(name="triggered")
        profile.capture.trigger = "timer"
        profile.capture.change_threshold = 0.02
        profile.capture.max_idle_seconds = 3.0
        restored = load_profile_file(save_profile(profile))
        assert restored.capture.trigger == "timer"
        assert restored.capture.change_threshold == 0.02
        assert restored.capture.max_idle_seconds == 3.0

    def test_shipped_profiles_declare_a_valid_trigger(self):
        from gamevoice.config import SHIPPED_PROFILES

        for path in SHIPPED_PROFILES.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["capture"]["trigger"] in ("change", "timer")
