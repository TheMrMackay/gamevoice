"""Profile loading and persistence.

The shipped-profile test exists because a profile that fails to parse degrades
silently to defaults - the app still starts, so a broken file is easy to miss.
"""
from __future__ import annotations

import json

import pytest

from gamevoice.config import (
    SHIPPED_PROFILES,
    AppSettings,
    CaptureSettings,
    Profile,
    Region,
    load_profile_file,
    safe_name,
)


class TestShippedProfiles:
    def test_every_shipped_profile_parses(self):
        files = sorted(SHIPPED_PROFILES.glob("*.json"))
        assert files, "no profiles are shipped"
        for path in files:
            raw = path.read_text(encoding="utf-8")
            json.loads(raw)  # fails loudly on a bad escape
            assert load_profile_file(path) is not None, f"{path.name} did not load"

    def test_a_default_profile_is_shipped(self):
        names = {load_profile_file(p).name for p in SHIPPED_PROFILES.glob("*.json")}
        assert "default" in names

    def test_shipped_regex_patterns_compile(self):
        import re

        for path in SHIPPED_PROFILES.glob("*.json"):
            profile = load_profile_file(path)
            for pattern in profile.detect.ignore_patterns + profile.detect.strip_patterns:
                re.compile(pattern)  # raises re.error if the escaping is wrong

    def test_default_profile_reads_the_bottom_of_the_screen(self):
        profile = load_profile_file(SHIPPED_PROFILES / "default.json")
        assert profile.capture.text_region is None, "default must be automatic"
        assert 0.1 <= profile.capture.bottom_band <= 0.6


class TestRoundTrip:
    def test_profile_survives_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GAMEVOICE_HOME", str(tmp_path))
        from gamevoice import config

        profile = Profile(
            name="test-game",
            title="Test Game",
            match_exe=["testgame.exe"],
            capture=CaptureSettings(text_region=Region(10, 20, 300, 80), fps=6.0),
            voice_overrides={"elena": "model-medium#7"},
            gender_hints={"zyrthax": "female"},
        )
        path = config.save_profile(profile)
        restored = config.load_profile_file(path)

        assert restored is not None
        assert restored.name == "test-game"
        assert restored.match_exe == ["testgame.exe"]
        assert restored.capture.text_region == Region(10, 20, 300, 80)
        assert restored.capture.fps == 6.0
        assert restored.capture.auto_found_region is None
        assert restored.voice_overrides == {"elena": "model-medium#7"}
        assert restored.gender_hints == {"zyrthax": "female"}

    def test_a_learned_region_survives_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GAMEVOICE_HOME", str(tmp_path))
        from gamevoice import config

        profile = Profile(
            name="test-game",
            capture=CaptureSettings(auto_found_region=Region(250, 500, 500, 120)),
        )
        path = config.save_profile(profile)
        restored = config.load_profile_file(path)

        assert restored is not None
        assert restored.capture.auto_found_region == Region(250, 500, 500, 120)

    def test_unknown_keys_are_ignored(self, tmp_path):
        path = tmp_path / "old.json"
        path.write_text(
            json.dumps({"name": "old", "title": "Old", "removed_setting": 42}),
            encoding="utf-8",
        )
        profile = load_profile_file(path)
        assert profile is not None and profile.name == "old"

    def test_malformed_json_returns_none_rather_than_raising(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"name": "bad", "detect": {"ignore_patterns": ["\\s"]}}',
                        encoding="utf-8")
        assert load_profile_file(path) is None

    def test_settings_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GAMEVOICE_HOME", str(tmp_path))
        settings = AppSettings(hotkey_toggle="ctrl+alt+p", engine="sapi")
        settings.save()
        assert AppSettings.load().hotkey_toggle == "ctrl+alt+p"
        assert AppSettings.load().engine == "sapi"

    def test_window_placement_defaults_to_unset(self):
        settings = AppSettings()
        assert settings.window_x is None and settings.window_y is None

    def test_window_placement_round_trips(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GAMEVOICE_HOME", str(tmp_path))
        settings = AppSettings(window_x=-1500, window_y=142, window_w=954, window_h=832)
        settings.save()
        restored = AppSettings.load()
        assert (restored.window_x, restored.window_y) == (-1500, 142)
        assert (restored.window_w, restored.window_h) == (954, 832)


class TestRegion:
    def test_tiny_region_is_invalid(self):
        assert not Region(0, 0, 4, 4).is_valid()

    def test_usable_region_is_valid(self):
        assert Region(10, 10, 400, 90).is_valid()

    def test_mss_shape(self):
        assert Region(1, 2, 3, 4).as_mss() == {
            "left": 1, "top": 2, "width": 3, "height": 4
        }


class TestSafeName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("The Game: Chapter II", "the-game-chapter-ii"),
            ("  spaces  ", "spaces"),
            ("../../etc/passwd", "etc-passwd"),
            ("!!!", "profile"),
        ],
    )
    def test_names_are_filesystem_safe(self, raw, expected):
        assert safe_name(raw) == expected
