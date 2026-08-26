"""Hotkey editing and honest voice reporting."""
from __future__ import annotations

import pytest

from gamevoice.config import AppSettings
from gamevoice.engine import SpokenLine
from gamevoice.dialogue import Utterance
from gamevoice.gui.hotkeys import HotkeyError, parse
from gamevoice.voices import VoiceAssignment, VoiceRef


class TestHotkeyParsing:
    @pytest.mark.parametrize(
        "spec,expected_key",
        [("ctrl+alt+v", ord("V")), ("ctrl+shift+f5", 0x74), ("alt+space", 0x20)],
    )
    def test_valid_specs_parse(self, spec, expected_key):
        _modifiers, key = parse(spec)
        assert key == expected_key

    def test_modifiers_combine(self):
        from gamevoice.gui.hotkeys import MOD_ALT, MOD_CONTROL, MOD_SHIFT

        modifiers, _key = parse("ctrl+alt+shift+k")
        assert modifiers == MOD_CONTROL | MOD_ALT | MOD_SHIFT

    @pytest.mark.parametrize("spec", ["", "ctrl+alt", "ctrl+v+x", "ctrl+notakey"])
    def test_bad_specs_are_rejected(self, spec):
        with pytest.raises(HotkeyError):
            parse(spec)

    def test_a_bare_key_needs_no_modifier(self):
        _modifiers, key = parse("f9")
        assert key == 0x78


class TestKeySequenceConversion:
    """The Settings tab captures a QKeySequence; registration needs a string."""

    @pytest.fixture(autouse=True)
    def qt(self):
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])

    @pytest.mark.parametrize(
        "spec", ["ctrl+alt+v", "ctrl+shift+f5", "alt+space", "ctrl+alt+shift+k"]
    )
    def test_spec_survives_a_round_trip(self, spec):
        from gamevoice.gui.settings_tab import sequence_to_spec, spec_to_sequence

        assert sequence_to_spec(spec_to_sequence(spec)) == spec

    def test_a_round_tripped_spec_still_registers(self, spec="ctrl+alt+j"):
        from gamevoice.gui.settings_tab import sequence_to_spec, spec_to_sequence

        parse(sequence_to_spec(spec_to_sequence(spec)))  # raises if malformed

    def test_an_empty_sequence_means_unbound(self):
        from PySide6.QtGui import QKeySequence

        from gamevoice.gui.settings_tab import sequence_to_spec

        assert sequence_to_spec(QKeySequence()) == ""

    def test_the_windows_key_is_translated(self):
        """Qt calls it Meta, the registration code calls it win."""
        from gamevoice.gui.settings_tab import sequence_to_spec, spec_to_sequence

        assert "win" in sequence_to_spec(spec_to_sequence("win+alt+p"))


class TestHotkeySettings:
    def test_the_new_actions_default_to_unbound(self):
        settings = AppSettings()
        assert settings.hotkey_skip == ""
        assert settings.hotkey_replay == ""

    def test_all_four_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GAMEVOICE_HOME", str(tmp_path))
        settings = AppSettings(
            hotkey_toggle="ctrl+alt+1",
            hotkey_stop="ctrl+alt+2",
            hotkey_skip="ctrl+alt+3",
            hotkey_replay="ctrl+alt+4",
        )
        settings.save()
        restored = AppSettings.load()
        assert restored.hotkey_skip == "ctrl+alt+3"
        assert restored.hotkey_replay == "ctrl+alt+4"

    def test_rebind_skips_unbound_entries(self):
        pytest.importorskip("PySide6")
        from gamevoice.gui.hotkeys import HotkeyManager

        manager = HotkeyManager()
        try:
            # Empty specs must be ignored rather than raising.
            failures = manager.rebind([("", lambda: None), ("   ", lambda: None)])
            assert failures == {}
        finally:
            manager.unregister_all()


class TestVoiceReporting:
    """A line must name the voice that was actually heard."""

    def assignment(self) -> VoiceAssignment:
        return VoiceAssignment(
            VoiceRef("en_US-libritts_r-medium", 42, "male", 120.0), "male", 1.0, "auto"
        )

    def test_spoken_line_prefers_the_engine_label(self):
        line = SpokenLine(
            Utterance("Roland", "Hold fast.", "Hold fast."),
            self.assignment(),
            voice_label="Microsoft David Desktop (male)",
        )
        assert line.voice == "Microsoft David Desktop (male)"

    def test_it_falls_back_to_the_router_choice(self):
        line = SpokenLine(
            Utterance("Roland", "Hold fast.", "Hold fast."), self.assignment()
        )
        assert line.voice == "en_US-libritts_r-medium#42"

    def test_the_base_engine_reports_the_router_voice(self):
        from gamevoice.tts import TtsEngine

        assert "libritts" in TtsEngine().describe(self.assignment())

    @pytest.mark.integration
    def test_sapi_reports_a_windows_voice_not_a_piper_one(self):
        """The bug this fixes: SAPI spoke, but the log named a Piper voice."""
        from gamevoice.tts import SapiEngine, SynthesisError

        try:
            engine = SapiEngine()
        except SynthesisError as exc:
            pytest.skip(str(exc))

        described = engine.describe(self.assignment())
        assert "libritts" not in described
        assert any(
            name in described for name, _gender in engine.available()
        ), f"described a voice that is not installed: {described}"

    @pytest.mark.integration
    def test_sapi_honours_the_requested_gender(self):
        from gamevoice.tts import SapiEngine, SynthesisError

        try:
            engine = SapiEngine()
        except SynthesisError as exc:
            pytest.skip(str(exc))
        if not any(g == "female" for _n, g in engine.available()):
            pytest.skip("no female Windows voice installed")

        female = VoiceAssignment(
            VoiceRef("en_US-libritts_r-medium", 7, "female", 210.0),
            "female", 1.0, "auto",
        )
        assert "female" in engine.describe(female)
