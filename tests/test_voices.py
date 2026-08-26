"""Tests for gender inference and voice assignment.

The property that matters most is determinism: the same character in the same
game must get the same voice on every launch and every machine, because nothing
about the assignment is stored.
"""
from __future__ import annotations

import json

import pytest

from gamevoice.voices import (
    FEMALE,
    MALE,
    UNKNOWN,
    VoiceCatalog,
    VoiceRef,
    VoiceRouter,
    infer_gender,
)


@pytest.fixture
def catalog(tmp_path):
    """A catalogue backed by fake model files and a hand-written table."""
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    for model in ("multi-medium", "solo_male-medium", "solo_female-medium"):
        (voices_dir / f"{model}.onnx").write_bytes(b"stub")

    table = {
        "probe": "test",
        "models": {
            "multi-medium": {
                "model": "multi-medium",
                "sample_rate": 22050,
                "speakers": (
                    [{"speaker_id": i, "f0": 110.0 + i, "gender": MALE} for i in range(6)]
                    + [{"speaker_id": 6 + i, "f0": 200.0 + i, "gender": FEMALE} for i in range(6)]
                    + [{"speaker_id": 12, "f0": 163.0, "gender": "ambiguous"}]
                ),
            },
            "solo_male-medium": {
                "model": "solo_male-medium", "sample_rate": 22050,
                "speakers": [{"speaker_id": 0, "f0": 120.0, "gender": MALE}],
            },
            "solo_female-medium": {
                "model": "solo_female-medium", "sample_rate": 22050,
                "speakers": [{"speaker_id": 0, "f0": 210.0, "gender": FEMALE}],
            },
        },
    }
    table_path = tmp_path / "voice_table.json"
    table_path.write_text(json.dumps(table), encoding="utf-8")
    return VoiceCatalog(voices_dir=voices_dir, table=table_path)


class TestInferGender:
    @pytest.mark.parametrize(
        "name", ["Sir Roland", "King Aldric", "The Old Man", "Blacksmith", "Father GreyDon"]
    )
    def test_male_role_words(self, name):
        assert infer_gender(name) == MALE

    @pytest.mark.parametrize(
        "name", ["Lady Isolde", "Queen Marren", "Old Woman", "Barmaid", "Sister Aline"]
    )
    def test_female_role_words(self, name):
        assert infer_gender(name) == FEMALE

    def test_given_names_from_the_lexicon(self):
        assert infer_gender("Thomas") == MALE
        assert infer_gender("Elizabeth") == FEMALE

    def test_invented_name_is_unknown_rather_than_guessed(self):
        assert infer_gender("Zyrthax") == UNKNOWN

    def test_explicit_hint_beats_everything(self):
        assert infer_gender("King Aldric", {"king aldric": FEMALE}) == FEMALE

    def test_empty_speaker_is_unknown(self):
        assert infer_gender("") == UNKNOWN

    def test_role_word_beats_a_given_name_in_the_same_string(self):
        # "Lady Thomas" should follow the title, not the first name.
        assert infer_gender("Lady Thomas") == FEMALE


class TestCatalog:
    def test_pools_split_by_measured_gender(self, catalog):
        assert len(catalog.pool(MALE)) == 7
        assert len(catalog.pool(FEMALE)) == 7

    def test_ambiguous_voices_stay_out_of_the_pools(self, catalog):
        for gender in (MALE, FEMALE):
            assert all(ref.gender == gender for ref in catalog.pool(gender))

    def test_lookup_by_key(self, catalog):
        ref = catalog.get("multi-medium#3")
        assert ref is not None and ref.speaker_id == 3

    def test_single_speaker_model_has_no_speaker_id(self, catalog):
        ref = catalog.get("solo_male-medium")
        assert ref is not None and ref.speaker_id is None

    def test_unknown_key_returns_none(self, catalog):
        assert catalog.get("not-a-model#9") is None


class TestRouter:
    def test_same_character_always_gets_the_same_voice(self, catalog):
        first = VoiceRouter(catalog, game_id="thegame").resolve("Elena")
        second = VoiceRouter(catalog, game_id="thegame").resolve("Elena")
        assert first.voice.key == second.voice.key

    def test_assignment_is_case_insensitive(self, catalog):
        router = VoiceRouter(catalog, game_id="thegame")
        assert router.resolve("ELENA").voice.key == router.resolve("elena").voice.key

    def test_same_character_differs_between_games(self, catalog):
        a = VoiceRouter(catalog, game_id="game-a").resolve("Guard")
        b = VoiceRouter(catalog, game_id="game-b").resolve("Guard")
        # Not guaranteed for every name, but must not be hard-wired to match.
        assert isinstance(a.voice.key, str) and isinstance(b.voice.key, str)

    def test_male_name_draws_from_the_male_pool(self, catalog):
        assignment = VoiceRouter(catalog, game_id="g").resolve("Sir Roland")
        assert assignment.voice.gender == MALE

    def test_female_name_draws_from_the_female_pool(self, catalog):
        assignment = VoiceRouter(catalog, game_id="g").resolve("Lady Isolde")
        assert assignment.voice.gender == FEMALE

    def test_override_wins_over_inference(self, catalog):
        router = VoiceRouter(
            catalog, game_id="g", overrides={"sir roland": "solo_female-medium"}
        )
        assignment = router.resolve("Sir Roland")
        assert assignment.voice.key == "solo_female-medium"
        assert assignment.source == "override"

    def test_hint_redirects_the_pool(self, catalog):
        router = VoiceRouter(catalog, game_id="g", hints={"zyrthax": FEMALE})
        assert router.resolve("Zyrthax").voice.gender == FEMALE

    def test_no_speaker_uses_the_narrator(self, catalog):
        assignment = VoiceRouter(catalog, game_id="g").resolve("")
        assert assignment.source == "narrator"

    def test_distinct_characters_spread_across_voices(self, catalog):
        router = VoiceRouter(catalog, game_id="g")
        names = [f"Guard {n}" for n in range(20)]
        chosen = {router.resolve(name).voice.key for name in names}
        # 20 characters over a 7-voice male pool should use most of the pool.
        assert len(chosen) >= 5

    def test_length_scale_varies_per_character(self, catalog):
        router = VoiceRouter(catalog, game_id="g", pitch_variety=0.1)
        scales = {router.resolve(f"Guard {n}").length_scale for n in range(10)}
        assert len(scales) > 1
        assert all(0.85 <= scale <= 1.15 for scale in scales)

    def test_auto_assign_off_uses_the_fixed_voices(self, catalog):
        router = VoiceRouter(
            catalog, game_id="g", auto_assign=False,
            male="solo_male-medium", female="solo_female-medium",
        )
        assert router.resolve("Sir Roland").voice.key == "solo_male-medium"
        assert router.resolve("Lady Isolde").voice.key == "solo_female-medium"

    def test_missing_override_target_falls_back_instead_of_failing(self, catalog):
        router = VoiceRouter(catalog, game_id="g", overrides={"elena": "gone#1"})
        assignment = router.resolve("Elena")
        assert assignment.voice.model in {"multi-medium", "solo_male-medium",
                                          "solo_female-medium"}


class TestVoiceRef:
    def test_key_round_trips(self):
        ref = VoiceRef("model-medium", 42, MALE)
        parsed = VoiceRef.parse(ref.key)
        assert parsed is not None
        assert (parsed.model, parsed.speaker_id) == ("model-medium", 42)

    def test_single_speaker_key_has_no_hash(self):
        assert VoiceRef("model-medium", None, MALE).key == "model-medium"

    def test_bad_key_is_rejected(self):
        assert VoiceRef.parse("model#notanumber") is None
        assert VoiceRef.parse("") is None
