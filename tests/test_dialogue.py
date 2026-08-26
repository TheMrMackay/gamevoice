"""Tests for the two decisions that decide whether the product feels right:
when a line is finished, and who said it."""
from __future__ import annotations

import pytest

from gamevoice.config import DetectSettings
from gamevoice.dialogue import (
    SPEECH_CHUNK_CHARS,
    Stabilizer,
    clean_for_speech,
    normalize,
    parse,
    speaker_from_layout,
    split_for_speech,
    split_speaker,
)
from gamevoice.ocr import OcrLine, OcrOutput


def out(*lines: str) -> OcrOutput:
    """An OcrOutput with plausible geometry, one line per row."""
    built = tuple(
        OcrLine(text=text, top=float(i * 40), left=10.0, height=30.0)
        for i, text in enumerate(lines)
    )
    return OcrOutput("\n".join(lines), built)


class TestSpeakerSplit:
    @pytest.mark.parametrize(
        "raw,speaker,body",
        [
            ("Aria: We should go now.", "Aria", "We should go now."),
            ("CAPTAIN: Hold the line.", "Captain", "Hold the line."),
            ("[Merchant] Finest steel in the province.", "Merchant", "Finest steel in the province."),
            ("(Old Woman) Mind the road after dark.", "Old Woman", "Mind the road after dark."),
            ("Marcus - I told you it was a trap.", "Marcus", "I told you it was a trap."),
        ],
    )
    def test_extracts_name_and_body(self, raw, speaker, body):
        assert split_speaker(raw) == (speaker, body)

    @pytest.mark.parametrize(
        "raw",
        [
            "Objective: Reach the north gate.",
            "Warning: structural failure imminent.",
            "Note: this door is locked.",
            "12:30: the train departs.",
            "Level: 14",
        ],
    )
    def test_interface_labels_are_not_speakers(self, raw):
        speaker, body = split_speaker(raw)
        assert speaker == ""
        assert body == raw

    def test_plain_sentence_has_no_speaker(self):
        assert split_speaker("The gate groans open.") == ("", "The gate groans open.")

    def test_overlong_prefix_is_rejected(self):
        raw = "A very long stretch of narration that happens to contain: a colon."
        assert split_speaker(raw)[0] == ""


class TestLayoutSpeaker:
    def test_short_name_line_above_body(self):
        speaker, body = speaker_from_layout(out("Elena", "The bridge will not hold much longer."))
        assert speaker == "Elena"
        assert body == "The bridge will not hold much longer."

    def test_punctuated_first_line_is_dialogue_not_a_name(self):
        speaker, _ = speaker_from_layout(out("Run.", "They are already inside the walls."))
        assert speaker == ""

    def test_single_line_has_no_layout_speaker(self):
        speaker, _ = speaker_from_layout(out("Just one line of text here."))
        assert speaker == ""

    def test_body_shorter_than_name_is_rejected(self):
        speaker, _ = speaker_from_layout(out("Bartholomew", "Yes"))
        assert speaker == ""


class TestCleanForSpeech:
    def test_strips_button_prompts(self):
        assert "continue" not in clean_for_speech("Well met. [E] Continue").lower()

    def test_strips_press_prompts(self):
        cleaned = clean_for_speech("The vault is sealed. Press SPACE to advance")
        assert "space" not in cleaned.lower()
        assert "vault is sealed" in cleaned

    def test_prompt_only_text_becomes_empty(self):
        assert clean_for_speech("[E] Continue") == ""

    def test_custom_pattern_is_applied(self):
        assert "autosave" not in clean_for_speech(
            "Autosave complete. We ride at dawn.", (r"autosave complete\.",)
        ).lower()


class TestSplitForSpeech:
    """Long passages must be spoken in full. Truncation was the bug."""

    SENTENCE = "The scouts came back before dawn with word of a second column. "

    def test_short_text_is_one_chunk(self):
        assert split_for_speech("Hold the gate.") == ["Hold the gate."]

    def test_empty_text_yields_nothing(self):
        assert split_for_speech("") == []
        assert split_for_speech("   ") == []

    @pytest.mark.parametrize("repeats", [2, 5, 12, 40])
    def test_no_word_is_ever_lost(self, repeats):
        text = (self.SENTENCE * repeats).strip()
        chunks = split_for_speech(text)
        assert " ".join(chunks).split() == text.split()

    @pytest.mark.parametrize("repeats", [2, 5, 12, 40])
    def test_every_chunk_respects_the_limit(self, repeats):
        text = (self.SENTENCE * repeats).strip()
        assert all(len(chunk) <= SPEECH_CHUNK_CHARS for chunk in split_for_speech(text))

    def test_it_prefers_sentence_boundaries(self):
        text = " ".join(f"Sentence number {n} is here." for n in range(30))
        for chunk in split_for_speech(text):
            assert chunk.endswith(".") or len(chunk) > SPEECH_CHUNK_CHARS - 40

    def test_a_long_sentence_splits_on_clauses(self):
        text = ", ".join(f"clause number {n} runs on for a while" for n in range(30))
        chunks = split_for_speech(text)
        assert " ".join(chunks).split() == text.split()
        assert len(chunks) > 1

    def test_a_sentence_with_no_punctuation_still_splits(self):
        text = " ".join(["word"] * 400)
        chunks = split_for_speech(text)
        assert " ".join(chunks).split() == text.split()
        assert all(len(chunk) <= SPEECH_CHUNK_CHARS for chunk in chunks)

    def test_one_enormous_token_is_kept_not_dropped(self):
        text = "x" * 900
        chunks = split_for_speech(text)
        assert "".join(chunks) == text

    def test_custom_limit_is_honoured(self):
        text = (self.SENTENCE * 6).strip()
        chunks = split_for_speech(text, limit=100)
        assert all(len(chunk) <= 100 for chunk in chunks)
        assert " ".join(chunks).split() == text.split()


class TestLongTextIsNotTruncated:
    def settings(self, **kwargs) -> DetectSettings:
        base = dict(settle_frames=2, similarity_threshold=0.90, min_chars=3,
                    max_chars=4000, history=8)
        base.update(kwargs)
        return DetectSettings(**base)

    def test_a_long_passage_survives_the_stabilizer_intact(self):
        text = ("A long stretch of narration that keeps going. " * 20).strip()
        assert len(text) > 700, "fixture must exceed the old truncation point"

        stabilizer = Stabilizer(self.settings())
        emitted = None
        for _ in range(4):
            emitted = stabilizer.feed(text) or emitted
        assert emitted == text

    def test_a_block_over_the_ceiling_is_skipped_whole_not_cut(self):
        """Better to say nothing than to read half a sentence and stop."""
        stabilizer = Stabilizer(self.settings(max_chars=200))
        text = "Far too much text to be dialogue. " * 20
        for _ in range(5):
            assert stabilizer.feed(text) is None


class TestNormalize:
    def test_collapses_whitespace_and_quotes(self):
        assert normalize("  He   said  “no”\n ") == 'He said "no"'

    def test_removes_arrow_glyphs(self):
        assert "▼" not in normalize("Carry on ▼")


class TestParse:
    def test_builds_attributed_utterance(self):
        utterance = parse(out("Aria: The road is watched."))
        assert utterance is not None
        assert utterance.speaker == "Aria"
        assert utterance.text == "The road is watched."
        assert utterance.has_speaker

    def test_separate_name_region_wins(self):
        utterance = parse(out("The road is watched."), speaker_text="Aria")
        assert utterance is not None
        assert utterance.speaker == "Aria"

    def test_narration_has_no_speaker(self):
        utterance = parse(out("The lantern gutters and dies."))
        assert utterance is not None
        assert utterance.speaker == ""

    def test_empty_input_returns_none(self):
        assert parse(out("")) is None

    def test_prompt_only_input_returns_none(self):
        assert parse(out("[E] Continue")) is None


class TestStabilizer:
    def settings(self, **kwargs) -> DetectSettings:
        base = dict(settle_frames=2, similarity_threshold=0.90, min_chars=3, history=8)
        base.update(kwargs)
        return DetectSettings(**base)

    def test_waits_for_text_to_settle(self):
        stabilizer = Stabilizer(self.settings())
        assert stabilizer.feed("We ride at dawn.") is None
        assert stabilizer.feed("We ride at dawn.") is None
        assert stabilizer.feed("We ride at dawn.") == "We ride at dawn."

    def test_typewriter_effect_is_not_spoken_in_fragments(self):
        stabilizer = Stabilizer(self.settings())
        emitted = [
            stabilizer.feed(partial)
            for partial in ("We ri", "We ride at", "We ride at dawn.",
                            "We ride at dawn.", "We ride at dawn.")
        ]
        assert [e for e in emitted if e] == ["We ride at dawn."]

    def test_same_line_is_not_repeated(self):
        stabilizer = Stabilizer(self.settings())
        for _ in range(6):
            stabilizer.feed("Hold the gate.")
        again = [stabilizer.feed("Hold the gate.") for _ in range(4)]
        assert not any(again)

    def test_ocr_jitter_does_not_reset_the_settle_count(self):
        stabilizer = Stabilizer(self.settings())
        stabilizer.feed("The keep is lost, my lord.")
        # One character misread between frames must still count as settled.
        stabilizer.feed("The keep is Iost, my lord.")
        assert stabilizer.feed("The keep is lost, my lord.") is not None

    def test_paged_text_speaks_only_the_new_part(self):
        stabilizer = Stabilizer(self.settings())
        for _ in range(3):
            first = stabilizer.feed("The scouts returned at midnight.")
        assert first == "The scouts returned at midnight."

        grown = "The scouts returned at midnight. Only two of them."
        emitted = None
        for _ in range(3):
            emitted = stabilizer.feed(grown) or emitted
        assert emitted == "Only two of them."

    def test_too_short_is_ignored(self):
        stabilizer = Stabilizer(self.settings(min_chars=5))
        for _ in range(5):
            assert stabilizer.feed("Ok") is None

    def test_ignore_pattern_suppresses_a_line(self):
        stabilizer = Stabilizer(self.settings(ignore_patterns=[r"^loading\b"]))
        for _ in range(5):
            assert stabilizer.feed("Loading the next area") is None

    def test_reset_forgets_history(self):
        stabilizer = Stabilizer(self.settings())
        for _ in range(3):
            stabilizer.feed("Stand fast.")
        stabilizer.reset()
        emitted = None
        for _ in range(3):
            emitted = stabilizer.feed("Stand fast.") or emitted
        assert emitted == "Stand fast."

    def test_distinct_lines_both_emit(self):
        stabilizer = Stabilizer(self.settings())
        spoken = []
        for text in ("First thing said.", "Second thing said."):
            for _ in range(3):
                got = stabilizer.feed(text)
                if got:
                    spoken.append(got)
        assert spoken == ["First thing said.", "Second thing said."]
