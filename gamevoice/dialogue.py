"""Turning raw OCR output into finished, attributed lines of dialogue.

Two problems live here, and both decide whether the product feels right:

1. *When* is a block of text finished? Games type dialogue out a character at a
   time, repaint it every frame, and page it. Speaking too early clips the
   line; speaking on every sample repeats it.
2. *Who* is speaking? Most games put the name in the text ("NAME: hello") or in
   a separate box. Getting this right is what makes per-character voices work
   on a game nobody wrote a profile for.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .config import DetectSettings
from .ocr import OcrOutput

log = logging.getLogger(__name__)

# "NAME:", "NAME -", "NAME >" and the bracketed forms games use for captions.
_SPEAKER_COLON = re.compile(
    r"^\s*[\[\(<]?\s*([A-Za-z][A-Za-z0-9'’.\- ]{0,30}?)\s*[\]\)>]?\s*[:：]\s*(.+)$",
    re.DOTALL,
)
_SPEAKER_BRACKET = re.compile(
    r"^\s*[\[\(<]\s*([A-Za-z][A-Za-z0-9'’.\- ]{0,30}?)\s*[\]\)>]\s*(.+)$", re.DOTALL
)
_SPEAKER_DASH = re.compile(
    r"^\s*([A-Z][A-Za-z0-9'’.\- ]{0,30}?)\s+[—–-]{1,2}\s+(.+)$", re.DOTALL
)

# Interface furniture that reaches the OCR but is not dialogue.
_BUTTON_PROMPT = re.compile(
    r"\[(?:[A-Z0-9]{1,6}|[^\]]{1,12})\]\s*(?:continue|next|skip|advance|press|select)?",
    re.IGNORECASE,
)
_PRESS_PROMPT = re.compile(
    r"\b(?:press|hold|tap|click)\s+(?:the\s+)?[\[\(]?[A-Za-z0-9 ]{1,14}[\]\)]?"
    r"(?:\s+to\s+\w+)?",
    re.IGNORECASE,
)
_PAGE_MARK = re.compile(r"^\s*\(?\d{1,2}\s*[/of]{1,2}\s*\d{1,2}\)?\s*$", re.IGNORECASE)
_ARROWS = re.compile(r"[▼▲►◄▶◀→←↓↑»«▸]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Words that look like a speaker but label a UI panel.
_NOT_SPEAKERS = frozenset(
    """
    note notes warning caution tip tips hint hints objective objectives quest
    mission task goal error loading level score health ammo new tutorial
    location area zone chapter act part day time date weather status inventory
    item items gold exp xp hp mp save autosave checkpoint achievement trophy
    unlocked complete completed failed updated received acquired equipped
    subtitles subtitle caption captions menu options settings paused
    """.split()
)

_MIN_NAME_CHARS = 2
_MAX_NAME_WORDS = 4
_MAX_NAME_CHARS = 32
# A name line has to be clearly shorter than the speech under it.
_NAME_LINE_MAX = 28

# How much text to hand the synthesiser at once. Long passages are split into
# units this size and spoken in order, so nothing is dropped and the first
# words start sooner than they would if the whole block were synthesised first.
SPEECH_CHUNK_CHARS = 320

_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])[\"')\]]*\s+")
_CLAUSE_BREAK = re.compile(r"(?<=[,;:—–])\s+")


@dataclass(frozen=True)
class Utterance:
    """One finished thing for one voice to say."""

    speaker: str
    text: str
    raw: str

    @property
    def has_speaker(self) -> bool:
        return bool(self.speaker)


def normalize(text: str) -> str:
    """Collapse OCR noise so two readings of one line compare equal."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub(" ", text)
    text = _ARROWS.sub(" ", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_for_speech(text: str, strip_patterns: tuple[str, ...] = ()) -> str:
    """Remove interface text a voice should not read aloud."""
    for pattern in strip_patterns:
        try:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        except re.error as exc:
            log.warning("ignoring bad strip pattern %r: %s", pattern, exc)

    text = _BUTTON_PROMPT.sub(" ", text)
    text = _PRESS_PROMPT.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # A line that was only a prompt is now punctuation; treat it as empty.
    if not re.search(r"[A-Za-z0-9]", text):
        return ""
    return text


def _greedy_pack(pieces: list[str], limit: int) -> list[str]:
    """Join pieces into runs no longer than ``limit``, keeping their order."""
    packed: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= limit:
            current = f"{current} {piece}"
        else:
            packed.append(current)
            current = piece
    if current:
        packed.append(current)
    return packed


def split_for_speech(text: str, limit: int = SPEECH_CHUNK_CHARS) -> list[str]:
    """Break a long passage into speakable chunks, losing nothing.

    Truncating long text is the wrong answer - a codex entry, a chat message or
    a wall of narration should be read to the end, not cut off mid-sentence.
    Splitting happens at the largest natural boundary that fits: sentences
    first, then clauses, then words, so prosody survives wherever possible.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    for sentence in _greedy_pack(
        [s for s in _SENTENCE_BREAK.split(text) if s.strip()], limit
    ):
        if len(sentence) <= limit:
            chunks.append(sentence)
            continue

        for clause in _greedy_pack(
            [c for c in _CLAUSE_BREAK.split(sentence) if c.strip()], limit
        ):
            if len(clause) <= limit:
                chunks.append(clause)
                continue

            for run in _greedy_pack(clause.split(), limit):
                if len(run) <= limit:
                    chunks.append(run)
                else:
                    # A single unbroken token longer than the limit. Rare, but
                    # it must still be spoken rather than dropped.
                    chunks.extend(
                        run[i: i + limit] for i in range(0, len(run), limit)
                    )

    return [chunk for chunk in chunks if chunk.strip()]


def _plausible_name(name: str) -> bool:
    name = name.strip(" .'-")
    if not (_MIN_NAME_CHARS <= len(name) <= _MAX_NAME_CHARS):
        return False
    if len(name.split()) > _MAX_NAME_WORDS:
        return False
    if not re.search(r"[A-Za-z]", name):
        return False
    if any(word.lower().strip(".:") in _NOT_SPEAKERS for word in name.split()):
        return False
    # "12:30" and similar timestamps arrive as a colon match.
    if re.fullmatch(r"[\d\s.:-]+", name):
        return False
    return True


def split_speaker(text: str) -> tuple[str, str]:
    """Pull a "NAME: line" style speaker off the front of a block."""
    for pattern in (_SPEAKER_COLON, _SPEAKER_BRACKET, _SPEAKER_DASH):
        match = pattern.match(text)
        if not match:
            continue
        name, remainder = match.group(1).strip(), match.group(2).strip()
        if _plausible_name(name) and remainder:
            return _tidy_name(name), remainder
    return "", text


def _tidy_name(name: str) -> str:
    name = name.strip(" .:'-")
    # Games shout names in caps; title case reads better in a UI and hashes
    # the same either way because the router lowercases.
    if name.isupper() and len(name) > 3:
        return name.title()
    return name


def speaker_from_layout(output: OcrOutput) -> tuple[str, str]:
    """Detect a name rendered on its own line above the dialogue.

    Common in RPGs where the name sits in a small box. Requires the first line
    to be short, unpunctuated, and followed by something longer, so a genuine
    short sentence of dialogue is not mistaken for a label.
    """
    lines = [line for line in output.lines if line.text.strip()]
    if len(lines) < 2:
        return "", output.text

    head, *rest = lines
    name = head.text.strip()
    body = " ".join(line.text.strip() for line in rest)

    if len(name) > _NAME_LINE_MAX or not _plausible_name(name):
        return "", output.text
    if re.search(r"[.!?,;]$", name):
        return "", output.text
    if len(body) <= len(name):
        return "", output.text

    return _tidy_name(name), body


def parse(
    output: OcrOutput,
    speaker_text: str = "",
    strip_patterns: tuple[str, ...] = (),
) -> Utterance | None:
    """Build one utterance from an OCR reading, or None if there is nothing."""
    raw = normalize(output.text)
    if not raw:
        return None

    speaker = _tidy_name(normalize(speaker_text)) if speaker_text else ""
    body = raw

    if speaker and not _plausible_name(speaker):
        speaker = ""

    if not speaker:
        speaker, body = split_speaker(raw)
    if not speaker:
        speaker, body = speaker_from_layout(output)
        body = normalize(body)
        if not speaker:
            body = raw

    body = clean_for_speech(body, strip_patterns)
    if not body:
        return None

    return Utterance(speaker=speaker, text=body, raw=raw)


def similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


@dataclass
class Stabilizer:
    """Decides when on-screen text has stopped changing and is worth speaking.

    Holds a candidate until it survives ``settle_frames`` samples unchanged,
    which is what stops a typewriter effect being read out as fragments.
    """

    settings: DetectSettings
    _pending: str = ""
    _stable: int = 0
    _spoken: deque[str] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self._spoken = deque(maxlen=max(self.settings.history, 1))
        self._ignore = self._compile(self.settings.ignore_patterns)

    @staticmethod
    def _compile(patterns: list[str]) -> tuple[re.Pattern, ...]:
        compiled = []
        for pattern in patterns:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                log.warning("ignoring bad ignore pattern %r: %s", pattern, exc)
        return tuple(compiled)

    def reset(self) -> None:
        self._pending = ""
        self._stable = 0
        self._spoken.clear()

    def _should_ignore(self, text: str) -> bool:
        if _PAGE_MARK.match(text):
            return True
        return any(pattern.search(text) for pattern in self._ignore)

    def _already_spoken(self, text: str) -> bool:
        if text in self._spoken:
            return True
        threshold = self.settings.similarity_threshold
        return any(similarity(text, seen) >= threshold for seen in self._spoken)

    def _new_tail(self, text: str) -> str:
        """For paged text, the part that has appeared since the last read.

        Games that keep earlier sentences on screen and append to them would
        otherwise be re-read from the top every time a line is added.
        """
        for seen in reversed(self._spoken):
            if len(text) > len(seen) + 4 and text.startswith(seen):
                # Trim the separator that joined the two pages, but keep the
                # tail's own closing punctuation - it carries the prosody.
                return text[len(seen):].lstrip(" .,;:-—–").rstrip()
        return text

    def feed(self, text: str) -> str | None:
        """Offer the current screen text. Returns a line when one is ready."""
        candidate = normalize(text)

        if len(candidate) < self.settings.min_chars or self._should_ignore(candidate):
            self._pending = ""
            self._stable = 0
            return None

        if len(candidate) > self.settings.max_chars:
            # Not truncated. Text this long is a wall of interface rather than a
            # line of dialogue, so it is skipped whole and said so, instead of
            # being silently cut off in the middle of a sentence.
            log.info(
                "ignoring a %d character block (max_chars is %d); raise it if "
                "this game really does show passages that long",
                len(candidate), self.settings.max_chars,
            )
            self._pending = ""
            self._stable = 0
            return None

        if candidate == self._pending:
            self._stable += 1
        elif self._pending and (
            candidate.startswith(self._pending) or self._pending.startswith(candidate)
        ):
            # Still being typed out, or being erased. Not settled.
            self._pending = candidate
            self._stable = 0
        elif similarity(candidate, self._pending) >= self.settings.similarity_threshold:
            # Same line, one character read differently between frames.
            self._pending = candidate
            self._stable += 1
        else:
            self._pending = candidate
            self._stable = 0

        if self._stable < self.settings.settle_frames:
            return None
        if self._already_spoken(candidate):
            return None

        emit = self._new_tail(candidate)
        self._spoken.append(candidate)
        if emit != candidate:
            self._spoken.append(emit)
        self._stable = 0
        return emit or None
