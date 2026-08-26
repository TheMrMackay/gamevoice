"""The voice catalogue, and the rule that decides who sounds like what.

Piper's ``libritts_r`` model carries 904 distinct speakers in one file, which is
what makes "a different voice per character" practical without a download per
character. It ships no gender metadata, so ``tools/build_voice_table.py``
measures each speaker's median pitch once and writes ``data/voice_table.json``;
this module reads that table and uses it to build a male pool and a female pool.

Assignment is deterministic: the same character name in the same game always
hashes to the same voice, on any machine, with nothing stored. That is what
lets an unprofiled game sound consistent the first time it is played.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import DATA_DIR, VOICES_DIR

log = logging.getLogger(__name__)

MALE = "male"
FEMALE = "female"
NEUTRAL = "neutral"
UNKNOWN = "unknown"

VOICE_TABLE = DATA_DIR / "voice_table.json"
NAME_TABLE = DATA_DIR / "name_genders.json"


@dataclass(frozen=True)
class VoiceRef:
    model: str
    speaker_id: int | None
    gender: str
    f0: float = 0.0

    @property
    def key(self) -> str:
        if self.speaker_id is None:
            return self.model
        return f"{self.model}#{self.speaker_id}"

    @property
    def label(self) -> str:
        pretty = self.model.replace("_", " ")
        if self.speaker_id is None:
            return f"{pretty} ({self.gender})"
        return f"{pretty} #{self.speaker_id} ({self.gender}, {self.f0:.0f} Hz)"

    def model_path(self) -> Path:
        return VOICES_DIR / f"{self.model}.onnx"

    @staticmethod
    def parse(key: str) -> "VoiceRef | None":
        if not key:
            return None
        model, _, sid = key.partition("#")
        if not model:
            return None
        try:
            speaker = int(sid) if sid else None
        except ValueError:
            return None
        return VoiceRef(model=model, speaker_id=speaker, gender=UNKNOWN)


class VoiceCatalog:
    """Every voice on disk, split into pools by measured gender."""

    def __init__(self, voices_dir: Path = VOICES_DIR, table: Path = VOICE_TABLE) -> None:
        self._voices_dir = voices_dir
        self._by_key: dict[str, VoiceRef] = {}
        self._pools: dict[str, list[VoiceRef]] = {MALE: [], FEMALE: [], NEUTRAL: []}
        self._load(table)

    def _load(self, table: Path) -> None:
        measured: dict[str, dict[int, dict]] = {}
        if table.is_file():
            try:
                payload = json.loads(table.read_text(encoding="utf-8"))
                for model, entry in payload.get("models", {}).items():
                    measured[model] = {
                        item["speaker_id"]: item for item in entry.get("speakers", [])
                    }
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                log.error("voice table %s unreadable: %s", table, exc)

        installed = sorted(p.stem for p in self._voices_dir.glob("*.onnx"))
        if not installed:
            log.warning("no Piper voices found in %s", self._voices_dir)

        for model in installed:
            speakers = measured.get(model)
            if not speakers:
                # Present but never measured: usable, gender unknown.
                ref = VoiceRef(model, None, UNKNOWN)
                self._by_key[ref.key] = ref
                continue
            single = len(speakers) == 1 and 0 in speakers
            for sid, item in speakers.items():
                ref = VoiceRef(
                    model=model,
                    speaker_id=None if single else sid,
                    gender=item.get("gender", UNKNOWN),
                    f0=float(item.get("f0", 0.0)),
                )
                self._by_key[ref.key] = ref
                if ref.gender in self._pools:
                    self._pools[ref.gender].append(ref)

        # Sort by pitch so a pool has a predictable, listenable ordering.
        for pool in self._pools.values():
            pool.sort(key=lambda ref: (ref.f0, ref.key))

        self._pools[NEUTRAL] = self._pools[MALE] + self._pools[FEMALE]
        log.info(
            "voice catalogue: %d male, %d female, %d total",
            len(self._pools[MALE]),
            len(self._pools[FEMALE]),
            len(self._by_key),
        )

    @property
    def is_empty(self) -> bool:
        return not self._by_key

    def all_voices(self) -> list[VoiceRef]:
        return sorted(self._by_key.values(), key=lambda ref: (ref.model, ref.f0))

    def pool(self, gender: str) -> list[VoiceRef]:
        pool = self._pools.get(gender) or self._pools.get(NEUTRAL) or []
        if pool:
            return pool
        return self.all_voices()

    def get(self, key: str) -> VoiceRef | None:
        found = self._by_key.get(key)
        if found is not None:
            return found
        parsed = VoiceRef.parse(key)
        # A key can name a model that is installed but unmeasured.
        if parsed and parsed.model_path().is_file():
            return parsed
        return None

    def default_for(self, gender: str) -> VoiceRef | None:
        pool = self.pool(gender)
        if not pool:
            return None
        # Middle of the pitch range is the least remarkable, so it makes the
        # safest default narrator or fallback.
        return pool[len(pool) // 2]


# --- gender inference -------------------------------------------------------

# Role and honorific words carry gender far more reliably than a fantasy given
# name does, and games label generic NPCs by role constantly ("Old Woman",
# "Guard Captain", "Barmaid"), so these are checked first.
_MALE_WORDS = frozenset(
    """
    mr sir lord king prince duke baron count emperor father brother son uncle
    grandfather husband boy man men male gentleman guy lad boyfriend nephew
    monk priest friar abbot bishop pope knight squire blacksmith barkeep
    fisherman huntsman woodsman spokesman craftsman chairman policeman
    postman milkman handyman salesman watchman swordsman marksman
    """.split()
)
_FEMALE_WORDS = frozenset(
    """
    mrs ms miss madam madame lady queen princess duchess baroness countess
    empress mother sister daughter aunt grandmother wife girl woman women
    female gentlewoman gal lass girlfriend niece nun priestess abbess
    barmaid waitress actress seamstress huntress sorceress enchantress
    witch matron maiden maid widow bride mistress governess
    """.split()
)
_MALE_SUFFIX_WORDS = ("man", "men", "boy", "lord", "king", "father", "sir")
_FEMALE_SUFFIX_WORDS = ("woman", "women", "girl", "lady", "queen", "mother", "maid")

_WORD_SPLIT = re.compile(r"[^A-Za-z]+")


@lru_cache(maxsize=1)
def _name_lexicon() -> dict[str, str]:
    if not NAME_TABLE.is_file():
        return {}
    try:
        payload = json.loads(NAME_TABLE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.error("name table %s unreadable: %s", NAME_TABLE, exc)
        return {}
    lexicon: dict[str, str] = {}
    for gender in (MALE, FEMALE):
        for name in payload.get(gender, []):
            lexicon[str(name).lower()] = gender
    return lexicon


def infer_gender(speaker: str, hints: dict[str, str] | None = None) -> str:
    """Best guess at a speaker's voice gender.

    Order matters: an explicit hint from the profile always wins, then role
    words, then a given-name lookup. Returns ``unknown`` rather than guessing
    when nothing matches - the router treats that as "any voice", which sounds
    better than confidently picking the wrong one.
    """
    if not speaker:
        return UNKNOWN

    lowered = speaker.lower().strip()
    if hints:
        direct = hints.get(lowered)
        if direct in (MALE, FEMALE, NEUTRAL):
            return direct

    words = [word for word in _WORD_SPLIT.split(lowered) if word]
    if not words:
        return UNKNOWN

    for word in words:
        if word in _MALE_WORDS:
            return MALE
        if word in _FEMALE_WORDS:
            return FEMALE

    for word in words:
        # "Fisherman", "Swordswoman" and friends, without listing every one.
        if any(word.endswith(suffix) for suffix in _FEMALE_SUFFIX_WORDS):
            return FEMALE
        if any(word.endswith(suffix) for suffix in _MALE_SUFFIX_WORDS):
            return MALE

    lexicon = _name_lexicon()
    for word in words:
        found = lexicon.get(word)
        if found:
            return found

    return UNKNOWN


# --- assignment -------------------------------------------------------------


def _stable_hash(*parts: str) -> int:
    """A hash that is the same on every run and every machine.

    Python's built-in hash is salted per process, so a character would get a
    different voice each time the app restarted.
    """
    digest = hashlib.blake2b(
        "\x1f".join(parts).encode("utf-8", "replace"), digest_size=8
    )
    return int.from_bytes(digest.digest(), "big")


@dataclass
class VoiceAssignment:
    voice: VoiceRef
    gender: str
    length_scale: float
    source: str  # "override" | "auto" | "narrator" | "fallback"


class VoiceRouter:
    """Maps a speaker name to a voice, consistently and without setup."""

    def __init__(
        self,
        catalog: VoiceCatalog,
        game_id: str = "",
        overrides: dict[str, str] | None = None,
        hints: dict[str, str] | None = None,
        narrator: str = "",
        male: str = "",
        female: str = "",
        pitch_variety: float = 0.06,
        auto_assign: bool = True,
    ) -> None:
        self._catalog = catalog
        self._game_id = game_id or "generic"
        self._overrides = {k.lower(): v for k, v in (overrides or {}).items()}
        self._hints = {k.lower(): v for k, v in (hints or {}).items()}
        self._pitch_variety = max(0.0, min(pitch_variety, 0.25))
        self._auto = auto_assign
        self._narrator = catalog.get(narrator) or catalog.default_for(NEUTRAL)
        self._male = catalog.get(male) or catalog.default_for(MALE)
        self._female = catalog.get(female) or catalog.default_for(FEMALE)
        self._cache: dict[str, VoiceAssignment] = {}

    @property
    def catalog(self) -> VoiceCatalog:
        return self._catalog

    def set_override(self, speaker: str, voice_key: str) -> None:
        self._overrides[speaker.lower()] = voice_key
        self._cache.pop(speaker.lower(), None)

    def set_hint(self, speaker: str, gender: str) -> None:
        self._hints[speaker.lower()] = gender
        self._cache.pop(speaker.lower(), None)

    def known_speakers(self) -> list[str]:
        return sorted(self._cache)

    def _fixed_voice(self, gender: str) -> VoiceRef | None:
        if gender == MALE:
            return self._male
        if gender == FEMALE:
            return self._female
        return self._narrator

    def resolve(self, speaker: str) -> VoiceAssignment:
        key = (speaker or "").lower().strip()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        assignment = self._resolve_uncached(speaker, key)
        self._cache[key] = assignment
        return assignment

    def _resolve_uncached(self, speaker: str, key: str) -> VoiceAssignment:
        if not key:
            voice = self._narrator or self._catalog.default_for(NEUTRAL)
            return VoiceAssignment(voice, NEUTRAL, 1.0, "narrator") if voice else _silent()

        override = self._overrides.get(key)
        if override:
            voice = self._catalog.get(override)
            if voice is not None:
                return VoiceAssignment(voice, voice.gender, 1.0, "override")
            log.warning("override for %r names an unavailable voice %r", speaker, override)

        gender = infer_gender(speaker, self._hints)

        if not self._auto:
            voice = self._fixed_voice(gender)
            return (
                VoiceAssignment(voice, gender, 1.0, "fallback") if voice else _silent()
            )

        pool = self._catalog.pool(gender if gender in (MALE, FEMALE) else NEUTRAL)
        if not pool:
            voice = self._fixed_voice(gender)
            return (
                VoiceAssignment(voice, gender, 1.0, "fallback") if voice else _silent()
            )

        seed = _stable_hash(self._game_id, key)
        voice = pool[seed % len(pool)]

        # A small, stable speed offset makes two characters who land on the
        # same voice still sound like different people.
        spread = self._pitch_variety
        offset = (((seed >> 17) % 2001) / 1000.0 - 1.0) * spread
        return VoiceAssignment(voice, voice.gender or gender, 1.0 + offset, "auto")


def _silent() -> VoiceAssignment:
    return VoiceAssignment(VoiceRef("", None, UNKNOWN), UNKNOWN, 1.0, "fallback")
