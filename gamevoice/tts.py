"""Speech synthesis.

Two engines, one interface. Both hand back 16-bit mono PCM so everything
downstream - device choice, volume, interruption - works the same either way.

* **Piper** is the real one: neural voices, 904 of them in a single model, and
  fast enough on CPU that it never competes with the game for the GPU. Measured
  on this machine at a real-time factor near 0.025, so a three-second line takes
  under a tenth of a second to produce.
* **SAPI** is the fallback that needs no download at all, so the app is useful
  the moment it is installed and while voices are still downloading.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .voices import VoiceAssignment, VoiceRef

log = logging.getLogger(__name__)

SAMPLE_RATE = 22050
# Loading a Piper model costs about a second and a few hundred MB, so keep a
# small cache. In practice one multi-speaker model covers most characters.
MODEL_CACHE_SIZE = 3
# A backstop, not a policy. Callers split long passages into chunks well under
# this (see dialogue.split_for_speech); anything reaching it is a bug, so it is
# logged rather than quietly cut.
MAX_SPEECH_CHARS = 5000


def _fit(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= MAX_SPEECH_CHARS:
        return text
    log.warning(
        "text of %d chars reached the synthesis backstop and was cut; it should "
        "have been split before now", len(text)
    )
    return text[:MAX_SPEECH_CHARS]


class SynthesisError(RuntimeError):
    """Raised when an engine cannot be started at all."""


@dataclass(frozen=True)
class Clip:
    audio: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.audio) / float(self.sample_rate or SAMPLE_RATE)


class TtsEngine:
    name = "base"

    def synthesize(self, text: str, assignment: VoiceAssignment) -> Clip | None:
        raise NotImplementedError

    def warm_up(self, assignment: VoiceAssignment) -> None:
        """Pay one-off costs before the first real line arrives."""

    def close(self) -> None:
        pass


class PiperEngine(TtsEngine):
    name = "piper"

    def __init__(self, rate: float = 1.0, volume: float = 1.0) -> None:
        try:
            from piper import PiperVoice, SynthesisConfig  # noqa: F401
        except ImportError as exc:
            raise SynthesisError(
                "Piper is not installed. Run: pip install piper-tts"
            ) from exc
        self._rate = max(0.5, min(rate, 2.0))
        self._volume = max(0.0, min(volume, 1.0))
        self._models: OrderedDict[str, object] = OrderedDict()
        self._lock = threading.Lock()

    def set_rate(self, rate: float) -> None:
        self._rate = max(0.5, min(rate, 2.0))

    def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(volume, 1.0))

    def _load(self, ref: VoiceRef):
        with self._lock:
            cached = self._models.get(ref.model)
            if cached is not None:
                self._models.move_to_end(ref.model)
                return cached

            path = ref.model_path()
            if not path.is_file():
                raise SynthesisError(
                    f"Voice model {path.name} is not in {path.parent}. "
                    "Download it from the Voices tab."
                )

            from piper import PiperVoice

            try:
                voice = PiperVoice.load(str(path))
            except Exception as exc:
                raise SynthesisError(
                    f"Could not load voice {ref.model}: {exc}"
                ) from exc

            self._models[ref.model] = voice
            while len(self._models) > MODEL_CACHE_SIZE:
                self._models.popitem(last=False)
            log.info("loaded voice model %s", ref.model)
            return voice

    def _config(self, assignment: VoiceAssignment):
        from piper import SynthesisConfig

        # length_scale stretches time, so it is the reciprocal of speed. The
        # per-character offset rides on top of the user's rate setting.
        length = (1.0 / self._rate) * max(0.6, min(assignment.length_scale, 1.6))
        return SynthesisConfig(
            speaker_id=assignment.voice.speaker_id,
            length_scale=length,
            volume=self._volume,
            normalize_audio=True,
        )

    def warm_up(self, assignment: VoiceAssignment) -> None:
        try:
            self.synthesize("Ready.", assignment)
        except SynthesisError as exc:
            log.warning("warm-up failed: %s", exc)

    def synthesize(self, text: str, assignment: VoiceAssignment) -> Clip | None:
        text = _fit(text)
        if not text or not assignment.voice.model:
            return None

        voice = self._load(assignment.voice)
        try:
            chunks = list(voice.synthesize(text, self._config(assignment)))
        except Exception as exc:
            log.error("synthesis failed for %r: %s", text[:60], exc)
            return None
        if not chunks:
            return None

        audio = np.concatenate(
            [np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in chunks]
        )
        rate = getattr(chunks[0], "sample_rate", None) or voice.config.sample_rate
        return Clip(audio, int(rate))

    def close(self) -> None:
        with self._lock:
            self._models.clear()


class SapiEngine(TtsEngine):
    """Windows' built-in voices. Always available, no download.

    Renders into a memory stream rather than speaking directly so its output
    joins the same playback queue as Piper's and honours the same device,
    volume and interrupt behaviour.
    """

    name = "sapi"
    _FORMAT_22K_16BIT_MONO = 22

    def __init__(self, rate: float = 1.0, volume: float = 1.0) -> None:
        self._rate = rate
        self._volume = max(0.0, min(volume, 1.0))
        self._lock = threading.Lock()
        self._local = threading.local()
        self._voice_tokens = self._enumerate()
        if not self._voice_tokens:
            raise SynthesisError("Windows reported no installed SAPI voices.")

    def _dispatch(self, prog_id: str):
        import pythoncom
        import win32com.client

        if not getattr(self._local, "com_ready", False):
            pythoncom.CoInitialize()
            self._local.com_ready = True
        return win32com.client.Dispatch(prog_id)

    def _enumerate(self) -> list[tuple[str, str]]:
        """(name, gender) for each installed voice, gender from SAPI metadata."""
        try:
            speaker = self._dispatch("SAPI.SpVoice")
        except Exception as exc:
            raise SynthesisError(f"Could not start SAPI: {exc}") from exc

        found = []
        try:
            for token in speaker.GetVoices():
                name = token.GetDescription()
                try:
                    gender = token.GetAttribute("Gender").lower()
                except Exception:
                    gender = "unknown"
                found.append((name, gender))
        except Exception as exc:
            log.error("could not enumerate SAPI voices: %s", exc)
        return found

    def available(self) -> list[tuple[str, str]]:
        return list(self._voice_tokens)

    def _pick_token(self, speaker, gender: str):
        tokens = list(speaker.GetVoices())
        for token in tokens:
            try:
                if token.GetAttribute("Gender").lower() == gender:
                    return token
            except Exception:
                continue
        return tokens[0] if tokens else None

    def set_rate(self, rate: float) -> None:
        self._rate = rate

    def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(volume, 1.0))

    def synthesize(self, text: str, assignment: VoiceAssignment) -> Clip | None:
        text = _fit(text)
        if not text:
            return None

        with self._lock:
            try:
                speaker = self._dispatch("SAPI.SpVoice")
                stream = self._dispatch("SAPI.SpMemoryStream")
                fmt = self._dispatch("SAPI.SpAudioFormat")
                fmt.Type = self._FORMAT_22K_16BIT_MONO
                stream.Format = fmt
                speaker.AudioOutputStream = stream

                token = self._pick_token(speaker, assignment.gender)
                if token is not None:
                    speaker.Voice = token
                # SAPI rate is -10..10 on a roughly logarithmic scale.
                speaker.Rate = int(max(-10, min((self._rate - 1.0) * 10, 10)))
                speaker.Volume = int(self._volume * 100)
                speaker.Speak(text, 0)

                raw = bytes(bytearray(stream.GetData()))
            except Exception as exc:
                log.error("SAPI synthesis failed for %r: %s", text[:60], exc)
                return None

        if not raw:
            return None
        return Clip(np.frombuffer(raw, dtype=np.int16), SAMPLE_RATE)


def create_engine(name: str, rate: float = 1.0, volume: float = 1.0) -> TtsEngine:
    """Build the requested engine, falling back to SAPI if Piper cannot start."""
    if name == "sapi":
        return SapiEngine(rate, volume)
    try:
        return PiperEngine(rate, volume)
    except SynthesisError as exc:
        log.warning("Piper unavailable (%s); falling back to Windows voices", exc)
        return SapiEngine(rate, volume)
