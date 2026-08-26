"""The running system: capture, recognise, attribute, speak.

Two threads do the work. The *reader* samples the screen and decides when a
line is finished; the *speaker* turns finished lines into audio. Splitting them
matters because synthesis of a long line must never stall the next screen
sample - if it did, the reader would miss dialogue while talking about the
previous one.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from . import profiles as profile_matching
from .audio import AudioPlayer
from .capture import (
    ChangeDetector,
    ScreenGrabber,
    WindowInfo,
    foreground_window,
    looks_blank,
    resolve_regions,
)
from .config import AppSettings, Profile
from .dialogue import Stabilizer, Utterance, parse, split_for_speech
from .ocr import (
    AdaptiveUpscale,
    OcrOutput,
    WindowsOcr,
    clamp_scale,
    create_engine as create_ocr,
    preprocess,
)
from .tts import SynthesisError, TtsEngine, create_engine as create_tts
from .voices import VoiceAssignment, VoiceCatalog, VoiceRouter

log = logging.getLogger(__name__)

WINDOW_POLL_SECONDS = 1.0
SYNTH_QUEUE_MAX = 16


@dataclass
class SpokenLine:
    utterance: Utterance
    assignment: VoiceAssignment
    # What the engine actually spoke with. Not always the router's choice: the
    # SAPI engine picks from Windows' own voices by gender instead.
    voice_label: str = ""
    at: float = field(default_factory=time.time)

    @property
    def voice(self) -> str:
        return self.voice_label or self.assignment.voice.key or "-"


@dataclass
class EngineStats:
    lines_seen: int = 0
    lines_spoken: int = 0
    chunks_spoken: int = 0
    frames_skipped: int = 0
    ocr_calls: int = 0
    ocr_ms: float = 0.0
    synth_ms: float = 0.0
    upscale: float = 0.0
    last_error: str = ""

    @property
    def average_ocr_ms(self) -> float:
        return self.ocr_ms / self.ocr_calls if self.ocr_calls else 0.0


class GameVoiceEngine:
    """Owns the worker threads and the state they share."""

    def __init__(
        self,
        catalog: VoiceCatalog,
        settings: AppSettings,
        profile: Profile | None = None,
        on_line: Callable[[SpokenLine], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        on_profile: Callable[[Profile], None] | None = None,
    ) -> None:
        self._catalog = catalog
        self._settings = settings
        self._profile = profile or Profile()
        self._on_line = on_line
        self._on_status = on_status
        self._on_error = on_error
        self._on_profile = on_profile

        self._grabber = ScreenGrabber()
        self._ocr: WindowsOcr | None = None
        self._tts: TtsEngine | None = None
        self._player: AudioPlayer | None = None
        self._router = self._build_router(self._profile)
        self._stabilizer = Stabilizer(self._profile.detect)
        self._scaler = self._build_scaler(self._profile)
        self._changes = self._build_detector(self._profile)
        # Every piece of one spoken line shares a group id, so a skip drops the
        # whole line rather than just the sentence being said at that instant.
        self._group = 0
        self._last_line: Utterance | None = None

        self._queue: queue.Queue[Utterance | None] = queue.Queue(SYNTH_QUEUE_MAX)
        self._threads: list[threading.Thread] = []
        self._running = threading.Event()
        self._active = threading.Event()
        self._lock = threading.RLock()
        self.stats = EngineStats()
        self.last_window: WindowInfo | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._running.is_set():
            return

        try:
            self._ocr = create_ocr("en-US")
        except Exception as exc:
            self._fail(str(exc))
            raise

        try:
            self._tts = create_tts(
                self._settings.engine,
                self._profile.speech.rate,
                self._profile.speech.volume,
            )
        except SynthesisError as exc:
            self._fail(str(exc))
            raise

        self._player = AudioPlayer(device=self._settings.output_device)
        try:
            self._player.start()
        except RuntimeError as exc:
            self._fail(str(exc))
            raise

        self._running.set()
        if not self._settings.start_paused:
            self._active.set()

        self._threads = [
            threading.Thread(target=self._reader_loop, name="gv-reader", daemon=True),
            threading.Thread(target=self._speaker_loop, name="gv-speaker", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

        self._status(
            f"Ready - {self._tts.name} engine, profile '{self._profile.name}'"
        )

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        self._active.clear()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        for thread in self._threads:
            thread.join(timeout=3.0)
        self._threads.clear()

        if self._player is not None:
            self._player.close()
            self._player = None
        if self._tts is not None:
            self._tts.close()
            self._tts = None
        if self._ocr is not None:
            self._ocr.close()
            self._ocr = None
        self._grabber.close()
        self._status("Stopped")

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def is_active(self) -> bool:
        return self._active.is_set()

    def resume(self) -> None:
        self._stabilizer.reset()
        self._active.set()
        self._status("Listening")

    def pause(self) -> None:
        self._active.clear()
        self.silence()
        self._status("Paused")

    def toggle(self) -> bool:
        if self._active.is_set():
            self.pause()
        else:
            self.resume()
        return self._active.is_set()

    def silence(self) -> None:
        """Cut off whatever is being said right now, and everything waiting."""
        if self._player is not None:
            self._player.stop()
        self._drain_queue()

    def skip_line(self) -> bool:
        """Abandon the line being spoken and start the next one.

        Unlike ``silence`` this keeps the queue, so a backlog of dialogue keeps
        playing - it moves on rather than stopping.
        """
        if self._player is None:
            return False
        return self._player.skip_group()

    @property
    def last_line(self) -> Utterance | None:
        """The most recent line, for replaying it in full."""
        with self._lock:
            return self._last_line

    def replay(self, utterance: Utterance | None = None) -> bool:
        """Say a line again from the beginning, whole.

        Defaults to the most recent line, which is what "read the whole line"
        means after a line was skipped or missed.
        """
        target = utterance or self.last_line
        if target is None:
            return False
        if self._player is not None:
            self._player.stop()
        self._drain_queue()
        try:
            self._queue.put_nowait(target)
        except queue.Full:
            return False
        return True

    # -- configuration -----------------------------------------------------

    @staticmethod
    def _build_detector(profile: Profile) -> ChangeDetector:
        capture = profile.capture
        return ChangeDetector(
            threshold=capture.change_threshold,
            tolerance=capture.change_tolerance,
            max_idle=capture.max_idle_seconds,
        )

    @staticmethod
    def _build_scaler(profile: Profile) -> AdaptiveUpscale:
        capture = profile.capture
        if not capture.auto_upscale:
            # Fixed scale: a zero-width band around the configured value.
            return AdaptiveUpscale(
                initial=capture.upscale,
                minimum=capture.upscale,
                maximum=capture.upscale,
            )
        return AdaptiveUpscale(
            initial=capture.upscale, maximum=max(capture.max_upscale, capture.upscale)
        )

    def _build_router(self, profile: Profile) -> VoiceRouter:
        speech = profile.speech
        return VoiceRouter(
            catalog=self._catalog,
            game_id=profile.name,
            overrides=profile.voice_overrides,
            hints=profile.gender_hints,
            narrator=speech.narrator_voice,
            male=speech.male_voice,
            female=speech.female_voice,
            pitch_variety=speech.pitch_variety,
            auto_assign=speech.auto_assign,
        )

    @property
    def profile(self) -> Profile:
        with self._lock:
            return self._profile

    @property
    def router(self) -> VoiceRouter:
        with self._lock:
            return self._router

    def set_profile(self, profile: Profile, announce: bool = True) -> None:
        with self._lock:
            self._profile = profile
            self._router = self._build_router(profile)
            self._stabilizer = Stabilizer(profile.detect)
            self._scaler = self._build_scaler(profile)
            self._changes = self._build_detector(profile)
        if self._tts is not None and hasattr(self._tts, "set_rate"):
            self._tts.set_rate(profile.speech.rate)
            self._tts.set_volume(profile.speech.volume)
        if self._player is not None:
            self._player.set_volume(profile.speech.volume)
        if announce:
            self._status(f"Profile: {profile.title or profile.name}")
        if self._on_profile is not None:
            self._on_profile(profile)

    def speak_sample(self, text: str, speaker: str = "") -> None:
        """Say one line immediately, for previewing a voice choice."""
        self._enqueue(Utterance(speaker=speaker, text=text, raw=text))

    # -- reader ------------------------------------------------------------

    def _reader_loop(self) -> None:
        last_window_check = 0.0
        while self._running.is_set():
            profile = self.profile
            interval = 1.0 / max(profile.capture.fps, 0.5)
            started = time.perf_counter()

            if not self._active.is_set():
                time.sleep(0.15)
                continue

            try:
                now = time.time()
                window = self.last_window
                if now - last_window_check >= WINDOW_POLL_SECONDS:
                    window = foreground_window()
                    self.last_window = window
                    last_window_check = now
                    if self._settings.auto_switch_profile:
                        self._maybe_switch(window)
                        profile = self.profile

                self._read_once(profile, window)
            except Exception as exc:
                log.exception("reader loop error")
                self._fail(f"Reader error: {exc}")
                time.sleep(0.5)

            elapsed = time.perf_counter() - started
            time.sleep(max(0.0, interval - elapsed))

    def _maybe_switch(self, window: WindowInfo) -> None:
        chosen = profile_matching.select(window)
        if chosen.name != self.profile.name:
            self.set_profile(chosen)

    def _read_once(self, profile: Profile, window: WindowInfo | None) -> None:
        text_region, speaker_region = resolve_regions(
            profile.capture, self._grabber, window
        )
        frame = self._grabber.grab(text_region)
        if frame is None or looks_blank(frame):
            return

        capture = profile.capture
        # Grabbing is cheap, recognising is not. In change mode the expensive
        # half only runs when the pixels actually moved.
        if capture.trigger == "change":
            if not self._changes.changed(frame, time.time()):
                self.stats.frames_skipped += 1
                return

        applied_scale = clamp_scale(
            frame.shape[1], frame.shape[0], self._scaler.scale
        )
        prepared = preprocess(
            frame,
            upscale=applied_scale,
            contrast=capture.contrast,
            grayscale=capture.grayscale,
            invert=capture.invert,
        )

        started = time.perf_counter()
        output = self._ocr.recognize(prepared) if self._ocr else OcrOutput("", ())
        self.stats.ocr_calls += 1
        self.stats.ocr_ms += (time.perf_counter() - started) * 1000.0
        self.stats.upscale = applied_scale

        if output.is_empty:
            return

        # Learn the right scale from the size the recogniser reported.
        self._scaler.observe(output.lines, applied_scale)

        settled = self._stabilizer.feed(output.text)
        if not settled:
            return

        speaker_text = ""
        if speaker_region is not None:
            name_frame = self._grabber.grab(speaker_region)
            if name_frame is not None and not looks_blank(name_frame):
                name_scale = clamp_scale(
                    name_frame.shape[1], name_frame.shape[0], self._scaler.scale
                )
                name_out = self._ocr.recognize(
                    preprocess(name_frame, upscale=name_scale,
                               contrast=capture.contrast,
                               grayscale=capture.grayscale, invert=capture.invert)
                ) if self._ocr else OcrOutput("", ())
                speaker_text = name_out.text

        utterance = parse(
            OcrOutput(settled, output.lines),
            speaker_text=speaker_text,
            strip_patterns=tuple(profile.detect.strip_patterns),
        )
        if utterance is None:
            return

        self.stats.lines_seen += 1
        self._enqueue(utterance)

    def _enqueue(self, utterance: Utterance) -> None:
        if self.profile.speech.interrupt_on_new_line:
            self.silence()
        try:
            self._queue.put_nowait(utterance)
        except queue.Full:
            log.warning("synthesis queue full; dropping %r", utterance.text[:40])

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    # -- speaker -----------------------------------------------------------

    def _speaker_loop(self) -> None:
        while self._running.is_set():
            try:
                utterance = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if utterance is None:
                return
            try:
                self._speak(utterance)
            except Exception as exc:
                log.exception("speaker loop error")
                self._fail(f"Speech error: {exc}")

    def _speak(self, utterance: Utterance) -> None:
        assignment = self.router.resolve(utterance.speaker)
        if not assignment.voice.model:
            self._fail("No voices installed. Open the Voices tab and download one.")
            return

        spoken = utterance.text
        if self.profile.speech.speak_speaker_name and utterance.has_speaker:
            spoken = f"{utterance.speaker} says, {spoken}"

        # Long passages are spoken in sentence-sized pieces rather than
        # truncated. Each piece is queued as it is produced, so playback starts
        # after the first one instead of after the whole block, and the pieces
        # play back to back with no gap or overlap.
        chunks = split_for_speech(spoken)
        if not chunks:
            return
        if len(chunks) > 1:
            log.debug("speaking %d chars as %d pieces", len(spoken), len(chunks))

        with self._lock:
            self._group += 1
            group = self._group
            self._last_line = utterance

        announced = False
        for chunk in chunks:
            if not self._running.is_set():
                return
            started = time.perf_counter()
            try:
                clip = self._tts.synthesize(chunk, assignment) if self._tts else None
            except SynthesisError as exc:
                self._fail(str(exc))
                return
            self.stats.synth_ms += (time.perf_counter() - started) * 1000.0

            if clip is None or self._player is None:
                continue
            self._player.enqueue(clip.audio, clip.sample_rate, group=group)

            # Show the line as soon as it starts, not when the last piece is
            # finished synthesising.
            self.stats.chunks_spoken += 1
            if not announced:
                announced = True
                self.stats.lines_spoken += 1
                if self._on_line is not None:
                    label = self._tts.describe(assignment) if self._tts else ""
                    self._on_line(SpokenLine(utterance, assignment, label))

    # -- notifications -----------------------------------------------------

    def _status(self, message: str) -> None:
        log.info(message)
        if self._on_status is not None:
            self._on_status(message)

    def _fail(self, message: str) -> None:
        log.error(message)
        self.stats.last_error = message
        if self._on_error is not None:
            self._on_error(message)
