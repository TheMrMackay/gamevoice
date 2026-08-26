"""Playback: a queue of clips, one output stream, instant interruption.

Games advance dialogue faster than speech can be read aloud, so the ability to
cut off the current line the moment a new one appears matters more here than it
would in a normal reader. ``stop`` drops everything already queued and silences
the stream within one callback period.
"""
from __future__ import annotations

import logging
import threading
from collections import deque

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_RATE = 22050
BLOCK_FRAMES = 512
# Ceiling on how much unspoken audio can pile up; past this the reader is so
# far behind the game that older lines are stale anyway.
MAX_QUEUED_SECONDS = 45.0


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear resample. Adequate for speech and cheap enough to run inline."""
    if source_rate == target_rate or audio.size == 0:
        return audio
    duration = audio.size / float(source_rate)
    target_length = max(1, int(duration * target_rate))
    source_x = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(target_x, source_x, audio.astype(np.float32)).astype(np.float32)


def list_output_devices() -> list[tuple[int, str]]:
    try:
        import sounddevice as sd

        return [
            (index, device["name"])
            for index, device in enumerate(sd.query_devices())
            if device.get("max_output_channels", 0) > 0
        ]
    except Exception as exc:
        log.error("could not list audio devices: %s", exc)
        return []


class AudioPlayer:
    """A single output stream fed from a queue of float32 mono blocks."""

    def __init__(self, device: str | int | None = None, rate: int = DEFAULT_RATE) -> None:
        self._device = device
        self._rate = rate
        # Each entry is (group, samples). A long line is synthesised as several
        # clips sharing one group, so "skip this line" can drop the rest of it
        # without touching the line queued behind it.
        self._queue: deque[tuple[int, np.ndarray]] = deque()
        self._current: np.ndarray | None = None
        self._current_group = 0
        self._cursor = 0
        self._lock = threading.Lock()
        self._stream = None
        self._gain = 1.0
        self._queued_frames = 0
        self._drained = threading.Event()
        self._drained.set()

    @property
    def sample_rate(self) -> int:
        return self._rate

    def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        try:
            self._stream = sd.OutputStream(
                samplerate=self._rate,
                channels=1,
                dtype="float32",
                blocksize=BLOCK_FRAMES,
                device=self._device,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise RuntimeError(
                f"Could not open the audio output device: {exc}"
            ) from exc

    def _callback(self, outdata, frames, time_info, status) -> None:
        if status:
            log.debug("audio stream status: %s", status)

        produced = 0
        buffer = outdata.reshape(-1)

        # The whole body runs under the lock. Without it, a stop() landing
        # between the size check and the slice below clears _current and the
        # audio thread dies on None - silently, because an exception here only
        # kills the stream. The work is a memcpy, so holding it is cheap.
        with self._lock:
            while produced < frames:
                current = self._current
                if current is None or self._cursor >= current.size:
                    if not self._queue:
                        self._current = None
                        self._cursor = 0
                        self._queued_frames = 0
                        self._drained.set()
                        break
                    # One clip is finished before the next begins, so two lines
                    # can never be heard on top of each other.
                    self._current_group, current = self._queue.popleft()
                    self._current = current
                    self._cursor = 0

                take = min(frames - produced, current.size - self._cursor)
                buffer[produced: produced + take] = (
                    current[self._cursor: self._cursor + take] * self._gain
                )
                self._cursor += take
                produced += take
                self._queued_frames = max(0, self._queued_frames - take)

        if produced < frames:
            buffer[produced:] = 0.0

    def enqueue(self, audio: np.ndarray, source_rate: int, group: int = 0) -> None:
        """Queue 16-bit or float audio for playback.

        ``group`` ties the pieces of one spoken line together so the whole line
        can be skipped as a unit.
        """
        if audio is None or audio.size == 0:
            return

        if audio.dtype == np.int16:
            samples = audio.astype(np.float32) / 32768.0
        else:
            samples = audio.astype(np.float32)

        samples = resample(samples, source_rate, self._rate)
        samples = np.clip(samples, -1.0, 1.0)

        with self._lock:
            if self._queued_frames / float(self._rate) > MAX_QUEUED_SECONDS:
                log.warning("playback backlog exceeded %.0fs; dropping oldest",
                            MAX_QUEUED_SECONDS)
                while self._queue and (
                    self._queued_frames / float(self._rate) > MAX_QUEUED_SECONDS / 2
                ):
                    self._queued_frames -= self._queue.popleft()[1].size
            self._queue.append((group, samples))
            self._queued_frames += samples.size
            self._drained.clear()

    def stop(self) -> None:
        """Drop everything queued and go quiet immediately."""
        with self._lock:
            self._queue.clear()
            self._current = None
            self._cursor = 0
            self._queued_frames = 0
            self._drained.set()

    def skip_group(self, group: int | None = None) -> bool:
        """Abandon the line being spoken and move to the next one.

        Everything sharing the current group is dropped - the rest of the clip
        playing now and any later pieces of the same line - while whatever was
        queued behind it is left to play. Returns True if anything was skipped.
        """
        with self._lock:
            target = self._current_group if group is None else group
            skipped = False

            if self._current is not None and self._current_group == target:
                self._queued_frames = max(
                    0, self._queued_frames - (self._current.size - self._cursor)
                )
                self._current = None
                self._cursor = 0
                skipped = True

            remaining = deque()
            for entry_group, samples in self._queue:
                if entry_group == target:
                    self._queued_frames = max(0, self._queued_frames - samples.size)
                    skipped = True
                else:
                    remaining.append((entry_group, samples))
            self._queue = remaining

            if not self._queue and self._current is None:
                self._queued_frames = 0
                self._drained.set()
            return skipped

    @property
    def current_group(self) -> int:
        with self._lock:
            return self._current_group

    def set_volume(self, gain: float) -> None:
        self._gain = max(0.0, min(gain, 1.0))

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return bool(self._queue) or self._current is not None

    @property
    def queued_seconds(self) -> float:
        with self._lock:
            return self._queued_frames / float(self._rate)

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        return self._drained.wait(timeout)

    def close(self) -> None:
        self.stop()
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                log.debug("error closing audio stream: %s", exc)
