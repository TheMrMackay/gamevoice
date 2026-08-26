"""One-time: measure every Piper speaker's pitch and tag gender acoustically.

Piper's libritts_r model exposes 904 speakers with no gender metadata. Rather
than trusting an external table, synthesize one probe phrase per speaker and
measure median F0. Male speech sits near 85-155 Hz, female near 165-255 Hz;
anything in between is marked ambiguous and kept out of the automatic pool.

Writes data/voice_table.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from piper import PiperVoice, SynthesisConfig

ROOT = Path(__file__).resolve().parent.parent
PROBE = "The road ahead is long, and the night comes on quickly."

# F0 search window. Below 60 Hz is rumble, above 400 Hz is not speech.
F0_MIN, F0_MAX = 60.0, 400.0
# Decision boundary. The gap is deliberate: voices inside it are not classified.
MALE_MAX, FEMALE_MIN = 155.0, 172.0


def median_f0(audio: np.ndarray, rate: int) -> float:
    """Median fundamental frequency over voiced frames, by autocorrelation."""
    frame = int(rate * 0.040)
    hop = int(rate * 0.020)
    if audio.size < frame:
        return 0.0

    audio = audio.astype(np.float32)
    peak = float(np.abs(audio).max()) or 1.0
    audio = audio / peak

    lag_lo = max(2, int(rate / F0_MAX))
    lag_hi = min(frame - 1, int(rate / F0_MIN))
    if lag_hi <= lag_lo:
        return 0.0

    energies = []
    frames = []
    for start in range(0, audio.size - frame, hop):
        window = audio[start:start + frame]
        energy = float(np.sqrt(np.mean(window ** 2)))
        energies.append(energy)
        frames.append(window)
    if not frames:
        return 0.0

    # Voiced frames are the loud ones. A fixed threshold misfires across
    # speakers of different loudness, so gate on this speaker's own level.
    gate = max(0.06, float(np.median(energies)))

    pitches = []
    for window, energy in zip(frames, energies):
        if energy < gate:
            continue
        window = window - window.mean()
        corr = np.correlate(window, window, mode="full")[frame - 1:]
        if corr[0] <= 0:
            continue
        corr = corr / corr[0]
        segment = corr[lag_lo:lag_hi]
        if segment.size == 0:
            continue
        lag = int(np.argmax(segment)) + lag_lo
        # A weak correlation peak means the frame was not periodic.
        if corr[lag] < 0.30:
            continue
        pitches.append(rate / lag)

    return float(np.median(pitches)) if pitches else 0.0


def classify(f0: float) -> str:
    if f0 <= 0:
        return "unknown"
    if f0 <= MALE_MAX:
        return "male"
    if f0 >= FEMALE_MIN:
        return "female"
    return "ambiguous"


def measure_model(model: Path, limit: int | None = None) -> dict:
    voice = PiperVoice.load(str(model))
    rate = voice.config.sample_rate
    speakers = voice.config.num_speakers
    ids = list(range(speakers))[:limit] if limit else list(range(speakers))

    entries = []
    started = time.perf_counter()
    for n, sid in enumerate(ids, 1):
        cfg = SynthesisConfig(speaker_id=sid if speakers > 1 else None)
        chunks = list(voice.synthesize(PROBE, cfg))
        audio = np.concatenate(
            [np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in chunks]
        )
        f0 = median_f0(audio, rate)
        entries.append({"speaker_id": sid, "f0": round(f0, 1), "gender": classify(f0)})
        if n % 50 == 0 or n == len(ids):
            rate_s = n / (time.perf_counter() - started)
            print(f"  {model.stem}: {n}/{len(ids)}  ({rate_s:.1f}/s)", flush=True)

    return {"model": model.stem, "sample_rate": rate, "speakers": entries}


def main() -> int:
    voices_dir = ROOT / "voices"
    models = sorted(voices_dir.glob("*.onnx"))
    if not models:
        print("no voice models in voices/", file=sys.stderr)
        return 1

    table = {"probe": PROBE, "models": {}}
    for model in models:
        result = measure_model(model)
        table["models"][result["model"]] = result
        counts: dict[str, int] = {}
        for entry in result["speakers"]:
            counts[entry["gender"]] = counts.get(entry["gender"], 0) + 1
        print(f"{result['model']}: {counts}", flush=True)

    out = ROOT / "data" / "voice_table.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
