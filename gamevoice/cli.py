"""Headless command line. Useful on its own, and the fastest way to diagnose.

``gamevoice doctor`` is the one to reach for first when something is not being
read: it reports what the capture actually sees, rather than leaving the user to
guess whether the problem is the region, the OCR or the voice.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from . import setup_logging
from .native import ensure_load_order

ensure_load_order()  # before anything below pulls in WinRT; see native.py

from .audio import list_output_devices
from .capture import ScreenGrabber, foreground_window, resolve_regions, set_dpi_aware, looks_blank
from .config import AppSettings, DEFAULT_PROFILE_NAME, VOICES_DIR, list_profiles
from .dialogue import parse
from .engine import GameVoiceEngine, SpokenLine
from .ocr import available_languages, create_engine as create_ocr, preprocess
from .voices import VoiceCatalog

log = logging.getLogger(__name__)


def _catalog() -> VoiceCatalog:
    catalog = VoiceCatalog()
    if catalog.is_empty:
        print(
            f"No voices found in {VOICES_DIR}.\n"
            "Download a starter set with:  gamevoice download --starter",
            file=sys.stderr,
        )
    return catalog


def cmd_devices(_: argparse.Namespace) -> int:
    print("OCR languages installed:", ", ".join(available_languages()) or "none")
    print("\nAudio output devices:")
    for index, name in list_output_devices():
        print(f"  [{index}] {name}")
    return 0


def cmd_voices(args: argparse.Namespace) -> int:
    catalog = _catalog()
    voices = catalog.all_voices()
    if args.gender:
        voices = [v for v in voices if v.gender == args.gender]
    print(f"{len(voices)} voices installed")
    for voice in voices[: args.limit]:
        print(f"  {voice.key:38s} {voice.gender:9s} {voice.f0:6.1f} Hz")
    if len(voices) > args.limit:
        print(f"  ... and {len(voices) - args.limit} more (use --limit)")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from piper.download_voices import download_voice

    wanted = list(args.voice)
    if args.starter or not wanted:
        wanted = ["en_US-ryan-high", "en_US-hfc_female-medium", "en_US-libritts_r-medium"]

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        print(f"downloading {name} ...", flush=True)
        try:
            download_voice(name, VOICES_DIR)
        except Exception as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            return 1
    print("done. Run 'gamevoice measure' to tag new voices by gender.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report what the capture pipeline actually sees right now."""
    set_dpi_aware()
    profiles = list_profiles()
    grabber = ScreenGrabber()

    if args.delay:
        print(f"switch to your game; reading in {args.delay}s ...")
        time.sleep(args.delay)

    window = foreground_window()
    print(f"foreground : {window.title!r}  exe={window.exe or '?'}")
    print(f"window rect: {window.rect}")

    from . import profiles as matching

    profile = matching.select(window, profiles)
    print(f"profile    : {profile.name} ({profile.title})")

    region, speaker_region = resolve_regions(profile.capture, grabber, window)
    print(f"text region: {region}")
    if speaker_region:
        print(f"name region: {speaker_region}")

    frame = grabber.grab(region)
    if frame is None:
        print("\nFAIL: the screen grab returned nothing.")
        return 1
    if looks_blank(frame):
        print(
            "\nFAIL: the captured area is a flat colour.\n"
            "  If the game is in exclusive fullscreen, switch it to borderless "
            "windowed - a fullscreen swap chain cannot be captured this way."
        )
        return 1

    engine = create_ocr("en-US")
    prepared = preprocess(
        frame,
        upscale=profile.capture.upscale,
        contrast=profile.capture.contrast,
        grayscale=profile.capture.grayscale,
        invert=profile.capture.invert,
    )
    started = time.perf_counter()
    output = engine.recognize(prepared)
    elapsed = (time.perf_counter() - started) * 1000

    print(f"\nOCR took {elapsed:.1f} ms and found {len(output.lines)} line(s):")
    for line in output.lines:
        print(f"  | {line.text}")

    from .ocr import clipped_edges

    edges = clipped_edges(output, prepared.shape[0])
    if edges:
        print(
            f"\n>> Text touches the {' and '.join(edges)} of the capture area.\n"
            "   The passage probably continues outside it, so only part is read.\n"
            "   Use a taller region, or raise bottom_band in the profile."
        )

    utterance = parse(output, strip_patterns=tuple(profile.detect.strip_patterns))
    if utterance is None:
        print("\nNothing that looks like dialogue. Try a tighter region.")
        return 0

    catalog = VoiceCatalog()
    print(f"\nspeaker : {utterance.speaker or '(narrator)'}")
    print(f"text    : {utterance.text}")
    if not catalog.is_empty:
        from .voices import VoiceRouter

        router = VoiceRouter(catalog, game_id=profile.name,
                             overrides=profile.voice_overrides,
                             hints=profile.gender_hints)
        assignment = router.resolve(utterance.speaker)
        print(f"voice   : {assignment.voice.label}  [{assignment.source}]")
    return 0


def cmd_say(args: argparse.Namespace) -> int:
    from .audio import AudioPlayer
    from .tts import create_engine as create_tts
    from .voices import VoiceRouter

    catalog = _catalog()
    if catalog.is_empty:
        return 1

    router = VoiceRouter(catalog, game_id=args.game)
    assignment = router.resolve(args.speaker)
    print(f"{args.speaker or '(narrator)'} -> {assignment.voice.label} [{assignment.source}]")

    engine = create_tts(args.engine)
    clip = engine.synthesize(args.text, assignment)
    if clip is None:
        print("synthesis produced no audio", file=sys.stderr)
        return 1

    player = AudioPlayer()
    player.start()
    player.enqueue(clip.audio, clip.sample_rate)
    player.wait_until_idle(timeout=clip.duration + 5)
    time.sleep(0.2)
    player.close()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    set_dpi_aware()
    settings = AppSettings.load()
    settings.start_paused = False
    if args.engine:
        settings.engine = args.engine

    catalog = _catalog()
    if catalog.is_empty:
        return 1

    profiles = list_profiles()
    profile = profiles.get(args.profile or DEFAULT_PROFILE_NAME) or profiles[DEFAULT_PROFILE_NAME]

    def on_line(line: SpokenLine) -> None:
        who = line.utterance.speaker or "narrator"
        print(f"[{who}] {line.utterance.text}   <{line.assignment.voice.key}>", flush=True)

    engine = GameVoiceEngine(
        catalog,
        settings,
        profile,
        on_line=on_line,
        on_status=lambda m: print(f"-- {m}", flush=True),
        on_error=lambda m: print(f"!! {m}", file=sys.stderr, flush=True),
    )
    engine.start()
    engine.resume()
    print("reading. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstopping ...")
    finally:
        engine.stop()
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    from pathlib import Path
    import subprocess

    script = Path(__file__).resolve().parent.parent / "tools" / "build_voice_table.py"
    return subprocess.call([sys.executable, str(script)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gamevoice", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="list OCR languages and audio outputs").set_defaults(
        func=cmd_devices
    )

    voices = sub.add_parser("voices", help="list installed voices")
    voices.add_argument("--gender", choices=["male", "female", "ambiguous", "unknown"])
    voices.add_argument("--limit", type=int, default=40)
    voices.set_defaults(func=cmd_voices)

    download = sub.add_parser("download", help="download Piper voice models")
    download.add_argument("voice", nargs="*")
    download.add_argument("--starter", action="store_true", help="the recommended set")
    download.set_defaults(func=cmd_download)

    doctor = sub.add_parser("doctor", help="show what the capture pipeline sees")
    doctor.add_argument("--delay", type=int, default=0, help="seconds to switch windows")
    doctor.set_defaults(func=cmd_doctor)

    say = sub.add_parser("say", help="speak one line as a given character")
    say.add_argument("text")
    say.add_argument("--speaker", default="")
    say.add_argument("--game", default="demo")
    say.add_argument("--engine", default="piper", choices=["piper", "sapi"])
    say.set_defaults(func=cmd_say)

    run = sub.add_parser("run", help="read the screen aloud until stopped")
    run.add_argument("--profile", default=None)
    run.add_argument("--engine", default=None, choices=["piper", "sapi"])
    run.set_defaults(func=cmd_run)

    sub.add_parser("measure", help="re-measure voice genders").set_defaults(func=cmd_measure)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.WARNING, to_file=True)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        log.exception("command failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
