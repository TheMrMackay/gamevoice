"""Settings model and persistence.

A *profile* is everything GameVoice needs to read one game: where on screen the
dialogue is, how to tell who is speaking, and which voice each speaker gets.
Shipped profiles live in ``profiles/``; anything the user edits is written to
``%LOCALAPPDATA%\\GameVoice\\profiles`` and wins over a shipped file of the same
name, so an update never overwrites a tuned region.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

APP_NAME = "GameVoice"
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

SHIPPED_PROFILES = PROJECT_ROOT / "profiles"
VOICES_DIR = PROJECT_ROOT / "voices"
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"

DEFAULT_PROFILE_NAME = "default"

# Profile name -> filename needs to survive arbitrary window titles.
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def user_data_dir() -> Path:
    """Per-user writable state. Honours GAMEVOICE_HOME for portable installs."""
    override = os.environ.get("GAMEVOICE_HOME")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return Path(base) / APP_NAME


def user_profiles_dir() -> Path:
    return user_data_dir() / "profiles"


def safe_name(name: str) -> str:
    """A filename-safe slug. Window titles are arbitrary user-facing text.

    Separators are already gone by the time the dot handling runs, so this
    cannot escape the profiles directory; collapsing runs of dots and dashes
    just stops a path-like title turning into a "..-.."-shaped filename.
    """
    cleaned = _UNSAFE_NAME.sub("-", name.strip())
    cleaned = re.sub(r"[-.]{2,}", "-", cleaned).strip("-.")
    return (cleaned or "profile").lower()[:64]


@dataclass
class Region:
    """A screen rectangle in virtual-desktop coordinates."""

    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0

    def is_valid(self) -> bool:
        return self.width > 8 and self.height > 8

    def as_mss(self) -> dict[str, int]:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "width": int(self.width),
            "height": int(self.height),
        }


@dataclass
class CaptureSettings:
    """Where to look, and how hard to squint.

    ``text_region`` of None means "use the bottom band of the chosen monitor",
    which is where the large majority of games put subtitles. That default is
    what lets the tool do something useful on a game it has never seen.
    """

    monitor: int = 1
    text_region: Region | None = None
    speaker_region: Region | None = None
    bottom_band: float = 0.34
    fps: float = 8.0
    # "change" recognises only when the captured pixels actually move, which is
    # both faster to react and far cheaper than recognising on every tick.
    # "timer" is the old fixed-interval behaviour, kept for anything with a
    # constantly animating dialogue box.
    trigger: str = "change"
    change_threshold: float = 0.004
    change_tolerance: int = 10
    # Read anyway after this long with no detected change, so a difference too
    # subtle for the threshold cannot leave a line unread forever. 0 disables.
    max_idle_seconds: float = 8.0
    # Starting scale. With auto_upscale on this is only the first guess - the
    # engine measures the glyph height the recogniser reports and converges on
    # whatever actually reads well, so small subtitles get scaled up hard and
    # large ones are left alone.
    upscale: float = 2.0
    auto_upscale: bool = True
    max_upscale: float = 6.0
    contrast: float = 1.6
    invert: bool = False
    grayscale: bool = True
    # Auto mode: when no text_region is set, periodically scan the whole
    # foreground window for a block of dialogue-shaped text and read just that
    # box, wherever the game happens to draw it. Falls back to the bottom band
    # while nothing has been found.
    auto_region: bool = True
    scan_interval: float = 1.0
    hold_scan_interval: float = 3.0
    lost_seconds: float = 4.0
    scan_width: int = 1280
    max_region_fraction: float = 0.6
    # The dialogue box auto-detection found last time, window-relative. Seeded
    # back in on the next launch so a game starts tracked instead of searched,
    # and re-learned if the game has moved its box since.
    auto_found_region: Region | None = None


@dataclass
class DetectSettings:
    """When a block of on-screen text counts as a finished line."""

    settle_frames: int = 2
    similarity_threshold: float = 0.90
    min_chars: int = 3
    # A ceiling for rejecting a screen full of interface text, not a limit on
    # how much dialogue gets read. Anything under this is spoken in full, split
    # into sentence-sized pieces; anything over is skipped whole and logged.
    max_chars: int = 4000
    history: int = 24
    strip_patterns: list[str] = field(default_factory=list)
    ignore_patterns: list[str] = field(default_factory=list)


@dataclass
class SpeechSettings:
    engine: str = "piper"
    narrator_voice: str = ""
    male_voice: str = ""
    female_voice: str = ""
    rate: float = 1.0
    volume: float = 0.9
    pitch_variety: float = 0.06
    # Off by default: a line is spoken to the end and the next one waits its
    # turn. Cutting a line off the moment new text appears loses the end of
    # almost every sentence, because games advance the box while it is still
    # being read aloud.
    interrupt_on_new_line: bool = False
    speak_speaker_name: bool = False
    output_device: str | None = None
    auto_assign: bool = True


@dataclass
class Profile:
    name: str = DEFAULT_PROFILE_NAME
    title: str = "Default (any game)"
    match_exe: list[str] = field(default_factory=list)
    match_title: list[str] = field(default_factory=list)
    capture: CaptureSettings = field(default_factory=CaptureSettings)
    detect: DetectSettings = field(default_factory=DetectSettings)
    speech: SpeechSettings = field(default_factory=SpeechSettings)
    # speaker name (lowercased) -> voice key, e.g. "en_US-libritts_r-medium#313"
    voice_overrides: dict[str, str] = field(default_factory=dict)
    # speaker name (lowercased) -> "male" | "female" | "neutral"
    gender_hints: dict[str, str] = field(default_factory=dict)

    def path(self) -> Path:
        return user_profiles_dir() / f"{safe_name(self.name)}.json"


def _build(cls: type, payload: Any) -> Any:
    """Rebuild a dataclass from JSON, ignoring keys the schema no longer has.

    Being lenient here matters: a profile file written by an older build must
    still load rather than taking the app down at startup.
    """
    if payload is None or not is_dataclass(cls):
        return payload

    kwargs: dict[str, Any] = {}
    for spec in fields(cls):
        if spec.name not in payload:
            continue
        raw = payload[spec.name]
        if spec.name in ("text_region", "speaker_region", "auto_found_region"):
            kwargs[spec.name] = _build(Region, raw) if raw else None
        elif spec.name == "capture":
            kwargs[spec.name] = _build(CaptureSettings, raw)
        elif spec.name == "detect":
            kwargs[spec.name] = _build(DetectSettings, raw)
        elif spec.name == "speech":
            kwargs[spec.name] = _build(SpeechSettings, raw)
        else:
            kwargs[spec.name] = raw
    return cls(**kwargs)


def load_profile_file(path: Path) -> Profile | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.error("could not read profile %s: %s", path, exc)
        return None
    try:
        return _build(Profile, payload)
    except (TypeError, ValueError) as exc:
        log.error("profile %s has an unusable shape: %s", path, exc)
        return None


def save_profile(profile: Profile) -> Path:
    """Write a profile to the user directory. Raises OSError on failure."""
    target = profile.path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(profile), indent=2)
        # Write-then-rename so a crash mid-write cannot leave a truncated file
        # that fails to parse on next launch.
        scratch = target.with_suffix(".json.tmp")
        scratch.write_text(payload, encoding="utf-8")
        scratch.replace(target)
    except OSError as exc:
        log.error("could not save profile %s: %s", target, exc)
        raise OSError(f"Could not save profile to {target}: {exc}") from exc
    return target


def is_shipped(profile: Profile) -> bool:
    """True if a profile of this name is built in.

    Matters for deletion: removing a user copy of a shipped profile does not
    remove the profile, it reverts it to the version that came with the app.
    """
    return (SHIPPED_PROFILES / f"{safe_name(profile.name)}.json").is_file()


def delete_profile(profile: Profile) -> str:
    """Remove a profile's user copy. Returns what actually happened.

    Shipped profiles cannot be removed, only reset - saying so plainly beats a
    delete button that appears to work and leaves the entry in the list.
    """
    target = profile.path()
    existed = target.is_file()
    if existed:
        try:
            target.unlink()
        except OSError as exc:
            log.error("could not delete profile %s: %s", target, exc)
            raise OSError(f"Could not delete {target}: {exc}") from exc

    if is_shipped(profile):
        return "reset" if existed else "unchanged"
    return "deleted" if existed else "unchanged"


def list_profiles() -> dict[str, Profile]:
    """All known profiles, user copies shadowing shipped ones by name."""
    found: dict[str, Profile] = {}
    for source in (SHIPPED_PROFILES, user_profiles_dir()):
        if not source.is_dir():
            continue
        for path in sorted(source.glob("*.json")):
            profile = load_profile_file(path)
            if profile is not None:
                found[profile.name] = profile
    if DEFAULT_PROFILE_NAME not in found:
        found[DEFAULT_PROFILE_NAME] = Profile()
    return found


@dataclass
class AppSettings:
    """Settings that are not per-game."""

    # An empty string means "not bound". Every one of these is meant to be
    # usable with a game in the foreground, which is why they are global.
    hotkey_toggle: str = "ctrl+alt+v"
    hotkey_stop: str = "ctrl+alt+x"
    hotkey_skip: str = ""
    hotkey_replay: str = ""
    auto_switch_profile: bool = True
    start_paused: bool = True
    log_lines: bool = True
    engine: str = "piper"
    output_device: str | None = None
    # Window placement. None means "not chosen yet" - the window centres itself
    # on the primary monitor rather than wherever Qt would have put it, which
    # on a multi-monitor desktop is often the screen the user is not using.
    window_x: int | None = None
    window_y: int | None = None
    window_w: int | None = None
    window_h: int | None = None

    @staticmethod
    def path() -> Path:
        return user_data_dir() / "settings.json"

    @classmethod
    def load(cls) -> "AppSettings":
        path = cls.path()
        if not path.is_file():
            return cls()
        try:
            return _build(cls, json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            log.error("settings unreadable, using defaults: %s", exc)
            return cls()

    def save(self) -> None:
        path = self.path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        except OSError as exc:
            log.error("could not save settings to %s: %s", path, exc)
            raise OSError(f"Could not save settings to {path}: {exc}") from exc
