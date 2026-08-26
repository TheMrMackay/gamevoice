"""Choosing the right profile for whatever game is in front.

Matching is deliberately simple and predictable: an executable name beats a
window-title fragment, a longer title fragment beats a shorter one, and the
built-in ``default`` catches everything else. The default existing at all is
what satisfies "works with every game" - an unrecognised game still gets read,
just with generic settings until someone tunes it.
"""
from __future__ import annotations

import logging
import re
from dataclasses import replace

from .capture import WindowInfo
from .config import (
    DEFAULT_PROFILE_NAME,
    Profile,
    list_profiles,
    save_profile,
)

log = logging.getLogger(__name__)

# Windows that are never a game, so auto-switching should not react to them.
_IGNORED_EXES = frozenset(
    """
    explorer.exe searchapp.exe shellexperiencehost.exe startmenuexperiencehost.exe
    textinputhost.exe applicationframehost.exe systemsettings.exe taskmgr.exe
    python.exe pythonw.exe cmd.exe powershell.exe windowsterminal.exe
    """.split()
)

_TITLE_NOISE = re.compile(r"\s*[-|–—]\s*(?:steam|epic games|gog galaxy)\s*$", re.I)


def is_plausible_game(window: WindowInfo) -> bool:
    if not window.is_usable:
        return False
    if window.exe in _IGNORED_EXES:
        return False
    # A tiny window is a dialog or a launcher, not the game being played.
    return window.rect.width >= 640 and window.rect.height >= 400


def normalize_title(title: str) -> str:
    return _TITLE_NOISE.sub("", title or "").strip()


def score_match(profile: Profile, window: WindowInfo) -> int:
    """How well a profile fits a window. Zero means no match at all."""
    if profile.name == DEFAULT_PROFILE_NAME:
        return 0

    exe = (window.exe or "").lower()
    for candidate in profile.match_exe:
        if candidate and candidate.lower() == exe:
            # An exe match is exact and unambiguous, so it outranks any title.
            return 1000

    title = normalize_title(window.title).lower()
    best = 0
    for fragment in profile.match_title:
        fragment = (fragment or "").lower().strip()
        if fragment and fragment in title:
            best = max(best, len(fragment))
    return best


def select(window: WindowInfo, known: dict[str, Profile] | None = None) -> Profile:
    """The best profile for this window, never None."""
    profiles = known if known is not None else list_profiles()
    fallback = profiles.get(DEFAULT_PROFILE_NAME) or Profile()

    if not is_plausible_game(window):
        return fallback

    best, best_score = fallback, 0
    for profile in profiles.values():
        score = score_match(profile, window)
        if score > best_score:
            best, best_score = profile, score
    return best


def profile_for_window(window: WindowInfo, base: Profile | None = None) -> Profile:
    """Build a new profile pre-filled to match the window in front.

    Used by "Create a profile for this game", so the user never has to know
    what an executable name is.
    """
    source = base or Profile()
    title = normalize_title(window.title) or window.exe or "Untitled game"
    name = title.lower().replace(" ", "-")[:48] or "game"
    return replace(
        source,
        name=name,
        title=title,
        match_exe=[window.exe] if window.exe else [],
        match_title=[title] if title else [],
        voice_overrides=dict(source.voice_overrides),
        gender_hints=dict(source.gender_hints),
    )


def remember(profile: Profile) -> Profile:
    """Persist a profile, logging rather than raising on a write failure."""
    try:
        save_profile(profile)
    except OSError as exc:
        log.error("could not persist profile %s: %s", profile.name, exc)
    return profile
