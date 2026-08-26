"""System-wide hotkeys.

The whole point of this app is that the user is inside a game with the window
focused, so a shortcut that only works when GameVoice has focus would be
useless. ``RegisterHotKey`` is the Windows facility for a genuinely global key,
and it needs a native message filter to be seen by Qt.
"""
from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes
from typing import Callable

from PySide6.QtCore import QAbstractNativeEventFilter

log = logging.getLogger(__name__)

_user32 = ctypes.WinDLL("user32", use_last_error=True)

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312

_MODIFIERS = {
    "ctrl": MOD_CONTROL,
    "control": MOD_CONTROL,
    "alt": MOD_ALT,
    "shift": MOD_SHIFT,
    "win": MOD_WIN,
    "super": MOD_WIN,
}

_NAMED_KEYS = {
    "space": 0x20, "esc": 0x1B, "escape": 0x1B, "tab": 0x09,
    "enter": 0x0D, "return": 0x0D, "backspace": 0x08, "insert": 0x2D,
    "delete": 0x2E, "home": 0x24, "end": 0x23, "pageup": 0x21,
    "pagedown": 0x22, "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "pause": 0x13, "scrolllock": 0x91,
    **{f"f{n}": 0x6F + n for n in range(1, 25)},
}


class HotkeyError(ValueError):
    """Raised when a hotkey string cannot be understood or registered."""


def parse(spec: str) -> tuple[int, int]:
    """"ctrl+alt+v" -> (modifiers, virtual key code)."""
    parts = [part.strip().lower() for part in (spec or "").split("+") if part.strip()]
    if not parts:
        raise HotkeyError("Empty hotkey")

    modifiers = 0
    key_name = ""
    for part in parts:
        if part in _MODIFIERS:
            modifiers |= _MODIFIERS[part]
        elif key_name:
            raise HotkeyError(f"'{spec}' names more than one key")
        else:
            key_name = part

    if not key_name:
        raise HotkeyError(f"'{spec}' has modifiers but no key")

    if key_name in _NAMED_KEYS:
        return modifiers, _NAMED_KEYS[key_name]
    if len(key_name) == 1 and key_name.isalnum():
        return modifiers, ord(key_name.upper())
    raise HotkeyError(f"'{spec}' is not a key GameVoice recognises")


class HotkeyManager(QAbstractNativeEventFilter):
    """Registers global hotkeys and routes them to callbacks.

    Install with ``QApplication.instance().installNativeEventFilter(manager)``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._callbacks: dict[int, Callable[[], None]] = {}
        self._specs: dict[int, str] = {}
        self._next_id = 1

    def register(self, spec: str, callback: Callable[[], None]) -> int:
        """Register one hotkey. Raises HotkeyError if Windows refuses it."""
        modifiers, key = parse(spec)
        hotkey_id = self._next_id
        self._next_id += 1

        if not _user32.RegisterHotKey(
            None, hotkey_id, modifiers | MOD_NOREPEAT, key
        ):
            error = ctypes.get_last_error()
            if error == 1409:  # ERROR_HOTKEY_ALREADY_REGISTERED
                raise HotkeyError(
                    f"'{spec}' is already claimed by another application. "
                    "Pick a different combination in Settings."
                )
            raise HotkeyError(f"Windows refused the hotkey '{spec}' (error {error})")

        self._callbacks[hotkey_id] = callback
        self._specs[hotkey_id] = spec
        log.info("registered hotkey %s", spec)
        return hotkey_id

    def unregister_all(self) -> None:
        for hotkey_id in list(self._callbacks):
            try:
                _user32.UnregisterHotKey(None, hotkey_id)
            except Exception as exc:
                log.debug("could not unregister hotkey %s: %s", hotkey_id, exc)
        self._callbacks.clear()
        self._specs.clear()

    def nativeEventFilter(self, event_type, message):
        if event_type not in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
            return False, 0
        msg = ctypes.cast(int(message), ctypes.POINTER(wintypes.MSG)).contents
        if msg.message != WM_HOTKEY:
            return False, 0

        callback = self._callbacks.get(int(msg.wParam))
        if callback is None:
            return False, 0
        try:
            callback()
        except Exception:
            log.exception("hotkey handler failed")
        return True, 0
