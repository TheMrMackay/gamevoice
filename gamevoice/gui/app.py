"""Main window, tray icon, and the wiring between the UI and the engine.

Engine callbacks arrive on worker threads, so everything they report is
re-emitted as a Qt signal and handled on the GUI thread. Touching widgets from
the reader thread would work most of the time and crash the rest.
"""
from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSystemTrayIcon,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import __version__, setup_logging
from ..capture import set_dpi_aware
from ..config import (
    ASSETS_DIR,
    DEFAULT_PROFILE_NAME,
    AppSettings,
    Profile,
    Region,
    delete_profile,
    is_shipped,
    list_profiles,
    save_profile,
)
from ..engine import GameVoiceEngine, SpokenLine
from ..ocr import OcrUnavailable
from ..profiles import profile_for_window
from ..tts import SynthesisError
from ..voices import VoiceCatalog
from .capture_tab import CaptureTab
from .download_dialog import DownloadDialog
from .hotkeys import HotkeyManager
from .profiles_tab import ProfilesTab
from .region_picker import pick_region
from .settings_tab import SettingsTab
from .status_tab import StatusTab
from .voices_tab import VoicesTab

log = logging.getLogger(__name__)

APP_TITLE = "GameVoice"
STATS_INTERVAL_MS = 1500


_ICON_CACHE: dict[bool, QIcon] = {}


def _drawn_icon(active: bool) -> QIcon:
    """Fallback mark, drawn at runtime if the icon files are missing."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#10B981") if active else QColor("#4338CA"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(6, 10, 52, 38, 10, 10)
    painter.drawPolygon([QPoint(20, 44), QPoint(34, 44), QPoint(22, 58)])
    painter.setBrush(QColor("#ffffff"))
    for index, height in enumerate((10, 18, 12)):
        painter.drawRoundedRect(19 + index * 11, 29 - height // 2, 6, height, 3, 3)
    painter.end()
    return QIcon(pixmap)


def build_icon(active: bool = False) -> QIcon:
    """The application icon: green while listening, indigo when idle.

    Loaded from the multi-resolution .ico built by tools/make_icon.py, which
    carries hand-simplified artwork at the small sizes; Windows would otherwise
    downscale the 256 and turn the level meter to mush in the tray.
    """
    cached = _ICON_CACHE.get(active)
    if cached is not None:
        return cached

    path = ASSETS_DIR / ("icon-active.ico" if active else "icon.ico")
    icon = QIcon(str(path)) if path.is_file() else QIcon()
    if icon.isNull():
        log.warning("icon %s missing; using the drawn fallback", path.name)
        icon = _drawn_icon(active)

    _ICON_CACHE[active] = icon
    return icon


class MainWindow(QMainWindow):
    line_spoken = Signal(object)
    status_changed = Signal(str)
    error_raised = Signal(str)
    profile_switched = Signal(object)

    def __init__(self, settings: AppSettings, catalog: VoiceCatalog) -> None:
        super().__init__()
        self._settings = settings
        self._catalog = catalog
        self._profiles = list_profiles()
        self._profile = self._profiles.get(DEFAULT_PROFILE_NAME) or Profile()
        self._engine: GameVoiceEngine | None = None
        self._picker = None
        self._hotkeys = HotkeyManager()
        self._dirty = False
        self._loaded_name = self._profile.name

        self.setWindowTitle(f"{APP_TITLE} {__version__}")
        self.setWindowIcon(build_icon())
        self._restore_geometry()

        self._build()
        self._connect()
        self._load_profile(self._profile)
        self._install_hotkeys()
        self._build_tray()

        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._refresh_stats)
        self._stats_timer.start(STATS_INTERVAL_MS)

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Game profile:"))
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(280)
        self._refill_profiles()
        bar.addWidget(self.profile_combo)

        self.new_profile_button = QPushButton("New for current game")
        self.save_profile_button = QPushButton("Save")
        bar.addWidget(self.new_profile_button)
        bar.addWidget(self.save_profile_button)
        bar.addStretch(1)
        layout.addLayout(bar)

        self.tabs = QTabWidget()
        self.status_tab = StatusTab()
        self.voices_tab = VoicesTab(self._catalog)
        self.capture_tab = CaptureTab()
        self.profiles_tab = ProfilesTab()
        self.settings_tab = SettingsTab(self._settings)
        self.tabs.addTab(self.status_tab, "Listening")
        self.tabs.addTab(self.voices_tab, "Voices")
        self.tabs.addTab(self.capture_tab, "Capture")
        self.tabs.addTab(self.profiles_tab, "Profiles")
        self.tabs.addTab(self.settings_tab, "Settings")
        layout.addWidget(self.tabs, stretch=1)

        hint = QLabel(
            "Hotkeys work while a game has focus - set them on the Settings "
            "tab. Closing this window keeps GameVoice running in the tray."
        )
        hint.setStyleSheet("color: #888;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.setCentralWidget(central)

    def _refill_profiles(self) -> None:
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for name, profile in sorted(self._profiles.items()):
            self.profile_combo.addItem(profile.title or name, name)
        self.profile_combo.blockSignals(False)

    def _connect(self) -> None:
        self.status_tab.toggle_button.clicked.connect(self._toggle)
        self.status_tab.silence_button.clicked.connect(self._silence)
        self.status_tab.clear_button.clicked.connect(self.status_tab.clear)
        self.status_tab.skip_button.clicked.connect(self._skip_line)
        self.status_tab.replay_button.clicked.connect(self._replay_line)

        self.profiles_tab.selected.connect(self._select_profile_by_name)
        self.profiles_tab.changed.connect(self._mark_dirty)
        self.profiles_tab.create_requested.connect(self._new_profile)
        self.profiles_tab.duplicate_requested.connect(self._duplicate_profile)
        self.profiles_tab.delete_requested.connect(self._delete_profile)
        self.profiles_tab.detect_requested.connect(self._detect_into_profile)

        self.settings_tab.hotkeys_changed.connect(self._apply_hotkeys)
        self.settings_tab.settings_changed.connect(self._on_settings_changed)

        self.voices_tab.changed.connect(self._mark_dirty)
        self.voices_tab.preview_requested.connect(self._preview)
        self.voices_tab.download_requested.connect(self._open_downloads)

        self.capture_tab.changed.connect(self._mark_dirty)
        self.capture_tab.pick_text_region.connect(lambda: self._pick("text"))
        self.capture_tab.pick_speaker_region.connect(lambda: self._pick("speaker"))

        self.profile_combo.currentIndexChanged.connect(self._profile_selected)
        self.new_profile_button.clicked.connect(self._new_profile)
        self.save_profile_button.clicked.connect(self._save_profile)

        self.line_spoken.connect(self._on_line)
        self.status_changed.connect(lambda m: self.status_tab.set_status(m))
        self.error_raised.connect(self._on_error)
        self.profile_switched.connect(self._on_engine_switched_profile)

    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(build_icon(), self)
        self.tray.setToolTip(APP_TITLE)

        menu = QMenu()
        self.tray_toggle = QAction("Start listening", self)
        self.tray_toggle.triggered.connect(self._toggle)
        menu.addAction(self.tray_toggle)

        show = QAction("Show window", self)
        show.triggered.connect(self._restore)
        menu.addAction(show)
        menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self._quit)
        menu.addAction(quit_action)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda reason: self._restore()
            if reason == QSystemTrayIcon.ActivationReason.DoubleClick
            else None
        )
        self.tray.show()

    def _install_hotkeys(self) -> None:
        app = QApplication.instance()
        app.installNativeEventFilter(self._hotkeys)
        self._apply_hotkeys()

    def _apply_hotkeys(self) -> None:
        """(Re)claim every hotkey and report which ones Windows refused."""
        failures = self._hotkeys.rebind(
            [
                (self._settings.hotkey_toggle, self._toggle),
                (self._settings.hotkey_stop, self._silence),
                (self._settings.hotkey_skip, self._skip_line),
                (self._settings.hotkey_replay, self._replay_line),
            ]
        )
        self.settings_tab.report(failures)
        if failures:
            first = next(iter(failures.values()))
            self.status_tab.set_status(first, error=True)
        try:
            self._settings.save()
        except OSError as exc:
            log.error("could not save settings: %s", exc)

    # -- engine ------------------------------------------------------------

    def _ensure_engine(self) -> bool:
        if self._engine is not None:
            return True
        self._apply_ui_to_profile()
        try:
            engine = GameVoiceEngine(
                self._catalog,
                self._settings,
                self._profile,
                on_line=self.line_spoken.emit,
                on_status=self.status_changed.emit,
                on_error=self.error_raised.emit,
                on_profile=self.profile_switched.emit,
            )
            engine.start()
        except OcrUnavailable as exc:
            QMessageBox.critical(self, "No text recognition", str(exc))
            return False
        except SynthesisError as exc:
            QMessageBox.critical(self, "No voices", str(exc))
            return False
        except RuntimeError as exc:
            QMessageBox.critical(self, "Audio problem", str(exc))
            return False
        self._engine = engine
        return True

    def _toggle(self) -> None:
        if not self._ensure_engine():
            return
        active = self._engine.toggle()
        self.status_tab.set_running(active)
        self.tray_toggle.setText("Pause" if active else "Start listening")
        self.tray.setIcon(build_icon(active))
        self.setWindowIcon(build_icon(active))

    def _silence(self) -> None:
        if self._engine is not None:
            self._engine.silence()

    def _on_settings_changed(self) -> None:
        try:
            self._settings.save()
        except OSError as exc:
            log.error("could not save settings: %s", exc)
            return
        if self._engine is not None:
            self.status_tab.set_status(
                "Saved. The audio device takes effect next time reading starts."
            )

    def _skip_line(self) -> None:
        if self._engine is None:
            return
        if self._engine.skip_line():
            self.status_tab.set_status("Skipped to the next line")
        else:
            self.status_tab.set_status("Nothing is being spoken")

    def _replay_line(self) -> None:
        """Read a line again in full - the selected row, or the last one."""
        if not self._ensure_engine():
            return
        chosen = self.status_tab.selected_utterance()
        if self._engine.replay(chosen):
            source = "selected line" if chosen is not None else "last line"
            self.status_tab.set_status(f"Reading the {source} in full")
        else:
            self.status_tab.set_status("No line to read yet")

    def _preview(self, text: str, speaker: str) -> None:
        if not self._ensure_engine():
            return
        self._engine.speak_sample(text, speaker)

    def _refresh_stats(self) -> None:
        if self._engine is None:
            return
        stats = self._engine.stats
        self.status_tab.set_stats(
            f"{stats.lines_spoken} lines spoken - "
            f"{stats.lines_seen} detected - "
            f"OCR {stats.average_ocr_ms:.0f} ms average - "
            f"upscale {stats.upscale:.2f}x - "
            f"{stats.frames_skipped} frames unchanged - "
            f"{stats.scans} scans"
        )
        self.capture_tab.show_auto_region(self._engine.detected_region)
        window = self._engine.last_window
        if window is not None:
            self.status_tab.set_window(
                f"{window.title or '(untitled)'}  [{window.exe or '?'}]"
            )

    # -- signal handlers ---------------------------------------------------

    def _on_line(self, line: SpokenLine) -> None:
        self.status_tab.add_line(line)
        if line.utterance.has_speaker:
            self.voices_tab.note_speaker(
                line.utterance.speaker, line.assignment.voice.key
            )

    def _on_error(self, message: str) -> None:
        self.status_tab.set_status(message, error=True)

    def _on_engine_switched_profile(self, profile: Profile) -> None:
        """The engine auto-detected a different game; follow it in the UI."""
        if profile.name == self._profile.name:
            return
        self._profile = profile
        self._profiles[profile.name] = profile
        index = self.profile_combo.findData(profile.name)
        if index < 0:
            self._refill_profiles()
            index = self.profile_combo.findData(profile.name)
        self.profile_combo.blockSignals(True)
        self.profile_combo.setCurrentIndex(max(index, 0))
        self.profile_combo.blockSignals(False)
        self._load_profile(profile, push_to_engine=False)

    # -- profiles ----------------------------------------------------------

    def _load_profile(self, profile: Profile, push_to_engine: bool = True) -> None:
        self._profile = profile
        # Remembered so a rename can remove the file the profile used to live
        # in, rather than leaving a duplicate behind under the old name.
        self._loaded_name = profile.name
        self.status_tab.set_profile(profile.title or profile.name)
        self.voices_tab.load_profile(profile)
        self.capture_tab.load_profile(profile)
        self.profiles_tab.set_profiles(self._profiles, profile.name)
        self.profiles_tab.load_profile(profile)
        self._dirty = False
        if push_to_engine and self._engine is not None:
            self._engine.set_profile(profile, announce=False)

    def _select_profile_by_name(self, name: str) -> None:
        profile = self._profiles.get(name)
        if profile is None or profile.name == self._profile.name:
            return
        index = self.profile_combo.findData(name)
        if index >= 0:
            self.profile_combo.blockSignals(True)
            self.profile_combo.setCurrentIndex(index)
            self.profile_combo.blockSignals(False)
        self._load_profile(profile)

    def _duplicate_profile(self) -> None:
        from copy import deepcopy

        self._apply_ui_to_profile()
        copy = deepcopy(self._profile)
        base = f"{self._profile.name}-copy"
        name = base
        suffix = 2
        while name in self._profiles:
            name = f"{base}-{suffix}"
            suffix += 1
        copy.name = name
        copy.title = f"{self._profile.title or self._profile.name} (copy)"

        self._profiles[copy.name] = copy
        self._refill_profiles()
        index = self.profile_combo.findData(copy.name)
        self.profile_combo.blockSignals(True)
        self.profile_combo.setCurrentIndex(max(index, 0))
        self.profile_combo.blockSignals(False)
        self._load_profile(copy)
        self._save_profile()

    def _delete_profile(self) -> None:
        target = self._profile
        shipped = is_shipped(target)
        if target.name == DEFAULT_PROFILE_NAME and not shipped:
            QMessageBox.information(
                self, "Cannot delete",
                "The Default profile is what every unrecognised game falls back "
                "to, so it cannot be removed.",
            )
            return

        question = (
            f"Reset '{target.title or target.name}' to the version that ships "
            "with GameVoice? Your changes to it are discarded."
            if shipped
            else f"Delete the profile '{target.title or target.name}'? "
            "Its capture area, voice choices and overrides go with it."
        )
        if QMessageBox.question(
            self, "Reset profile" if shipped else "Delete profile", question
        ) != QMessageBox.StandardButton.Yes:
            return

        try:
            outcome = delete_profile(target)
        except OSError as exc:
            QMessageBox.critical(self, "Could not delete", str(exc))
            return

        self._dirty = False
        self._profiles = list_profiles()
        self._refill_profiles()
        fallback = self._profiles.get(target.name) or self._profiles.get(
            DEFAULT_PROFILE_NAME
        ) or Profile()
        index = self.profile_combo.findData(fallback.name)
        self.profile_combo.blockSignals(True)
        self.profile_combo.setCurrentIndex(max(index, 0))
        self.profile_combo.blockSignals(False)
        self._load_profile(fallback)
        self.status_tab.set_status(
            {"reset": "Profile reset to the built-in version.",
             "deleted": "Profile deleted.",
             "unchanged": "Nothing to remove - that profile was never saved."}[outcome]
        )

    def _detect_into_profile(self) -> None:
        from ..capture import foreground_window
        from ..profiles import normalize_title

        window = foreground_window()
        if not window.is_usable:
            QMessageBox.information(
                self, "No game found",
                "Bring the game to the front, then press this again.",
            )
            return
        self.profiles_tab.fill_from_window(
            window.exe, normalize_title(window.title)
        )
        self.status_tab.set_status(
            f"Filled from {window.exe or 'the current window'}"
        )

    def _apply_ui_to_profile(self) -> None:
        self.voices_tab.apply_to(self._profile)
        self.capture_tab.apply_to(self._profile)
        self.profiles_tab.apply_to(self._profile)
        self._settings.engine = self._profile.speech.engine

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._apply_ui_to_profile()
        self.profiles_tab.refresh_summary()
        if self._engine is not None:
            self._engine.set_profile(self._profile, announce=False)

    def _profile_selected(self, _index: int) -> None:
        name = self.profile_combo.currentData()
        profile = self._profiles.get(name)
        if profile is not None:
            self._load_profile(profile)

    def _new_profile(self) -> None:
        from ..capture import foreground_window

        window = foreground_window()
        if not window.is_usable:
            QMessageBox.information(
                self, "No game found",
                "Bring the game to the front, then use this button.\n"
                "GameVoice reads whichever window was in front last.",
            )
            return

        self._apply_ui_to_profile()
        created = profile_for_window(window, self._profile)
        if created.name in self._profiles:
            answer = QMessageBox.question(
                self, "Profile exists",
                f"A profile called '{created.name}' already exists. Replace it?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self._profiles[created.name] = created
        self._refill_profiles()
        index = self.profile_combo.findData(created.name)
        self.profile_combo.setCurrentIndex(max(index, 0))
        self._load_profile(created)
        self._save_profile()

    def _save_profile(self) -> None:
        self._apply_ui_to_profile()
        previous = getattr(self, "_loaded_name", self._profile.name)
        renamed = previous and previous != self._profile.name

        try:
            path = save_profile(self._profile)
        except OSError as exc:
            QMessageBox.critical(self, "Could not save", str(exc))
            return

        if renamed:
            # Drop the file it used to live in, or the old name lingers in the
            # list as a stale duplicate.
            old = self._profiles.pop(previous, None)
            if old is not None and not is_shipped(old):
                try:
                    delete_profile(old)
                except OSError as exc:
                    log.warning("could not remove the old profile file: %s", exc)

        self._profiles[self._profile.name] = self._profile
        self._loaded_name = self._profile.name
        self._dirty = False
        self._refill_profiles()
        index = self.profile_combo.findData(self._profile.name)
        self.profile_combo.blockSignals(True)
        self.profile_combo.setCurrentIndex(max(index, 0))
        self.profile_combo.blockSignals(False)
        self.profiles_tab.set_profiles(self._profiles, self._profile.name)
        self.status_tab.set_status(
            f"{'Renamed and saved' if renamed else 'Saved'} to {path}"
        )

    # -- region picking ----------------------------------------------------

    def _pick(self, which: str) -> None:
        current = (
            self._profile.capture.text_region
            if which == "text"
            else self._profile.capture.speaker_region
        )
        was_visible = self.isVisible()
        self.hide()

        def done(region: Region | None) -> None:
            self._picker = None
            if was_visible:
                self.show()
                self.raise_()
            if region is not None:
                self.capture_tab.set_region(region, which)

        # A short delay lets this window finish hiding, so the overlay does not
        # capture GameVoice itself in the area being framed.
        QTimer.singleShot(220, lambda: self._show_picker(current, done))

    def _show_picker(self, current: Region | None, done) -> None:
        self._picker = pick_region(current, done)

    def _open_downloads(self) -> None:
        dialog = DownloadDialog(self)
        if dialog.exec() == DownloadDialog.DialogCode.Accepted:
            QMessageBox.information(
                self, "Voices installed",
                "Restart GameVoice to load the new voices.",
            )

    # -- window lifecycle --------------------------------------------------

    DEFAULT_SIZE = (940, 760)

    def _restore_geometry(self) -> None:
        """Put the window somewhere the user can actually see it.

        A remembered position is only honoured if it still lands on a connected
        screen - unplugging a monitor would otherwise strand the window off the
        desktop with no way back. With nothing remembered, centre on the primary
        screen rather than letting Qt choose, which on a multi-monitor desktop
        tends to land on whichever screen the user is not looking at.
        """
        width = self._settings.window_w or self.DEFAULT_SIZE[0]
        height = self._settings.window_h or self.DEFAULT_SIZE[1]
        self.resize(width, height)

        left, top = self._settings.window_x, self._settings.window_y
        if left is not None and top is not None:
            remembered = QRect(left, top, width, height)
            for screen in QGuiApplication.screens():
                # Require a real overlap, not a single shared pixel, so a
                # barely-visible sliver still counts as lost.
                overlap = screen.availableGeometry().intersected(remembered)
                if overlap.width() > 120 and overlap.height() > 60:
                    self.move(left, top)
                    return
            log.info("saved window position %s is off-screen; recentring", remembered)

        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self.move(
            area.left() + max((area.width() - width) // 2, 0),
            area.top() + max((area.height() - height) // 3, 0),
        )

    def _save_geometry(self) -> None:
        if self.isMinimized() or self.isMaximized():
            return
        frame = self.geometry()
        self._settings.window_x = frame.left()
        self._settings.window_y = frame.top()
        self._settings.window_w = frame.width()
        self._settings.window_h = frame.height()

    def _restore(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event) -> None:
        if self.tray.isVisible():
            self.hide()
            self.tray.showMessage(
                APP_TITLE,
                "Still running. Use the tray icon to stop it.",
                build_icon(),
                3000,
            )
            event.ignore()
            return
        self._shutdown()
        super().closeEvent(event)

    def _quit(self) -> None:
        self._shutdown()
        QApplication.instance().quit()

    def _shutdown(self) -> None:
        self._stats_timer.stop()
        self._hotkeys.unregister_all()
        self._save_geometry()
        try:
            self._settings.save()
        except OSError as exc:
            log.error("could not save settings on exit: %s", exc)
        if self._dirty:
            try:
                save_profile(self._profile)
            except OSError as exc:
                log.error("could not save profile on exit: %s", exc)
        if self._engine is not None:
            self._engine.stop()
            self._engine = None
        self.tray.hide()


def _claim_taskbar_identity() -> None:
    """Give the app its own taskbar identity.

    A pythonw.exe process inherits Python's icon and groups under it. Setting an
    explicit AppUserModelID makes Windows treat GameVoice as its own
    application, so the taskbar and any pinned shortcut use our icon.
    """
    try:
        import ctypes

        ctypes.WinDLL("shell32").SetCurrentProcessExplicitAppUserModelID(
            "TheMrMackay.GameVoice.Reader.1"
        )
    except (OSError, AttributeError) as exc:
        log.debug("could not set the taskbar identity: %s", exc)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    set_dpi_aware()
    _claim_taskbar_identity()

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setWindowIcon(build_icon())
    app.setQuitOnLastWindowClosed(False)

    settings = AppSettings.load()
    catalog = VoiceCatalog()

    window = MainWindow(settings, catalog)
    window.show()

    if catalog.is_empty:
        QMessageBox.information(
            window,
            "No voices yet",
            "GameVoice has no neural voices installed.\n\n"
            "Open the Voices tab and choose 'Download more voices'. "
            "Until then it will fall back to the Windows built-in voices.",
        )

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
