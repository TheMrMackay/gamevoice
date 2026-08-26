"""Creating, editing, renaming and removing game profiles.

Everything about *which* game a profile belongs to lives here - its name, the
executable and window titles that select it. The voice and capture settings for
the selected profile stay on their own tabs, so this one answers a single
question: which game is this, and how is it recognised.

Deleting has two meanings and the interface says which applies. A profile you
made is removed. A profile that ships with GameVoice cannot be removed, only
reset to the version that came with the app.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import Profile, is_shipped

log = logging.getLogger(__name__)


def _lines_to_list(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


class ProfilesTab(QWidget):
    selected = Signal(str)          # profile name
    changed = Signal()              # the edited profile was modified
    create_requested = Signal()
    duplicate_requested = Signal()
    delete_requested = Signal()
    detect_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._profile: Profile | None = None
        self._loading = False
        self._build()

    def _build(self) -> None:
        layout = QHBoxLayout(self)

        left = QVBoxLayout()
        left.addWidget(QLabel("Profiles"))
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setMinimumWidth(240)
        left.addWidget(self.list, stretch=1)

        buttons = QHBoxLayout()
        self.new_button = QPushButton("New")
        self.new_button.setToolTip(
            "Create a profile for whichever game was in front last."
        )
        self.duplicate_button = QPushButton("Duplicate")
        self.delete_button = QPushButton("Delete")
        for button in (self.new_button, self.duplicate_button, self.delete_button):
            buttons.addWidget(button)
        left.addLayout(buttons)
        layout.addLayout(left)

        right = QVBoxLayout()

        identity = QGroupBox("This profile")
        form = QFormLayout(identity)

        self.name_edit = QLineEdit()
        self.name_edit.setToolTip(
            "The profile's identity. Changing it renames the profile; the old "
            "file is removed when you save."
        )
        form.addRow("Name", self.name_edit)

        self.title_edit = QLineEdit()
        self.title_edit.setToolTip("What this profile is called in the list.")
        form.addRow("Shown as", self.title_edit)

        self.origin_label = QLabel("")
        self.origin_label.setStyleSheet("color: #888;")
        self.origin_label.setWordWrap(True)
        form.addRow("", self.origin_label)
        right.addWidget(identity)

        matching = QGroupBox("Recognise this game by")
        match_form = QFormLayout(matching)

        self.exe_edit = QPlainTextEdit()
        self.exe_edit.setMaximumHeight(80)
        self.exe_edit.setPlaceholderText("thegame.exe\nlauncher.exe")
        self.exe_edit.setToolTip(
            "One executable name per line. An exact match here beats any title "
            "match, so it is the most reliable way to pin a profile to a game."
        )
        match_form.addRow("Executable", self.exe_edit)

        self.title_match_edit = QPlainTextEdit()
        self.title_match_edit.setMaximumHeight(80)
        self.title_match_edit.setPlaceholderText("Part of the window title")
        self.title_match_edit.setToolTip(
            "One fragment per line; a window title containing any of them "
            "matches. The longest matching fragment wins."
        )
        match_form.addRow("Window title", self.title_match_edit)

        self.detect_button = QPushButton("Fill from the current game window")
        match_form.addRow("", self.detect_button)

        hint = QLabel(
            "Leave both empty and this profile is only ever chosen by hand. "
            "The Default profile is used whenever nothing else matches."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888;")
        match_form.addRow("", hint)
        right.addWidget(matching)

        summary = QGroupBox("Settings in this profile")
        summary_layout = QVBoxLayout(summary)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        right.addWidget(summary)

        right.addStretch(1)
        layout.addLayout(right, stretch=1)

        self._connect()

    def _connect(self) -> None:
        self.list.currentItemChanged.connect(self._on_selected)
        self.new_button.clicked.connect(self.create_requested.emit)
        self.duplicate_button.clicked.connect(self.duplicate_requested.emit)
        self.delete_button.clicked.connect(self.delete_requested.emit)
        self.detect_button.clicked.connect(self.detect_requested.emit)
        for widget in (self.name_edit, self.title_edit):
            widget.textEdited.connect(self._on_edited)
        for widget in (self.exe_edit, self.title_match_edit):
            widget.textChanged.connect(self._on_edited)

    # -- population --------------------------------------------------------

    def set_profiles(self, profiles: dict[str, Profile], current: str) -> None:
        self._loading = True
        try:
            self.list.clear()
            for name, profile in sorted(
                profiles.items(), key=lambda pair: pair[1].title or pair[0]
            ):
                label = profile.title or name
                if is_shipped(profile):
                    label = f"{label}   (built in)"
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, name)
                self.list.addItem(item)
                if name == current:
                    self.list.setCurrentItem(item)
        finally:
            self._loading = False

    def load_profile(self, profile: Profile) -> None:
        self._loading = True
        try:
            self._profile = profile
            self.name_edit.setText(profile.name)
            self.title_edit.setText(profile.title)
            self.exe_edit.setPlainText("\n".join(profile.match_exe))
            self.title_match_edit.setPlainText("\n".join(profile.match_title))

            shipped = is_shipped(profile)
            self.delete_button.setText("Reset" if shipped else "Delete")
            self.origin_label.setText(
                "Built in. Your changes are saved separately and override it; "
                "Reset discards them and restores the original."
                if shipped
                else f"Your profile. Saved as {profile.path().name}."
            )
            self._update_summary(profile)
        finally:
            self._loading = False

    def _update_summary(self, profile: Profile) -> None:
        capture = profile.capture
        speech = profile.speech
        area = (
            f"{capture.text_region.width}x{capture.text_region.height}"
            if capture.text_region and capture.text_region.is_valid()
            else f"automatic, bottom {capture.bottom_band:.0%}"
        )
        trigger = (
            "when the picture changes" if capture.trigger == "change"
            else f"every {1 / max(capture.fps, 0.1):.2f}s"
        )
        self.summary_label.setText(
            f"Capture area: {area}\n"
            f"Reads: {trigger}, checked {capture.fps:g} times a second\n"
            f"Upscale: {'automatic' if capture.auto_upscale else f'{capture.upscale:g}x fixed'}\n"
            f"Voices: {'per character' if speech.auto_assign else 'fixed male/female'}, "
            f"{len(profile.voice_overrides)} override(s), "
            f"{len(profile.gender_hints)} gender hint(s)\n"
            f"Interrupts lines: {'yes' if speech.interrupt_on_new_line else 'no'}"
        )

    # -- editing -----------------------------------------------------------

    def _on_selected(self, current: QListWidgetItem | None, _previous) -> None:
        if self._loading or current is None:
            return
        name = current.data(Qt.ItemDataRole.UserRole)
        if name:
            self.selected.emit(name)

    def _on_edited(self, *_args) -> None:
        if self._loading or self._profile is None:
            return
        self.apply_to(self._profile)
        self.changed.emit()

    def apply_to(self, profile: Profile) -> None:
        name = self.name_edit.text().strip()
        if name:
            profile.name = name
        profile.title = self.title_edit.text().strip() or profile.name
        profile.match_exe = _lines_to_list(self.exe_edit.toPlainText())
        profile.match_title = _lines_to_list(self.title_match_edit.toPlainText())

    def fill_from_window(self, exe: str, title: str) -> None:
        if exe:
            self.exe_edit.setPlainText(exe)
        if title:
            self.title_match_edit.setPlainText(title)
            if not self.title_edit.text().strip():
                self.title_edit.setText(title)

    def refresh_summary(self) -> None:
        if self._profile is not None:
            self._update_summary(self._profile)
