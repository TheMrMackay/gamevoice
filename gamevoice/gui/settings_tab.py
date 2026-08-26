"""Application settings: hotkeys, audio output, and start-up behaviour.

These are not per-game, so they live apart from the profile tabs.

Hotkeys are the reason this tab exists. They have to be global - the whole point
is pressing them with a game in front - and Windows refuses a combination another
application already holds. Typing one into a text file and finding out later that
it never worked is a bad way to learn that, so every change is registered
immediately and the result is shown next to the field.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..audio import list_output_devices
from ..config import AppSettings

log = logging.getLogger(__name__)

# Qt spells the Windows key "Meta"; the registration code calls it "win".
_QT_TO_SPEC = {"meta": "win"}
_SPEC_TO_QT = {"win": "meta", "super": "meta"}

HOTKEYS = (
    ("hotkey_toggle", "Start / pause", "Begin reading, or stop reading."),
    ("hotkey_stop", "Silence", "Stop speaking and drop everything queued."),
    ("hotkey_skip", "Next line", "Abandon this line and start the next one waiting."),
    ("hotkey_replay", "Read whole line", "Read the most recent line again, in full."),
)


def sequence_to_spec(sequence: QKeySequence) -> str:
    """QKeySequence -> the "ctrl+alt+v" form the registration code parses."""
    text = sequence.toString(QKeySequence.SequenceFormat.PortableText).strip()
    if not text:
        return ""
    # Only the first chord is usable; Windows hotkeys are a single combination.
    text = text.split(",")[0].strip()
    parts = [_QT_TO_SPEC.get(p.strip().lower(), p.strip().lower())
             for p in text.split("+") if p.strip()]
    return "+".join(parts)


def spec_to_sequence(spec: str) -> QKeySequence:
    if not spec:
        return QKeySequence()
    parts = [_SPEC_TO_QT.get(p.strip().lower(), p.strip().lower())
             for p in spec.split("+") if p.strip()]
    return QKeySequence("+".join(part.capitalize() for part in parts))


class SettingsTab(QWidget):
    hotkeys_changed = Signal()
    settings_changed = Signal()

    def __init__(self, settings: AppSettings, parent=None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._loading = False
        self._editors: dict[str, QKeySequenceEdit] = {}
        self._status: dict[str, QLabel] = {}
        self._build()
        self.load()

    def _build(self) -> None:
        layout = QVBoxLayout(self)

        keys = QGroupBox("Global hotkeys")
        form = QFormLayout(keys)
        for field, label, tip in HOTKEYS:
            editor = QKeySequenceEdit()
            editor.setMaximumSequenceLength(1)
            editor.setToolTip(tip)
            editor.keySequenceChanged.connect(
                lambda _seq, f=field: self._on_key_edited(f)
            )

            clear = QPushButton("Clear")
            clear.setFixedWidth(64)
            clear.setToolTip("Leave this action without a hotkey.")
            clear.clicked.connect(lambda _c=False, f=field: self._clear(f))

            status = QLabel("")
            status.setMinimumWidth(280)
            status.setWordWrap(True)

            row = QHBoxLayout()
            row.addWidget(editor, stretch=1)
            row.addWidget(clear)
            row.addWidget(status, stretch=2)
            holder = QWidget()
            holder.setLayout(row)
            form.addRow(label, holder)

            self._editors[field] = editor
            self._status[field] = status

        note = QLabel(
            "These work while a game has focus, which is why Windows has to "
            "grant them exclusively. If a combination is already taken by "
            "something else, it is reported here rather than failing silently."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888;")
        form.addRow("", note)
        layout.addWidget(keys)

        audio = QGroupBox("Audio")
        audio_form = QFormLayout(audio)
        self.device_combo = QComboBox()
        self.device_combo.addItem("System default", None)
        for index, name in list_output_devices():
            self.device_combo.addItem(f"[{index}] {name}", index)
        self.device_combo.setToolTip(
            "Where speech is played. Takes effect the next time reading starts."
        )
        audio_form.addRow("Output device", self.device_combo)
        layout.addWidget(audio)

        behaviour = QGroupBox("Behaviour")
        behaviour_form = QFormLayout(behaviour)
        self.auto_switch_check = QCheckBox(
            "Switch profile automatically when the game changes"
        )
        self.start_paused_check = QCheckBox("Start paused")
        behaviour_form.addRow("", self.auto_switch_check)
        behaviour_form.addRow("", self.start_paused_check)
        layout.addWidget(behaviour)

        self.location_label = QLabel("")
        self.location_label.setStyleSheet("color: #888;")
        self.location_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.location_label.setWordWrap(True)
        layout.addWidget(self.location_label)
        layout.addStretch(1)

        for widget in (self.device_combo,):
            widget.currentIndexChanged.connect(self._on_setting_edited)
        for widget in (self.auto_switch_check, self.start_paused_check):
            widget.stateChanged.connect(self._on_setting_edited)

    # -- binding -----------------------------------------------------------

    def load(self) -> None:
        self._loading = True
        try:
            for field, _label, _tip in HOTKEYS:
                self._editors[field].setKeySequence(
                    spec_to_sequence(getattr(self._settings, field, ""))
                )
            index = self.device_combo.findData(self._settings.output_device)
            self.device_combo.setCurrentIndex(index if index >= 0 else 0)
            self.auto_switch_check.setChecked(self._settings.auto_switch_profile)
            self.start_paused_check.setChecked(self._settings.start_paused)
            self.location_label.setText(f"Saved in {AppSettings.path()}")
        finally:
            self._loading = False

    def _on_key_edited(self, field: str) -> None:
        if self._loading:
            return
        setattr(self._settings, field, sequence_to_spec(
            self._editors[field].keySequence()
        ))
        self.hotkeys_changed.emit()

    def _clear(self, field: str) -> None:
        self._editors[field].clear()
        setattr(self._settings, field, "")
        self._status[field].setText("Not bound")
        self._status[field].setStyleSheet("color: #888;")
        self.hotkeys_changed.emit()

    def _on_setting_edited(self, *_args) -> None:
        if self._loading:
            return
        self._settings.output_device = self.device_combo.currentData()
        self._settings.auto_switch_profile = self.auto_switch_check.isChecked()
        self._settings.start_paused = self.start_paused_check.isChecked()
        self.settings_changed.emit()

    def report(self, failures: dict[str, str]) -> None:
        """Show, per hotkey, whether Windows actually granted it."""
        for field, _label, _tip in HOTKEYS:
            spec = getattr(self._settings, field, "")
            status = self._status[field]
            if not spec:
                status.setText("Not bound")
                status.setStyleSheet("color: #888;")
            elif spec in failures:
                status.setText(failures[spec])
                status.setStyleSheet("color: #c0392b;")
            else:
                status.setText("Active")
                status.setStyleSheet("color: #1e8449;")
