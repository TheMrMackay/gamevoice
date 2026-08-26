"""Voice settings and per-character overrides.

Automatic assignment gets most characters right on its own, so this tab is
built around correcting the ones it does not: every speaker the tool has heard
appears in a table with the gender it guessed and the voice it chose, and both
are a dropdown away from being changed. Corrections are saved into the game's
profile, so they only ever have to be made once.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import Profile
from ..voices import FEMALE, MALE, NEUTRAL, UNKNOWN, VoiceCatalog, infer_gender

log = logging.getLogger(__name__)

PREVIEW_LINE = "We should not have come this way, and you know it."
_GENDER_CHOICES = [("Auto", ""), ("Male", MALE), ("Female", FEMALE), ("Neutral", NEUTRAL)]


class VoicesTab(QWidget):
    """Engine settings, default voices, and the per-speaker override table."""

    changed = Signal()
    preview_requested = Signal(str, str)  # text, speaker
    download_requested = Signal()

    def __init__(self, catalog: VoiceCatalog, parent=None) -> None:
        super().__init__(parent)
        self._catalog = catalog
        self._profile: Profile | None = None
        self._loading = False
        self._speakers: dict[str, str] = {}  # lowered name -> display name
        self._build()

    # -- construction ------------------------------------------------------

    def _voice_combo(self, allow_auto: bool = True) -> QComboBox:
        combo = QComboBox()
        combo.setMaxVisibleItems(24)
        if allow_auto:
            combo.addItem("Automatic", "")
        for voice in self._catalog.all_voices():
            combo.addItem(voice.label, voice.key)
        return combo

    def _build(self) -> None:
        layout = QVBoxLayout(self)

        engine_box = QGroupBox("Speech")
        form = QFormLayout(engine_box)

        self.engine_combo = QComboBox()
        self.engine_combo.addItem("Piper - neural voices (recommended)", "piper")
        self.engine_combo.addItem("Windows built-in voices", "sapi")
        form.addRow("Engine", self.engine_combo)

        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setRange(0.5, 2.0)
        self.rate_spin.setSingleStep(0.05)
        self.rate_spin.setValue(1.0)
        form.addRow("Speed", self.rate_spin)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(90)
        form.addRow("Volume", self.volume_slider)

        self.interrupt_check = QCheckBox(
            "Cut off the current line when new text appears"
        )
        self.interrupt_check.setChecked(False)
        self.interrupt_check.setToolTip(
            "Off by default. Lines are spoken to the end and the next one waits "
            "its turn, so nothing is clipped or overlapped. Turn this on only if "
            "you would rather keep pace with fast dialogue than hear every line "
            "in full."
        )
        form.addRow("", self.interrupt_check)

        self.announce_check = QCheckBox("Say the speaker's name before each line")
        form.addRow("", self.announce_check)
        layout.addWidget(engine_box)

        assign_box = QGroupBox("Voice assignment")
        assign_form = QFormLayout(assign_box)

        self.auto_check = QCheckBox(
            "Give every character their own voice automatically"
        )
        self.auto_check.setChecked(True)
        assign_form.addRow("", self.auto_check)

        self.variety_spin = QDoubleSpinBox()
        self.variety_spin.setRange(0.0, 0.25)
        self.variety_spin.setSingleStep(0.01)
        self.variety_spin.setValue(0.06)
        self.variety_spin.setToolTip(
            "How much the speaking pace varies between characters who share a voice."
        )
        assign_form.addRow("Pace variety", self.variety_spin)

        self.narrator_combo = self._voice_combo()
        self.male_combo = self._voice_combo()
        self.female_combo = self._voice_combo()
        assign_form.addRow("Narrator / unattributed", self.narrator_combo)
        assign_form.addRow("Default male", self.male_combo)
        assign_form.addRow("Default female", self.female_combo)

        pool = QLabel(
            f"{len(self._catalog.pool(MALE))} male and "
            f"{len(self._catalog.pool(FEMALE))} female voices available."
        )
        pool.setStyleSheet("color: #888;")
        assign_form.addRow("", pool)

        buttons = QHBoxLayout()
        self.download_button = QPushButton("Download more voices...")
        buttons.addWidget(self.download_button)
        buttons.addStretch(1)
        assign_form.addRow("", self._wrap(buttons))
        layout.addWidget(assign_box)

        speaker_box = QGroupBox("Characters heard so far")
        speaker_layout = QVBoxLayout(speaker_box)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Character", "Gender", "Voice", ""])
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        speaker_layout.addWidget(self.table)

        hint = QLabel(
            "Characters appear here as they speak. Changing a gender or a voice "
            "is remembered in this game's profile."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888;")
        speaker_layout.addWidget(hint)
        layout.addWidget(speaker_box, stretch=1)

        self._connect()

    @staticmethod
    def _wrap(inner) -> QWidget:
        holder = QWidget()
        holder.setLayout(inner)
        return holder

    def _connect(self) -> None:
        self.download_button.clicked.connect(self.download_requested.emit)
        for widget, signal in (
            (self.engine_combo, "currentIndexChanged"),
            (self.rate_spin, "valueChanged"),
            (self.volume_slider, "valueChanged"),
            (self.interrupt_check, "stateChanged"),
            (self.announce_check, "stateChanged"),
            (self.auto_check, "stateChanged"),
            (self.variety_spin, "valueChanged"),
            (self.narrator_combo, "currentIndexChanged"),
            (self.male_combo, "currentIndexChanged"),
            (self.female_combo, "currentIndexChanged"),
        ):
            getattr(widget, signal).connect(self._emit_changed)

    def _emit_changed(self, *_args) -> None:
        if not self._loading:
            self.changed.emit()

    # -- profile binding ---------------------------------------------------

    def load_profile(self, profile: Profile) -> None:
        self._loading = True
        try:
            self._profile = profile
            speech = profile.speech
            self._select(self.engine_combo, speech.engine)
            self.rate_spin.setValue(speech.rate)
            self.volume_slider.setValue(int(speech.volume * 100))
            self.interrupt_check.setChecked(speech.interrupt_on_new_line)
            self.announce_check.setChecked(speech.speak_speaker_name)
            self.auto_check.setChecked(speech.auto_assign)
            self.variety_spin.setValue(speech.pitch_variety)
            self._select(self.narrator_combo, speech.narrator_voice)
            self._select(self.male_combo, speech.male_voice)
            self._select(self.female_combo, speech.female_voice)

            self.table.setRowCount(0)
            self._speakers.clear()
            for name in sorted(set(profile.voice_overrides) | set(profile.gender_hints)):
                self.note_speaker(name)
        finally:
            self._loading = False

    def apply_to(self, profile: Profile) -> None:
        speech = profile.speech
        speech.engine = self.engine_combo.currentData() or "piper"
        speech.rate = self.rate_spin.value()
        speech.volume = self.volume_slider.value() / 100.0
        speech.interrupt_on_new_line = self.interrupt_check.isChecked()
        speech.speak_speaker_name = self.announce_check.isChecked()
        speech.auto_assign = self.auto_check.isChecked()
        speech.pitch_variety = self.variety_spin.value()
        speech.narrator_voice = self.narrator_combo.currentData() or ""
        speech.male_voice = self.male_combo.currentData() or ""
        speech.female_voice = self.female_combo.currentData() or ""

    @staticmethod
    def _select(combo: QComboBox, value: str) -> None:
        index = combo.findData(value or "")
        combo.setCurrentIndex(index if index >= 0 else 0)

    # -- speaker table -----------------------------------------------------

    def note_speaker(self, name: str, assigned_key: str = "") -> None:
        """Add a character to the table the first time they say something."""
        if not name:
            return
        key = name.lower()
        if key in self._speakers:
            return
        self._speakers[key] = name

        profile = self._profile or Profile()
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(name))

        gender_combo = QComboBox()
        for label, value in _GENDER_CHOICES:
            gender_combo.addItem(label, value)
        hinted = profile.gender_hints.get(key, "")
        gender_index = gender_combo.findData(hinted)
        gender_combo.setCurrentIndex(gender_index if gender_index >= 0 else 0)
        if not hinted:
            guessed = infer_gender(name)
            gender_combo.setItemText(
                0, "Auto" if guessed == UNKNOWN else f"Auto ({guessed})"
            )
        gender_combo.currentIndexChanged.connect(
            lambda _index, k=key, c=gender_combo: self._set_hint(k, c.currentData())
        )
        self.table.setCellWidget(row, 1, gender_combo)

        voice_combo = self._voice_combo()
        chosen = profile.voice_overrides.get(key, "")
        voice_index = voice_combo.findData(chosen)
        voice_combo.setCurrentIndex(voice_index if voice_index >= 0 else 0)
        if not chosen and assigned_key:
            voice_combo.setItemText(0, f"Automatic ({assigned_key})")
        voice_combo.currentIndexChanged.connect(
            lambda _index, k=key, c=voice_combo: self._set_override(k, c.currentData())
        )
        self.table.setCellWidget(row, 2, voice_combo)

        preview = QPushButton("Play")
        preview.clicked.connect(
            lambda _checked=False, n=name: self.preview_requested.emit(PREVIEW_LINE, n)
        )
        self.table.setCellWidget(row, 3, preview)

    def _set_hint(self, speaker: str, gender: str) -> None:
        if self._profile is None:
            return
        if gender:
            self._profile.gender_hints[speaker] = gender
        else:
            self._profile.gender_hints.pop(speaker, None)
        self._emit_changed()

    def _set_override(self, speaker: str, voice_key: str) -> None:
        if self._profile is None:
            return
        if voice_key:
            self._profile.voice_overrides[speaker] = voice_key
        else:
            self._profile.voice_overrides.pop(speaker, None)
        self._emit_changed()
