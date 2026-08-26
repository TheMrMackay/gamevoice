"""Downloading extra Piper voices.

A newly downloaded model has no gender measurements, and the router needs those
to build its male and female pools, so this dialog runs the measurement pass
straight after a download rather than leaving the voices unusable-looking.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ..config import VOICES_DIR

log = logging.getLogger(__name__)

VOICES_JSON = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json?download=true"
)
NETWORK_TIMEOUT = 30

# Multi-speaker models are what make per-character voices practical, so they are
# called out rather than buried in an alphabetical list.
RECOMMENDED = {
    "en_US-libritts_r-medium": "904 voices in one model - best for many characters",
    "en_GB-vctk-medium": "109 British voices in one model",
    "en_US-ryan-high": "Male, high quality",
    "en_US-hfc_female-medium": "Female, clear",
    "en_GB-alan-medium": "Male, British",
    "en_GB-jenny_dioco-medium": "Female, British",
}


class _Worker(QObject):
    progress = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, names: list[str]) -> None:
        super().__init__()
        self._names = names

    def run(self) -> None:
        try:
            from piper.download_voices import download_voice
        except ImportError as exc:
            self.finished.emit(False, f"Piper is not installed: {exc}")
            return

        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        for name in self._names:
            self.progress.emit(f"Downloading {name} ...")
            try:
                download_voice(name, VOICES_DIR)
            except Exception as exc:
                self.finished.emit(False, f"Could not download {name}: {exc}")
                return

        self.progress.emit("Measuring voice pitch to sort male and female ...")
        try:
            self._measure()
        except Exception as exc:
            self.finished.emit(
                False,
                f"Voices downloaded, but measuring them failed: {exc}\n"
                "Run 'gamevoice measure' to finish.",
            )
            return

        self.finished.emit(True, "Done. Restart GameVoice to use the new voices.")

    @staticmethod
    def _measure() -> None:
        import subprocess
        import sys
        from pathlib import Path

        script = Path(__file__).resolve().parents[2] / "tools" / "build_voice_table.py"
        result = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip()[:400] or "measurement failed")


class DownloadDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Download voices")
        self.resize(620, 480)
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        self._build()
        self._populate()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Voices are downloaded once and used offline. A multi-speaker "
                "model gives hundreds of distinct characters from a single file."
            )
        )

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        layout.addWidget(self.list, stretch=1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        row = QHBoxLayout()
        self.refresh_button = QPushButton("Show all available voices")
        row.addWidget(self.refresh_button)
        row.addStretch(1)
        layout.addLayout(row)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Download")
        self.buttons.accepted.connect(self._start)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.refresh_button.clicked.connect(self._load_full_list)

    def _populate(self) -> None:
        installed = {path.stem for path in VOICES_DIR.glob("*.onnx")}
        for name, description in RECOMMENDED.items():
            item = QListWidgetItem(f"{name} - {description}")
            item.setData(256, name)
            if name in installed:
                item.setText(f"{name} - installed")
                item.setFlags(item.flags() & ~item.flags().ItemIsEnabled)
            self.list.addItem(item)

    def _load_full_list(self) -> None:
        self.status.setText("Fetching the voice list ...")
        try:
            with urllib.request.urlopen(VOICES_JSON, timeout=NETWORK_TIMEOUT) as response:
                catalogue = json.load(response)
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            self.status.setText(f"Could not fetch the voice list: {exc}")
            return

        installed = {path.stem for path in VOICES_DIR.glob("*.onnx")}
        self.list.clear()
        for name, entry in sorted(catalogue.items()):
            speakers = entry.get("num_speakers", 1)
            note = f"{speakers} speakers" if speakers > 1 else entry.get("quality", "")
            item = QListWidgetItem(f"{name} - {note}")
            item.setData(256, name)
            if name in installed:
                item.setText(f"{name} - installed")
                item.setFlags(item.flags() & ~item.flags().ItemIsEnabled)
            self.list.addItem(item)
        self.status.setText(f"{len(catalogue)} voices available.")

    def _start(self) -> None:
        names = [item.data(256) for item in self.list.selectedItems()]
        if not names:
            self.status.setText("Select at least one voice.")
            return

        self.buttons.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.progress.setVisible(True)

        self._thread = QThread(self)
        self._worker = _Worker(names)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status.setText)
        self._worker.finished.connect(self._done)
        self._thread.start()

    def _done(self, ok: bool, message: str) -> None:
        self.status.setText(message)
        self.progress.setVisible(False)
        self.buttons.setEnabled(True)
        self.refresh_button.setEnabled(True)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None
        self._worker = None
        if ok:
            self.accept()

    def closeEvent(self, event) -> None:
        if self._thread is not None and self._thread.isRunning():
            self.status.setText("Waiting for the download to finish ...")
            event.ignore()
            return
        super().closeEvent(event)
