"""The running view: one big button, and a transcript of what was spoken.

The transcript matters more than it looks. When a game is not being read
correctly the user needs to see *what the tool heard*, and seeing the wrong
speaker attached to a line is what tells them to set an override.
"""
from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..engine import SpokenLine

MAX_ROWS = 300


class StatusTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)

        self.toggle_button = QPushButton("Start listening")
        self.toggle_button.setMinimumHeight(52)
        font = self.toggle_button.font()
        font.setPointSize(12)
        font.setBold(True)
        self.toggle_button.setFont(font)
        layout.addWidget(self.toggle_button)

        row = QHBoxLayout()
        self.skip_button = QPushButton("Next line")
        self.skip_button.setToolTip(
            "Stop this line and start the next one waiting. The queue keeps "
            "playing, unlike Silence."
        )
        self.replay_button = QPushButton("Read whole line")
        self.replay_button.setToolTip(
            "Read a line again from the start, in full. Uses the row selected "
            "in the transcript, or the most recent line if nothing is selected."
        )
        self.silence_button = QPushButton("Silence")
        self.silence_button.setToolTip("Stop speaking and drop everything queued.")
        self.clear_button = QPushButton("Clear transcript")
        for button in (
            self.skip_button, self.replay_button,
            self.silence_button, self.clear_button,
        ):
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)

        self.profile_label = QLabel("Profile: -")
        self.window_label = QLabel("Game window: -")
        self.window_label.setStyleSheet("color: #888;")
        layout.addWidget(self.profile_label)
        layout.addWidget(self.window_label)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Time", "Speaker", "Voice", "Line"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setWordWrap(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, stretch=1)

        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #888;")
        layout.addWidget(self.stats_label)

        self.status_label = QLabel("Not started")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    # -- updates -----------------------------------------------------------

    def set_running(self, active: bool) -> None:
        self.toggle_button.setText("Pause" if active else "Start listening")

    def set_status(self, message: str, error: bool = False) -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #c0392b;" if error else "color: #444;")

    def set_profile(self, title: str) -> None:
        self.profile_label.setText(f"Profile: {title}")

    def set_window(self, description: str) -> None:
        self.window_label.setText(f"Game window: {description}")

    def set_stats(self, text: str) -> None:
        self.stats_label.setText(text)

    def add_line(self, line: SpokenLine) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

        speaker = line.utterance.speaker or "narrator"
        cells = [
            time.strftime("%H:%M:%S", time.localtime(line.at)),
            speaker,
            line.voice,
            line.utterance.text,
        ]
        for column, text in enumerate(cells):
            item = QTableWidgetItem(text)
            if column == 1 and not line.utterance.has_speaker:
                item.setForeground(QColor("#888"))
                item.setFont(_italic(item.font()))
            if column == 3:
                item.setToolTip(text)
            if column == 0:
                # Keep the whole utterance on the row so it can be replayed in
                # full, not just the text as it was truncated into the column.
                item.setData(Qt.ItemDataRole.UserRole, line.utterance)
            self.table.setItem(row, column, item)

        while self.table.rowCount() > MAX_ROWS:
            self.table.removeRow(0)
        self.table.scrollToBottom()

    def selected_utterance(self):
        """The utterance on the selected transcript row, or None."""
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def clear(self) -> None:
        self.table.setRowCount(0)


def _italic(font: QFont) -> QFont:
    italic = QFont(font)
    italic.setItalic(True)
    return italic
