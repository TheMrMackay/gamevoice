"""Drag a box over the screen to choose the capture area.

Covers the whole virtual desktop so a region can be drawn on any monitor, and
stays translucent so the game underneath is still visible while choosing - the
user needs to see the dialogue box they are framing.

**Coordinates are taken from Windows, not from Qt.** On a display at anything
other than 100% scaling the two disagree: at 125% Qt calls a 2560x1440 screen
2048x1152, so a rectangle drawn here and handed to the screen grabber would
capture the wrong area at four fifths of the resolution. ``GetCursorPos``
reports true pixels in a per-monitor DPI-aware process, so the drag is tracked
with it and Qt's coordinates are used only for drawing the overlay.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QKeyEvent, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..config import Region

_HINT = "Drag over the dialogue text.    Enter accepts    Esc cancels"

_user32 = ctypes.WinDLL("user32", use_last_error=True)


def cursor_position() -> tuple[int, int]:
    """The mouse position in real screen pixels."""
    point = wintypes.POINT()
    if not _user32.GetCursorPos(ctypes.byref(point)):
        return 0, 0
    return int(point.x), int(point.y)


class RegionPicker(QWidget):
    """A full-desktop overlay that reports the rectangle the user drew."""

    picked = Signal(object)  # Region or None

    def __init__(self, initial: Region | None = None) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setWindowOpacity(0.35)
        self.setCursor(Qt.CursorShape.CrossCursor)

        desktop = QRect()
        for screen in QGuiApplication.screens():
            desktop = desktop.united(screen.geometry())
        self._desktop = desktop
        self.setGeometry(desktop)

        self._origin = QPoint()
        self._current = QRect()
        self._dragging = False
        # The same drag, tracked in real screen pixels. This is what is
        # returned; self._current exists only to draw with.
        self._origin_px = (0, 0)
        self._current_px = (0, 0)

        if initial and initial.is_valid():
            self._current = QRect(
                initial.left - desktop.left(),
                initial.top - desktop.top(),
                initial.width,
                initial.height,
            )
            self._origin_px = (initial.left, initial.top)
            self._current_px = (
                initial.left + initial.width,
                initial.top + initial.height,
            )

    def _to_region(self) -> Region | None:
        (x1, y1), (x2, y2) = self._origin_px, self._current_px
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        if right - left < 12 or bottom - top < 12:
            return None
        return Region(left=left, top=top, width=right - left, height=bottom - top)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.position().toPoint()
            self._current = QRect(self._origin, self._origin)
            self._origin_px = cursor_position()
            self._current_px = self._origin_px
            self._dragging = True
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self._current = QRect(self._origin, event.position().toPoint())
            self._current_px = cursor_position()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._dragging:
            return
        self._dragging = False
        self._current = QRect(self._origin, event.position().toPoint())
        self._current_px = cursor_position()
        self.update()
        # Releasing is the natural "done" gesture; Enter is there for anyone
        # who wants to adjust before committing.
        region = self._to_region()
        if region is not None:
            self._finish(region)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._finish(None)
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._finish(self._to_region())

    def _finish(self, region: Region | None) -> None:
        self.picked.emit(region)
        self.close()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 10, 16))

        rect = self._current.normalized()
        if rect.width() > 2 and rect.height() > 2:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(rect, Qt.GlobalColor.transparent)
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver
            )
            painter.setPen(QPen(QColor(120, 210, 255), 2))
            painter.drawRect(rect)
            painter.setPen(QColor(230, 240, 255))
            # Report the size that will actually be captured, which on a scaled
            # display is larger than the rectangle Qt just drew.
            region = self._to_region()
            size = f"{region.width} x {region.height}" if region else "too small"
            painter.drawText(rect.left(), max(rect.top() - 8, 14), f"{size} px")

        painter.setPen(QColor(235, 240, 250))
        font = painter.font()
        font.setPointSize(15)
        painter.setFont(font)
        painter.drawText(
            self.rect().adjusted(0, 40, 0, 0), Qt.AlignmentFlag.AlignHCenter, _HINT
        )


def pick_region(initial: Region | None, on_done) -> RegionPicker:
    """Show the overlay. ``on_done`` receives a Region or None."""
    picker = RegionPicker(initial)
    picker.picked.connect(on_done)
    picker.showFullScreen()
    picker.raise_()
    picker.activateWindow()
    return picker
