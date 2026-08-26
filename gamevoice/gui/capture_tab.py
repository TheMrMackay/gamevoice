"""Where to look on screen, and a way to prove it is looking in the right place.

The "Test capture now" button is the important control. Without it, a region
that is slightly wrong produces silence, and silence is indistinguishable from
every other failure. With it, the user sees the exact crop and the exact text
the recogniser read back.
"""
from __future__ import annotations

import logging
import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..capture import ScreenGrabber, foreground_window, looks_blank, resolve_regions
from ..config import Profile, Region
from ..dialogue import parse
from ..ocr import (
    AdaptiveUpscale,
    OcrUnavailable,
    clamp_scale,
    clipped_edges,
    create_engine,
    preprocess,
)

log = logging.getLogger(__name__)

# The useful range of change thresholds spans two orders of magnitude, so the
# slider is logarithmic - a linear one would spend most of its travel in values
# nobody wants.
_LOOSEST_CHANGE = 0.05
_TIGHTEST_CHANGE = 0.0002


def slider_to_threshold(value: int) -> float:
    span = math.log10(_TIGHTEST_CHANGE) - math.log10(_LOOSEST_CHANGE)
    fraction = (min(max(value, 1), 100) - 1) / 99.0
    return 10 ** (math.log10(_LOOSEST_CHANGE) + fraction * span)


def threshold_to_slider(threshold: float) -> int:
    threshold = min(max(threshold, _TIGHTEST_CHANGE), _LOOSEST_CHANGE)
    span = math.log10(_TIGHTEST_CHANGE) - math.log10(_LOOSEST_CHANGE)
    fraction = (math.log10(threshold) - math.log10(_LOOSEST_CHANGE)) / span
    return int(round(1 + fraction * 99))


class CaptureTab(QWidget):
    changed = Signal()
    pick_text_region = Signal()
    pick_speaker_region = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._profile: Profile | None = None
        self._loading = False
        self._grabber = ScreenGrabber()
        self._ocr = None
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)

        area = QGroupBox("Capture area")
        form = QFormLayout(area)

        self.monitor_combo = QComboBox()
        for index, monitor in enumerate(self._grabber.monitors()):
            if index == 0:
                self.monitor_combo.addItem("All monitors combined", 0)
            else:
                self.monitor_combo.addItem(
                    f"Monitor {index} - {monitor['width']}x{monitor['height']}", index
                )
        form.addRow("Monitor", self.monitor_combo)

        self.text_region_label = QLabel("Automatic (bottom of the game window)")
        text_row = QHBoxLayout()
        self.pick_text_button = QPushButton("Pick dialogue area...")
        self.clear_text_button = QPushButton("Use automatic")
        text_row.addWidget(self.pick_text_button)
        text_row.addWidget(self.clear_text_button)
        text_row.addStretch(1)
        form.addRow("Dialogue text", self.text_region_label)
        form.addRow("", self._wrap(text_row))

        self.speaker_region_label = QLabel("Not set - name is read from the text")
        speaker_row = QHBoxLayout()
        self.pick_speaker_button = QPushButton("Pick name box...")
        self.clear_speaker_button = QPushButton("Clear")
        speaker_row.addWidget(self.pick_speaker_button)
        speaker_row.addWidget(self.clear_speaker_button)
        speaker_row.addStretch(1)
        form.addRow("Speaker name box", self.speaker_region_label)
        form.addRow("", self._wrap(speaker_row))

        self.band_slider = QSlider(Qt.Orientation.Horizontal)
        self.band_slider.setRange(10, 100)
        self.band_slider.setValue(34)
        self.band_slider.setToolTip(
            "How much of the bottom of the screen to read when no area is picked."
        )
        form.addRow("Automatic band %", self.band_slider)
        layout.addWidget(area)

        tuning = QGroupBox("Recognition")
        tuning_form = QFormLayout(tuning)

        self.trigger_combo = QComboBox()
        self.trigger_combo.addItem("When the picture changes (recommended)", "change")
        self.trigger_combo.addItem("On a fixed timer", "timer")
        self.trigger_combo.setToolTip(
            "Change mode compares each grab with the last and only reads when "
            "something moved. It reacts sooner and costs far less, because "
            "grabbing pixels is cheap and recognising them is not."
        )
        tuning_form.addRow("Read", self.trigger_combo)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 30.0)
        self.fps_spin.setSingleStep(0.5)
        self.fps_spin.setValue(8.0)
        self.fps_spin.setToolTip(
            "How often the screen is grabbed. In change mode this is how often "
            "it is checked, not how often it is read."
        )
        tuning_form.addRow("Checks per second", self.fps_spin)

        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(1, 100)
        self.sensitivity_slider.setValue(40)
        self.sensitivity_slider.setToolTip(
            "How much of the picture must move to count as a change. Lower it "
            "if lines are missed; raise it if an animated background keeps "
            "triggering a re-read."
        )
        tuning_form.addRow("Change sensitivity", self.sensitivity_slider)

        self.max_idle_spin = QDoubleSpinBox()
        self.max_idle_spin.setRange(0.0, 60.0)
        self.max_idle_spin.setSingleStep(1.0)
        self.max_idle_spin.setValue(8.0)
        self.max_idle_spin.setToolTip(
            "Read anyway after this many seconds with no change detected, so a "
            "difference too subtle to notice cannot leave a line unread. "
            "0 turns it off."
        )
        tuning_form.addRow("Re-read after (s)", self.max_idle_spin)

        self.auto_upscale_check = QCheckBox(
            "Upscale automatically until the text is big enough to read"
        )
        self.auto_upscale_check.setChecked(True)
        self.auto_upscale_check.setToolTip(
            "Measures how tall the text actually is and scales to suit it, so "
            "small subtitles get magnified and large ones are left alone."
        )
        tuning_form.addRow("", self.auto_upscale_check)

        self.upscale_spin = QDoubleSpinBox()
        self.upscale_spin.setRange(1.0, 6.0)
        self.upscale_spin.setSingleStep(0.25)
        self.upscale_spin.setValue(2.0)
        self.upscale_spin.setToolTip(
            "Starting scale. With automatic upscaling on this is only the "
            "first guess; turn it off to pin the scale here."
        )
        tuning_form.addRow("Upscale", self.upscale_spin)

        self.max_upscale_spin = QDoubleSpinBox()
        self.max_upscale_spin.setRange(1.0, 8.0)
        self.max_upscale_spin.setSingleStep(0.5)
        self.max_upscale_spin.setValue(6.0)
        tuning_form.addRow("Upscale limit", self.max_upscale_spin)

        self.contrast_spin = QDoubleSpinBox()
        self.contrast_spin.setRange(0.5, 3.0)
        self.contrast_spin.setSingleStep(0.1)
        self.contrast_spin.setValue(1.6)
        tuning_form.addRow("Contrast", self.contrast_spin)

        self.grayscale_check = QCheckBox("Convert to grey before reading")
        self.grayscale_check.setChecked(True)
        tuning_form.addRow("", self.grayscale_check)

        self.invert_check = QCheckBox("Invert (for dark text on a light box)")
        tuning_form.addRow("", self.invert_check)
        layout.addWidget(tuning)

        test = QGroupBox("Test")
        test_layout = QVBoxLayout(test)
        self.test_button = QPushButton("Test capture now")
        test_layout.addWidget(self.test_button)

        self.preview_label = QLabel("The captured area will appear here.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(110)
        self.preview_label.setStyleSheet(
            "border: 1px solid #555; background: #1a1a1a; color: #888;"
        )
        test_layout.addWidget(self.preview_label)

        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(120)
        test_layout.addWidget(self.result_text)
        layout.addWidget(test, stretch=1)

        self._connect()

    @staticmethod
    def _wrap(inner) -> QWidget:
        holder = QWidget()
        holder.setLayout(inner)
        return holder

    def _connect(self) -> None:
        self.pick_text_button.clicked.connect(self.pick_text_region.emit)
        self.pick_speaker_button.clicked.connect(self.pick_speaker_region.emit)
        self.clear_text_button.clicked.connect(lambda: self.set_region(None, "text"))
        self.clear_speaker_button.clicked.connect(lambda: self.set_region(None, "speaker"))
        self.test_button.clicked.connect(self.run_test)
        for widget, signal in (
            (self.monitor_combo, "currentIndexChanged"),
            (self.band_slider, "valueChanged"),
            (self.trigger_combo, "currentIndexChanged"),
            (self.fps_spin, "valueChanged"),
            (self.sensitivity_slider, "valueChanged"),
            (self.max_idle_spin, "valueChanged"),
            (self.auto_upscale_check, "stateChanged"),
            (self.upscale_spin, "valueChanged"),
            (self.max_upscale_spin, "valueChanged"),
            (self.contrast_spin, "valueChanged"),
            (self.grayscale_check, "stateChanged"),
            (self.invert_check, "stateChanged"),
        ):
            getattr(widget, signal).connect(self._emit_changed)

    def _update_trigger_enabled(self) -> None:
        """Grey out the change controls when reading on a fixed timer."""
        on_change = self.trigger_combo.currentData() == "change"
        self.sensitivity_slider.setEnabled(on_change)
        self.max_idle_spin.setEnabled(on_change)

    def _emit_changed(self, *_args) -> None:
        self._update_trigger_enabled()
        if not self._loading:
            self.changed.emit()

    # -- profile binding ---------------------------------------------------

    def load_profile(self, profile: Profile) -> None:
        self._loading = True
        try:
            self._profile = profile
            capture = profile.capture
            index = self.monitor_combo.findData(capture.monitor)
            self.monitor_combo.setCurrentIndex(index if index >= 0 else 0)
            self.band_slider.setValue(int(capture.bottom_band * 100))
            self.fps_spin.setValue(capture.fps)
            index = self.trigger_combo.findData(capture.trigger)
            self.trigger_combo.setCurrentIndex(index if index >= 0 else 0)
            self.sensitivity_slider.setValue(
                threshold_to_slider(capture.change_threshold)
            )
            self.max_idle_spin.setValue(capture.max_idle_seconds)
            self._update_trigger_enabled()
            self.auto_upscale_check.setChecked(capture.auto_upscale)
            self.upscale_spin.setValue(capture.upscale)
            self.max_upscale_spin.setValue(capture.max_upscale)
            self.contrast_spin.setValue(capture.contrast)
            self.grayscale_check.setChecked(capture.grayscale)
            self.invert_check.setChecked(capture.invert)
            self._show_region(capture.text_region, "text")
            self._show_region(capture.speaker_region, "speaker")
        finally:
            self._loading = False

    def apply_to(self, profile: Profile) -> None:
        capture = profile.capture
        capture.monitor = self.monitor_combo.currentData() or 1
        capture.bottom_band = self.band_slider.value() / 100.0
        capture.fps = self.fps_spin.value()
        capture.trigger = self.trigger_combo.currentData() or "change"
        capture.change_threshold = slider_to_threshold(self.sensitivity_slider.value())
        capture.max_idle_seconds = self.max_idle_spin.value()
        capture.auto_upscale = self.auto_upscale_check.isChecked()
        capture.upscale = self.upscale_spin.value()
        capture.max_upscale = self.max_upscale_spin.value()
        capture.contrast = self.contrast_spin.value()
        capture.grayscale = self.grayscale_check.isChecked()
        capture.invert = self.invert_check.isChecked()

    def set_region(self, region: Region | None, which: str) -> None:
        if self._profile is None:
            return
        if which == "text":
            self._profile.capture.text_region = region
        else:
            self._profile.capture.speaker_region = region
        self._show_region(region, which)
        self._emit_changed()

    def _show_region(self, region: Region | None, which: str) -> None:
        if region is not None and region.is_valid():
            text = f"{region.width} x {region.height} at ({region.left}, {region.top})"
        elif which == "text":
            text = "Automatic (bottom of the game window)"
        else:
            text = "Not set - name is read from the text"
        label = self.text_region_label if which == "text" else self.speaker_region_label
        label.setText(text)

    # -- live test ---------------------------------------------------------

    def run_test(self) -> None:
        if self._profile is None:
            return
        self.apply_to(self._profile)
        capture = self._profile.capture

        window = foreground_window()
        region, _ = resolve_regions(capture, self._grabber, window)
        frame = self._grabber.grab(region)

        if frame is None:
            self._report("The screen grab returned nothing. Try a different monitor.")
            return
        if looks_blank(frame):
            self._report(
                "The captured area is a flat colour.\n\n"
                "If the game is running in exclusive fullscreen, switch it to "
                "borderless windowed - that display mode cannot be captured."
            )
            return

        self._show_preview(frame)

        if self._ocr is None:
            try:
                self._ocr = create_engine("en-US")
            except OcrUnavailable as exc:
                self._report(str(exc))
                return

        # Run the same adaptive pass the engine uses, so the scale reported
        # here is the scale the game will actually be read at.
        scaler = (
            AdaptiveUpscale(initial=capture.upscale, maximum=capture.max_upscale)
            if capture.auto_upscale
            else AdaptiveUpscale(
                initial=capture.upscale,
                minimum=capture.upscale,
                maximum=capture.upscale,
            )
        )
        output = None
        applied = capture.upscale
        for _ in range(3 if capture.auto_upscale else 1):
            applied = clamp_scale(frame.shape[1], frame.shape[0], scaler.scale)
            output = self._ocr.recognize(
                preprocess(
                    frame,
                    upscale=applied,
                    contrast=capture.contrast,
                    grayscale=capture.grayscale,
                    invert=capture.invert,
                )
            )
            if output.is_empty:
                break
            scaler.observe(output.lines, applied)

        if output is None or output.is_empty:
            self._report(
                f"Captured {region.width}x{region.height} at {applied:.2f}x "
                "but read no text.\n"
                "Try picking a tighter area around the words, or raise the "
                "upscale limit."
            )
            return

        heights = sorted(line.height for line in output.lines if line.height > 1)
        glyph = heights[len(heights) // 2] / applied if heights else 0.0

        utterance = parse(output, strip_patterns=tuple(self._profile.detect.strip_patterns))
        lines = "\n".join(f"  {line.text}" for line in output.lines)
        summary = (
            f"Captured {region.width}x{region.height} real pixels, "
            f"read at {applied:.2f}x "
            f"({region.width * applied:.0f}x{region.height * applied:.0f}).\n"
            f"Text height {glyph:.0f} px on screen, "
            f"{glyph * applied:.0f} px as read.\n\n"
            f"Found {len(output.lines)} line(s):\n{lines}"
        )
        edges = clipped_edges(output, int(frame.shape[0] * applied))
        if edges:
            summary += (
                f"\n\n>> Text is touching the {' and '.join(edges)} of the capture "
                "area, so the passage probably continues outside it and is being "
                "read only in part.\n"
                "   Drag a taller area with 'Pick dialogue area', or raise the "
                "automatic band percentage."
            )

        if utterance is not None:
            summary += (
                f"\n\nSpeaker: {utterance.speaker or '(none - narrator voice)'}"
                f"\nWould say ({len(utterance.text)} chars): {utterance.text}"
            )
        else:
            summary += "\n\nNothing here looks like dialogue."
        self._report(summary)

    def _report(self, message: str) -> None:
        self.result_text.setPlainText(message)

    def _show_preview(self, frame) -> None:
        rgb = frame[:, :, :3][:, :, ::-1].copy()
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview_label.width(),
            160,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(pixmap)
