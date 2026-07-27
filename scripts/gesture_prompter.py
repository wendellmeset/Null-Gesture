#!/usr/bin/env python3
"""Gesture prompt display — cycles through gestures with countdown + live IMU plot + NN detection.

Usage:
    python scripts/gesture_prompter.py --serial /dev/ttyACM0
    python -m null_gesture prompter --serial /dev/ttyACM0
"""

from __future__ import annotations

import sys
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

GESTURES: list[str] = [
    "Pull",
    "Push",
    "Clockwise",
    "Anti-Clockwise",
    "Left",
    "Right",
    "Bye-Bye",
    "Clapping",
    "One Arm Boxing",
    "T-Arms",
    "Raise Arms",
    "Palm Up and Down",
]

# Map prompter display name → NN model label(s)
# Model uses lowercase underscore names; prompter shows title case
GESTURE_KEY: dict[str, list[str]] = {
    "Pull": ["pull"],
    "Push": ["push"],
    "Clockwise": ["clockwise"],
    "Anti-Clockwise": ["anti_clockwise"],
    "Left": ["left"],
    "Right": ["right"],
    "Bye-Bye": ["bye_bye"],
    "Clapping": ["clapping"],
    "One Arm Boxing": ["one_arm_boxing"],
    "T-Arms": ["t_arms"],
    "Raise Arms": ["raise_arms"],
    "Palm Up and Down": ["palm_up", "palm_down"],
}

INTERVAL_MS = 3000
TICK_MS = 33
PLOT_HISTORY = 200


class GesturePrompter(QtWidgets.QWidget):
    def __init__(self, serial: str | None = None, host: str = "127.0.0.1", port: int = 9999) -> None:
        super().__init__()
        self._idx = 0
        self._remaining_ms = INTERVAL_MS
        self._gestures = GESTURES
        self._serial = serial
        self._host = host
        self._port = port

        # IMU
        from null_gesture.sensors.imu_sensor import IMUClient
        self._imu = IMUClient()
        self._imu_connected = False
        self._plot_bufs: list[deque[float]] = [deque(maxlen=PLOT_HISTORY) for _ in range(6)]
        self._curves: list[pg.PlotDataItem] = []

        # NN detector
        from null_gesture.models.nn_detector import NNDetector
        self._detector = NNDetector()
        self._detected_label: str = ""
        self._detected_conf: float = 0.0

        self._init_ui()
        self._init_timer()
        self._connect_imu()

    # ── IMU ────────────────────────────────────────────────────────

    def _connect_imu(self) -> None:
        ok = False
        if self._serial:
            ok = self._imu.connect_serial(self._serial)
        if not ok and self._host:
            ok = self._imu.connect_tcp(self._host, self._port)
        self._imu_connected = ok

    # ── UI ─────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        self.setWindowTitle("Gesture Prompter")
        self.resize(900, 620)
        self.setStyleSheet("background-color: #0d1117;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── IMU plot ───────────────────────────────────────────────
        self._plot = pg.PlotWidget()
        self._plot.setBackground("#161b22")
        self._plot.showGrid(x=True, y=True, alpha=0.12)
        self._plot.getPlotItem().hideAxis("left")
        self._plot.setMouseEnabled(x=False, y=False)
        colors = ["#58a6ff", "#3fb950", "#d2991d", "#f85149", "#a371f7", "#79c0ff"]
        for i, ch in enumerate(["ax", "ay", "az", "gx", "gy", "gz"]):
            c = self._plot.plot([], [], pen=pg.mkPen(colors[i], width=1.2), name=ch)
            self._curves.append(c)
        self._plot.addLegend(offset=(1, 1))
        layout.addWidget(self._plot, stretch=3)

        # ── Center area ────────────────────────────────────────────
        center = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center)
        center_layout.setContentsMargins(0, 8, 0, 4)
        center_layout.setSpacing(2)

        # Prompt gesture (large)
        self._label = QtWidgets.QLabel(self._gestures[0])
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(
            "color: #e6edf3; font-size: 56px; font-weight: bold; font-family: sans-serif;"
        )
        center_layout.addWidget(self._label)

        # Detected gesture + match indicator (smaller, below prompt)
        detect_row = QtWidgets.QWidget()
        dr_layout = QtWidgets.QHBoxLayout(detect_row)
        dr_layout.setContentsMargins(0, 0, 0, 0)
        dr_layout.setSpacing(8)
        dr_layout.addStretch()

        self._match_label = QtWidgets.QLabel("")
        self._match_label.setStyleSheet("font-size: 18px;")
        dr_layout.addWidget(self._match_label)

        self._detect_label = QtWidgets.QLabel("")
        self._detect_label.setStyleSheet(
            "color: #8b949e; font-size: 16px; font-family: monospace;"
        )
        dr_layout.addWidget(self._detect_label)

        dr_layout.addStretch()
        center_layout.addWidget(detect_row)

        layout.addWidget(center, stretch=2)

        # ── Bottom bar ─────────────────────────────────────────────
        bottom = QtWidgets.QWidget()
        bottom.setFixedHeight(40)
        bottom.setStyleSheet("background-color: #161b22;")
        bottom_layout = QtWidgets.QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(14, 0, 14, 0)
        bottom_layout.setSpacing(10)

        self._countdown_label = QtWidgets.QLabel("3.0")
        self._countdown_label.setStyleSheet(
            "color: #58a6ff; font-size: 13px; font-family: monospace;"
        )
        bottom_layout.addWidget(self._countdown_label)

        self._progress_label = QtWidgets.QLabel(f"1/{len(self._gestures)}")
        self._progress_label.setStyleSheet(
            "color: #8b949e; font-size: 11px; font-family: sans-serif;"
        )
        bottom_layout.addWidget(self._progress_label)

        bottom_layout.addStretch()

        next_idx = (self._idx + 1) % len(self._gestures)
        self._next_label = QtWidgets.QLabel(f"next → {self._gestures[next_idx]}")
        self._next_label.setStyleSheet(
            "color: #58a6ff; font-size: 10px; font-family: monospace;"
        )
        bottom_layout.addWidget(self._next_label)

        bottom_layout.addStretch()

        layout.addWidget(bottom)

    # ── Timer ──────────────────────────────────────────────────────

    def _init_timer(self) -> None:
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(TICK_MS)

    def _tick(self) -> None:
        # ── IMU + detection ───────────────────────────────────────
        if self._imu_connected:
            self._imu.ingest(max_samples=10)
            window = self._imu.get_window()
            if window.sum() != 0:
                # Plot
                for i in range(6):
                    self._plot_bufs[i].append(float(window[-1, i]))
                for i, buf in enumerate(self._plot_bufs):
                    b = list(buf)
                    if b:
                        self._curves[i].setData(range(len(b)), b)

                # NN detection (onset/offset — result appears when gesture completes)
                det_label, det_conf = self._detector.update(window)
                if det_label != "standing_still":
                    self._detected_label = det_label
                    self._detected_conf = det_conf

        # ── Countdown ─────────────────────────────────────────────
        self._remaining_ms -= TICK_MS

        if self._remaining_ms <= 0:
            self._idx = (self._idx + 1) % len(self._gestures)
            self._remaining_ms = INTERVAL_MS

            self._label.setText(self._gestures[self._idx])
            self._progress_label.setText(f"{self._idx + 1}/{len(self._gestures)}")

            next_idx = (self._idx + 1) % len(self._gestures)
            self._next_label.setText(f"next → {self._gestures[next_idx]}")

            # Reset detection for new gesture
            self._detected_label = ""
            self._detected_conf = 0.0
            self._detector._ema_probs = None
            self._detector._state = "still"
            self._detector._raw.clear()

        # ── Update detection display ──────────────────────────────
        prompt = self._gestures[self._idx]
        expected = GESTURE_KEY.get(prompt, [])

        if self._detected_label:
            display_name = self._detected_label.replace("_", " ").title()
            self._detect_label.setText(f"detected: {display_name} ({self._detected_conf:.0%})")

            if not expected:
                # Untrained gesture — show neutral
                self._match_label.setText("—")
                self._match_label.setStyleSheet("color: #484f58; font-size: 18px;")
            elif self._detected_label in expected:
                self._match_label.setText("✅")
                self._match_label.setStyleSheet("color: #3fb950; font-size: 18px;")
            else:
                self._match_label.setText("❌")
                self._match_label.setStyleSheet("color: #f85149; font-size: 18px;")
        else:
            self._detect_label.setText("")
            self._match_label.setText("")

        # Countdown display
        seconds = self._remaining_ms / 1000.0
        self._countdown_label.setText(f"{seconds:.1f}s")

    def closeEvent(self, event: QtCore.QEvent) -> None:
        if self._imu_connected:
            self._imu.disconnect()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Gesture prompter with IMU plot + NN detection")
    parser.add_argument("--serial", help="Serial port (e.g. /dev/ttyACM0)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args(argv)

    app = QtWidgets.QApplication(sys.argv)
    win = GesturePrompter(serial=args.serial, host=args.host, port=args.port)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
