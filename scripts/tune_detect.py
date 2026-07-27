#!/usr/bin/env python3
"""Tunable gesture detector — see what's happening, adjust thresholds live.

Shows gyro magnitude plot, state, extracted features, and template distances.
Sliders for onset/offset thresholds let you tune in real-time.

Usage:
    PYTHONPATH=. venv/bin/python scripts/tune_detect.py --serial /dev/ttyACM0
"""

from __future__ import annotations

import sys
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from null_gesture.models.calibrated_detector import (
    CalibratedDetector, extract_features,
)


class TuneWindow(QtWidgets.QWidget):
    def __init__(self, serial: str | None = None):
        super().__init__()
        self._serial = serial

        self._detector = CalibratedDetector(onset_thresh=20, offset_thresh=10, offset_frames=10)

        # Real-time gyro buffer for plot
        self._gyro_hist: deque[float] = deque(maxlen=300)
        self._state_hist: deque[int] = deque(maxlen=300)  # 0=still, 1=collecting, 2=classified

        from null_gesture.sensors.imu_sensor import IMUClient
        self._imu = IMUClient()
        self._imu_ok = False

        self._init_ui()
        self._init_timer()
        self._connect()

    def _connect(self) -> None:
        ok = False
        if self._serial:
            ok = self._imu.connect_serial(self._serial)
        if not ok:
            ok = self._imu.connect_tcp("127.0.0.1", 9999)
        self._imu_ok = ok

    # ── UI ─────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        self.setWindowTitle("Gesture Detector — Tune Mode")
        self.resize(800, 600)
        self.setStyleSheet("background-color: #0d1117; color: #e6edf3; font-family: sans-serif; font-size: 12px;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # ── Gyro plot ──────────────────────────────────────────────
        self._plot = pg.PlotWidget()
        self._plot.setBackground("#161b22")
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setMaximumHeight(150)
        self._plot.setLabel("left", "gyro dps")
        self._gyro_curve = self._plot.plot([], [], pen=pg.mkPen("#58a6ff", width=2), name="gyro mag")
        self._onset_line = pg.InfiniteLine(pos=20, angle=0, pen=pg.mkPen("#f85149", width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
        self._offset_line = pg.InfiniteLine(pos=10, angle=0, pen=pg.mkPen("#d2991d", width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
        self._plot.addItem(self._onset_line)
        self._plot.addItem(self._offset_line)
        layout.addWidget(self._plot)

        # ── Status row ─────────────────────────────────────────────
        row1 = QtWidgets.QHBoxLayout()
        self._state_label = QtWidgets.QLabel("⏸ STILL")
        self._state_label.setStyleSheet("color: #3fb950; font-size: 28px; font-weight: bold;")
        row1.addWidget(self._state_label)

        self._result_label = QtWidgets.QLabel("")
        self._result_label.setStyleSheet("color: #58a6ff; font-size: 28px; font-weight: bold;")
        row1.addWidget(self._result_label)

        self._conf_label = QtWidgets.QLabel("")
        self._conf_label.setStyleSheet("color: #8b949e; font-size: 16px;")
        row1.addWidget(self._conf_label)
        row1.addStretch()
        layout.addLayout(row1)

        # ── Feature info ───────────────────────────────────────────
        self._feat_label = QtWidgets.QLabel("")
        self._feat_label.setStyleSheet("color: #484f58; font-size: 10px; font-family: monospace;")
        self._feat_label.setWordWrap(True)
        layout.addWidget(self._feat_label)

        # ── Template distances ─────────────────────────────────────
        self._dist_label = QtWidgets.QLabel("")
        self._dist_label.setStyleSheet("color: #30363d; font-size: 10px; font-family: monospace;")
        self._dist_label.setWordWrap(True)
        layout.addWidget(self._dist_label)

        # ── Sliders ────────────────────────────────────────────────
        sliders = QtWidgets.QWidget()
        sl = QtWidgets.QHBoxLayout(sliders)
        sl.setContentsMargins(0, 0, 0, 0)

        sl.addWidget(QtWidgets.QLabel("Onset:"))
        self._onset_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._onset_slider.setRange(5, 80)
        self._onset_slider.setValue(20)
        self._onset_slider.valueChanged.connect(self._on_onset)
        sl.addWidget(self._onset_slider)
        self._onset_val = QtWidgets.QLabel("20")
        sl.addWidget(self._onset_val)

        sl.addSpacing(20)

        sl.addWidget(QtWidgets.QLabel("Offset:"))
        self._offset_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._offset_slider.setRange(3, 40)
        self._offset_slider.setValue(10)
        self._offset_slider.valueChanged.connect(self._on_offset)
        sl.addWidget(self._offset_slider)
        self._offset_val = QtWidgets.QLabel("10")
        sl.addWidget(self._offset_val)

        sl.addSpacing(20)

        sl.addWidget(QtWidgets.QLabel("Frames:"))
        self._frames_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._frames_slider.setRange(3, 25)
        self._frames_slider.setValue(10)
        self._frames_slider.valueChanged.connect(self._on_frames)
        sl.addWidget(self._frames_slider)
        self._frames_val = QtWidgets.QLabel("10")
        sl.addWidget(self._frames_val)

        sl.addStretch()
        layout.addWidget(sliders)

        # ── Buffer size ────────────────────────────────────────────
        self._buf_label = QtWidgets.QLabel("buffer: 0 samples")
        self._buf_label.setStyleSheet("color: #484f58; font-size: 10px; font-family: monospace;")
        layout.addWidget(self._buf_label)

    def _on_onset(self, v):
        self._onset_val.setText(str(v))
        self._detector.onset_thresh = float(v)
        self._onset_line.setPos(v)

    def _on_offset(self, v):
        self._offset_val.setText(str(v))
        self._detector.offset_thresh = float(v)
        self._offset_line.setPos(v)

    def _on_frames(self, v):
        self._frames_val.setText(str(v))
        self._detector.offset_frames = v

    # ── Timer ──────────────────────────────────────────────────────

    def _init_timer(self) -> None:
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    def _tick(self) -> None:
        if not self._imu_ok:
            return
        self._imu.ingest(max_samples=10)
        window = self._imu.get_window()
        if window.sum() == 0:
            return

        gyro_mag = float(np.linalg.norm(window[-1, 3:6]))
        self._gyro_hist.append(gyro_mag)

        # Update plot
        xs = list(range(len(self._gyro_hist)))
        ys = list(self._gyro_hist)
        self._gyro_curve.setData(xs, ys)

        # Feed detector
        label, conf = self._detector.update(window[:, :6])

        # State display
        state = self._detector._state
        buf_size = len(self._detector._raw) - self._detector._onset_idx
        if state == "still":
            self._state_label.setText("⏸ STILL")
            self._state_label.setStyleSheet("color: #3fb950; font-size: 28px; font-weight: bold;")
            self._state_hist.append(0)
        elif state == "collecting":
            self._state_label.setText(f"🔴 RECORDING ({buf_size})")
            self._state_label.setStyleSheet("color: #f85149; font-size: 28px; font-weight: bold;")
            self._state_hist.append(1)
        else:
            self._state_label.setText("✅ CLASSIFIED")
            self._state_label.setStyleSheet("color: #d2991d; font-size: 28px; font-weight: bold;")
            self._state_hist.append(2)

        # Result
        if label != "standing_still":
            name = label.replace("_", " ").title()
            self._result_label.setText(name)
            self._conf_label.setText(f"{conf:.0%}")
        elif state == "still":
            self._result_label.setText("")
            self._conf_label.setText("")

        # Show features of last classification
        if state == "classified" and hasattr(self._detector, '_last_features'):
            f = self._detector._last_features
            self._feat_label.setText(
                f"g=({f[0]:+.0f},{f[1]:+.0f},{f[2]:+.0f})  "
                f"mag_mean={f[3]:.0f}  peak={f[4]:.0f}  "
                f"osc={f[5]*100:.0f}  acc_var={f[6]:.3f}  "
                f"dom_ratio={f[7]:.1f}  dom_axis={f[8]:.0f}  sign={f[9]:+.0f}  "
                f"sym={f[10]:.2f}  peak_pos={f[11]:.2f}"
            )

        # Buffer info
        self._buf_label.setText(f"buffer: {buf_size} samples  |  total: {self._detector._sample_count}")

    def closeEvent(self, event):
        if self._imu_ok:
            self._imu.disconnect()
        super().closeEvent(event)


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--serial", default="/dev/ttyACM0")
    args = p.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    win = TuneWindow(serial=args.serial)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
