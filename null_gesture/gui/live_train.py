"""Zero-training live IMU gesture display.

Shows standing_still / pull / push in real time using physics-based detection.
No model, no training — connect and go.
"""

from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np

logger = logging.getLogger("null_gesture.gui.live")


def _try_qt_imports():
    from PyQt6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True, useNumba=False)
    return QtCore, QtGui, QtWidgets, pg


class LiveDetectWindow:
    """Real-time IMU gesture display — no training needed."""

    WINDOW_MS = 40  # ~25Hz UI refresh

    def __init__(self, imu_host: str = "127.0.0.1"):
        self.QtCore, self.QtGui, self.QtWidgets, self.pg = _try_qt_imports()
        self.imu_host = imu_host

        from null_gesture.sensors.imu_sensor import IMUClient
        self.imu = IMUClient()
        self.imu_connected = False

        from null_gesture.models.simple_detector import SimpleIMUDetector
        self.detector = SimpleIMUDetector()

        self._current_label = "standing_still"
        self._current_conf = 0.0
        self._history: deque[tuple[str, float]] = deque(maxlen=100)

        self._init_ui()
        self._init_timers()

    # ── UI ──────────────────────────────────────────────────────────────

    def _init_ui(self):
        QtWidgets = self.QtWidgets
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("Null-Gesture — Live IMU Detection")
        self.win.resize(900, 600)

        D = "#0d1117"; C = "#161b22"; B = "#30363d"; T = "#e6edf3"
        A = "#58a6ff"; G = "#3fb950"; O = "#d2991d"; R = "#f85149"; P = "#a371f7"

        self.win.setStyleSheet(f"""
            QMainWindow {{ background-color: {D}; }}
            QWidget {{ color: {T}; font-family: sans-serif; font-size: 12px; }}
            QPushButton {{
                background-color: {C}; border: 1px solid {B};
                border-radius: 6px; padding: 8px 16px; color: {T};
            }}
            QPushButton:hover {{ border-color: {A}; }}
            QLabel#big {{ font-size: 48px; font-weight: bold; }}
            QLabel#conf {{ font-size: 18px; }}
            QLabel#sub {{ font-size: 14px; color: #8b949e; }}
        """)

        central = QtWidgets.QWidget()
        self.win.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Header
        hdr = QtWidgets.QLabel("Null-Gesture — IMU Detection")
        hdr.setStyleSheet(f"color: {A}; font-size: 20px; font-weight: bold;")
        layout.addWidget(hdr)

        # ── Big prediction ───────────────────────────────────────────
        pred_card = QtWidgets.QWidget()
        pred_card.setStyleSheet(f"background: {C}; border-radius: 10px; padding: 24px;")
        pbox = QtWidgets.QVBoxLayout(pred_card)

        self.pred_label = QtWidgets.QLabel("Standing Still")
        self.pred_label.setObjectName("big")
        self.pred_label.setAlignment(self.QtCore.Qt.AlignmentFlag.AlignCenter)
        pbox.addWidget(self.pred_label)

        self.conf_label = QtWidgets.QLabel("")
        self.conf_label.setObjectName("conf")
        self.conf_label.setAlignment(self.QtCore.Qt.AlignmentFlag.AlignCenter)
        pbox.addWidget(self.conf_label)

        self.sub_label = QtWidgets.QLabel("Connect IMU to begin")
        self.sub_label.setObjectName("sub")
        self.sub_label.setAlignment(self.QtCore.Qt.AlignmentFlag.AlignCenter)
        pbox.addWidget(self.sub_label)

        layout.addWidget(pred_card)

        # ── IMU plot ─────────────────────────────────────────────────
        self.plot = self.pg.PlotWidget()
        self.plot.setBackground(C)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setMaximumHeight(140)
        self.curves = []
        self.plot_bufs = [deque(maxlen=200) for _ in range(6)]
        colors = ["#58a6ff", "#3fb950", "#d2991d", "#f85149", "#a371f7", "#79c0ff"]
        for i, ch in enumerate(["ax", "ay", "az", "gx", "gy", "gz"]):
            c = self.plot.plot([], [], pen=self.pg.mkPen(colors[i], width=1.5), name=ch)
            self.curves.append(c)
        self.plot.addLegend(offset=(1, 1))
        layout.addWidget(self.plot)

        # ── Detection history plot ───────────────────────────────────
        self.hist_plot = self.pg.PlotWidget()
        self.hist_plot.setBackground(C)
        self.hist_plot.showGrid(x=False, y=True, alpha=0.1)
        self.hist_plot.setMaximumHeight(80)
        self.hist_plot.hideAxis("left")
        self.hist_plot.setLabel("bottom", "")
        layout.addWidget(self.hist_plot)

        # ── Connect button ───────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton("🔌 Connect IMU")
        self.connect_btn.clicked.connect(self._connect_imu)
        btn_row.addWidget(self.connect_btn)

        self.status_lbl = QtWidgets.QLabel("Disconnected")
        self.status_lbl.setStyleSheet("color: #8b949e;")
        btn_row.addWidget(self.status_lbl)
        btn_row.addStretch()

        # Threshold sliders
        btn_row.addWidget(QtWidgets.QLabel("Threshold:"))
        self.thresh_slider = QtWidgets.QSlider(self.QtCore.Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(5, 50)
        self.thresh_slider.setValue(15)
        self.thresh_slider.setFixedWidth(100)
        self.thresh_slider.valueChanged.connect(self._on_threshold)
        btn_row.addWidget(self.thresh_slider)
        self.thresh_label = QtWidgets.QLabel("0.15g")
        btn_row.addWidget(self.thresh_label)

        layout.addLayout(btn_row)

    def _init_timers(self):
        self._timer = self.QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(self.WINDOW_MS)

    # ── Actions ──────────────────────────────────────────────────────────

    def _connect_imu(self):
        if self.imu_connected:
            self.imu.disconnect()
            self.imu_connected = False
            self.connect_btn.setText("🔌 Connect IMU")
            self.status_lbl.setText("Disconnected")
            return
        self.sub_label.setText("Connecting...")
        self.QtWidgets.QApplication.processEvents()
        if self.imu.connect():
            self.imu_connected = True
            self.connect_btn.setText("⚡ Disconnect")
            self.sub_label.setText("Connected — move your hand!")
            self.status_lbl.setText(f"Connected to {self.imu_host}:9999")
        else:
            self.sub_label.setText("❌ Connection failed — is esp32_reader.py running?")

    def _on_threshold(self, value: int):
        g = value / 100.0
        self.thresh_label.setText(f"{g:.2f}g")
        self.detector.movement_threshold = g

    def _tick(self):
        if not self.imu_connected:
            return

        self.imu.ingest(max_samples=20)
        window = self.imu.get_window()

        if window.sum() == 0:
            return

        # Update IMU plot
        for i in range(6):
            self.plot_bufs[i].append(float(window[-1, i]))
        for i, buf in enumerate(self.plot_bufs):
            b = list(buf)
            if b: self.curves[i].setData(range(len(b)), b)

        # Run detector
        label, conf = self.detector.update(window)
        self._current_label = label
        self._current_conf = conf
        self._history.append((label, conf))

        # Update display — all 11 gestures
        names = {
            "standing_still": "Standing Still", "pull": "Pulling", "push": "Pushing",
            "left": "Left", "right": "Right",
            "clockwise": "Clockwise", "anti_clockwise": "Anti-Clockwise",
            "raise_arms": "Raise Arms", "bye_bye": "Bye-Bye",
            "one_arm_boxing": "One-Arm Boxing", "palm_up_down": "Palm Up/Down",
        }
        colors_map = {
            "standing_still": "#58a6ff", "pull": "#3fb950", "push": "#f85149",
            "left": "#d2991d", "right": "#d2991d",
            "clockwise": "#a371f7", "anti_clockwise": "#a371f7",
            "raise_arms": "#79c0ff", "bye_bye": "#ff7b72",
            "one_arm_boxing": "#f0883e", "palm_up_down": "#56d364",
        }

        display = names.get(label, label.replace("_", " ").title())
        color = colors_map.get(label, "#e6edf3")

        self.pred_label.setText(display)
        self.pred_label.setStyleSheet(f"color: {color}; font-size: 44px; font-weight: bold;")
        self.conf_label.setText(f"Confidence: {conf:.0%}")

        # Status line
        recent = window[-20:, :3]
        mag = float(np.mean(np.linalg.norm(recent, axis=1)))
        self.sub_label.setText(
            f"Accel: {mag:.3f}g | Thresh: {self.detector.motion_threshold:.2f}g | "
            f"Samples: {self.detector._sample_count}"
        )

        # Update history plot — color-coded by gesture
        self.hist_plot.clear()
        if len(self._history) > 1:
            xs = list(range(len(self._history)))
            y_map = {g: i for i, g in enumerate(self.detector.GESTURES)}
            ys = [y_map.get(h[0], 0) for h in self._history]
            c_hist = [colors_map.get(h[0], "#555") for h in self._history]
            self.hist_plot.plot(xs, ys, pen=None, symbol='o', symbolSize=3, symbolBrush=c_hist)

    def show(self):
        self.win.show()

    @property
    def app(self):
        return self.QtWidgets.QApplication.instance()
