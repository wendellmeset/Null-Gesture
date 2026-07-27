"""Live IMU gesture display — supports heuristic and neural detectors."""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger("null_gesture.gui.live")

DEFAULT_NN_MODEL = Path(__file__).resolve().parent.parent / "models" / "saved" / "gesture_cnn.pt"


def _try_qt_imports() -> tuple:
    import pyqtgraph as pg
    from PyQt6 import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets, pg


class LiveDetectWindow:
    WINDOW_MS = 40

    def __init__(self, imu_host: str = "127.0.0.1"):
        self.QtCore, self.QtGui, self.QtWidgets, self.pg = _try_qt_imports()
        self.imu_host = imu_host
        self.serial_port: str | None = None

        from null_gesture.sensors.imu_sensor import IMUClient
        self.imu = IMUClient()
        self.imu_connected = False

        from null_gesture.models.simple_detector import SimpleIMUDetector
        from null_gesture.models.nn_detector import NNDetector

        self.simple_detector = SimpleIMUDetector()
        self.nn_detector = NNDetector(model_path=DEFAULT_NN_MODEL) if DEFAULT_NN_MODEL.exists() else None
        self.use_nn = self.nn_detector is not None
        self.detector = self.nn_detector if self.use_nn else self.simple_detector

        self._current_label = "standing_still"
        self._current_conf = 0.0
        self._history: deque[tuple[str, float]] = deque(maxlen=200)
        self._plot_bufs: list[deque[float]] = [deque(maxlen=200) for _ in range(6)]
        self._curves: list = []

        self._init_ui()
        self._init_timers()

    def _init_ui(self) -> None:
        Q = self.QtWidgets
        D, C, B, T = "#0d1117", "#161b22", "#30363d", "#e6edf3"
        A = "#58a6ff"

        self.win = Q.QMainWindow()
        self.win.setWindowTitle("Null-Gesture — Live IMU Detection")
        self.win.resize(900, 620)
        self.win.setStyleSheet(f"""
            QMainWindow {{ background-color: {D}; }}
            QWidget {{ color: {T}; font-family: sans-serif; font-size: 12px; }}
            QPushButton {{ background-color: {C}; border: 1px solid {B}; border-radius: 6px; padding: 8px 16px; color: {T}; }}
            QPushButton:hover {{ border-color: {A}; }}
            QLabel#big {{ font-size: 48px; font-weight: bold; }}
            QLabel#conf {{ font-size: 18px; }}
            QLabel#sub {{ font-size: 14px; color: #8b949e; }}
        """)

        central = Q.QWidget()
        self.win.setCentralWidget(central)
        layout = Q.QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        hdr = Q.QLabel("Null-Gesture — IMU Detection")
        hdr.setStyleSheet(f"color: {A}; font-size: 20px; font-weight: bold;")
        layout.addWidget(hdr)

        card = Q.QWidget()
        card.setStyleSheet(f"background: {C}; border-radius: 10px; padding: 24px;")
        cb = Q.QVBoxLayout(card)
        self.pred_label = Q.QLabel("Standing Still")
        self.pred_label.setObjectName("big")
        self.pred_label.setAlignment(self.QtCore.Qt.AlignmentFlag.AlignCenter)
        cb.addWidget(self.pred_label)
        self.conf_label = Q.QLabel("")
        self.conf_label.setObjectName("conf")
        self.conf_label.setAlignment(self.QtCore.Qt.AlignmentFlag.AlignCenter)
        cb.addWidget(self.conf_label)
        self.sub_label = Q.QLabel("Connect IMU to begin")
        self.sub_label.setObjectName("sub")
        self.sub_label.setAlignment(self.QtCore.Qt.AlignmentFlag.AlignCenter)
        cb.addWidget(self.sub_label)
        layout.addWidget(card)

        self.plot = self.pg.PlotWidget()
        self.plot.setBackground(C)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setMaximumHeight(140)
        colors = ["#58a6ff", "#3fb950", "#d2991d", "#f85149", "#a371f7", "#79c0ff"]
        for i, ch in enumerate(["ax", "ay", "az", "gx", "gy", "gz"]):
            c = self.plot.plot([], [], pen=self.pg.mkPen(colors[i], width=1.5), name=ch)
            self._curves.append(c)
        self.plot.addLegend(offset=(1, 1))
        layout.addWidget(self.plot)

        self.hist_plot = self.pg.PlotWidget()
        self.hist_plot.setBackground(C)
        self.hist_plot.showGrid(x=False, y=True, alpha=0.1)
        self.hist_plot.setMaximumHeight(80)
        self.hist_plot.hideAxis("left")
        layout.addWidget(self.hist_plot)

        btn_row = Q.QHBoxLayout()
        self.connect_btn = Q.QPushButton("🔌 Connect IMU")
        self.connect_btn.clicked.connect(self._connect_imu)
        btn_row.addWidget(self.connect_btn)
        self.status_lbl = Q.QLabel("Disconnected")
        self.status_lbl.setStyleSheet("color: #8b949e;")
        btn_row.addWidget(self.status_lbl)
        btn_row.addStretch()
        btn_row.addWidget(Q.QLabel("Thr:"))
        self.thresh_slider = Q.QSlider(self.QtCore.Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(10, 100)
        self.thresh_slider.setValue(18)
        self.thresh_slider.setFixedWidth(100)
        self.thresh_slider.valueChanged.connect(self._on_threshold)
        btn_row.addWidget(self.thresh_slider)
        self.thresh_label = Q.QLabel("12 dps")
        btn_row.addWidget(self.thresh_label)
        layout.addLayout(btn_row)

    def _init_timers(self) -> None:
        self._timer = self.QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(self.WINDOW_MS)

    def _connect_imu(self) -> None:
        if self.imu_connected:
            self.imu.disconnect()
            self.imu_connected = False
            self.connect_btn.setText("🔌 Connect IMU")
            self.status_lbl.setText("Disconnected")
            return
        self.sub_label.setText("Connecting...")
        self.QtWidgets.QApplication.processEvents()
        ok = False
        if self.serial_port:
            ok = self.imu.connect_serial(self.serial_port)
        if not ok and self.imu_host:
            ok = self.imu.connect_tcp(self.imu_host)
        if ok:
            self.imu_connected = True
            self.connect_btn.setText("⚡ Disconnect")
            self.sub_label.setText("Connected — move your hand!")
            self.status_lbl.setText("Connected")
        else:
            self.sub_label.setText("❌ Connection failed")

    def _on_threshold(self, value: int) -> None:
        self.thresh_label.setText(f"{value} dps")
        self.simple_detector.gyro_onset = float(value)
        if self.nn_detector is not None:
            self.nn_detector.gyro_onset = float(value)
            self.nn_detector.gyro_offset = float(value) * 0.35

    def _tick(self) -> None:
        if not self.imu_connected:
            return
        self.imu.ingest(max_samples=20)
        window = self.imu.get_window()
        if window.sum() == 0:
            return
        for i in range(6):
            self._plot_bufs[i].append(float(window[-1, i]))
        for i, buf in enumerate(self._plot_bufs):
            b = list(buf)
            if b:
                self._curves[i].setData(range(len(b)), b)

        label, conf = self.detector.update(window)
        self._current_label = label
        self._current_conf = conf
        self._history.append((label, conf))

        names = {
            "standing_still": "Standing Still", "push": "Pushing", "pull": "Pulling",
            "left": "Left", "right": "Right",
            "up": "Up", "down": "Down",
            "clockwise": "Clockwise", "anti_clockwise": "Anti-Clockwise",
            "bye_bye": "Bye Bye", "palm_up": "Palm Up", "palm_down": "Palm Down",
        }
        colors_map = {
            "standing_still": "#58a6ff", "push": "#f85149", "pull": "#3fb950",
            "left": "#d2991d", "right": "#d2991d",
            "up": "#79c0ff", "down": "#79c0ff",
            "clockwise": "#a371f7", "anti_clockwise": "#a371f7",
            "bye_bye": "#f778ba", "palm_up": "#56d364", "palm_down": "#e5534b",
        }

        display = names.get(label, label.replace("_", " ").title())
        color = colors_map.get(label, "#e6edf3")
        self.pred_label.setText(display)
        self.pred_label.setStyleSheet(f"color: {color}; font-size: 48px; font-weight: bold;")
        self.conf_label.setText(f"Confidence: {conf:.0%}")

        gyro_mag = float(np.linalg.norm(window[-5:, 3:]))
        mode = "NN" if self.use_nn else "Heuristic"
        if self.use_nn and self.nn_detector:
            state = self.nn_detector._state
            state_icon = {"still": "⏸", "collecting": "🔴", "classified": "✅"}.get(state, "?")
            self.sub_label.setText(f"Gyro: {gyro_mag:.0f} dps | Onset: {self.nn_detector.gyro_onset:.0f} dps | {state_icon} {state} | {mode}")
        else:
            thr = self.simple_detector.gyro_onset
            self.sub_label.setText(f"Gyro: {gyro_mag:.0f} dps | Thr: {thr:.0f} dps | {mode} | {self.simple_detector._sample_count}")

        self.hist_plot.clear()
        if len(self._history) > 1:
            xs = list(range(len(self._history)))
            all_gestures = (self.nn_detector.gestures if self.use_nn and self.nn_detector else self.simple_detector.GESTURES)
            if "standing_still" not in all_gestures:
                all_gestures = ["standing_still"] + list(all_gestures)
            y_map = {g: i for i, g in enumerate(all_gestures)}
            ys = [y_map.get(h[0], 0) for h in self._history]
            c_hist = [colors_map.get(h[0], "#555") for h in self._history]
            self.hist_plot.plot(xs, ys, pen=None, symbol='o', symbolSize=3, symbolBrush=c_hist)

    def show(self) -> None:
        self.win.show()
