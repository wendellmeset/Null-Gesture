"""Live mmWave gesture detection GUI."""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from null_gesture.sensors.mmwave import MMWaveSensor
from null_gesture.pipeline.detector import MMWaveDetector


class MMWaveLiveWindow(QtWidgets.QWidget):
    def __init__(self, port: str = "/dev/ttyACM0"):
        super().__init__()
        self._port = port

        self._radar = MMWaveSensor()
        self._radar_ok = False

        self._detector = MMWaveDetector()
        self._detector.load_model()

        self._current = "standing_still"

        self._init_ui()
        self._init_timer()
        self._connect()

    def _connect(self) -> None:
        self._radar_ok = self._radar.connect(self._port)

    def _init_ui(self) -> None:
        self.setWindowTitle("Null-Gesture — mmWave")
        self.resize(500, 350)
        self.setStyleSheet(
            "background-color: #0d1117; color: #e6edf3; "
            "font-family: sans-serif; font-size: 13px;"
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("Null-Gesture — mmWave Radar")
        title.setStyleSheet("color: #58a6ff; font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        card = QtWidgets.QWidget()
        card.setStyleSheet("background: #161b22; border-radius: 10px; padding: 24px;")
        cl = QtWidgets.QVBoxLayout(card)
        cl.setSpacing(6)

        self._label = QtWidgets.QLabel("Standing Still")
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("color: #58a6ff; font-size: 42px; font-weight: bold;")
        cl.addWidget(self._label)

        self._conf_label = QtWidgets.QLabel("")
        self._conf_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._conf_label.setStyleSheet("color: #8b949e; font-size: 16px;")
        cl.addWidget(self._conf_label)

        self._detail = QtWidgets.QLabel("Connect mmWave radar to begin")
        self._detail.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._detail.setStyleSheet("color: #484f58; font-size: 12px;")
        cl.addWidget(self._detail)

        self._pos_label = QtWidgets.QLabel("")
        self._pos_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._pos_label.setStyleSheet("color: #30363d; font-size: 11px; font-family: monospace;")
        cl.addWidget(self._pos_label)

        layout.addWidget(card)
        layout.addStretch()

        self._status = QtWidgets.QLabel("Disconnected")
        self._status.setStyleSheet("color: #8b949e; font-size: 11px;")
        layout.addWidget(self._status)

    def _init_timer(self) -> None:
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    def _tick(self) -> None:
        if not self._radar_ok:
            return

        self._radar.ingest()
        pt = self._radar.get_dominant_point()
        label, conf = self._detector.feed(pt)

        if pt is not None:
            self._pos_label.setText(
                f"hand: ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})m  |  "
                f"points: {len(self._radar._points)}"
            )

        if label != self._current:
            self._current = label
            name = label.replace("_", " ").title()
            colors = {
                "standing_still": "#58a6ff",
                "push": "#f85149", "pull": "#3fb950",
                "left": "#d2991d", "right": "#d2991d",
                "clockwise": "#a371f7", "anti_clockwise": "#a371f7",
                "bye_bye": "#f778ba", "clapping": "#56d364",
                "one_arm_boxing": "#e5534b",
                "t_arms": "#79c0ff", "raise_arms": "#ff7b72",
                "palm_up": "#a5d6ff", "palm_down": "#ffa198",
            }
            c = colors.get(label, "#e6edf3")
            self._label.setText(name)
            self._label.setStyleSheet(f"color: {c}; font-size: 42px; font-weight: bold;")
            self._conf_label.setText(f"{conf:.0%}")
            self._detail.setText(
                f"state: {self._detector.state}  |  "
                f"model: {'✓' if self._detector._classifier.fitted else '✗'}"
            )

        self._status.setText(
            f"Connected  |  frame: {self._radar.frame_number}"
            if self._radar_ok else "Disconnected"
        )

    def closeEvent(self, event):
        if self._radar_ok:
            self._radar.disconnect()
        super().closeEvent(event)
