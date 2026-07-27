"""Live gesture detection GUI — connects to IMU, shows results."""
from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from null_gesture.pipeline.detector import GestureDetector
from null_gesture.sensors.imu import IMUSensor


class LiveWindow(QtWidgets.QWidget):
    """Real-time gesture display."""

    def __init__(self, serial: str | None = None, host: str = "127.0.0.1", port: int = 9999):
        super().__init__()
        self._serial = serial
        self._host = host
        self._port = port

        self._imu = IMUSensor()
        self._imu_ok = False

        self._detector = GestureDetector()
        self._detector.load_model()  # loads if exists, silent if not

        self._current = "standing_still"
        self._conf = 1.0

        self._init_ui()
        self._init_timer()
        self._connect()

    def _connect(self) -> None:
        ok = False
        if self._serial:
            ok = self._imu.connect_serial(self._serial)
        if not ok:
            ok = self._imu.connect_tcp(self._host, self._port)
        self._imu_ok = ok

    # ── UI ─────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        self.setWindowTitle("Null-Gesture")
        self.resize(500, 350)
        self.setStyleSheet(
            "background-color: #0d1117; color: #e6edf3; "
            "font-family: sans-serif; font-size: 13px;"
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("Null-Gesture")
        title.setStyleSheet("color: #58a6ff; font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # Result card
        card = QtWidgets.QWidget()
        card.setStyleSheet("background: #161b22; border-radius: 10px; padding: 24px;")
        cl = QtWidgets.QVBoxLayout(card)
        cl.setSpacing(6)

        self._label = QtWidgets.QLabel("Standing Still")
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(
            "color: #58a6ff; font-size: 42px; font-weight: bold;"
        )
        cl.addWidget(self._label)

        self._conf_label = QtWidgets.QLabel("")
        self._conf_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._conf_label.setStyleSheet("color: #8b949e; font-size: 16px;")
        cl.addWidget(self._conf_label)

        self._detail = QtWidgets.QLabel("Connect IMU to begin")
        self._detail.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._detail.setStyleSheet("color: #484f58; font-size: 12px;")
        cl.addWidget(self._detail)

        layout.addWidget(card)
        layout.addStretch()

        # Status
        self._status = QtWidgets.QLabel("Disconnected")
        self._status.setStyleSheet("color: #8b949e; font-size: 11px;")
        layout.addWidget(self._status)

    # ── Timer ──────────────────────────────────────────────────────

    def _init_timer(self) -> None:
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    def _tick(self) -> None:
        if not self._imu_ok:
            return

        self._imu.ingest(max_samples=5)
        window = self._imu.get_window()
        if window.sum() == 0:
            return

        # Feed latest sample to detector
        sample = window[-1]
        label, conf = self._detector.feed(sample)

        if label != self._current:
            self._current = label
            self._conf = conf

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

            state = self._detector.state
            trained = "✓" if self._detector._classifier.fitted else "✗"
            self._detail.setText(f"state: {state}  |  model: {trained}")

        if self._imu_ok:
            self._status.setText("Connected")
        else:
            self._status.setText("Disconnected")

    def closeEvent(self, a0):
        if self._imu_ok:
            self._imu.disconnect()
        super().closeEvent(a0)
