"""Real-time gesture prediction GUI using PyQt6 and pyqtgraph.

Displays live sensor feeds, model predictions with confidence bars,
and a gesture probability distribution chart.
"""

from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np

from null_gesture.config import GESTURES, NUM_GESTURES

logger = logging.getLogger("null_gesture.gui")

# Lazy imports — GUI may not be available in headless environments
_QT_AVAILABLE = False
try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg

    pg.setConfigOptions(antialias=True, useNumba=False)
    _QT_AVAILABLE = True
except ImportError:
    pass


# ── Color palette ───────────────────────────────────────────────────────────
PALETTE = {
    "bg_dark": "#0d1117",
    "bg_card": "#161b22",
    "border": "#30363d",
    "text_primary": "#e6edf3",
    "text_secondary": "#8b949e",
    "accent": "#58a6ff",
    "accent_green": "#3fb950",
    "accent_orange": "#d2991d",
    "accent_red": "#f85149",
    "accent_purple": "#a371f7",
}

# Gesture-specific colors — cycling through a palette
GESTURE_COLORS = [
    "#58a6ff", "#3fb950", "#d2991d", "#f85149", "#a371f7",
    "#79c0ff", "#56d364", "#e3b341", "#ff7b72", "#d2a8ff",
    "#7ee787", "#f0883e", "#ffa198", "#bc8cff", "#39d353",
]


class SensorWidget(QtWidgets.QWidget):
    """Real-time sensor data plot for one modality."""

    def __init__(
        self,
        title: str,
        channels: list[str],
        max_points: int = 300,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.title = title
        self.channels = channels
        self.max_points = max_points

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Title
        title_label = QtWidgets.QLabel(self.title)
        title_label.setStyleSheet(
            f"color: {PALETTE['text_primary']}; font-weight: bold; font-size: 13px;"
        )
        layout.addWidget(title_label)

        # Plot widget
        self.plot = pg.PlotWidget()
        self.plot.setBackground(PALETTE["bg_card"])
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel("bottom", "")
        self.plot.setLabel("left", "")
        self.plot.setMaximumHeight(150)

        colors = ["#58a6ff", "#3fb950", "#d2991d", "#f85149", "#a371f7", "#79c0ff"]
        self.curves: list[pg.PlotDataItem] = []
        self.buffers: list[deque] = []

        for i, ch in enumerate(self.channels):
            color = colors[i % len(colors)]
            curve = self.plot.plot(
                [], [],
                pen=pg.mkPen(color, width=1.5),
                name=ch,
            )
            self.curves.append(curve)
            self.buffers.append(deque(maxlen=self.max_points))

        if len(self.channels) > 1:
            self.plot.addLegend(offset=(1, 1))

        layout.addWidget(self.plot)

    def update_data(self, data: np.ndarray) -> None:
        """Push new data point(s). data shape: (timesteps, channels) or (channels,)."""
        if data.ndim == 2:
            data = data[-1]  # Take the latest timestep
        for i in range(min(len(self.channels), len(data))):
            self.buffers[i].append(float(data[i]))

        for i, curve in enumerate(self.curves):
            buf = list(self.buffers[i])
            if buf:
                curve.setData(range(len(buf)), buf)


class PredictionWidget(QtWidgets.QWidget):
    """Displays current prediction with confidence bars for all classes."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Current prediction label
        pred_layout = QtWidgets.QHBoxLayout()
        self.pred_icon = QtWidgets.QLabel("●")
        self.pred_icon.setStyleSheet(
            f"color: {PALETTE['accent_green']}; font-size: 18px;"
        )
        self.pred_label = QtWidgets.QLabel("Waiting...")
        self.pred_label.setStyleSheet(
            f"color: {PALETTE['text_primary']}; font-size: 22px; font-weight: bold;"
        )
        self.conf_label = QtWidgets.QLabel("")
        self.conf_label.setStyleSheet(f"color: {PALETTE['text_secondary']}; font-size: 14px;")
        pred_layout.addWidget(self.pred_icon)
        pred_layout.addWidget(self.pred_label)
        pred_layout.addWidget(self.conf_label)
        pred_layout.addStretch()
        layout.addLayout(pred_layout)

        # Confidence bar chart
        self.bar_widget = pg.PlotWidget()
        self.bar_widget.setBackground(PALETTE["bg_card"])
        self.bar_widget.hideAxis("left")
        self.bar_widget.hideAxis("bottom")
        self.bar_widget.setMaximumHeight(200)
        self.bar_widget.setMinimumHeight(150)
        layout.addWidget(self.bar_widget)

        self.bar_items: list[pg.BarGraphItem] = []

    def update_prediction(self, probs: np.ndarray | None) -> None:
        """Update with probability distribution over all gestures."""
        if probs is None or len(probs) == 0:
            self.pred_label.setText("No data")
            self.conf_label.setText("")
            return

        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        gesture = GESTURES[top_idx].replace("_", " ").title()

        self.pred_label.setText(gesture)

        if top_conf > 0.75:
            color = PALETTE["accent_green"]
        elif top_conf > 0.4:
            color = PALETTE["accent_orange"]
        else:
            color = PALETTE["accent_red"]
        self.pred_icon.setStyleSheet(f"color: {color}; font-size: 18px;")
        self.conf_label.setText(f"{top_conf:.1%}")

        # Bar chart — sort by probability descending
        indices = np.argsort(probs)[::-1]
        self._draw_bars(indices, probs)

    def _draw_bars(self, indices: np.ndarray, probs: np.ndarray) -> None:
        self.bar_widget.clear()
        y = list(range(len(indices)))
        widths = [float(probs[i]) for i in indices]
        colors = [GESTURE_COLORS[i % len(GESTURE_COLORS)] for i in indices]
        labels = [GESTURES[i].replace("_", " ").title() for i in indices]

        bar = pg.BarGraphItem(
            x0=0, y=y, width=widths, height=0.7, brushes=colors,
        )
        self.bar_widget.addItem(bar)

        # Label the top 5 bars
        for j, (idx, label, w) in enumerate(zip(indices[:5], labels[:5], widths[:5])):
            text = pg.TextItem(
                f"{label}  ({w:.0%})",
                color=PALETTE["text_primary"],
                anchor=(0, 0.5),
            )
            text.setPos(w + 0.02, j)
            self.bar_widget.addItem(text)

        self.bar_widget.setXRange(0, 1.15)
        self.bar_widget.setYRange(-0.5, len(indices) - 0.5)


class GestureGUI(QtWidgets.QMainWindow):
    """Main application window for real-time gesture prediction."""

    PREDICTION_INTERVAL_MS = 100  # 10 Hz prediction updates

    def __init__(
        self,
        *,
        prediction_callback=None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if not _QT_AVAILABLE:
            raise ImportError(
                "PyQt6 and pyqtgraph required. Install: pip install PyQt6 pyqtgraph"
            )
        super().__init__(parent)
        self.prediction_callback = prediction_callback
        self._imu_data: np.ndarray | None = None
        self._rfid_data: np.ndarray | None = None
        self._uwb_data: np.ndarray | None = None
        self._probs: np.ndarray | None = None

        self._init_ui()
        self._init_timer()

    def _init_ui(self) -> None:
        self.setWindowTitle("Null-Gesture — Real-Time Prediction")
        self.resize(1200, 850)

        # Dark theme
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {PALETTE['bg_dark']};
            }}
            QWidget {{
                background-color: {PALETTE['bg_dark']};
                color: {PALETTE['text_primary']};
                font-family: "SF Pro Display", "Inter", "Segoe UI", sans-serif;
            }}
            QStatusBar {{
                background-color: {PALETTE['bg_card']};
                color: {PALETTE['text_secondary']};
                border-top: 1px solid {PALETTE['border']};
            }}
        """)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Header
        header = QtWidgets.QLabel("Null-Gesture")
        header.setStyleSheet(
            f"color: {PALETTE['accent']}; font-size: 24px; font-weight: bold;"
        )
        main_layout.addWidget(header)

        subtitle = QtWidgets.QLabel(
            "Multi-Modal Gesture Detection — IMU + RFID + UWB"
        )
        subtitle.setStyleSheet(f"color: {PALETTE['text_secondary']}; font-size: 13px;")
        main_layout.addWidget(subtitle)

        # ── Sensor panels ────────────────────────────────────────────
        sensors_layout = QtWidgets.QHBoxLayout()
        sensors_layout.setSpacing(12)

        self.imu_widget = SensorWidget(
            "IMU — Accelerometer + Gyroscope",
            ["ax", "ay", "az", "gx", "gy", "gz"],
        )
        self.rfid_widget = SensorWidget(
            "RFID — RSSI + Phase",
            ["RSSI", "Phase"],
        )
        self.uwb_widget = SensorWidget(
            "UWB — Inter-Hand Distance",
            ["Distance (cm)"],
        )

        sensors_layout.addWidget(self.imu_widget)
        sensors_layout.addWidget(self.rfid_widget)
        sensors_layout.addWidget(self.uwb_widget)
        main_layout.addLayout(sensors_layout)

        # ── Prediction panel ────────────────────────────────────────
        pred_card = QtWidgets.QWidget()
        pred_card.setStyleSheet(
            f"background-color: {PALETTE['bg_card']}; "
            f"border: 1px solid {PALETTE['border']}; "
            f"border-radius: 8px; padding: 12px;"
        )
        pred_layout = QtWidgets.QVBoxLayout(pred_card)
        self.pred_widget = PredictionWidget()
        pred_layout.addWidget(self.pred_widget)
        main_layout.addWidget(pred_card)

        # Status bar
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — waiting for sensor data...")

    def _init_timer(self) -> None:
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(self.PREDICTION_INTERVAL_MS)

    def _tick(self) -> None:
        """Called every PREDICTION_INTERVAL_MS to update UI and run inference."""
        if self.prediction_callback is not None:
            try:
                result = self.prediction_callback()
                if result is not None:
                    imu, rfid, uwb, probs = result
                    if imu is not None:
                        self._imu_data = imu
                        self.imu_widget.update_data(imu)
                    if rfid is not None:
                        self._rfid_data = rfid
                        self.rfid_widget.update_data(rfid)
                    if uwb is not None:
                        self._uwb_data = uwb
                        self.uwb_widget.update_data(uwb)
                    if probs is not None:
                        self._probs = probs
                        self.pred_widget.update_prediction(probs)
            except Exception as exc:
                logger.debug("Prediction tick error: %s", exc)

    def update_status(self, message: str) -> None:
        self.status_bar.showMessage(message)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._timer.stop()
        super().closeEvent(event)
