"""Live interactive training + prediction GUI.

Real-time gesture labeling: press gesture buttons while performing,
collects 2s windows continuously, trains on demand, shows live results.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("null_gesture.gui.live")


def _try_qt_imports():
    """Lazy import Qt — may not be installed."""
    from PyQt6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True, useNumba=False)
    return QtCore, QtGui, QtWidgets, pg


class LiveTrainWindow:
    """Interactive window for real-time gesture collection + training."""

    WINDOW_MS = 50  # UI refresh rate

    def __init__(
        self,
        gestures: list[str],
        imu_host: str = "127.0.0.1",
        window_seconds: float = 2.0,
    ):
        self.QtCore, self.QtGui, self.QtWidgets, self.pg = _try_qt_imports()

        self.gestures = gestures
        self.imu_host = imu_host
        self.window_seconds = window_seconds
        self.window_samples = int(window_seconds * 50)  # 50Hz → samples

        # Data buffers
        self.imu_windows: list[np.ndarray] = []
        self.labels: list[int] = []
        self._active_gesture: int | None = None  # Currently recording gesture index
        self._window_start: float | None = None
        self._current_buffer: deque[np.ndarray] = deque()

        # IMU
        from null_gesture.sensors.imu_sensor import IMUClient
        self.imu = IMUClient()
        self.imu_connected = False

        # Model
        self.model: "GestureFusionModel | None" = None
        self.preprocessor: "Preprocessor | None" = None
        self._pred_probs: np.ndarray | None = None

        # Build UI
        self._init_ui()
        self._init_timers()

    # ── UI ──────────────────────────────────────────────────────────────

    def _init_ui(self):
        QtWidgets = self.QtWidgets
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("Null-Gesture — Live Train")
        self.win.resize(1000, 750)

        dark = "#0d1117"
        card = "#161b22"
        border = "#30363d"
        text = "#e6edf3"
        accent = "#58a6ff"
        green = "#3fb950"
        red = "#f85149"

        self.win.setStyleSheet(f"""
            QMainWindow {{ background-color: {dark}; }}
            QWidget {{ color: {text}; font-family: sans-serif; font-size: 12px; }}
            QPushButton {{
                background-color: {card}; border: 1px solid {border};
                border-radius: 6px; padding: 8px 16px; color: {text};
            }}
            QPushButton:hover {{ border-color: {accent}; }}
            QPushButton:pressed {{ background-color: {accent}; color: {dark}; }}
            QPushButton:checked {{ background-color: {green}; color: {dark}; border-color: {green}; }}
            QLabel#count {{ font-size: 28px; font-weight: bold; }}
            QLabel#gesture {{ font-size: 14px; }}
        """)

        central = QtWidgets.QWidget()
        self.win.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # ── Header ──────────────────────────────────────────────────
        header = QtWidgets.QLabel("Live Training — IMU Gesture Collector")
        header.setStyleSheet(f"color: {accent}; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        status = QtWidgets.QLabel("Click Connect to start the IMU stream")
        status.setStyleSheet("color: #8b949e;")
        layout.addWidget(status)
        self.status_label = status

        # ── IMU plot ─────────────────────────────────────────────────
        self.plot = self.pg.PlotWidget()
        self.plot.setBackground(card)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setMaximumHeight(160)
        colors = ["#58a6ff", "#3fb950", "#d2991d", "#f85149", "#a371f7", "#79c0ff"]
        self.curves = []
        self.plot_buffers = [deque(maxlen=200) for _ in range(6)]
        for i, ch in enumerate(["ax", "ay", "az", "gx", "gy", "gz"]):
            c = self.plot.plot([], [], pen=self.pg.mkPen(colors[i], width=1.5), name=ch)
            self.curves.append(c)
        self.plot.addLegend(offset=(1, 1))
        layout.addWidget(self.plot)

        # ── Gesture buttons + counts ─────────────────────────────────
        btn_layout = QtWidgets.QHBoxLayout()
        self.gesture_btns: list[QtWidgets.QPushButton] = []
        self.count_labels: list[QtWidgets.QLabel] = []

        for i, g in enumerate(self.gestures):
            gbox = QtWidgets.QVBoxLayout()

            btn = QtWidgets.QPushButton(g.replace("_", " ").title())
            btn.setCheckable(True)
            btn.setMinimumHeight(48)
            btn.setMinimumWidth(120)
            btn.clicked.connect(lambda checked, idx=i: self._on_gesture_toggle(idx, checked))
            self.gesture_btns.append(btn)

            cnt = QtWidgets.QLabel("0")
            cnt.setObjectName("count")
            cnt.setAlignment(self.QtCore.Qt.AlignmentFlag.AlignCenter)
            self.count_labels.append(cnt)

            lbl = QtWidgets.QLabel(g.replace("_", " ").title())
            lbl.setObjectName("gesture")
            lbl.setAlignment(self.QtCore.Qt.AlignmentFlag.AlignCenter)

            gbox.addWidget(btn)
            gbox.addWidget(cnt)
            gbox.addWidget(lbl)
            btn_layout.addLayout(gbox)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # ── Action row ───────────────────────────────────────────────
        action_layout = QtWidgets.QHBoxLayout()

        self.connect_btn = QtWidgets.QPushButton("🔌 Connect IMU")
        self.connect_btn.clicked.connect(self._connect_imu)
        action_layout.addWidget(self.connect_btn)

        self.train_btn = QtWidgets.QPushButton("🎯 Train & Predict")
        self.train_btn.clicked.connect(self._train_and_predict)
        self.train_btn.setEnabled(False)
        action_layout.addWidget(self.train_btn)

        self.clear_btn = QtWidgets.QPushButton("🗑 Clear Data")
        self.clear_btn.clicked.connect(self._clear_data)
        action_layout.addWidget(self.clear_btn)

        action_layout.addStretch()
        layout.addLayout(action_layout)

        # ── Prediction bar ───────────────────────────────────────────
        self.pred_plot = self.pg.PlotWidget()
        self.pred_plot.setBackground(card)
        self.pred_plot.hideAxis("left")
        self.pred_plot.hideAxis("bottom")
        self.pred_plot.setMaximumHeight(120)
        layout.addWidget(self.pred_plot)

        self.pred_label = QtWidgets.QLabel("Train a model to see predictions")
        self.pred_label.setStyleSheet(f"color: {accent}; font-size: 16px;")
        layout.addWidget(self.pred_label)

    def _init_timers(self):
        self._timer = self.QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(self.WINDOW_MS)

    # ── Logic ────────────────────────────────────────────────────────────

    def _connect_imu(self):
        if self.imu_connected:
            self.imu.disconnect()
            self.imu_connected = False
            self.connect_btn.setText("🔌 Connect IMU")
            self.status_label.setText("Disconnected")
            return

        self.status_label.setText("Connecting...")
        self.QtWidgets.QApplication.processEvents()
        if self.imu.connect():
            self.imu_connected = True
            self.connect_btn.setText("⚡ Disconnect IMU")
            self.status_label.setText(f"Connected — press a gesture button to start collecting")
        else:
            self.status_label.setText("❌ Connection failed — is esp32_reader.py running?")

    def _on_gesture_toggle(self, idx: int, checked: bool):
        if checked:
            # Deactivate all other buttons
            for i, btn in enumerate(self.gesture_btns):
                if i != idx:
                    btn.setChecked(False)
            self._active_gesture = idx
            self._window_start = time.time()
            self._current_buffer.clear()
            self.status_label.setText(f"Recording: {self.gestures[idx].replace('_',' ').title()}...")
        else:
            if self._active_gesture == idx:
                self._active_gesture = None
                self._window_start = None
                self.status_label.setText("Recording stopped")

    def _tick(self):
        if not self.imu_connected:
            return

        # Ingest data
        self.imu.ingest(max_samples=20)
        window = self.imu.get_window()

        # Update plot
        if window.sum() != 0:
            for i in range(6):
                self.plot_buffers[i].append(float(window[-1, i]))
            for i, buf in enumerate(self.plot_buffers):
                b = list(buf)
                if b:
                    self.curves[i].setData(range(len(b)), b)

        # If recording, accumulate samples
        if self._active_gesture is not None and self._window_start is not None:
            self._current_buffer.append(window[-1].copy())
            elapsed = time.time() - self._window_start
            if elapsed >= self.window_seconds:
                # Build a full window
                samples = list(self._current_buffer)
                if len(samples) >= 10:
                    arr = np.array(samples[-self.window_samples:], dtype=np.float32)
                    if arr.shape[0] < self.window_samples:
                        padded = np.zeros((self.window_samples, 6), dtype=np.float32)
                        padded[-arr.shape[0]:] = arr
                        arr = padded
                    self.imu_windows.append(arr)
                    self.labels.append(self._active_gesture)
                    cnt = sum(1 for l in self.labels if l == self._active_gesture)
                    self.count_labels[self._active_gesture].setText(str(cnt))
                # Reset for next window
                self._window_start = time.time()
                self._current_buffer.clear()

        # If model is loaded, show prediction
        if self.model is not None and window.sum() != 0:
            self._update_prediction(window)

    def _update_prediction(self, window: np.ndarray):
        from null_gesture.config import GESTURES

        imu_t = self._preprocess_imu(window)
        with torch.no_grad():
            _, probs = self.model.predict(
                imu_t,
                torch.zeros(1, 60, 2),
                torch.zeros(1, 100, 1),
                modalities="imu",
            )
        self._pred_probs = probs[0].cpu().numpy()

        # Update bar chart
        self.pred_plot.clear()
        top5 = np.argsort(self._pred_probs)[::-1][:5]
        y = list(range(5))
        widths = [float(self._pred_probs[i]) for i in top5]
        c = ["#58a6ff", "#3fb950", "#d2991d", "#f85149", "#a371f7"]
        bar = self.pg.BarGraphItem(x0=0, y=y, width=widths, height=0.7, brushes=c)
        self.pred_plot.addItem(bar)
        for j, (idx, w) in enumerate(zip(top5, widths)):
            label = GESTURES[idx].replace("_", " ").title()
            t = self.pg.TextItem(f"{label} ({w:.0%})", color="#e6edf3", anchor=(0, 0.5))
            t.setPos(w + 0.02, j)
            self.pred_plot.addItem(t)
        self.pred_plot.setXRange(0, 1.15)

        top_gesture = GESTURES[top5[0]].replace("_", " ").title()
        top_conf = widths[0]
        if top_conf > 0.5:
            self.pred_label.setText(f"→ {top_gesture} ({top_conf:.0%})")
        else:
            self.pred_label.setText(f"Uncertain ({top_gesture} {top_conf:.0%})")

    def _preprocess_imu(self, window: np.ndarray) -> torch.Tensor:
        """Preprocess a raw IMU window for model input."""
        if self.preprocessor is not None:
            mean = self.preprocessor._imu_mean
            std = self.preprocessor._imu_std
            window = (window - mean) / std
        return torch.from_numpy(window).float().unsqueeze(0)

    def _train_and_predict(self):
        total = len(self.labels)
        if total < 10:
            self.status_label.setText(f"Need at least 10 samples, have {total}")
            return

        self.status_label.setText("Training...")
        self.QtWidgets.QApplication.processEvents()

        from null_gesture.config import ModelConfig, GESTURES
        from null_gesture.data.preprocessor import Preprocessor
        from null_gesture.data.dataset import GestureDataset
        from null_gesture.models.fusion import GestureFusionModel
        from null_gesture.models.trainer import Trainer

        # Build arrays
        imu_arr = np.array(self.imu_windows, dtype=np.float32)
        labels_arr = np.array(self.labels, dtype=np.int32)

        # Fit preprocessor
        self.preprocessor = Preprocessor()
        rfid_dummy = np.zeros((len(imu_arr), 60, 2), dtype=np.float32)
        uwb_dummy = np.zeros((len(imu_arr), 100, 1), dtype=np.float32)
        self.preprocessor.fit(imu_arr, rfid_dummy, uwb_dummy)

        imu_norm, _, _ = self.preprocessor.transform(imu_arr, rfid_dummy, uwb_dummy)

        # Split
        n = len(imu_arr)
        indices = np.random.default_rng(42).permutation(n)
        split = max(2, int(n * 0.75))
        train_idx = indices[:split]
        val_idx = indices[split:]

        train_ds = GestureDataset(
            imu_norm[train_idx], rfid_dummy[train_idx], uwb_dummy[train_idx], labels_arr[train_idx],
            augment=True if n > 20 else False,
        )
        val_ds = GestureDataset(
            imu_norm[val_idx], rfid_dummy[val_idx], uwb_dummy[val_idx], labels_arr[val_idx],
        )

        cfg = ModelConfig(epochs=40, batch_size=min(8, max(2, len(train_ds) // 2)),
                          early_stopping_patience=10, learning_rate=2e-3)

        self.model = GestureFusionModel(cfg)
        trainer = Trainer(self.model, config=cfg)
        history = trainer.train(train_ds, val_ds, model_name="live_model")

        best_acc = max(history["val_acc"])
        self.status_label.setText(f"✅ Trained! Val accuracy: {best_acc:.1%} | {total} samples, {split}/{n-split} split")
        self.train_btn.setText(f"🎯 Retrain ({best_acc:.0%})")

    def _clear_data(self):
        self.imu_windows.clear()
        self.labels.clear()
        for lbl in self.count_labels:
            lbl.setText("0")
        self.model = None
        self.preprocessor = None
        self.pred_plot.clear()
        self.pred_label.setText("Train a model to see predictions")
        self.train_btn.setText("🎯 Train & Predict")
        self.train_btn.setEnabled(False)
        self.status_label.setText("Data cleared")

    def show(self):
        self.win.show()

    @property
    def app(self):
        return self.QtWidgets.QApplication.instance()
