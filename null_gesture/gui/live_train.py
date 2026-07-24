"""Live interactive training GUI — guided repetition-based collection.

Workflow per gesture:
1. Click "Push" → starts guided sequence
2. "Push 1/10" prompt → record 2s → "Rest..." pause (1.5s) → "Push 2/10" → ...
3. After all reps, stops. Repeat for other gestures.
4. Click "Train & Predict" → trains immediately → live predictions.
"""

from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np
import torch

logger = logging.getLogger("null_gesture.gui.live")


def _try_qt_imports():
    from PyQt6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True, useNumba=False)
    return QtCore, QtGui, QtWidgets, pg


class GuidedCollector:
    """State machine for guided gesture collection."""

    def __init__(self, window_s: float, reps: int, rest_s: float = 1.5):
        self.window_s = window_s
        self.reps = reps
        self.rest_s = rest_s
        self.window_samples = int(window_s * 50)  # 50Hz target
        self.reset()

    def reset(self):
        self._gesture_idx: int | None = None
        self._gesture_name: str = ""
        self._state: str = "idle"  # idle | countdown | recording | resting | done
        self._rep: int = 0
        self._phase_start: float = 0.0
        self._buffer: deque[np.ndarray] = deque()
        self._countdown_remaining: float = 0.0

    def start_gesture(self, idx: int, name: str):
        self._gesture_idx = idx
        self._gesture_name = name
        self._rep = 0
        self._start_countdown()

    def cancel(self):
        self.reset()

    def _start_countdown(self):
        self._state = "countdown"
        self._countdown_remaining = 1.0
        self._phase_start = time.time()

    def _start_recording(self):
        self._state = "recording"
        self._rep += 1
        self._buffer.clear()
        self._phase_start = time.time()

    def _start_resting(self):
        self._state = "resting"
        self._phase_start = time.time()

    def update(self, window: np.ndarray) -> tuple[str, str, np.ndarray | None, int | None]:
        """Call every tick. Returns (state, display_text, collected_window or None, gesture_idx or None)."""
        now = time.time()

        if self._state == "idle":
            return "idle", "Click a gesture button to start", None, None

        elif self._state == "countdown":
            elapsed = now - self._phase_start
            remaining = max(0, self._countdown_remaining - elapsed)
            if remaining <= 0:
                self._start_recording()
                return "recording", f"{self._gesture_name} {self._rep}/{self.reps} — GO!", None, None
            return "countdown", f"{self._gesture_name} {self._rep+1}/{self.reps} in {remaining:.0f}...", None, None

        elif self._state == "recording":
            if window.sum() != 0:
                self._buffer.append(window[-1].copy())
            elapsed = now - self._phase_start
            remaining = max(0, self.window_s - elapsed)
            if remaining <= 0:
                # Build window
                samples = list(self._buffer)
                arr = None
                if len(samples) >= 10:
                    arr = np.array(samples[-self.window_samples:], dtype=np.float32)
                    if arr.shape[0] < self.window_samples:
                        padded = np.zeros((self.window_samples, 6), dtype=np.float32)
                        padded[-arr.shape[0]:] = arr
                        arr = padded
                if self._rep >= self.reps:
                    self._state = "done"
                    return "done", f"✅ {self._gesture_name} complete ({self.reps} reps)", arr, self._gesture_idx
                self._start_resting()
                label = self._gesture_idx
                return "resting", f"Rest...", arr, label
            return "recording", f"{self._gesture_name} {self._rep}/{self.reps} — {remaining:.1f}s", None, None

        elif self._state == "resting":
            elapsed = now - self._phase_start
            if elapsed >= self.rest_s:
                self._start_countdown()
                return "countdown", f"{self._gesture_name} {self._rep+1}/{self.reps} in 1...", None, None
            return "resting", f"Rest... ({self.rest_s - elapsed:.1f}s)", None, None

        elif self._state == "done":
            return "done", f"✅ {self._gesture_name} complete ({self.reps} reps)", None, None

        return "idle", "", None, None


class LiveTrainWindow:
    """Interactive window for guided gesture collection + training."""

    WINDOW_MS = 50
    DEFAULT_REPS = 10

    def __init__(
        self,
        gestures: list[str],
        imu_host: str = "127.0.0.1",
        window_seconds: float = 2.0,
        reps: int = 10,
    ):
        self.QtCore, self.QtGui, self.QtWidgets, self.pg = _try_qt_imports()

        self.gestures = gestures
        self.imu_host = imu_host
        self.window_seconds = window_seconds
        self.reps = reps

        self.collector = GuidedCollector(window_seconds, reps)

        # Data
        self.imu_windows: list[np.ndarray] = []
        self.labels: list[int] = []

        # IMU
        from null_gesture.sensors.imu_sensor import IMUClient
        self.imu = IMUClient()
        self.imu_connected = False

        # Model + prediction
        self.model: "GestureFusionModel | None" = None
        self.preprocessor: "Preprocessor | None" = None

        self._init_ui()
        self._init_timers()

    # ── UI ──────────────────────────────────────────────────────────────

    def _init_ui(self):
        QtWidgets = self.QtWidgets
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("Null-Gesture — Live Train")
        self.win.resize(1000, 750)

        D = "#0d1117"; C = "#161b22"; B = "#30363d"; T = "#e6edf3"
        A = "#58a6ff"; G = "#3fb950"; O = "#d2991d"; R = "#f85149"

        self.win.setStyleSheet(f"""
            QMainWindow {{ background-color: {D}; }}
            QWidget {{ color: {T}; font-family: sans-serif; font-size: 12px; }}
            QPushButton {{
                background-color: {C}; border: 1px solid {B};
                border-radius: 6px; padding: 10px 20px; color: {T};
            }}
            QPushButton:hover {{ border-color: {A}; }}
            QPushButton:disabled {{ color: #555; border-color: #333; }}
            QPushButton#startBtn {{ font-size: 16px; font-weight: bold; padding: 14px 28px; }}
            QLabel#big {{ font-size: 32px; font-weight: bold; }}
            QLabel#count {{ font-size: 20px; }}
            QLabel#gesture {{ font-size: 13px; }}
        """)

        central = QtWidgets.QWidget()
        self.win.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Header
        hdr = QtWidgets.QLabel("Live Training — IMU Gesture Collector")
        hdr.setStyleSheet(f"color: {A}; font-size: 20px; font-weight: bold;")
        layout.addWidget(hdr)

        # ── Prompt area (big text) ────────────────────────────────────
        self.prompt = QtWidgets.QLabel("Click Connect IMU to begin")
        self.prompt.setObjectName("big")
        self.prompt.setAlignment(self.QtCore.Qt.AlignmentFlag.AlignCenter)
        self.prompt.setStyleSheet(f"color: {A}; padding: 20px; background: {C}; border-radius: 8px;")
        layout.addWidget(self.prompt)

        # ── IMU plot ─────────────────────────────────────────────────
        self.plot = self.pg.PlotWidget()
        self.plot.setBackground(C)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setMaximumHeight(150)
        colors = ["#58a6ff", "#3fb950", "#d2991d", "#f85149", "#a371f7", "#79c0ff"]
        self.curves = []
        self.plot_bufs = [deque(maxlen=200) for _ in range(6)]
        for i, ch in enumerate(["ax", "ay", "az", "gx", "gy", "gz"]):
            c = self.plot.plot([], [], pen=self.pg.mkPen(colors[i], width=1.5), name=ch)
            self.curves.append(c)
        self.plot.addLegend(offset=(1, 1))
        layout.addWidget(self.plot)

        # ── Gesture buttons + counts ─────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self.gesture_btns: list[QtWidgets.QPushButton] = []
        self.count_labels: list[QtWidgets.QLabel] = []

        for i, g in enumerate(self.gestures):
            gbox = QtWidgets.QVBoxLayout()
            btn = QtWidgets.QPushButton(f"▶ {g.replace('_', ' ').title()}")
            btn.setObjectName("startBtn")
            btn.setMinimumHeight(60)
            btn.setMinimumWidth(160)
            btn.clicked.connect(lambda checked, idx=i: self._start_gesture(idx))
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
            btn_row.addLayout(gbox)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── Action row ───────────────────────────────────────────────
        act = QtWidgets.QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton("🔌 Connect IMU")
        self.connect_btn.clicked.connect(self._connect_imu)
        act.addWidget(self.connect_btn)

        self.train_btn = QtWidgets.QPushButton("🎯 Train & Predict")
        self.train_btn.clicked.connect(self._train_and_predict)
        self.train_btn.setEnabled(False)
        act.addWidget(self.train_btn)

        self.clear_btn = QtWidgets.QPushButton("🗑 Clear All")
        self.clear_btn.clicked.connect(self._clear_data)
        act.addWidget(self.clear_btn)
        act.addStretch()
        layout.addLayout(act)

        # ── Prediction bar ───────────────────────────────────────────
        self.pred_plot = self.pg.PlotWidget()
        self.pred_plot.setBackground(C)
        self.pred_plot.hideAxis("left"); self.pred_plot.hideAxis("bottom")
        self.pred_plot.setMaximumHeight(120)
        layout.addWidget(self.pred_plot)

        self.pred_label = QtWidgets.QLabel("Collect data then click Train & Predict")
        self.pred_label.setStyleSheet(f"color: {A}; font-size: 15px;")
        layout.addWidget(self.pred_label)

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
            self.prompt.setText("Click Connect IMU to begin")
            return
        self.prompt.setText("Connecting...")
        self.QtWidgets.QApplication.processEvents()
        if self.imu.connect():
            self.imu_connected = True
            self.connect_btn.setText("⚡ Disconnect")
            self.prompt.setText("Ready — click a gesture button to start")
        else:
            self.prompt.setText("❌ Connection failed — is esp32_reader.py running?")

    def _start_gesture(self, idx: int):
        if not self.imu_connected:
            self.prompt.setText("❌ Connect IMU first!")
            return
        # Disable all buttons during collection
        for b in self.gesture_btns:
            b.setEnabled(False)
        self.collector.start_gesture(idx, self.gestures[idx].replace("_", " ").title())

    def _tick(self):
        if not self.imu_connected:
            return

        self.imu.ingest(max_samples=20)
        window = self.imu.get_window()

        # Update plot
        if window.sum() != 0:
            for i in range(6):
                self.plot_bufs[i].append(float(window[-1, i]))
            for i, buf in enumerate(self.plot_bufs):
                b = list(buf)
                if b: self.curves[i].setData(range(len(b)), b)

        # Update collector state machine
        state, text, collected, label_idx = self.collector.update(window)
        self.prompt.setText(text)

        if collected is not None and label_idx is not None:
            self.imu_windows.append(collected)
            self.labels.append(label_idx)
            count = sum(1 for l in self.labels if l == label_idx)
            self.count_labels[label_idx].setText(str(count))

        if state == "done":
            # Re-enable buttons
            for b in self.gesture_btns:
                b.setEnabled(True)
            total = len(self.labels)
            self.train_btn.setEnabled(total >= 10)

        # Live prediction if model loaded
        if self.model is not None and window.sum() != 0:
            self._update_prediction(window)

    def _update_prediction(self, window: np.ndarray):
        from null_gesture.config import GESTURES
        imu_t = self._preprocess_imu(window)
        with torch.no_grad():
            _, probs = self.model.predict(
                imu_t, torch.zeros(1, 60, 2), torch.zeros(1, 100, 1), modalities="imu",
            )
        probs_np = probs[0].cpu().numpy()
        self.pred_plot.clear()
        top5 = np.argsort(probs_np)[::-1][:5]
        widths = [float(probs_np[i]) for i in top5]
        bar = self.pg.BarGraphItem(
            x0=0, y=list(range(5)), width=widths, height=0.7,
            brushes=["#58a6ff", "#3fb950", "#d2991d", "#f85149", "#a371f7"],
        )
        self.pred_plot.addItem(bar)
        for j, (idx, w) in enumerate(zip(top5, widths)):
            t = self.pg.TextItem(f"{GESTURES[idx].replace('_',' ').title()} ({w:.0%})", color="#e6edf3", anchor=(0, 0.5))
            t.setPos(w + 0.02, j)
            self.pred_plot.addItem(t)
        self.pred_plot.setXRange(0, 1.15)
        top = GESTURES[top5[0]].replace("_", " ").title()
        self.pred_label.setText(f"→ {top} ({widths[0]:.0%})")

    def _preprocess_imu(self, window: np.ndarray) -> torch.Tensor:
        if self.preprocessor is not None:
            window = (window - self.preprocessor._imu_mean) / self.preprocessor._imu_std
        return torch.from_numpy(window).float().unsqueeze(0)

    def _train_and_predict(self):
        total = len(self.labels)
        if total < 10:
            self.prompt.setText(f"Need ≥10 samples, have {total}")
            return

        self.prompt.setText("Training model...")
        self.QtWidgets.QApplication.processEvents()

        from null_gesture.config import ModelConfig
        from null_gesture.data.preprocessor import Preprocessor
        from null_gesture.data.dataset import GestureDataset
        from null_gesture.models.fusion import GestureFusionModel
        from null_gesture.models.trainer import Trainer

        imu_arr = np.array(self.imu_windows, dtype=np.float32)
        labels_arr = np.array(self.labels, dtype=np.int32)
        rfid_dummy = np.zeros((len(imu_arr), 60, 2), dtype=np.float32)
        uwb_dummy = np.zeros((len(imu_arr), 100, 1), dtype=np.float32)

        self.preprocessor = Preprocessor()
        self.preprocessor.fit(imu_arr, rfid_dummy, uwb_dummy)
        imu_norm, _, _ = self.preprocessor.transform(imu_arr, rfid_dummy, uwb_dummy)

        n = len(imu_arr)
        idx = np.random.default_rng(42).permutation(n)
        split = max(2, int(n * 0.75))
        train_ds = GestureDataset(
            imu_norm[idx[:split]], rfid_dummy[idx[:split]], uwb_dummy[idx[:split]], labels_arr[idx[:split]],
            augment=n > 20,
        )
        val_ds = GestureDataset(
            imu_norm[idx[split:]], rfid_dummy[idx[split:]], uwb_dummy[idx[split:]], labels_arr[idx[split:]],
        )

        cfg = ModelConfig(epochs=40, batch_size=min(8, max(2, len(train_ds)//2)),
                          early_stopping_patience=10, learning_rate=2e-3)
        self.model = GestureFusionModel(cfg)
        trainer = Trainer(self.model, config=cfg)
        history = trainer.train(train_ds, val_ds, model_name="live_model")

        best = max(history["val_acc"])
        self.prompt.setText(f"✅ Trained! Val acc: {best:.1%} — {total} samples")
        self.train_btn.setText(f"🎯 Retrain ({best:.0%})")

    def _clear_data(self):
        self.imu_windows.clear()
        self.labels.clear()
        self.collector.reset()
        for lbl in self.count_labels: lbl.setText("0")
        self.model = None; self.preprocessor = None
        self.pred_plot.clear()
        self.pred_label.setText("Collect data then click Train & Predict")
        self.train_btn.setText("🎯 Train & Predict")
        self.train_btn.setEnabled(False)
        for b in self.gesture_btns: b.setEnabled(True)
        self.prompt.setText("Ready — click a gesture button to start")

    def show(self):
        self.win.show()

    @property
    def app(self):
        return self.QtWidgets.QApplication.instance()
