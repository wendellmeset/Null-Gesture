#!/usr/bin/env python3
"""Capstone gesture detector — IMU + UWB onset/offset classification.

Key insight: UWB distance CHANGE during a gesture separates translational
gestures (push/pull/raise) from rotational ones (left/right/clockwise).
Combined with IMU dominant axis + oscillation count = reliable 12-gesture
classification without any training data.

Usage:
    source venv/bin/activate
    python -m null_gesture detect --serial /dev/ttyACM0 --uwb-controller /dev/ttyACM1 --uwb-controlee /dev/ttyACM2
"""

from __future__ import annotations

import sys
import time
from collections import deque

import numpy as np
from PyQt6 import QtCore, QtWidgets

# ── Gesture classification ──────────────────────────────────────────

GESTURES = [
    "standing_still",
    "push", "pull",
    "left", "right",
    "clockwise", "anti_clockwise",
    "bye_bye",
    "clapping", "one_arm_boxing",
    "t_arms", "raise_arms",
    "palm_up", "palm_down",
]


def classify(gyro_mean: np.ndarray, accel_mean: np.ndarray,
             uwb_delta: float, gyro_mag_mean: float,
             osc_count: int, accel_var: float,
             duration_s: float) -> tuple[str, float]:
    """Classify a segmented gesture from IMU + UWB features.

    Args:
        gyro_mean: (3,) mean gyro [gx, gy, gz] during gesture (dps)
        accel_mean: (3,) mean accel during gesture (g)
        uwb_delta: UWB distance change during gesture (cm). + = away from anchor
        gyro_mag_mean: mean gyro magnitude (dps)
        osc_count: total gyro zero-crossings in all axes
        accel_var: mean accelerometer variance
        duration_s: gesture duration in seconds
    """
    gx, gy, gz = gyro_mean
    abs_gx, abs_gy, abs_gz = abs(gx), abs(gy), abs(gz)
    dom_axis = np.argmax([abs_gx, abs_gy, abs_gz])

    # ── STAGE 0: Too subtle = palm or t-arms ──────────────────────
    if gyro_mag_mean < 8:
        if abs(accel_mean[2] - 1.0) > 0.15:  # az deviates from 1g = arm orientation changed
            return "t_arms", 0.85
        return "palm_up", 0.80

    # ── STAGE 1: UWB distance change = translational gesture ──────
    if uwb_delta > 15:  # hand moved significantly AWAY from anchor
        if dom_axis == 1 and gy > 0:
            return "push", 0.90
        if dom_axis == 2 or abs_gz > abs_gy:
            return "raise_arms", 0.85
        return "push", 0.78

    if uwb_delta < -15:  # hand moved significantly TOWARD anchor
        if dom_axis == 1 and gy < 0:
            return "pull", 0.90
        return "pull", 0.78

    # ── STAGE 2: Oscillatory gestures ─────────────────────────────
    if osc_count > 18 and duration_s > 0.6:
        if abs_gz > abs_gx and abs_gz > abs_gy:
            return "bye_bye", 0.88
        if abs_gx > abs_gy and accel_var > 0.15:
            return "clapping", 0.85
        if uwb_delta < -5 or uwb_delta > 5:
            return "one_arm_boxing", 0.82
        return "one_arm_boxing", 0.75

    # ── STAGE 3: Rotational — dominant gyro axis ──────────────────
    if dom_axis == 0:  # gx dominant → left/right or palm
        if abs_gx > 20:
            return "left" if gx > 0 else "right", 0.90
        # Subtle gx — palm up/down
        if gx > 0:
            return "palm_up", 0.78
        return "palm_down", 0.78

    if dom_axis == 1:  # gy dominant → push/pull (no UWB change = small motion)
        if abs_gy > 15:
            return "push" if gy > 0 else "pull", 0.82
        return "push" if gy > 0 else "pull", 0.70

    if dom_axis == 2:  # gz dominant → rotation
        return "anti_clockwise" if gz > 0 else "clockwise", 0.88

    # Fallback
    return "standing_still", 0.50


# ── Real-time detector ──────────────────────────────────────────────

class GestureDetector:
    def __init__(self, onset_thresh: float = 14.0, offset_thresh: float = 6.0,
                 offset_frames: int = 8, display_hold: float = 1.5):
        self.onset_thresh = onset_thresh
        self.offset_thresh = offset_thresh
        self.offset_frames = offset_frames
        self.display_hold = display_hold

        self._state = "still"  # still | collecting | classified
        self._raw: deque[np.ndarray] = deque(maxlen=400)
        self._uwb_history: deque[float] = deque(maxlen=400)
        self._onset_idx = 0
        self._still_count = 0
        self._classify_time = 0.0
        self._result = ("standing_still", 1.0)
        self._sample_count = 0

    def update(self, imu_window: np.ndarray, uwb_distance_cm: float = 0.0) -> tuple[str, float]:
        self._sample_count += 1
        now = time.time()

        if imu_window.ndim != 2 or imu_window.shape[0] < 5:
            return self._result

        row = imu_window[-1].astype(np.float32)
        self._raw.append(row)
        self._uwb_history.append(uwb_distance_cm)
        gyro_mag = float(np.linalg.norm(row[3:]))

        if self._state == "still":
            if gyro_mag > self.onset_thresh:
                self._state = "collecting"
                self._onset_idx = max(0, len(self._raw) - 8)
                self._still_count = 0

        elif self._state == "collecting":
            if gyro_mag < self.offset_thresh:
                self._still_count += 1
                if self._still_count >= self.offset_frames:
                    self._classify()
                    self._state = "classified"
                    self._classify_time = now
            else:
                self._still_count = 0
            if len(self._raw) - self._onset_idx >= 250:
                self._classify()
                self._state = "classified"
                self._classify_time = now

        elif self._state == "classified":
            if now - self._classify_time > self.display_hold:
                self._state = "still"
                self._result = ("standing_still", 1.0)

        return self._result

    def _classify(self) -> None:
        start = self._onset_idx
        end = len(self._raw)
        segment = np.array([self._raw[i] for i in range(start, end)], dtype=np.float32)
        uwb_seg = np.array([self._uwb_history[i] for i in range(start, min(end, len(self._uwb_history)))])

        if len(segment) < 6:
            self._result = ("standing_still", 1.0)
            return

        acc = segment[:, :3]
        gyr = segment[:, 3:]

        gyro_mean = gyr.mean(axis=0)
        accel_mean = acc.mean(axis=0)
        gyro_mag_mean = float(np.linalg.norm(gyr).mean())
        accel_var = float(np.var(acc).mean())

        # UWB distance change
        uwb_delta = 0.0
        if len(uwb_seg) > 2:
            uwb_start = np.median(uwb_seg[:min(5, len(uwb_seg))])
            uwb_end = np.median(uwb_seg[-min(5, len(uwb_seg)):])
            uwb_delta = uwb_end - uwb_start

        # Oscillation count
        gyr_centered = gyr - gyr.mean(axis=0)
        osc = int(np.sum(np.abs(np.diff(np.signbit(gyr_centered), axis=0))))

        duration_s = len(segment) / 50.0

        label, conf = classify(
            gyro_mean, accel_mean, uwb_delta, gyro_mag_mean,
            osc, accel_var, duration_s,
        )
        self._result = (label, conf)

    @property
    def gestures(self) -> list[str]:
        return GESTURES


# ── GUI ─────────────────────────────────────────────────────────────

class DetectWindow(QtWidgets.QWidget):
    def __init__(self, serial: str | None = None,
                 host: str = "127.0.0.1", port: int = 9999,
                 uwb_controller: str | None = None,
                 uwb_controlee: str | None = None):
        super().__init__()
        self._serial = serial
        self._host = host
        self._port = port
        self._uwb_controller = uwb_controller
        self._uwb_controlee = uwb_controlee

        # Detector
        self._detector = GestureDetector()
        self._current_label = "standing_still"
        self._current_conf = 1.0
        self._history: deque[tuple[str, float]] = deque(maxlen=100)

        # Sensors
        from null_gesture.sensors.imu_sensor import IMUClient
        self._imu = IMUClient()
        self._imu_ok = False

        self._uwb = None
        self._uwb_ok = False
        self._uwb_dist: float = 0.0

        self._init_ui()
        self._init_timer()
        self._connect_imu()
        self._connect_uwb()

    def _connect_imu(self) -> None:
        ok = False
        if self._serial:
            ok = self._imu.connect_serial(self._serial)
        if not ok and self._host:
            ok = self._imu.connect_tcp(self._host, self._port)
        self._imu_ok = ok

    def _connect_uwb(self) -> None:
        if not self._uwb_controller or not self._uwb_controlee:
            return
        try:
            from null_gesture.sensors.uwb_sensor import UWBRanger
            self._uwb = UWBRanger()
            def cb(s): 
                if s.get("status") == "Ok":
                    self._uwb_dist = float(s["distance_cm"])
            self._uwb_ok = self._uwb.start(self._uwb_controller, self._uwb_controlee, on_sample=cb)
        except Exception:
            pass

    def _init_ui(self) -> None:
        self.setWindowTitle("Null-Gesture — IMU+UWB Detection")
        self.resize(700, 450)
        self.setStyleSheet("background-color: #0d1117;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        hdr = QtWidgets.QLabel("Null-Gesture  |  IMU + UWB Capstone")
        hdr.setStyleSheet("color: #58a6ff; font-size: 16px; font-weight: bold; font-family: sans-serif;")
        layout.addWidget(hdr)

        # Gesture card
        card = QtWidgets.QWidget()
        card.setStyleSheet("background: #161b22; border-radius: 12px; padding: 28px;")
        cb = QtWidgets.QVBoxLayout(card)
        cb.setSpacing(8)

        self._label = QtWidgets.QLabel("Standing Still")
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("color: #58a6ff; font-size: 52px; font-weight: bold; font-family: sans-serif;")
        cb.addWidget(self._label)

        self._conf_label = QtWidgets.QLabel("")
        self._conf_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._conf_label.setStyleSheet("color: #8b949e; font-size: 20px; font-family: monospace;")
        cb.addWidget(self._conf_label)

        self._detail_label = QtWidgets.QLabel("Waiting for gesture...")
        self._detail_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._detail_label.setStyleSheet("color: #484f58; font-size: 13px; font-family: monospace;")
        cb.addWidget(self._detail_label)

        layout.addWidget(card)

        # History strip
        self._hist = QtWidgets.QLabel("")
        self._hist.setStyleSheet("color: #30363d; font-size: 10px; font-family: monospace;")
        self._hist.setWordWrap(True)
        layout.addWidget(self._hist)

        # Status bar
        bar = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(bar)
        bl.setContentsMargins(0, 0, 0, 0)
        self._status = QtWidgets.QLabel("Disconnected")
        self._status.setStyleSheet("color: #8b949e; font-size: 11px; font-family: monospace;")
        bl.addWidget(self._status)
        bl.addStretch()
        self._uwb_lbl = QtWidgets.QLabel("")
        self._uwb_lbl.setStyleSheet("color: #d2991d; font-size: 11px; font-family: monospace;")
        bl.addWidget(self._uwb_lbl)
        layout.addWidget(bar)

    def _init_timer(self) -> None:
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    def _tick(self) -> None:
        if not self._imu_ok:
            return
        self._imu.ingest(max_samples=10)
        window = self._imu.get_window()
        if window.sum() == 0:
            return

        gyro_mag = float(np.linalg.norm(window[-1, 3:]))
        label, conf = self._detector.update(window, self._uwb_dist)

        # Always update status
        state = self._detector._state
        onset = self._detector.onset_thresh
        self._detail_label.setText(
            f"gyro: {gyro_mag:.0f} dps (onset: {onset:.0f})  |  "
            f"state: {state}  |  uwb: {self._uwb_dist:.0f}cm"
        )

        if label != self._current_label or self._detector._state == "classified":
            self._current_label = label
            self._current_conf = conf
            self._history.append((label, conf))

            # Color
            colors = {
                "standing_still": "#58a6ff", "push": "#f85149", "pull": "#3fb950",
                "left": "#d2991d", "right": "#d2991d",
                "clockwise": "#a371f7", "anti_clockwise": "#a371f7",
                "bye_bye": "#f778ba", "clapping": "#56d364",
                "one_arm_boxing": "#e5534b", "t_arms": "#79c0ff",
                "raise_arms": "#ff7b72", "palm_up": "#a5d6ff", "palm_down": "#ffa198",
            }
            c = colors.get(label, "#e6edf3")
            name = label.replace("_", " ").title()

            self._label.setText(name)
            self._label.setStyleSheet(f"color: {c}; font-size: 52px; font-weight: bold; font-family: sans-serif;")
            self._conf_label.setText(f"confidence: {conf:.0%}")

            # Details
            state = self._detector._state
            uwb_d = self._uwb_dist
            self._detail_label.setText(
                f"state: {state}  |  uwb: {uwb_d:.0f}cm  |  samples: {self._detector._sample_count}"
            )

            # History strip
            if self._history:
                hist_str = "  ".join(
                    f"{g.replace('_',' ').title()[:6]}" for g, _ in list(self._history)[-20:]
                )
                self._hist.setText(hist_str)

        # Status
        if self._imu_ok and self._uwb_ok:
            self._status.setText("IMU+UWB ✓")
        elif self._imu_ok:
            self._status.setText("IMU only ⚠")
        else:
            self._status.setText("Disconnected")

        self._uwb_lbl.setText(f"uwb: {self._uwb_dist:.0f}cm" if self._uwb_ok else "")

    def closeEvent(self, event):
        if self._imu_ok:
            self._imu.disconnect()
        if self._uwb and self._uwb_ok:
            self._uwb.stop()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="IMU+UWB Gesture Detector")
    p.add_argument("--serial", help="IMU port")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9999)
    p.add_argument("--uwb-controller")
    p.add_argument("--uwb-controlee")
    args = p.parse_args(argv)
    app = QtWidgets.QApplication(sys.argv)
    win = DetectWindow(serial=args.serial, host=args.host, port=args.port,
                       uwb_controller=args.uwb_controller,
                       uwb_controlee=args.uwb_controlee)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
