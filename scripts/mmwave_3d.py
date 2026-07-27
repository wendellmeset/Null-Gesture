#!/usr/bin/env python3
"""Simple mmWave 3D viewer — raw point cloud, no fancy models.

Usage:
    PYTHONPATH=. venv/bin/python scripts/mmwave_3d.py --port /dev/ttyUSB0
"""

from __future__ import annotations

import sys
import time
from collections import deque

import numpy as np
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from null_gesture.sensors.mmwave import MMWaveSensor


class MMWave3DWindow(QtWidgets.QWidget):
    def __init__(self, port: str = "/dev/ttyUSB0", config_file: str | None = None) -> None:
        super().__init__()
        self._port = port
        self._config_file = config_file
        self._radar = MMWaveSensor()
        self._radar_ok = False
        self._last_tick = time.time()
        self._trail: deque[np.ndarray] = deque(maxlen=100)

        self._init_ui()
        QtCore.QTimer.singleShot(100, self._connect)
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(80)

    def _connect(self) -> None:
        try:
            self._radar_ok = self._radar.connect(self._port, 115200, self._config_file)
        except Exception as e:
            print(f"Config failed: {e}")
            self._radar.disconnect()
            self._radar_ok = self._radar.connect(self._port, 115200)
        self._status.setText("Connected" if self._radar_ok else "Failed")

    def _init_ui(self) -> None:
        self.setWindowTitle("mmWave 3D Viewer")
        self.resize(900, 700)
        self.setStyleSheet("background-color: #0d1117;")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._view = gl.GLViewWidget()
        self._view.setBackgroundColor("#0d1117")
        self._view.setCameraPosition(distance=1.5, elevation=15, azimuth=-70)

        # Grid
        g = gl.GLGridItem(); g.setSize(2, 2); g.setSpacing(0.2, 0.2); g.setColor("#30363d44")
        self._view.addItem(g)

        # Axes
        for i, c in enumerate([(1,0,0,.5),(0,1,0,.5),(0,0,1,.5)]):
            pts = np.zeros((2,3)); pts[1,i] = 1.2
            self._view.addItem(gl.GLLinePlotItem(pos=pts, color=c, width=1.5, antialias=True))

        # Point cloud scatter
        self._scatter = gl.GLScatterPlotItem(pos=np.zeros((1,3)), color=(0.3,0.6,0.9,0.7), size=5, pxMode=True)
        self._view.addItem(self._scatter)

        # Trail
        self._trail_line = gl.GLLinePlotItem(pos=np.zeros((1,3)), color=(1,0.5,0.1,0.6), width=2, antialias=True)
        self._view.addItem(self._trail_line)

        # Hand dot
        self._hand_dot = gl.GLScatterPlotItem(pos=np.zeros((1,3)), color=(1,0.55,0.1,1), size=10, pxMode=True)
        self._view.addItem(self._hand_dot)

        layout.addWidget(self._view)

        # Bottom bar
        bar = QtWidgets.QWidget(); bar.setFixedHeight(30)
        bar.setStyleSheet("background-color: #161b22;")
        bl = QtWidgets.QHBoxLayout(bar); bl.setContentsMargins(8,0,8,0)
        self._status = QtWidgets.QLabel("Disconnected")
        self._status.setStyleSheet("color: #8b949e; font-size: 10px; font-family: monospace;")
        bl.addWidget(self._status)
        bl.addStretch()
        self._info = QtWidgets.QLabel("")
        self._info.setStyleSheet("color: #484f58; font-size: 10px; font-family: monospace;")
        bl.addWidget(self._info)
        layout.addWidget(bar)

    def _tick(self) -> None:
        if not self._radar_ok:
            return
        for _ in range(5):
            if not self._radar.read_frame(timeout_s=0.05):
                break

        pts = self._radar.points
        hand = self._radar.get_dominant_point() if len(pts) > 0 else None

        # Update scatter
        if len(pts) > 0:
            self._scatter.setData(pos=pts)
            self._scatter.setVisible(True)
        else:
            self._scatter.setVisible(False)

        # Hand dot + trail
        if hand is not None:
            self._hand_dot.setData(pos=hand.reshape(1, 3))
            self._hand_dot.setVisible(True)
            self._trail.append(hand.copy())
            self._trail_line.setData(pos=np.array(self._trail, dtype=np.float32))
        else:
            self._hand_dot.setVisible(False)

        now = time.time()
        fps = 1 / max(now - self._last_tick, 0.001)
        self._last_tick = now
        hstr = f"hand: ({hand[0]:.2f},{hand[1]:.2f})" if hand is not None else "hand: —"
        self._info.setText(f"{len(pts)} pts | {hstr} | {fps:.0f}fps")

    def closeEvent(self, event):
        if self._radar_ok:
            self._radar.disconnect()
        super().closeEvent(event)


def main() -> int:
    import argparse
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--config")
    args = p.parse_args()
    cfg = args.config or str(Path(__file__).resolve().parent.parent / "config" / "mmwave_point_cloud.cfg")
    app = QtWidgets.QApplication(sys.argv)
    win = MMWave3DWindow(port=args.port, config_file=cfg)
    win.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
