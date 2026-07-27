#!/usr/bin/env python3
"""3D mmWave point cloud viewer — real-time object visualization.

Shows detected points as colored spheres, highlights the dominant (hand) point,
draws a motion trail, and displays range/velocity info.

Usage:
    PYTHONPATH=. venv/bin/python scripts/mmwave_3d.py --port /dev/ttyACM1
"""
from __future__ import annotations

import sys
from collections import deque

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6 import QtCore, QtWidgets

from null_gesture.sensors.mmwave import MMWaveSensor


class MMWave3DWindow(QtWidgets.QWidget):
    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 921600,
                 config_file: str | None = None) -> None:
        super().__init__()
        self._port = port
        self._baud = baud
        self._config_file = config_file

        self._radar = MMWaveSensor()
        self._radar_ok = False

        # Point cloud display items (reusable pool)
        self._point_spheres: list[gl.GLMeshItem] = []
        self._max_points = 30

        # Hand point (larger, highlighted)
        self._hand_sphere: gl.GLMeshItem | None = None
        self._hand_trail_pts: deque[np.ndarray] = deque(maxlen=200)

        self._init_ui()
        self._init_timer()
        self._connect()

    def _connect(self) -> None:
        self._radar_ok = self._radar.connect(self._port, self._baud, self._config_file)
        if self._radar_ok:
            self._status.setText(f"Connected — {self._port} @ {self._baud}")
        else:
            self._status.setText(f"Failed — {self._port}")

    # ── UI ─────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        self.setWindowTitle("mmWave 3D — Point Cloud Viewer")
        self.resize(900, 750)
        self.setStyleSheet("background-color: #0d1117;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 3D view
        self._view = gl.GLViewWidget()
        self._view.setBackgroundColor("#0d1117")
        self._view.setCameraPosition(distance=3.5, elevation=25, azimuth=-50)

        # Floor grid
        g = gl.GLGridItem()
        g.setSize(4, 4)
        g.setSpacing(0.5, 0.5)
        g.setColor("#30363d55")
        g.rotate(90, 1, 0, 0)
        self._view.addItem(g)

        # Axes (X=red right, Y=green up, Z=blue forward)
        for i, c in enumerate([(1, 0, 0, 0.7), (0, 1, 0, 0.7), (0, 0, 1, 0.7)]):
            pts = np.zeros((2, 3))
            pts[1, i] = 2.0
            self._view.addItem(gl.GLLinePlotItem(pos=pts, color=c, width=2, antialias=True))

        # Origin
        self._view.addItem(gl.GLMeshItem(
            meshdata=gl.MeshData.sphere(rows=8, cols=8, radius=0.03),
            color=(1, 1, 1, 0.5), shader="shaded", smooth=True))

        # Hand trail line
        self._trail_line = gl.GLLinePlotItem(
            pos=np.zeros((1, 3)), color=(1.0, 0.55, 0.1, 0.7), width=2, antialias=True)
        self._view.addItem(self._trail_line)

        # Point sphere pool
        for i in range(self._max_points):
            sph = gl.GLMeshItem(
                meshdata=gl.MeshData.sphere(rows=6, cols=6, radius=0.03),
                color=(0.35, 0.55, 0.85, 0.6), shader="shaded", smooth=True)
            sph.setVisible(False)
            self._point_spheres.append(sph)
            self._view.addItem(sph)

        # Hand sphere (larger, orange)
        self._hand_sphere = gl.GLMeshItem(
            meshdata=gl.MeshData.sphere(rows=10, cols=10, radius=0.05),
            color=(1.0, 0.55, 0.1, 0.95), shader="shaded", smooth=True)
        self._hand_sphere.setVisible(False)
        self._view.addItem(self._hand_sphere)

        layout.addWidget(self._view)

        # ── Info overlay ───────────────────────────────────────────
        self._overlay = QtWidgets.QLabel(self._view)
        self._overlay.setStyleSheet(
            "color: #8b949e; font-size: 11px; font-family: monospace; "
            "background: transparent; padding: 8px;"
        )
        self._overlay.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # ── Bottom bar ────────────────────────────────────────────
        bar = QtWidgets.QWidget()
        bar.setFixedHeight(34)
        bar.setStyleSheet("background-color: #161b22;")
        bl = QtWidgets.QHBoxLayout(bar)
        bl.setContentsMargins(12, 0, 12, 0)
        bl.setSpacing(14)

        self._status = QtWidgets.QLabel("Disconnected")
        self._status.setStyleSheet("color: #8b949e; font-size: 11px; font-family: monospace;")
        bl.addWidget(self._status)

        bl.addStretch()

        for label in ["points", "frame", "hand_pos"]:
            lbl = QtWidgets.QLabel(f"{label}: —")
            lbl.setStyleSheet("color: #484f58; font-size: 10px; font-family: monospace;")
            setattr(self, f"_lbl_{label}", lbl)
            bl.addWidget(lbl)

        layout.addWidget(bar)

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        if a0 is not None:
            self._overlay.setGeometry(0, 0, a0.size().width(), 30)

    # ── Timer ──────────────────────────────────────────────────────

    def _init_timer(self) -> None:
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)  # 20 fps

    def _tick(self) -> None:
        if not self._radar_ok:
            return

        self._radar.ingest()
        points = self._radar.get_point_cloud()
        hand = self._radar.get_dominant_point()

        n_pts = len(points)

        # ── Update point spheres ───────────────────────────────────
        for i in range(self._max_points):
            if i < n_pts:
                pt = points[i]
                T = np.eye(4, dtype=np.float32)
                T[:3, 3] = pt
                self._point_spheres[i].setTransform(T)
                self._point_spheres[i].setVisible(True)
            else:
                self._point_spheres[i].setVisible(False)

        # ── Update hand sphere ─────────────────────────────────────
        if hand is not None and self._hand_sphere is not None:
            T = np.eye(4, dtype=np.float32)
            T[:3, 3] = hand
            self._hand_sphere.setTransform(T)
            self._hand_sphere.setVisible(True)

            # Trail
            self._hand_trail_pts.append(hand.copy())
            trail = np.array(self._hand_trail_pts, dtype=np.float32)
            self._trail_line.setData(pos=trail)

        # ── Labels ─────────────────────────────────────────────────
        self._lbl_points.setText(f"points: {n_pts}")  # type: ignore[attr-defined]
        self._lbl_frame.setText(f"frame: {self._radar.frame_number}")  # type: ignore[attr-defined]

        if hand is not None:
            self._lbl_hand_pos.setText(f"hand: ({hand[0]:.2f}, {hand[1]:.2f}, {hand[2]:.2f})")  # type: ignore[attr-defined]
        else:
            self._lbl_hand_pos.setText("hand: —")  # type: ignore[attr-defined]

        # Overlay
        vel_str = ""
        if len(self._radar._velocities) > 0:
            v = self._radar._velocities
            vel_str = f"  |  vel: {v.min():+.1f}..{v.max():+.1f} m/s"
        self._overlay.setText(
            f"frame {self._radar.frame_number}  |  {n_pts} points{vel_str}"
        )

    def closeEvent(self, a0):
        if self._radar_ok:
            self._radar.disconnect()
        super().closeEvent(a0)


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="mmWave 3D point cloud viewer")
    p.add_argument("--port", default="/dev/ttyACM1")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--config", help="Path to .cfg file for radar configuration")
    p.add_argument("--dump", action="store_true", help="Dump raw bytes and exit")
    args = p.parse_args()

    if args.dump:
        from null_gesture.sensors.mmwave import MMWaveSensor
        r = MMWaveSensor()
        if not r.connect(args.port, args.baud):
            print(f"Failed to connect to {args.port}")
            return 1
        print(f"Connected @ {args.baud}. Reading raw data for 3s...")
        data = r.dump_raw(1000)
        print(f"Got {len(data)} bytes:")
        for i in range(0, min(len(data), 512), 16):
            hex_str = " ".join(f"{b:02x}" for b in data[i:i+16])
            ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in data[i:i+16])
            print(f"  {i:04x}: {hex_str:<48s} {ascii_str}")
        r.disconnect()
        return 0

    app = QtWidgets.QApplication(sys.argv)
    win = MMWave3DWindow(port=args.port, baud=args.baud, config_file=args.config)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
