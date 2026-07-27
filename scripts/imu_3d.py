#!/usr/bin/env python3
"""3D IMU + UWB visualiser — absolute position via UWB anchor + IMU orientation.

UWB gives distance from anchor (±10cm). IMU gives hand orientation.
Fused: position = anchor_pos + distance × forward_direction.

Usage:
    python scripts/imu_3d.py --serial /dev/ttyACM0 --uwb-controller /dev/ttyACM1 --uwb-controlee /dev/ttyACM2
    python -m null_gesture imu3d --serial /dev/ttyACM0 --uwb-controller /dev/ttyACM1 --uwb-controlee /dev/ttyACM2

Press SPACE to calibrate: point IMU directly at the UWB anchor, press Space.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from null_gesture.sensors.tracking import (
    MadgwickAHRS, quat_to_matrix, quat_mult, quat_rotate,
)
from null_gesture.sensors.fusion import UWBPositionTracker


class IMU3DWindow(QtWidgets.QWidget):
    def __init__(
        self,
        serial: str | None = None,
        host: str = "127.0.0.1", port: int = 9999,
        uwb_controller: str | None = None,
        uwb_controlee: str | None = None,
    ) -> None:
        super().__init__()
        self._serial = serial
        self._host = host
        self._port = port
        self._uwb_controller = uwb_controller
        self._uwb_controlee = uwb_controlee

        # Madgwick orientation filter
        self._madgwick = MadgwickAHRS(beta=0.06, sample_freq=60.0)

        # UWB + IMU fusion tracker
        self._fusion = UWBPositionTracker()

        # UWB ranger
        self._uwb = None
        self._uwb_connected = False
        self._uwb_distance: float = 0.0  # latest distance in cm
        self._uwb_time: float = 0.0

        # Trail
        self._trail_pts = np.zeros((500, 3), dtype=np.float32)
        self._trail_idx = 0

        # IMU
        from null_gesture.sensors.imu_sensor import IMUClient
        self._imu = IMUClient()
        self._imu_connected = False

        self._init_ui()
        self._init_timer()
        self._connect_imu()
        self._connect_uwb()

    # ── Connections ────────────────────────────────────────────────

    def _connect_imu(self) -> None:
        ok = False
        if self._serial:
            ok = self._imu.connect_serial(self._serial)
        if not ok and self._host:
            ok = self._imu.connect_tcp(self._host, self._port)
        self._imu_connected = ok

    def _connect_uwb(self) -> None:
        if not self._uwb_controller or not self._uwb_controlee:
            return
        try:
            from null_gesture.sensors.uwb_sensor import UWBRanger
            self._uwb = UWBRanger()
            def on_sample(sample: dict) -> None:
                if sample.get("status") == "Ok":
                    self._uwb_distance = float(sample["distance_cm"])
                    self._uwb_time = time.time()
            ok = self._uwb.start(
                self._uwb_controller, self._uwb_controlee,
                on_sample=on_sample,
            )
            self._uwb_connected = ok
            if ok:
                print(f"[imu3d] UWB ranging active: {self._uwb_controller} → {self._uwb_controlee}")
            else:
                print("[imu3d] UWB failed to start")
        except ImportError:
            print("[imu3d] UWB tools not found — distance tracking disabled")
        except Exception as e:
            print(f"[imu3d] UWB error: {e}")

    # ── Calibration ────────────────────────────────────────────────

    def calibrate(self) -> None:
        """Calibrate: place hand at anchor position, press Space."""
        self._fusion.calibrate(self._uwb_distance / 100.0 if self._uwb_distance > 0 else 0.0)
        self._madgwick.reset()
        self._trail_pts.fill(0)
        self._trail_idx = 0
        print("[imu3d] Calibrated — place hand at anchor, press Space")

    # ── UI ─────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        self.setWindowTitle("IMU 3D — UWB Anchor + IMU Tracking")
        self.resize(900, 780)
        self.setStyleSheet("background-color: #0d1117;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._view = gl.GLViewWidget()
        self._view.setBackgroundColor("#0d1117")
        self._view.setCameraPosition(distance=5, elevation=30, azimuth=-45)

        # Floor grid
        g = gl.GLGridItem()
        g.setSize(6, 6); g.setSpacing(0.5, 0.5)
        g.setColor("#30363d55"); g.rotate(90, 1, 0, 0)
        self._view.addItem(g)

        # Axes
        for i, c in enumerate([(1,0,0,.7),(0,1,0,.7),(0,0,1,.7)]):
            pts = np.zeros((2,3)); pts[1,i] = 2.5
            self._view.addItem(gl.GLLinePlotItem(pos=pts, color=c, width=2, antialias=True))

        # ── Anchor sphere (orange, at origin) ──────────────────────
        self._anchor = gl.GLMeshItem(
            meshdata=gl.MeshData.sphere(rows=12, cols=12, radius=0.08),
            color=(1.0, 0.55, 0.1, 0.9), shader="shaded", smooth=True)
        self._view.addItem(self._anchor)

        # Anchor → tag line
        self._anchor_line = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), color=(1.0, 0.55, 0.1, 0.5), width=1.5, antialias=True)
        self._view.addItem(self._anchor_line)

        # ── Hand cube ──────────────────────────────────────────────
        self._cube = self._make_box(0.35, (0.30, 0.50, 0.80, 0.9))
        self._view.addItem(self._cube)
        self._dot = gl.GLMeshItem(
            meshdata=gl.MeshData.sphere(rows=6, cols=6, radius=0.05),
            color=(1.0, 0.35, 0.35, 0.9), shader="shaded", smooth=True)
        self._view.addItem(self._dot)

        # ── Position trail ─────────────────────────────────────────
        self._trail_line = gl.GLLinePlotItem(
            pos=np.zeros((1, 3)), color=(0.35, 0.55, 0.85, 0.5), width=2, antialias=True)
        self._view.addItem(self._trail_line)

        # ── UWB distance sphere (wireframe, shows possible positions) ──
        self._dist_sphere = gl.GLMeshItem(
            meshdata=gl.MeshData.sphere(rows=16, cols=16, radius=1.0),
            color=(1.0, 0.55, 0.1, 0.12), shader="shaded",
            smooth=True, drawEdges=False)
        self._dist_sphere.setGLOptions("translucent")
        self._dist_sphere.setVisible(False)
        self._view.addItem(self._dist_sphere)

        layout.addWidget(self._view)

        # ── Bottom bar ────────────────────────────────────────────
        bar = QtWidgets.QWidget()
        bar.setFixedHeight(38)
        bar.setStyleSheet("background-color: #161b22;")
        bl = QtWidgets.QHBoxLayout(bar)
        bl.setContentsMargins(12, 0, 12, 0); bl.setSpacing(10)

        self._status = QtWidgets.QLabel("Disconnected")
        self._status.setStyleSheet("color: #8b949e; font-size: 11px; font-family: monospace;")
        bl.addWidget(self._status)

        btn = QtWidgets.QPushButton("🎯 Calibrate (Space)")
        btn.setStyleSheet(
            "QPushButton{background:#30363d;color:#e6edf3;border:1px solid #484f58;"
            "border-radius:5px;padding:4px 12px;font-size:11px;}"
            "QPushButton:hover{border-color:#58a6ff;}")
        btn.clicked.connect(self.calibrate)
        btn.setShortcut("Space")
        bl.addWidget(btn)

        bl.addStretch()

        self._uwb_lbl = QtWidgets.QLabel("uwb: —")
        self._uwb_lbl.setStyleSheet("color: #d2991d; font-size: 11px; font-family: monospace;")
        bl.addWidget(self._uwb_lbl)

        for ch, c in [("gx","#f85149"),("gy","#3fb950"),("gz","#58a6ff")]:
            lbl = QtWidgets.QLabel(f"{ch}:0")
            lbl.setStyleSheet(f"color:{c};font-size:10px;font-family:monospace;")
            setattr(self, f"_lbl_{ch}", lbl); bl.addWidget(lbl)

        layout.addWidget(bar)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Space:
            self.calibrate()

    @staticmethod
    def _make_box(size: float, color: tuple) -> gl.GLMeshItem:
        s = size/2
        verts = np.array([
            [-s,-s,-s],[s,-s,-s],[s,s,-s],[-s,s,-s],
            [-s,-s,s],[s,-s,s],[s,s,s],[-s,s,s]], dtype=np.float32)
        faces = np.array([
            [0,1,2],[0,2,3],[4,5,6],[4,6,7],
            [0,1,5],[0,5,4],[2,3,7],[2,7,6],
            [0,3,7],[0,7,4],[1,2,6],[1,6,5]], dtype=np.int32)
        return gl.GLMeshItem(vertexes=verts, faces=faces, color=color, shader="shaded",
                             smooth=False, drawEdges=True, edgeColor=(1,1,1,0.12))

    # ── Timer ──────────────────────────────────────────────────────

    def _init_timer(self) -> None:
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    def _tick(self) -> None:
        if not self._imu_connected:
            return

        self._imu.ingest(max_samples=5)
        window = self._imu.get_window()
        if window.sum() == 0:
            return

        gyr_dps = window[-1, 3:].astype(np.float64)
        acc_g = window[-1, :3].astype(np.float64)
        gyro_mag = float(np.linalg.norm(gyr_dps))
        dt = 0.016

        # ── Madgwick orientation ───────────────────────────────────
        gyr_rad = np.deg2rad(gyr_dps)
        q = self._madgwick.update(gyr_rad, acc_g, dt)

        # ── UWB + IMU fusion → position ────────────────────────────
        # Get world-frame accel (gravity subtracted)
        acc_world = quat_rotate(q, acc_g)
        acc_world -= np.array([0.0, 0.0, 1.0])  # remove gravity
        acc_world *= 9.81  # g → m/s²

        uwb_cm = self._uwb_distance if self._uwb_connected else 0.0
        pos = self._fusion.update(acc_world, gyro_mag, uwb_cm, dt)

        # ── Distance sphere (show possible positions at current distance) ──
        if self._fusion.distance > 0.01:
            d = self._fusion.distance
            # Update sphere mesh radius
            sph = gl.MeshData.sphere(rows=16, cols=16, radius=d)
            self._dist_sphere.setMeshData(meshdata=sph)
            self._dist_sphere.setVisible(True)
        else:
            self._dist_sphere.setVisible(False)

        # ── Anchor → tag line ──────────────────────────────────────
        anchor = np.zeros(3, dtype=np.float32)
        line_pts = np.array([anchor, pos.astype(np.float32)])
        self._anchor_line.setData(pos=line_pts)

        # ── Trail ──────────────────────────────────────────────────
        self._trail_pts[self._trail_idx] = pos.astype(np.float32)
        self._trail_idx = (self._trail_idx + 1) % len(self._trail_pts)
        rolled = np.roll(self._trail_pts, -self._trail_idx, axis=0)
        self._trail_line.setData(pos=rolled)

        # ── Cube transform ─────────────────────────────────────────
        R = quat_to_matrix(q)
        R[:3, 3] = pos.astype(np.float32)
        self._cube.setTransform(R)
        # Front dot
        front = np.array([0.0, 0.0, 0.35/2])
        dot_world = quat_rotate(q, front) + pos
        R_dot = np.eye(4, dtype=np.float32)
        R_dot[:3, 3] = dot_world
        self._dot.setTransform(R_dot)

        # ── Labels ─────────────────────────────────────────────────
        self._lbl_gx.setText(f"gx:{gyr_dps[0]:+.0f}")
        self._lbl_gy.setText(f"gy:{gyr_dps[1]:+.0f}")
        self._lbl_gz.setText(f"gz:{gyr_dps[2]:+.0f}")

        if self._uwb_connected:
            self._uwb_lbl.setText(f"uwb: {self._uwb_distance:.0f}cm")
            self._status.setText("IMU+UWB")
        else:
            self._uwb_lbl.setText("uwb: —")
            self._status.setText("IMU only")

    def closeEvent(self, event: QtCore.QEvent) -> None:
        if self._imu_connected:
            self._imu.disconnect()
        if self._uwb and self._uwb_connected:
            self._uwb.stop()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="3D IMU + UWB anchor tracking")
    parser.add_argument("--serial", help="IMU serial port (e.g. /dev/ttyACM0)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--uwb-controller", help="UWB controller serial port")
    parser.add_argument("--uwb-controlee", help="UWB controlee serial port")
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication(sys.argv)
    win = IMU3DWindow(
        serial=args.serial, host=args.host, port=args.port,
        uwb_controller=args.uwb_controller,
        uwb_controlee=args.uwb_controlee,
    )
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
