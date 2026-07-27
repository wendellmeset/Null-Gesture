#!/usr/bin/env python3
"""2D IMU + UWB drawing canvas — tilt to draw.

Tilt your hand (pitch/roll) to move the cursor on a 2D plane.
UWB distance scales the reach. Stillness = new stroke.

Usage:
    source venv/bin/activate
    python -m null_gesture uwbdraw --serial /dev/ttyACM0 --uwb-controller /dev/ttyACM1 --uwb-controlee /dev/ttyACM2
"""

from __future__ import annotations

import sys

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from null_gesture.sensors.tracking import MadgwickAHRS


# ── Quaternion helpers ──────────────────────────────────────────────

def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mult(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


class DrawCanvas(QtWidgets.QWidget):
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

        # IMU orientation
        self._madgwick = MadgwickAHRS(beta=0.06, sample_freq=60.0)

        # Calibration — neutral orientation = origin
        self._q_neutral = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Tilt-to-position params
        self._tilt_scale = 2.5       # meters per radian of tilt
        self._smooth_alpha = 0.30    # EMA smoothing

        # State
        self._pos = np.zeros(2, dtype=np.float64)  # current XZ position
        self._was_still = True
        self._drawing = False

        # UWB
        self._uwb = None
        self._uwb_connected = False
        self._uwb_distance: float = 0.0

        # Drawing
        self._strokes: list[list[QtCore.QPointF]] = []
        self._current_stroke: list[QtCore.QPointF] = []
        self._cursor_pos = QtCore.QPointF(0, 0)

        # View
        self._scale = 250.0  # pixels per meter
        self._offset = QtCore.QPointF(450, 350)

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
            def on_sample(s: dict) -> None:
                if s.get("status") == "Ok":
                    self._uwb_distance = float(s["distance_cm"])
            self._uwb_connected = self._uwb.start(
                self._uwb_controller, self._uwb_controlee, on_sample=on_sample,
            )
        except Exception as e:
            print(f"UWB error: {e}")

    def calibrate(self) -> None:
        self._q_neutral = self._madgwick.q.copy()
        self._pos.fill(0)
        self._strokes.clear()
        self._current_stroke.clear()
        self.update()

    # ── UI ─────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        self.setWindowTitle("UWB Draw — Tilt to Draw")
        self.resize(900, 700)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #0d1117;")
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self._overlay = QtWidgets.QLabel(self)
        self._overlay.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignRight)
        self._overlay.setStyleSheet(
            "color: #8b949e; font-size: 11px; font-family: monospace; "
            "background: transparent; padding: 8px;"
        )
        self._overlay.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._offset = QtCore.QPointF(event.size().width() / 2, event.size().height() / 2)
        self._overlay.setGeometry(0, 0, event.size().width(), 30)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Space:
            self.calibrate()
        elif event.key() == QtCore.Qt.Key.Key_C:
            self._strokes.clear()
            self._current_stroke.clear()
            self.update()
        elif event.key() == QtCore.Qt.Key.Key_Z and self._strokes:
            self._strokes.pop()
            self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self._offset = event.position()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.buttons() & QtCore.Qt.MouseButton.RightButton:
            self._offset = event.position()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = 1.1 if event.angleDelta().y() > 0 else 0.9
        self._scale *= delta
        self._scale = max(20, min(2000, self._scale))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor("#0d1117"))

        # Grid
        p.setPen(QtGui.QPen(QtGui.QColor("#30363d"), 0.5))
        gs = 50
        for x in range(0, self.width(), gs):
            p.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), gs):
            p.drawLine(0, y, self.width(), y)

        # Crosshair
        ox, oy = int(self._offset.x()), int(self._offset.y())
        p.setPen(QtGui.QPen(QtGui.QColor("#484f58"), 1))
        p.drawLine(ox - 15, oy, ox + 15, oy)
        p.drawLine(ox, oy - 15, ox, oy + 15)

        # Completed strokes
        for i, stroke in enumerate(self._strokes):
            if len(stroke) < 2:
                continue
            age = len(self._strokes) - i - 1
            alpha = max(40, 200 - age * 40)
            color = QtGui.QColor(200, 200, 220, alpha)
            pen = QtGui.QPen(color, 2.5, QtCore.Qt.PenStyle.SolidLine,
                             QtCore.Qt.PenCapStyle.RoundCap,
                             QtCore.Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            self._draw_polyline(p, stroke)

        # Current stroke
        if len(self._current_stroke) > 1:
            pen = QtGui.QPen(QtGui.QColor("#58a6ff"), 3,
                             QtCore.Qt.PenStyle.SolidLine,
                             QtCore.Qt.PenCapStyle.RoundCap,
                             QtCore.Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            self._draw_polyline(p, self._current_stroke)

        # Cursor
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor("#58a6ff") if self._drawing else QtGui.QColor("#8b949e"))
        cp = self._world_to_screen(self._cursor_pos)
        p.drawEllipse(cp, 5, 5)

        p.end()

    def _draw_polyline(self, p: QtGui.QPainter, pts: list[QtCore.QPointF]) -> None:
        path = QtGui.QPainterPath()
        path.moveTo(pts[0])
        for pt in pts[1:]:
            path.lineTo(pt)
        p.drawPath(path)

    def _world_to_screen(self, pos: QtCore.QPointF) -> QtCore.QPointF:
        return QtCore.QPointF(
            pos.x() * self._scale + self._offset.x(),
            -pos.y() * self._scale + self._offset.y(),
        )

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

        # Madgwick orientation
        gyr_rad = np.deg2rad(gyr_dps)
        q = self._madgwick.update(gyr_rad, acc_g, dt)

        # ── Tilt → 2D position ─────────────────────────────────────
        # Relative orientation from neutral
        q_rel = quat_mult(quat_conj(self._q_neutral), q)

        # Gravity direction in body frame after relative rotation
        # This tells us how the device is tilted relative to neutral
        grav = np.array([
            2*(q_rel[1]*q_rel[3] - q_rel[0]*q_rel[2]),
            2*(q_rel[0]*q_rel[1] + q_rel[2]*q_rel[3]),
            q_rel[0]**2 - q_rel[1]**2 - q_rel[2]**2 + q_rel[3]**2,
        ])

        # pitch: forward/back tilt → Z axis on canvas
        # roll:  left/right tilt → X axis on canvas
        pitch = np.arctan2(grav[0], grav[2])
        roll  = np.arctan2(grav[1], grav[2])

        # UWB distance modulates reach (further = more sensitive)
        uwb_m = self._uwb_distance / 100.0 if (self._uwb_connected and self._uwb_distance > 0) else 1.5

        target_x = roll * self._tilt_scale * uwb_m
        target_z = pitch * self._tilt_scale * uwb_m

        # EMA smooth
        self._pos[0] += self._smooth_alpha * (target_x - self._pos[0])
        self._pos[1] += self._smooth_alpha * (target_z - self._pos[1])

        screen_pos = QtCore.QPointF(self._pos[0], self._pos[1])
        self._cursor_pos = screen_pos

        # ── Drawing state ──────────────────────────────────────────
        is_still = gyro_mag < 10.0

        if not is_still and self._was_still:
            self._current_stroke = [self._world_to_screen(screen_pos)]
            self._drawing = True
        elif is_still and not self._was_still:
            if len(self._current_stroke) > 3:
                self._strokes.append(self._current_stroke)
            self._current_stroke = []
            self._drawing = False
        elif self._drawing:
            pt = self._world_to_screen(screen_pos)
            if (not self._current_stroke or
                (pt - self._current_stroke[-1]).manhattanLength() > 1.5):
                self._current_stroke.append(pt)

        self._was_still = is_still

        # Overlay
        parts = []
        if self._uwb_connected:
            parts.append(f"uwb: {self._uwb_distance:.0f}cm")
        parts.append(f"tilt: pitch={pitch:+.2f} roll={roll:+.2f}")
        parts.append(f"pos: ({self._pos[0]:.2f}, {self._pos[1]:.2f})m")
        parts.append("🔴 drawing" if self._drawing else "⏸ still")
        parts.append(f"{len(self._strokes)} strokes")
        parts.append("Space=cal C=clear Z=undo")
        self._overlay.setText("  |  ".join(parts))

        self.update()


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="2D IMU+UWB tilt drawing")
    parser.add_argument("--serial", help="IMU serial port")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--uwb-controller", help="UWB controller port")
    parser.add_argument("--uwb-controlee", help="UWB controlee port")
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication(sys.argv)
    win = DrawCanvas(
        serial=args.serial, host=args.host, port=args.port,
        uwb_controller=args.uwb_controller,
        uwb_controlee=args.uwb_controlee,
    )
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
