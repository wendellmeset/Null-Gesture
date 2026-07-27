#!/usr/bin/env python3
"""Real-time 3D point cloud from webcam using Depth Anything V2.

Turns your webcam into a LiDAR-like 3D viewer.
Color mapping: close = red/orange, mid = yellow/green, far = blue/purple.

Usage:
    PYTHONPATH=. venv/bin/python scripts/real_viewer.py
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import torch


class DepthViewer(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()

        # Load depth model
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self._device}")
        print("Loading Depth Anything V2...")
        from transformers import pipeline
        self._pipe = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=self._device,
        )
        print("Model loaded.")

        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self._cap.set(cv2.CAP_PROP_FPS, 15)

        self._last_frame_time = time.time()
        self._point_cloud: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self._point_colors: np.ndarray = np.zeros((0, 4), dtype=np.float32)

        self._init_ui()
        self._init_timer()

    # ── UI ─────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        self.setWindowTitle("Real Viewer — Webcam 3D Point Cloud")
        self.resize(1000, 800)
        self.setStyleSheet("background-color: #0d1117;")

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── 3D View ────────────────────────────────────────────────
        view_container = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(view_container)
        vl.setContentsMargins(0, 0, 0, 0)

        self._view = gl.GLViewWidget()
        self._view.setBackgroundColor("#0d1117")
        self._view.setCameraPosition(distance=4, elevation=-20, azimuth=-60)

        # Floor
        g = gl.GLGridItem()
        g.setSize(4, 4); g.setSpacing(0.5, 0.5)
        g.setColor("#30363d44"); g.translate(0, -1.5, 0)
        self._view.addItem(g)

        # Point cloud
        self._scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), color=(1, 1, 1, 0.5), size=3, pxMode=True)
        self._view.addItem(self._scatter)

        vl.addWidget(self._view)

        # Overlay
        self._overlay = QtWidgets.QLabel(self._view)
        self._overlay.setStyleSheet(
            "color: #8b949e; font-size: 11px; font-family: monospace; background: transparent; padding: 8px;")
        self._overlay.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        layout.addWidget(view_container, stretch=3)

        # ── 2D camera preview ──────────────────────────────────────
        preview = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(preview)
        pl.setContentsMargins(8, 8, 8, 8)
        self._preview_label = QtWidgets.QLabel()
        self._preview_label.setFixedSize(280, 210)
        self._preview_label.setStyleSheet("border: 1px solid #30363d;")
        pl.addWidget(self._preview_label)
        pl.addStretch()

        # FPS label
        self._fps_label = QtWidgets.QLabel("FPS: —")
        self._fps_label.setStyleSheet("color: #8b949e; font-size: 11px; font-family: monospace;")
        pl.addWidget(self._fps_label)

        layout.addWidget(preview)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._overlay.setGeometry(0, 0, self._view.width(), 30)

    # ── Timer ──────────────────────────────────────────────────────

    def _init_timer(self) -> None:
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(100)  # ~10fps — depth model is heavy

    def _tick(self) -> None:
        ret, frame = self._cap.read()
        if not ret:
            return

        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL for the pipeline
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(rgb)
        result = self._pipe(pil_img)
        depth = np.array(result["depth"])  # (H, W) in meters

        H, W = depth.shape

        # ── Build 3D point cloud ───────────────────────────────────
        # Camera intrinsics (rough estimate for typical webcam)
        fx = fy = W  # rough focal length in pixels
        cx, cy = W / 2, H / 2

        # Create pixel grid
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)

        # Back-project to 3D
        z = depth
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy

        # Subsample randomly (avoids grid-line artifacts)
        step = 4
        n_total = (H // step) * (W // step)
        max_pts = 4000
        if n_total > max_pts:
            # Random sample
            idx = np.random.choice(H * W, max_pts, replace=False)
            u_flat = uu.flatten()[idx]
            v_flat = vv.flatten()[idx]
            z_flat = z.flatten()[idx]
        else:
            u_flat = uu[::step, ::step].flatten()
            v_flat = vv[::step, ::step].flatten()
            z_flat = z[::step, ::step].flatten()

        # Back-project
        x_s = (u_flat - cx) * z_flat / fx
        y_s = (v_flat - cy) * z_flat / fy

        # Filter depth
        valid = (z_flat > 0.3) & (z_flat < 6.0) & np.isfinite(z_flat)
        x_s, y_s, z_s = x_s[valid], y_s[valid], z_flat[valid]

        # Flip Y for display (camera Y=down → world Y=up)
        pts = np.column_stack([x_s, -y_s, z_s]).astype(np.float32)

        if len(pts) == 0:
            return

        # ── Color by depth ─────────────────────────────────────────
        # Normalize depth for color mapping
        z_norm = np.clip((z_s - z_s.min()) / (z_s.max() - z_s.min() + 1e-8), 0, 1)

        # Purple (close) → Red → Orange → Yellow → Green → Cyan → Blue (far)
        colors = np.zeros((len(pts), 4), dtype=np.float32)
        # Smooth rainbow: near=red, mid=yellow/green, far=blue
        t = np.clip(z_norm, 0, 1)
        colors[:, 0] = np.clip(2.0 - 2.0 * t, 0, 1)       # red: 1→0
        colors[:, 1] = 1.0 - np.abs(2.0 * t - 1.0)         # green: 0→1→0
        colors[:, 2] = np.clip(2.0 * t - 0.5, 0, 1)        # blue: 0→1
        colors[:, 3] = 0.6

        # Update scatter
        if len(pts) > 0:
            self._scatter.setData(pos=pts, color=colors)
            self._scatter.setVisible(True)

        # ── 2D preview ─────────────────────────────────────────────
        # Show depth as heatmap
        depth_viz = np.clip(depth / 5.0, 0, 1)
        depth_viz = (depth_viz * 255).astype(np.uint8)
        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_HOT)
        depth_viz = cv2.resize(depth_viz, (280, 210))
        h, w, ch = depth_viz.shape
        bytes_per_line = ch * w
        from PyQt6.QtGui import QImage, QPixmap
        qt_img = QImage(depth_viz.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        self._preview_label.setPixmap(QPixmap.fromImage(qt_img))

        # ── FPS ────────────────────────────────────────────────────
        fps = 1.0 / max(dt, 0.001)
        self._fps_label.setText(f"FPS: {fps:.1f}")
        self._overlay.setText(
            f"{len(pts)} points  |  depth: {depth.min():.2f}..{depth.max():.2f}m  |  {fps:.1f} fps"
        )

    def closeEvent(self, event):
        self._cap.release()
        super().closeEvent(event)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = DepthViewer()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
