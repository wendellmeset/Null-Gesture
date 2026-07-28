"""Live visualization (optional).

Provides real-time matplotlib plots of sensor data and gesture
detections for debugging and demos.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np


class LiveVisualizer:
    """Real-time matplotlib visualization of sensor streams and detections.

    Usage::

        viz = LiveVisualizer()
        viz.start()
        # ... in main loop ...
        viz.update(imu_sample, mmwave_result, gesture_label)
    """

    def __init__(
        self,
        history_sec: float = 5.0,
        imu_rate: float = 100.0,
        mmwave_rate: float = 20.0,
    ):
        """
        Args:
            history_sec: Seconds of history to display.
            imu_rate: IMU sample rate for buffer sizing.
            mmwave_rate: mmWave frame rate for buffer sizing.
        """
        self._history_sec = history_sec
        self._imu_buffer_size = int(history_sec * imu_rate)
        self._mmwave_buffer_size = int(history_sec * mmwave_rate)

        # Data buffers
        self._times: deque[float] = deque(maxlen=self._imu_buffer_size)
        self._accel_norms: deque[float] = deque(maxlen=self._imu_buffer_size)
        self._gyro_norms: deque[float] = deque(maxlen=self._imu_buffer_size)
        self._mmwave_times: deque[float] = deque(maxlen=self._mmwave_buffer_size)
        self._ranges: deque[float] = deque(maxlen=self._mmwave_buffer_size)
        self._velocities: deque[float] = deque(maxlen=self._mmwave_buffer_size)

        # Detected gestures
        self._detections: deque[tuple[float, str]] = deque(maxlen=50)

        # Matplotlib state
        self._fig = None
        self._axes = None
        self._running = False

    def start(self) -> None:
        """Initialize the matplotlib figure (non-blocking)."""
        try:
            import matplotlib
            matplotlib.use("TkAgg")  # Non-interactive backend
            import matplotlib.pyplot as plt

            self._fig, self._axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            self._fig.canvas.manager.set_window_title("Null-Gesture Live View")
            self._axes[0].set_ylabel("Linear Accel (g)")
            self._axes[0].set_title("IMU — Linear Acceleration Norm")
            self._axes[0].grid(True, alpha=0.3)

            self._axes[1].set_ylabel("Gyro (dps)")
            self._axes[1].set_title("IMU — Gyroscope Norm")
            self._axes[1].grid(True, alpha=0.3)

            self._axes[2].set_ylabel("Range (m) / Velocity (m/s)")
            self._axes[2].set_title("mmWave — Centroid Range & Velocity")
            self._axes[2].set_xlabel("Time (s)")
            self._axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.ion()
            plt.show()

            self._running = True
        except ImportError:
            print("[Visualizer] matplotlib not installed. Visualization disabled.")
            self._running = False

    def update(
        self,
        imu_sample: dict | None = None,
        mmwave_result: dict | None = None,
        gesture_label: str | None = None,
    ) -> None:
        """Add new data and refresh the plot.

        Args:
            imu_sample: Processed IMU sample dict (with 'linear_norm', 'gyro_norm', 't').
            mmwave_result: Processed mmWave frame dict (with 'centroid', 'velocity_centroid').
            gesture_label: Detected gesture name to annotate (or None).
        """
        if not self._running:
            return

        now = time.time()

        if imu_sample is not None:
            self._times.append(now)
            self._accel_norms.append(imu_sample.get("linear_norm", 0.0))
            self._gyro_norms.append(imu_sample.get("gyro_norm", 0.0))

        if mmwave_result is not None:
            centroid = mmwave_result.get("centroid")
            if centroid is not None:
                self._mmwave_times.append(now)
                r = float(np.linalg.norm(centroid))
                self._ranges.append(r)
                self._velocities.append(
                    mmwave_result.get("velocity_centroid", 0.0)
                )

        if gesture_label is not None:
            self._detections.append((now, gesture_label))

        # Redraw at most every 100ms
        if not hasattr(self, "_last_draw"):
            self._last_draw = 0.0
        if now - self._last_draw < 0.1:
            return
        self._last_draw = now

        self._redraw()

    def _redraw(self) -> None:
        """Refresh the matplotlib figure."""
        if self._fig is None or self._axes is None:
            return

        try:
            # Clear axes
            for ax in self._axes:
                ax.clear()
                ax.grid(True, alpha=0.3)

            rel_times = np.array(self._times) - self._times[0] if self._times else np.array([])

            # IMU plot
            if len(rel_times) > 0:
                self._axes[0].plot(rel_times, list(self._accel_norms), 'b-', linewidth=0.5)
                self._axes[0].set_ylabel("Linear Accel (g)")
                self._axes[0].set_title("IMU — Linear Acceleration Norm")

                self._axes[1].plot(rel_times, list(self._gyro_norms), 'r-', linewidth=0.5)
                self._axes[1].set_ylabel("Gyro (dps)")
                self._axes[1].set_title("IMU — Gyroscope Norm")

            # mmWave plot
            if self._mmwave_times:
                mm_rel = np.array(list(self._mmwave_times)) - self._mmwave_times[0] if self._mmwave_times else np.array([])
                if len(mm_rel) > 0:
                    self._axes[2].plot(mm_rel, list(self._ranges), 'g-', linewidth=0.8, label="Range (m)")
                    self._axes[2].plot(mm_rel, list(self._velocities), 'm-', linewidth=0.8, label="Velocity (m/s)")
                    self._axes[2].legend(loc="upper right", fontsize=7)
                    self._axes[2].set_xlabel("Time (s)")
                    self._axes[2].set_ylabel("Range (m) / Vel (m/s)")
                    self._axes[2].set_title("mmWave — Centroid Range & Velocity")

            # Annotate recent detections
            for det_time, gesture in list(self._detections)[-5:]:
                if self._times:
                    base = self._times[0]
                    rel = det_time - base
                    for ax in self._axes:
                        ax.axvline(x=rel, color="orange", linestyle="--", alpha=0.5)
                        ax.annotate(
                            gesture, (rel, ax.get_ylim()[1] * 0.9),
                            fontsize=8, color="orange",
                            ha="center", va="top",
                        )

            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
        except Exception:
            pass  # Silently ignore drawing errors

    def stop(self) -> None:
        """Close the visualization window."""
        self._running = False
        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._fig)
            except Exception:
                pass
            self._fig = None
            self._axes = None
