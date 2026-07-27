"""UWB sensor: DWM3001CDK FiRa TWR ranging via Qorvo UCI library.

Refactored for Python 3.11 from https://github.com/wshanmu/UWB_lab.

Requires the uwb-qorvo-tools/ subdirectory with the UCI Python library
and two DWM3001CDK boards connected over USB.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

from null_gesture.config import UWBConfig, uwb_config as default_uwb_config

logger = logging.getLogger("null_gesture.sensors.uwb")

# ── Resolve the UWB tools path ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_UWB_TOOLS = _PROJECT_ROOT / "uwb-qorvo-tools"

# ── FiRa TWR script ─────────────────────────────────────────────────────────
_TWR_SCRIPT = (
    _UWB_TOOLS / "scripts" / "fira" / "run_fira_twr" / "run_fira_twr.py"
)


def _has_uwb_tools() -> bool:
    return _UWB_TOOLS.exists() and _TWR_SCRIPT.exists()


# ── Output parser ───────────────────────────────────────────────────────────
# The TWR script with --stats prints per-measurement output like:
#   <measurement 1>:
#           mac:            0x....
#           status:         Ok
#           distance:       123.45 cm

_MEASUREMENT_HEADER_RE = re.compile(r"<\s*measurement\s+(\d+)\s*>:")
_DISTANCE_RE = re.compile(r"distance:\s+([\d.]+)\s+cm")
_STATUS_RE = re.compile(r"status:\s+(\w+)")
_SEQUENCE_RE = re.compile(r"sequence\s+n?:\s+(\d+)")


def _compute_ranging_span_ms(fps: float) -> int:
    """Convert desired FPS to ranging interval in milliseconds."""
    return max(1, int(round(1000.0 / fps)))


def _build_env() -> dict[str, str]:
    """Build environment dict with UWB tools on PYTHONPATH."""
    env = os.environ.copy()
    paths = [
        str(_UWB_TOOLS / "lib" / "uwb-uci"),
        str(_UWB_TOOLS / "lib" / "uqt-utils"),
        str(_UWB_TOOLS),
    ]
    existing = env.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = ":".join(paths)
    env["UWB_TOOLS"] = str(_UWB_TOOLS)
    return env


class UWBRanger:
    """Manages a UWB FiRa TWR ranging session between two DWM3001CDK boards."""

    def __init__(self, config: UWBConfig | None = None) -> None:
        self.config = config or default_uwb_config
        self._buffer: deque[dict] = deque()
        self._controller_proc: subprocess.Popen | None = None
        self._controlee_proc: subprocess.Popen | None = None
        self._running = False
        self._sample_count = 0
        self._on_sample: Callable[[dict], None] | None = None
        self._current_seq: int = -1
        self._env = _build_env()

    @property
    def connected(self) -> bool:
        return self._running

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def start(
        self,
        controller_port: str,
        controlee_port: str,
        duration: int = 0,
        *,
        on_sample: Callable[[dict], None] | None = None,
    ) -> bool:
        """Start a FiRa TWR session.

        Args:
            controller_port: Serial port of the controller board.
            controlee_port: Serial port of the controlee board.
            duration: Session duration in seconds. 0 = run until stop().
            on_sample: Optional callback for each range sample.
        """
        if not _has_uwb_tools():
            logger.error(
                "UWB tools not found at %s. Clone the UWB_lab repo's "
                "uwb-qorvo-tools/ directory.",
                _UWB_TOOLS,
            )
            return False

        self._on_sample = on_sample
        self._running = True

        if duration <= 0:
            duration = 3600  # Long default for indefinite streaming

        controlee_duration = duration + 10
        cfg = self.config
        python = sys.executable
        ranging_span = _compute_ranging_span_ms(cfg.sample_rate_hz)

        # Common args for both controller and controlee
        common = [
            python, "-u", str(_TWR_SCRIPT),
            "--channel", str(cfg.uwb_channel),
            "--preamble-idx", str(cfg.preamble_code),
            "--aoa-report", "all-disabled",
            "--slot-span", str(cfg.slot_span),
            "--slots-per-rr", str(cfg.slots_per_rr),
            "--ranging-span", str(ranging_span),
            "--stats",
        ]

        controlee_cmd = common + ["-p", controlee_port, "-t", str(controlee_duration), "--controlee"]
        controller_cmd = common + ["-p", controller_port, "-t", str(duration)]

        # Start controlee first
        try:
            self._controlee_proc = subprocess.Popen(
                controlee_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=self._env,
                start_new_session=True,
            )
            logger.info("UWB controlee started (PID %d)", self._controlee_proc.pid)
        except OSError as exc:
            logger.error("Failed to start controlee: %s", exc)
            self._running = False
            return False

        # Wait startup delay
        time.sleep(3.0)

        # Start controller (capture stdout for range data)
        try:
            self._controller_proc = subprocess.Popen(
                controller_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=self._env,
                start_new_session=True,
            )
            logger.info("UWB controller started (PID %d)", self._controller_proc.pid)
        except OSError as exc:
            logger.error("Failed to start controller: %s", exc)
            self._stop_controlee()
            self._running = False
            return False

        self._reader_thread = threading.Thread(
            target=self._read_loop,
            name="uwb-reader",
            daemon=True,
        )
        self._reader_thread.start()
        logger.info("UWB ranging session active (%d ms interval)", ranging_span)
        return True

    def _read_loop(self) -> None:
        """Read range samples from controller stdout."""
        assert self._controller_proc is not None
        assert self._controller_proc.stdout is not None

        current_seq = -1
        current_dist = None
        current_status = None

        try:
            for line in self._controller_proc.stdout:
                if not self._running:
                    break
                line = line.rstrip()

                # Track sequence number
                sm = _SEQUENCE_RE.search(line)
                if sm:
                    current_seq = int(sm.group(1))

                # Track distance
                dm = _DISTANCE_RE.search(line)
                if dm:
                    current_dist = float(dm.group(1))

                # Track status
                stm = _STATUS_RE.search(line)
                if stm:
                    current_status = stm.group(1)

                # When we have a complete measurement (distance + status),
                # emit the sample
                if current_dist is not None and current_status is not None:
                    sample = {
                        "sequence": current_seq,
                        "distance_cm": current_dist,
                        "status": current_status,
                        "timestamp": time.time(),
                    }
                    self._buffer.append(sample)
                    self._sample_count += 1
                    if self._on_sample:
                        try:
                            self._on_sample(sample)
                        except Exception as exc:
                            logger.debug("Sample callback error: %s", exc)
                    self._prune()
                    # Reset for next measurement
                    current_dist = None
                    current_status = None

        except (OSError, ValueError) as exc:
            logger.error("UWB read error: %s", exc)
        finally:
            self._running = False

    def _prune(self) -> None:
        cutoff = time.time() - self.config.window_seconds * 2
        while self._buffer and self._buffer[0].get("timestamp", 0) < cutoff:
            self._buffer.popleft()

    def _stop_controlee(self) -> None:
        if self._controlee_proc:
            try:
                self._controlee_proc.terminate()
                self._controlee_proc.wait(timeout=3)
            except Exception:
                try:
                    self._controlee_proc.kill()
                except Exception:
                    pass
            self._controlee_proc = None

    def stop(self) -> None:
        """Stop the ranging session and clean up subprocesses."""
        self._running = False
        if self._controller_proc:
            try:
                self._controller_proc.terminate()
                self._controller_proc.wait(timeout=3)
            except Exception:
                try:
                    self._controller_proc.kill()
                except Exception:
                    pass
            self._controller_proc = None
        self._stop_controlee()
        logger.info("UWB session stopped (%d samples)", self._sample_count)

    def get_window(self) -> np.ndarray:
        """Return (timesteps, 1) window of distance_cm values."""
        timesteps = self.config.timesteps
        now = time.time()
        window_s = self.config.window_seconds

        distances = [
            s["distance_cm"]
            for s in self._buffer
            if (
                s.get("timestamp", 0) >= now - window_s
                and s.get("status") == "Ok"
            )
        ]
        if not distances:
            return np.zeros((timesteps, 1), dtype=np.float32)

        arr = np.array(distances[-timesteps:], dtype=np.float32).reshape(-1, 1)
        if arr.shape[0] < timesteps:
            padded = np.zeros((timesteps, 1), dtype=np.float32)
            padded[-arr.shape[0] :] = arr
            return padded
        if arr.shape[0] > timesteps:
            indices = np.linspace(0, arr.shape[0] - 1, timesteps).astype(int)
            return arr[indices]
        return arr
