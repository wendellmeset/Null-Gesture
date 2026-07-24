"""UWB sensor: DWM3001CDK FiRa TWR ranging via Qorvo UCI library.

Refactored for Python 3.11 from https://github.com/wshanmu/UWB_lab.

Requires the uwb-qorvo-tools/ subdirectory with the UCI Python library
and two DWM3001CDK boards connected over USB.
"""

from __future__ import annotations

import logging
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

# ── FiRa TWR script path ────────────────────────────────────────────────────
_TWR_SCRIPT = (
    _UWB_TOOLS
    / "scripts"
    / "fira"
    / "run_fira_test_periodic_tx"
    / "run_fira_test_periodic_tx.py"
)


def _has_uwb_tools() -> bool:
    return _UWB_TOOLS.exists() and _TWR_SCRIPT.exists()


# ── Range log line parser ───────────────────────────────────────────────────
_RANGE_LINE_RE = re.compile(
    r"Range\s*\[(?P<seq>\d+)\].*?"
    r"distance\s*=\s*(?P<dist>[\d.]+)\s*cm.*?"
    r"status\s*=\s*(?P<status>\w+)"
)

RANGE_SAMPLE_KEYS = ("sequence", "distance_cm", "status", "timestamp")


class UWBRanger:
    """Manages a UWB FiRa TWR ranging session between two DWM3001CDK boards.

    The controller board is on one hand, controlee on the other.
    Distance measurements are streamed in real time via subprocess stdout.
    """

    def __init__(self, config: UWBConfig | None = None) -> None:
        self.config = config or default_uwb_config
        self._buffer: deque[dict] = deque()
        self._controller_proc: subprocess.Popen | None = None
        self._controlee_proc: subprocess.Popen | None = None
        self._running = False
        self._sample_count = 0
        self._session_dir: Path | None = None
        self._on_sample: Callable[[dict], None] | None = None
        self._parser_seq: int = -1

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
            controller_port: Serial port of the controller board (e.g. /dev/ttyACM0).
            controlee_port: Serial port of the controlee board.
            duration: Session duration in seconds. 0 = run until stop().
            on_sample: Optional callback for each range sample.
        """
        if not _has_uwb_tools():
            logger.error(
                "UWB tools not found at %s. Clone the UWB_lab repo's "
                "uwb-qorvo-tools/ directory into the project root.",
                _UWB_TOOLS,
            )
            return False

        self._on_sample = on_sample
        self._running = True

        if duration <= 0:
            duration = 3600  # Long default for indefinite streaming

        controlee_duration = duration + 5 + 3  # extra + startup delay

        python = sys.executable
        cfg = self.config

        # Build the UCI library PYTHONPATH components
        env_paths = [
            str(_UWB_TOOLS / "lib" / "uwb-uci"),
            str(_UWB_TOOLS / "lib" / "uqt-utils"),
            str(_UWB_TOOLS),
        ]
        if "PYTHONPATH" in sys.modules.get("os", __import__("os")).environ:
            env_paths.append(__import__("os").environ["PYTHONPATH"])

        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = ":".join(env_paths)
        env["UWB_TOOLS"] = str(_UWB_TOOLS)

        # Start controlee first
        controlee_cmd = [
            python,
            str(_TWR_SCRIPT),
            "-p", controlee_port,
            "--preamble-idx", str(cfg.preamble_code),
            "-d", str(controlee_duration),
            "--slot-span", str(cfg.slot_span),
            "--slots-per-rr", str(cfg.slots_per_rr),
            "--channel", str(cfg.uwb_channel),
            "--controlee",
            "--stats",
        ]

        try:
            self._controlee_proc = subprocess.Popen(
                controlee_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
            logger.info("UWB controlee started (PID %d)", self._controlee_proc.pid)
        except OSError as exc:
            logger.error("Failed to start controlee: %s", exc)
            self._running = False
            return False

        # Wait startup delay
        time.sleep(3.0)

        # Start controller
        controller_cmd = [
            python,
            str(_TWR_SCRIPT),
            "-p", controller_port,
            "--preamble-idx", str(cfg.preamble_code),
            "-d", str(duration),
            "--slot-span", str(cfg.slot_span),
            "--slots-per-rr", str(cfg.slots_per_rr),
            "--channel", str(cfg.uwb_channel),
            "--stats",
        ]

        try:
            self._controller_proc = subprocess.Popen(
                controller_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,
            )
            logger.info("UWB controller started (PID %d)", self._controller_proc.pid)
        except OSError as exc:
            logger.error("Failed to start controller: %s", exc)
            self._stop_controlee()
            self._running = False
            return False

        # Start reader thread
        self._reader_thread = threading.Thread(
            target=self._read_loop,
            name="uwb-reader",
            daemon=True,
        )
        self._reader_thread.start()
        logger.info("UWB ranging session active")
        return True

    def _read_loop(self) -> None:
        """Read range samples from controller stdout."""
        assert self._controller_proc is not None
        assert self._controller_proc.stdout is not None

        try:
            for line in self._controller_proc.stdout:
                if not self._running:
                    break
                line = line.strip()
                sample = self._parse_line(line)
                if sample is not None:
                    self._buffer.append(sample)
                    self._sample_count += 1
                    if self._on_sample:
                        try:
                            self._on_sample(sample)
                        except Exception as exc:
                            logger.debug("Sample callback error: %s", exc)
                    self._prune()
        except (OSError, ValueError) as exc:
            logger.error("UWB read error: %s", exc)
        finally:
            self._running = False

    def _parse_line(self, line: str) -> dict | None:
        """Parse a controller output line for range data."""
        m = _RANGE_LINE_RE.search(line)
        if m is None:
            return None
        try:
            seq = int(m.group("seq"))
            dist = float(m.group("dist"))
            status = m.group("status")
        except (ValueError, IndexError):
            return None

        return {
            "sequence": seq,
            "distance_cm": dist,
            "status": status,
            "timestamp": time.time(),
        }

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
