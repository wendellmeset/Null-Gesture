"""UWB Reader: DWM3001CDK FiRa TWR ranging via Qorvo UCI subprocess.

Launches run_fira_twr.py from uwb-qorvo-tools as a controller/controlee
pair and parses distance measurements from stdout.

Usage::

    from Readers import UWBReader

    uwb = UWBReader()
    uwb.connect(initiator="/dev/ttyACM1", responder="/dev/ttyACM0")
    for sample in uwb.stream():
        print(f"{sample['distance_m']:.2f} m")
    uwb.disconnect()
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import signal
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path

_log = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────
_TOOLS_DIR = Path(__file__).resolve().parent / "uwb-qorvo-tools"
_TWR_SCRIPT = _TOOLS_DIR / "scripts" / "fira" / "run_fira_twr" / "run_fira_twr.py"

# ── Output parser regex ─────────────────────────────────────────────────────
_SEQUENCE_RE = re.compile(r"sequence\s+n?:\s+(\d+)")
_DISTANCE_RE = re.compile(r"distance:\s+([\d.]+)\s+cm")
_STATUS_RE = re.compile(r"status:\s+(\w+)")

# ── Default FiRa parameters ─────────────────────────────────────────────────
DEFAULT_PREAMBLE = 10
DEFAULT_CHANNEL = 9
DEFAULT_SLOT_SPAN = 2400
DEFAULT_SLOTS_PER_RR = 6
DEFAULT_SAMPLE_RATE = 50  # Hz


def _build_env() -> dict[str, str]:
    """Build environment with UWB tools on PYTHONPATH."""
    env = os.environ.copy()
    paths = [
        str(_TOOLS_DIR / "lib" / "uwb-uci"),
        str(_TOOLS_DIR / "lib" / "uqt-utils"),
        str(_TOOLS_DIR),
    ]
    existing = env.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["UWB_TOOLS"] = str(_TOOLS_DIR)
    return env


def _find_python() -> str:
    """Find a Python that can import uci (conda env or system)."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = os.path.join(conda_prefix, "bin", "python")
        if os.path.isfile(candidate):
            return candidate
    for base in [os.path.expanduser("~/.conda/envs"), os.path.expanduser("~/miniconda3/envs")]:
        if os.path.isdir(base):
            for name in os.listdir(base):
                py = os.path.join(base, name, "bin", "python")
                if os.path.isfile(py):
                    return py
    return sys.executable


class UWBReader:
    """Minimal DWM3001CDK UWB ranging reader via Qorvo UCI subprocess.

    Provides the standard Readers interface: connect / read / stream / disconnect.
    """

    def __init__(
        self,
        port: str | None = None,
        initiator_port: str | None = None,
        responder_port: str | None = None,
        preamble_code: int = DEFAULT_PREAMBLE,
        channel: int = DEFAULT_CHANNEL,
        sample_rate_hz: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self._port = port
        self._initiator_port = initiator_port
        self._responder_port = responder_port
        self._preamble = preamble_code
        self._channel = channel
        self._sample_rate = sample_rate_hz

        self._connected = False
        self._running = False
        self._last_sample: dict | None = None
        self._samples: deque[dict] = deque(maxlen=2000)

        self._controller_proc: subprocess.Popen | None = None
        self._controlee_proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._sample_count = 0
        self._env = _build_env()

    @staticmethod
    def _kill_stale() -> None:
        """Kill any leftover UWB subprocesses from previous runs."""
        try:
            result = subprocess.run(
                ["pgrep", "-af", "run_fira_twr"],
                capture_output=True, text=True, timeout=2,
            )
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                pid_str = line.split()[0]
                try:
                    pid = int(pid_str)
                    os.kill(pid, signal.SIGTERM)
                except (ValueError, ProcessLookupError):
                    pass
        except Exception:
            pass

    # ── Connection ───────────────────────────────────────────────────────

    def connect(self, **kwargs) -> bool:
        """Start FiRa TWR session. Cleans up stale processes from previous runs."""
        UWBReader._kill_stale()
        initiator = kwargs.get("initiator") or self._initiator_port
        responder = kwargs.get("responder") or self._responder_port
        single = kwargs.get("port") or self._port

        if single:
            # Single-board mode not supported by run_fira_twr — need two boards
            _log.error("Single-board UWB mode not supported. Use initiator + responder.")
            return False

        if not initiator or not responder:
            _log.error("UWB requires initiator_port and responder_port.")
            return False

        if not _TWR_SCRIPT.exists():
            _log.error("UWB tools not found at %s", _TWR_SCRIPT)
            return False

        python = _find_python()
        ranging_span_ms = max(1, int(round(1000.0 / self._sample_rate)))
        controlee_duration = 3700  # ~1 hour for indefinite streaming

        common = [
            python, "-u", str(_TWR_SCRIPT),
            "--channel", str(self._channel),
            "--preamble-idx", str(self._preamble),
            "--aoa-report", "all-disabled",
            "--slot-span", str(DEFAULT_SLOT_SPAN),
            "--slots-per-rr", str(DEFAULT_SLOTS_PER_RR),
            "--ranging-span", str(ranging_span_ms),
            "--stats",
        ]

        controlee_cmd = common + ["-p", responder, "-t", str(controlee_duration), "--controlee"]
        controller_cmd = common + ["-p", initiator, "-t", str(controlee_duration)]

        # Start controlee first
        try:
            self._controlee_proc = subprocess.Popen(
                controlee_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=self._env,
                start_new_session=True,
            )
        except OSError as exc:
            _log.error("Failed to start UWB controlee: %s", exc)
            return False

        time.sleep(3.0)  # Startup delay for controlee

        # Start controller (capture stdout)
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
        except OSError as exc:
            _log.error("Failed to start UWB controller: %s", exc)
            self._stop_controlee()
            return False

        self._connected = True
        self._running = True
        self._reader_thread = threading.Thread(
            target=self._read_loop,
            name="uwb-reader",
            daemon=True,
        )
        self._reader_thread.start()
        return True

    def disconnect(self) -> None:
        """Stop ranging and clean up subprocesses."""
        self._running = False
        self._connected = False

        for proc in [self._controller_proc, self._controlee_proc]:
            if proc and proc.poll() is None:
                try:
                    # Kill entire process group (start_new_session=True)
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

        self._controller_proc = None
        self._controlee_proc = None

        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2)
        self._reader_thread = None

    # ── Reading ──────────────────────────────────────────────────────────

    def read(self) -> dict | None:
        """Return the latest distance sample, or None."""
        if self._samples:
            self._last_sample = self._samples[-1]
            return self._last_sample
        return self._last_sample

    def stream(self) -> Iterator[dict]:
        """Generator yielding distance samples as they arrive."""
        idx = 0
        while self._running or self._samples:
            while idx < len(self._samples):
                yield self._samples[idx]
                idx += 1
            if not self._running:
                break
            time.sleep(0.005)

    def __iter__(self) -> Iterator[dict]:
        return self.stream()

    # ── Internals ────────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        """Read and parse distance measurements from controller stdout."""
        assert self._controller_proc is not None
        assert self._controller_proc.stdout is not None

        current_dist: float | None = None
        current_status: str | None = None
        current_seq: int = -1

        try:
            for line in self._controller_proc.stdout:
                if not self._running:
                    break

                sm = _SEQUENCE_RE.search(line)
                if sm:
                    current_seq = int(sm.group(1))

                dm = _DISTANCE_RE.search(line)
                if dm:
                    current_dist = float(dm.group(1))

                stm = _STATUS_RE.search(line)
                if stm:
                    current_status = stm.group(1)

                if current_dist is not None and current_status is not None:
                    sample = {
                        "distance_m": current_dist / 100.0,
                        "distance_cm": current_dist,
                        "addr": f"seq_{current_seq}",
                        "status": current_status,
                        "timestamp": time.time(),
                    }
                    self._samples.append(sample)
                    self._last_sample = sample
                    self._sample_count += 1
                    current_dist = None
                    current_status = None

        except (OSError, ValueError) as exc:
            _log.error("UWB read error: %s", exc)
        finally:
            self._running = False

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

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_sample(self) -> dict | None:
        return self._last_sample

    @property
    def sample_count(self) -> int:
        return self._sample_count
