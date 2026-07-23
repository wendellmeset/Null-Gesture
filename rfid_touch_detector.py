#!/usr/bin/env -S /home/mattin/Documents/Projects/Null-Gesture/venv/bin/python
"""
RFID Touch Detector — M7E Hecto
================================

Detects touches by monitoring RSSI and read-rate changes from the
M7E Hecto UHF RFID reader.

Usage:
    ./rfid_touch_detector.py                    # auto-detect + live detection
    ./rfid_touch_detector.py --test             # verify reader and exit
    ./rfid_touch_detector.py --port /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, cast

import numpy as np

try:
    import mercury
except ImportError:
    print("python-mercuryapi not installed. Run: bash install_mercury.sh")
    sys.exit(1)
_MercuryReader = cast(Any, mercury).Reader

# ═══════════════════════════════════════════════════════════════════════════
#  Touch Detection Parameters
# ═══════════════════════════════════════════════════════════════════════════

CALIBRATION_SECONDS = 15.0
WINDOW_SECONDS = 3.0
NO_READ_SECONDS = 1.5

RSSI_DROP_TOUCH_DB = 18.0
RSSI_DROP_RELEASE_DB = 8.0
RATE_DROP_TOUCH_FRACTION = 0.70
RATE_DROP_RELEASE_FRACTION = 0.40
MIN_TOUCH_DURATION = 0.3
MIN_RELEASE_DURATION = 0.5
UI_REFRESH_MS = 200
FEATURE_COUNT = 9


def extract_features(rssi_history: deque, rate: float) -> np.ndarray:
    """Extract feature vector from a window of RSSI readings."""
    if len(rssi_history) < 2:
        return np.zeros(FEATURE_COUNT, dtype=np.float32)
    times = np.array([t for t, _ in rssi_history])
    rssis = np.array([r for _, r in rssi_history])
    now = time.time()
    feat = np.zeros(FEATURE_COUNT, dtype=np.float32)
    feat[0] = float(np.mean(rssis))
    feat[1] = float(np.std(rssis)) if len(rssis) > 1 else 0.0
    feat[2] = float(np.min(rssis))
    feat[3] = float(np.max(rssis))
    feat[4] = feat[3] - feat[2]
    feat[6] = rate
    if len(rssis) >= 3:
        dt = times - times[0]
        if dt[-1] > 0.01:
            feat[5] = float(np.clip(np.polyfit(dt, rssis, 1)[0] * 10, -5, 5))
    if len(rssis) >= 10:
        feat[7] = float(np.mean(rssis[-5:]) - np.mean(rssis[-10:-5]))
    elif len(rssis) >= 4:
        half = len(rssis) // 2
        feat[7] = float(np.mean(rssis[-half:]) - np.mean(rssis[:-half]))
    feat[8] = min((now - times[-1]), 5.0)
    return feat


class Stage:
    WAITING = "waiting"
    CALIBRATING = "calibrating"
    DETECTING = "detecting"


class LiveTagMetrics:
    def __init__(self, selected_epcs: list[str]):
        self.selected_epcs = selected_epcs
        self.selected_lookup = {epc.upper(): epc for epc in selected_epcs}
        self.window_samples: dict[str, deque] = {epc: deque() for epc in selected_epcs}
        self.smoothed_rssi: dict[str, float | None] = {epc: None for epc in selected_epcs}
        self.smooth_factor = 0.3
        self.calibration_samples: dict[str, list[int]] = {epc: [] for epc in selected_epcs}
        self.calibration_read_counts: dict[str, int] = {epc: 0 for epc in selected_epcs}
        self.baseline_rssi: dict[str, float | None] = {epc: None for epc in selected_epcs}
        self.baseline_rate: dict[str, float | None] = {epc: None for epc in selected_epcs}
        self.last_seen: dict[str, float | None] = {epc: None for epc in selected_epcs}
        self.touch_state: dict[str, bool] = {epc: False for epc in selected_epcs}
        self.touch_start_time: dict[str, float | None] = {epc: None for epc in selected_epcs}
        self.release_start_time: dict[str, float | None] = {epc: None for epc in selected_epcs}
        self.stage = Stage.WAITING
        self.calibration_start: float | None = None
        self.calibration_end: float | None = None
        self.detection_start: float | None = None

    def start_calibration(self, now: float) -> None:
        self.stage = Stage.CALIBRATING
        self.calibration_start = now
        self.calibration_end = now + CALIBRATION_SECONDS
        for d in [self.window_samples, self.smoothed_rssi, self.calibration_samples,
                  self.calibration_read_counts, self.baseline_rssi, self.baseline_rate,
                  self.last_seen, self.touch_state, self.touch_start_time, self.release_start_time]:
            d.clear()
            for epc in self.selected_epcs:
                if isinstance(d, dict):
                    if d is self.calibration_samples:
                        d[epc] = []
                    elif d is self.calibration_read_counts:
                        d[epc] = 0
                    elif d is self.touch_state:
                        d[epc] = False
                    elif d is self.smoothed_rssi or d is self.baseline_rssi or d is self.baseline_rate or d is self.last_seen or d is self.touch_start_time or d is self.release_start_time:
                        d[epc] = None
                    else:
                        d[epc] = deque()
        self.detection_start = None

    def finish_calibration(self, now: float) -> None:
        for epc in self.selected_epcs:
            vals = self.calibration_samples[epc]
            if vals:
                sv = sorted(vals)
                self.baseline_rssi[epc] = float(sv[len(sv) // 4])
                self.baseline_rate[epc] = self.calibration_read_counts[epc] / CALIBRATION_SECONDS
        self.window_samples = {epc: deque() for epc in self.selected_epcs}
        self.smoothed_rssi = {epc: None for epc in self.selected_epcs}
        self.last_seen = {epc: None for epc in self.selected_epcs}
        self.touch_state = {epc: False for epc in self.selected_epcs}
        self.touch_start_time = {epc: None for epc in self.selected_epcs}
        self.release_start_time = {epc: None for epc in self.selected_epcs}
        self.detection_start = now
        self.stage = Stage.DETECTING

    def add_record(self, epc: str, rssi: int, now: float) -> None:
        key = self.selected_lookup.get(epc.upper())
        if key is None:
            return
        cur = self.smoothed_rssi[key]
        self.smoothed_rssi[key] = float(rssi) if cur is None else self.smooth_factor * rssi + (1 - self.smooth_factor) * cur
        self.window_samples[key].append((now, self.smoothed_rssi[key], 1))
        self.last_seen[key] = now
        if self.stage == Stage.CALIBRATING:
            self.calibration_samples[key].append(rssi)
            self.calibration_read_counts[key] += 1
        self.prune(now)

    def prune(self, now: float) -> None:
        oldest = now - WINDOW_SECONDS
        for epc in self.selected_epcs:
            s = self.window_samples[epc]
            while s and s[0][0] < oldest:
                s.popleft()

    def calibration_progress(self, now: float) -> float:
        if self.stage != Stage.CALIBRATING or self.calibration_start is None:
            return 0.0
        return max(0.0, min((now - self.calibration_start) / CALIBRATION_SECONDS, 1.0))

    def update_stage(self, now: float) -> None:
        self.prune(now)
        if self.stage == Stage.CALIBRATING and self.calibration_end is not None and now >= self.calibration_end:
            self.finish_calibration(now)

    def current_values(self, epc: str, now: float) -> dict[str, Any]:
        samples = list(self.window_samples[epc])
        smoothed = self.smoothed_rssi[epc]
        rate = sum(s[2] for s in samples) / WINDOW_SECONDS if samples else 0.0
        bl_rssi = self.baseline_rssi[epc]
        bl_rate = self.baseline_rate[epc]
        rssi_drop = bl_rssi - smoothed if (bl_rssi is not None and smoothed is not None) else None
        rate_drop = max(0.0, 1.0 - rate / bl_rate) if (bl_rate is not None and bl_rate > 0) else None
        last = self.last_seen[epc]
        since = None if last is None else now - last
        no_read = since is None or since >= NO_READ_SECONDS
        if self._in_detection_grace_period(now) and not samples:
            no_read = False
            rate_drop = None
        status, reason = self._update_touch_status(epc, now, rssi_drop, rate_drop, no_read)
        return {"baseline_rssi": bl_rssi, "baseline_rate": bl_rate, "current_rssi": smoothed,
                "current_rate": rate, "rssi_drop": rssi_drop, "rate_drop": rate_drop,
                "seconds_since_seen": since, "status": status, "reason": reason}

    def _in_detection_grace_period(self, now: float) -> bool:
        return self.stage == Stage.DETECTING and self.detection_start is not None and now - self.detection_start < NO_READ_SECONDS

    def _update_touch_status(self, epc: str, now: float, rssi_drop: float | None,
                             rate_drop: float | None, no_read: bool) -> tuple[str, str]:
        if self.stage != Stage.DETECTING:
            return "CALIBRATING", "recording baseline"
        if self.baseline_rssi[epc] is None or self.baseline_rate[epc] is None:
            return "NO BASELINE", "not enough calibration reads"

        if not self.touch_state[epc]:
            triggered = no_read or (rssi_drop is not None and rssi_drop >= RSSI_DROP_TOUCH_DB) or (rate_drop is not None and rate_drop >= RATE_DROP_TOUCH_FRACTION)
            if triggered:
                tst = self.touch_start_time[epc]
                if tst is None:
                    self.touch_start_time[epc] = now
                    tst = now
                if now - tst >= MIN_TOUCH_DURATION:
                    self.touch_state[epc] = True
                    self.release_start_time[epc] = None
                    if no_read: return "TOUCHED", "tag not read"
                    if rssi_drop is not None and rssi_drop >= RSSI_DROP_TOUCH_DB: return "TOUCHED", f"RSSI drop {rssi_drop:.1f} dB"
                    return "TOUCHED", f"rate drop {rate_drop:.0%}"
                return "CLEAR", f"pending ({now - tst:.1f}/{MIN_TOUCH_DURATION:.1f}s)"
            self.touch_start_time[epc] = None
            return "CLEAR", "normal"

        recovered = not no_read and rssi_drop is not None and rssi_drop <= RSSI_DROP_RELEASE_DB and rate_drop is not None and rate_drop <= RATE_DROP_RELEASE_FRACTION
        if recovered:
            rst = self.release_start_time[epc]
            if rst is None:
                self.release_start_time[epc] = now
                rst = now
            if now - rst >= MIN_RELEASE_DURATION:
                self.touch_state[epc] = False
                self.touch_start_time[epc] = None
                self.release_start_time[epc] = None
                return "CLEAR", "normal"
            return "TOUCHED", f"releasing ({now - rst:.1f}/{MIN_RELEASE_DURATION:.1f}s)"
        self.release_start_time[epc] = None
        if no_read: return "TOUCHED", "tag not read"
        if rssi_drop is not None and rssi_drop > RSSI_DROP_RELEASE_DB: return "TOUCHED", f"RSSI drop {rssi_drop:.1f} dB"
        if rate_drop is not None and rate_drop > RATE_DROP_RELEASE_FRACTION: return "TOUCHED", f"rate drop {rate_drop:.0%}"
        return "TOUCHED", "waiting for recovery"


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _stop_streaming_via_serial(port: str) -> bool:
    try:
        import serial as _serial
        _ser = _serial.Serial(port, 115200, timeout=1)
        _ser.dtr = False; _ser.rts = False
        time.sleep(0.5)
        _ser.reset_input_buffer()
        _ser.write(b'\xff\x03\x2f\x00\x00\x02\x5e\x86')
        _ser.flush(); time.sleep(2)
        while _ser.in_waiting:
            _ser.read(_ser.in_waiting); time.sleep(0.1)
        _ser.close(); time.sleep(0.5)
        return True
    except OSError:
        return False


def _connect_reader(port: str):
    try:
        return _MercuryReader(f"tmr://{port}", baudrate=115200)
    except TypeError as exc:
        if "Streaming" not in str(exc):
            raise
        print("streaming detected, stopping...")
        _stop_streaming_via_serial(port)
        return _MercuryReader(f"tmr://{port}", baudrate=115200)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def format_dbm(v: float | None) -> str: return "-" if v is None else f"{v:.1f} dBm"
def format_rate(v: float | None) -> str: return "-" if v is None else f"{v:.1f} reads/s"
def format_drop(v: float | None) -> str: return "-" if v is None else f"{v:.1f} dB"
def format_pct(v: float | None) -> str: return "-" if v is None else f"{v:.0%}"
def format_sec(v: float | None) -> str:
    if v is None: return "never"
    if v < 0.1: return "now"
    return f"{v:.1f}s"


def do_test(port: str | None = None) -> None:
    import rfid_data_reader
    rfid_data_reader.do_test(port=port)


def do_detect(port: str | None = None, epc_targets: list[str] | None = None) -> None:
    import rfid_data_reader
    if port is None:
        port = rfid_data_reader.find_m7e_port()
        if port is None:
            sys.exit(1)

    print(f"\nConnecting to M7E reader on {port}...")
    reader = _connect_reader(port)
    model = reader.get_model()
    version = reader.get_software_version()
    print(f"✅ {model} — FW: {version}")

    reader.set_region("NA")
    reader.set_read_plan([1], "GEN2")

    if epc_targets:
        selected_epcs = epc_targets
    else:
        print("\nScanning for tags to monitor...")
        tags = reader.read(timeout=3000)
        if not tags:
            print("No tags found. Place a tag near the reader and retry.")
            return
        seen: list[str] = []
        for t in tags:
            epc = t.epc.hex().upper()
            if epc not in seen:
                seen.append(epc)
        selected_epcs = seen
        print(f"Found {len(selected_epcs)} tag(s): {', '.join(selected_epcs)}")

    metrics = LiveTagMetrics(selected_epcs)
    now = time.time()
    metrics.start_calibration(now)
    tag_count = 0

    def on_tag(tag):
        nonlocal tag_count, metrics, now
        tag_count += 1
        now = time.time()
        metrics.add_record(tag.epc.hex().upper(), tag.rssi, now)

    reader.start_reading(callback=on_tag, on_time=1000, off_time=0)

    try:
        while True:
            now = time.time()
            metrics.update_stage(now)

            if metrics.stage == Stage.CALIBRATING:
                pct = metrics.calibration_progress(now) * 100
                cs = metrics.calibration_start
                if cs is not None:
                    rem = max(0, int(CALIBRATION_SECONDS - (now - cs)))
                else:
                    rem = int(CALIBRATION_SECONDS)
                sys.stdout.write(f"\rCalibrating... {pct:.0f}% ({rem}s left)  ")
                sys.stdout.flush()

            elif metrics.stage == Stage.DETECTING:
                lines = []
                for epc in selected_epcs:
                    v = metrics.current_values(epc, now)
                    s, r = v["status"], v["reason"]
                    icon = "🔴" if s == "TOUCHED" else ("🟢" if s == "CLEAR" else "  ")
                    tag_label = f"{icon} {epc[:16]}.."
                    if s == "CLEAR":
                        lines.append(f"  {tag_label}  {s:12s}  RSSI {format_dbm(v['current_rssi'])}")
                    else:
                        lines.append(f"  {tag_label}  {s:12s}  {r}")
                    lines.append(f"             baseline {format_dbm(v['baseline_rssi'])} @ {format_rate(v['baseline_rate'])}  "
                                 f"drop {format_drop(v['rssi_drop'])} / {format_pct(v['rate_drop'])}  "
                                 f"seen {format_sec(v['seconds_since_seen'])}")

                sys.stdout.write("\033[J")
                sys.stdout.write(f"\rTags: {tag_count}  |  {metrics.stage}\n")
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()

            time.sleep(UI_REFRESH_MS / 1000)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        reader.stop_reading()
        print(f"Total tags seen: {tag_count}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RFID Touch Detector — M7E Hecto")
    p.add_argument("--port", "-p", help="Serial port (auto-detected).")
    p.add_argument("--test", action="store_true", help="Verify reader and exit.")
    p.add_argument("--epc", nargs="*", help="Specific EPCs to monitor.")
    p.add_argument("--model", default="touch_model.keras",
                    help="Trained model file (default: touch_model.keras). Omit for rule-based detection.")
    return p.parse_args()


def do_detect_nn(port: str | None = None, epc_targets: list[str] | None = None,
                  model_path: str = "touch_model.keras") -> None:
    """Touch detection using a trained neural network."""
    import json
    from pathlib import Path as _Path

    import numpy as np
    from keras import models as kmodels

    if not _Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Train one first: python rfid_train_touch.py --collect --train")
        sys.exit(1)

    print(f"Loading model: {model_path}...")
    model: Any = kmodels.load_model(model_path)
    norm_path = _Path(model_path).with_suffix(".norm.json")
    with open(norm_path) as f:
        norm = json.load(f)
    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)
    labels = norm["labels"]

    import rfid_data_reader
    if port is None:
        port = rfid_data_reader.find_m7e_port()
        if port is None:
            sys.exit(1)

    print(f"\nConnecting to reader on {port}...")
    reader = _connect_reader(port)
    print(f"✅ {reader.get_model()} — FW: {reader.get_software_version()}")
    reader.set_region("NA")
    reader.set_read_plan([1], "GEN2")

    if epc_targets:
        selected_epcs = epc_targets
    else:
        print("\nScanning for tags...")
        tags = reader.read(timeout=3000)
        if not tags:
            print("No tags found.")
            return
        seen: list[str] = []
        for t in tags:
            epc = t.epc.hex().upper()
            if epc not in seen:
                seen.append(epc)
        selected_epcs = seen
        print(f"Found {len(selected_epcs)} tag(s)")

    rssi_buf: dict[str, deque] = {epc: deque() for epc in selected_epcs}
    tag_count = 0

    def on_tag(tag):
        nonlocal tag_count
        tag_count += 1
        epc = tag.epc.hex().upper()
        if epc not in rssi_buf:
            return
        rssi_buf[epc].append((time.time(), float(tag.rssi)))
        cutoff = time.time() - WINDOW_SECONDS
        while rssi_buf[epc] and rssi_buf[epc][0][0] < cutoff:
            rssi_buf[epc].popleft()

    reader.start_reading(callback=on_tag, on_time=1000, off_time=0)
    print("\nNeural network detection running... Press Ctrl+C to stop.\n")

    try:
        while True:
            lines: list[str] = []
            for epc in selected_epcs:
                buf = rssi_buf[epc]
                rate = len(buf) / WINDOW_SECONDS if buf else 0.0
                feat_norm = (extract_features(buf, rate) - mean) / std
                probs = model.predict(feat_norm.reshape(1, -1), verbose=0)[0]
                pred = int(np.argmax(probs))
                confidence = float(probs[pred])
                pred_label = labels[pred] if pred < len(labels) else "UNKNOWN"

                icon = "🟢" if pred_label == "CLEAR" else "🔴"
                tag_label = f"{icon} {epc[:16]}.."

                rssi_str = f"RSSI: {buf[-1][1]:.1f} dBm" if buf else ""
                lines.append(f"  {tag_label}  {pred_label:12s}  {confidence:.0%}  {rssi_str}")
                probs_str = " | ".join(f"{labels[i]}={probs[i]:.0%}" for i in range(len(labels)) if i < len(probs))
                lines.append(f"             {probs_str}")

            sys.stdout.write("\033[J")
            sys.stdout.write(f"\rTags: {tag_count}  |  Model: {_Path(model_path).name}\n")
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()
            time.sleep(UI_REFRESH_MS / 1000)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        reader.stop_reading()
        print(f"Total tags: {tag_count}")


def main() -> None:
    args = parse_args()
    if args.test:
        do_test(port=args.port)
    elif args.model and Path(args.model).exists():
        do_detect_nn(port=args.port, epc_targets=args.epc, model_path=args.model)
    else:
        do_detect(port=args.port, epc_targets=args.epc)


if __name__ == "__main__":
    main()
