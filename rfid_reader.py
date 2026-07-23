from __future__ import annotations

import queue
import socket
import statistics
import threading
from dataclasses import dataclass
import time
from collections import deque
from pathlib import Path
import argparse
import sys


@dataclass(frozen=True)
class RfidRecord:
    epc: str
    timestamp: str
    rssi: int
    read_count: int
    line_number: int
    source_file: Path

def parse_rfid_line(line: str, line_number: int, source_file: Path) -> RfidRecord | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        rssi = int(parts[-2])
        read_count = int(parts[-1])
    except ValueError:
        return None

    return RfidRecord(
        epc=parts[0],
        timestamp=" ".join(parts[1:-2]),
        rssi=rssi,
        read_count=read_count,
        line_number=line_number,
        source_file=source_file,
    )

#_______________________________________________________________________________
# Reading RFID
# _______________________________________________________________________________

DEFAULT_HOST = "127.0.0.1"
# DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9055
DEFAULT_DURATION_SECONDS = 15.0
BUFFER_SIZE = 4096
CONNECT_TIMEOUT_SECONDS = 10


def positive_float(value):
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read RFID data from a TCP socket for a fixed time and save it to a log file."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="RFID reader IP address.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="RFID reader TCP port.")
    parser.add_argument(
        "--duration",
        type=positive_float,
        default=DEFAULT_DURATION_SECONDS,
        help="How many seconds to read data. Default: 15.",
    )
    parser.add_argument(
        "--output-dir",
        default="logs",
        help="Folder or subfolder where the log file will be saved. Default: logs.",
    )
    parser.add_argument(
        "--filename",
        required=True,
        help="Log filename, for example 0_5_m.log or trial_01.txt.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding used to decode socket bytes. Default: utf-8.",
    )
    return parser.parse_args()


def read_tcp_to_log(host, port, duration, output_path, encoding):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {host}:{port}...")
    start_time = time.monotonic()
    end_time = start_time + duration
    bytes_written = 0

    with socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_SECONDS) as tcp_socket:
        print(f"Connected. Reading for {duration:g} seconds.")

        with output_path.open("w", encoding=encoding, newline="") as log_file:
            while time.monotonic() < end_time:
                remaining = end_time - time.monotonic()
                tcp_socket.settimeout(min(0.5, max(remaining, 0.001)))

                try:
                    data = tcp_socket.recv(BUFFER_SIZE)
                except socket.timeout:
                    continue

                if not data:
                    print("\nConnection closed by remote host.")
                    break

                text = data.decode(encoding, errors="replace")
                log_file.write(text)
                log_file.flush()

                sys.stdout.write(text)
                sys.stdout.flush()
                bytes_written += len(data)

    return bytes_written






#_______________________________________________________________________________
# Detecting Touch
# _______________________________________________________________________________
# TODO - enter correct tag


SELECTED_EPCS = [
    "RFID_TAG_HERE"
]

CALIBRATION_SECONDS = 10.0
WINDOW_SECONDS = 2.0
NO_READ_SECONDS = 1.0

RSSI_DROP_TOUCH_DB = 8.0
RSSI_DROP_RELEASE_DB = 3.0
RATE_DROP_TOUCH_FRACTION = 0.50
RATE_DROP_RELEASE_FRACTION = 0.30

ENCODING = "utf-8"
BUFFER_SIZE = 4096
CONNECT_TIMEOUT_SECONDS = 10
UI_REFRESH_MS = 100


class Stage:
    WAITING = "waiting"
    CALIBRATING = "calibrating"
    DETECTING = "detecting"


class LiveTagMetrics:
    def __init__(self, selected_epcs: list[str]):
        self.selected_epcs = selected_epcs
        self.selected_lookup = {epc.upper(): epc for epc in selected_epcs}
        self.window_samples = {epc: deque() for epc in selected_epcs}
        self.calibration_samples = {epc: [] for epc in selected_epcs}
        self.calibration_read_counts = {epc: 0 for epc in selected_epcs}
        self.baseline_rssi: dict[str, float | None] = {epc: None for epc in selected_epcs}
        self.baseline_rate: dict[str, float | None] = {epc: None for epc in selected_epcs}
        self.last_seen: dict[str, float | None] = {epc: None for epc in selected_epcs}
        self.touch_state: dict[str, bool] = {epc: False for epc in selected_epcs}
        self.stage = Stage.WAITING
        self.calibration_start = None
        self.calibration_end = None
        self.detection_start = None

    def start_calibration(self, now: float):
        self.stage = Stage.CALIBRATING
        self.calibration_start = now
        self.calibration_end = now + CALIBRATION_SECONDS
        self.window_samples = {epc: deque() for epc in self.selected_epcs}
        self.calibration_samples = {epc: [] for epc in self.selected_epcs}
        self.calibration_read_counts = {epc: 0 for epc in self.selected_epcs}
        self.baseline_rssi = {epc: None for epc in self.selected_epcs}
        self.baseline_rate = {epc: None for epc in self.selected_epcs}
        self.last_seen = {epc: None for epc in self.selected_epcs}
        self.touch_state = {epc: False for epc in self.selected_epcs}
        self.detection_start = None

    def finish_calibration(self, now: float):
        for epc in self.selected_epcs:
            rssi_values = self.calibration_samples[epc]
            if rssi_values:
                self.baseline_rssi[epc] = statistics.median(rssi_values)
                self.baseline_rate[epc] = (
                    self.calibration_read_counts[epc] / CALIBRATION_SECONDS
                )
        self.window_samples = {epc: deque() for epc in self.selected_epcs}
        self.last_seen = {epc: None for epc in self.selected_epcs}
        self.touch_state = {epc: False for epc in self.selected_epcs}
        self.detection_start = now
        self.stage = Stage.DETECTING

    def add_record(self, record: RfidRecord, now: float):
        epc = self.selected_lookup.get(record.epc.upper())
        if epc is None:
            return

        self.window_samples[epc].append((now, record.rssi, record.read_count))
        self.last_seen[epc] = now

        if self.stage == Stage.CALIBRATING:
            self.calibration_samples[epc].append(record.rssi)
            self.calibration_read_counts[epc] += record.read_count

        self.prune(now)

    def prune(self, now: float):
        oldest_allowed = now - WINDOW_SECONDS
        for epc in self.selected_epcs:
            samples = self.window_samples[epc]
            while samples and samples[0][0] < oldest_allowed:
                samples.popleft()

    def calibration_progress(self, now: float) -> float:
        if self.stage != Stage.CALIBRATING or self.calibration_start is None:
            return 0.0
        elapsed = now - self.calibration_start
        return max(0.0, min(elapsed / CALIBRATION_SECONDS, 1.0))

    def update_stage(self, now: float):
        self.prune(now)
        if self.stage == Stage.CALIBRATING and self.calibration_end is not None:
            if now >= self.calibration_end:
                self.finish_calibration(now)

    def current_values(self, epc: str, now: float) -> dict[str, object]:
        samples = list(self.window_samples[epc])
        current_rssi = None
        current_rate = 0.0
        if samples:
            current_rssi = statistics.median([sample[1] for sample in samples])
            current_rate = sum(sample[2] for sample in samples) / WINDOW_SECONDS

        baseline_rssi = self.baseline_rssi[epc]
        baseline_rate = self.baseline_rate[epc]
        rssi_drop = None
        if baseline_rssi is not None and current_rssi is not None:
            rssi_drop = baseline_rssi - current_rssi

        rate_drop = None
        if baseline_rate is not None and baseline_rate > 0:
            rate_drop = max(0.0, 1.0 - current_rate / baseline_rate)

        last_seen = self.last_seen[epc]
        seconds_since_seen = None if last_seen is None else now - last_seen
        no_read = seconds_since_seen is None or seconds_since_seen >= NO_READ_SECONDS
        if self._in_detection_grace_period(now) and not samples:
            no_read = False
            rate_drop = None

        status, reason = self._update_touch_status(epc, rssi_drop, rate_drop, no_read)

        return {
            "baseline_rssi": baseline_rssi,
            "baseline_rate": baseline_rate,
            "current_rssi": current_rssi,
            "current_rate": current_rate,
            "rssi_drop": rssi_drop,
            "rate_drop": rate_drop,
            "seconds_since_seen": seconds_since_seen,
            "status": status,
            "reason": reason,
        }

    def _in_detection_grace_period(self, now: float) -> bool:
        if self.stage != Stage.DETECTING or self.detection_start is None:
            return False
        return now - self.detection_start < NO_READ_SECONDS

    def _update_touch_status(
        self,
        epc: str,
        rssi_drop: float | None,
        rate_drop: float | None,
        no_read: bool,
    ) -> tuple[str, str]:
        if self.stage != Stage.DETECTING:
            return "CALIBRATING", "recording baseline"

        if self.baseline_rssi[epc] is None or self.baseline_rate[epc] is None:
            return "NO BASELINE", "not enough calibration reads"

        was_touched = self.touch_state[epc]

        if not was_touched:
            if no_read:
                self.touch_state[epc] = True
                return "TOUCHED", "tag not read"
            if rssi_drop is not None and rssi_drop >= RSSI_DROP_TOUCH_DB:
                self.touch_state[epc] = True
                return "TOUCHED", f"RSSI drop {rssi_drop:.1f} dB"
            if rate_drop is not None and rate_drop >= RATE_DROP_TOUCH_FRACTION:
                self.touch_state[epc] = True
                return "TOUCHED", f"rate drop {rate_drop:.0%}"
            return "CLEAR", "normal"

        recovered_rssi = rssi_drop is not None and rssi_drop <= RSSI_DROP_RELEASE_DB
        recovered_rate = rate_drop is not None and rate_drop <= RATE_DROP_RELEASE_FRACTION
        if not no_read and recovered_rssi and recovered_rate:
            self.touch_state[epc] = False
            return "CLEAR", "normal"

        if no_read:
            return "TOUCHED", "tag not read"
        if rssi_drop is not None and rssi_drop > RSSI_DROP_RELEASE_DB:
            return "TOUCHED", f"RSSI drop {rssi_drop:.1f} dB"
        if rate_drop is not None and rate_drop > RATE_DROP_RELEASE_FRACTION:
            return "TOUCHED", f"rate drop {rate_drop:.0%}"
        return "TOUCHED", "waiting for recovery"


class TcpReader(threading.Thread):
    def __init__(self, events: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.events = events
        self.stop_event = stop_event

    def run(self):
        self.events.put(("status", f"Connecting to {DEFAULT_HOST}:{DEFAULT_PORT}..."))

        try:
            with socket.create_connection(
                (DEFAULT_HOST, DEFAULT_PORT), timeout=CONNECT_TIMEOUT_SECONDS
            ) as tcp_socket:
                tcp_socket.settimeout(0.5)
                self.events.put(("connected", f"Connected to {DEFAULT_HOST}:{DEFAULT_PORT}"))
                self._read_loop(tcp_socket)
        except socket.timeout:
            self.events.put(("error", f"Connection timed out: {DEFAULT_HOST}:{DEFAULT_PORT}"))
        except ConnectionRefusedError:
            self.events.put(("error", f"Connection refused: {DEFAULT_HOST}:{DEFAULT_PORT}"))
        except OSError as exc:
            self.events.put(("error", f"Socket error: {exc}"))

    def _read_loop(self, tcp_socket: socket.socket):
        buffer = ""
        line_number = 0

        while not self.stop_event.is_set():
            try:
                data = tcp_socket.recv(BUFFER_SIZE)
            except socket.timeout:
                continue

            if not data:
                self.events.put(("error", "Connection closed by remote host."))
                break

            buffer += data.decode(ENCODING, errors="replace")
            lines = buffer.splitlines(keepends=True)
            if lines and not lines[-1].endswith(("\n", "\r")):
                buffer = lines.pop()
            else:
                buffer = ""

            for raw_line in lines:
                line_number += 1
                record = parse_rfid_line(raw_line, line_number, Path("<tcp>"))
                if record is not None:
                    self.events.put(("record", record))

"""
class TouchDetectorGui:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RFID Touch Detector")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.metrics = LiveTagMetrics(SELECTED_EPCS)
        self.reader = None
        self.rows = {}

        self._build_ui()
        self.start_reader()
        self.root.after(UI_REFRESH_MS, self.update_ui)

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(12, 10, 12, 6))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        self.stage_var = tk.StringVar(value="Waiting for reader")
        self.status_var = tk.StringVar(value=f"Reader: {HOST}:{PORT}")
        self.stage_label = ttk.Label(header, textvariable=self.stage_var, font=("", 16, "bold"))
        self.stage_label.grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.status_var).grid(row=1, column=0, sticky="w")

        controls = ttk.Frame(header)
        controls.grid(row=0, column=1, rowspan=2, sticky="e")
        ttk.Button(controls, text="Recalibrate", command=self.recalibrate).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(controls, text="Quit", command=self.close).grid(row=0, column=1)

        self.progress = ttk.Progressbar(
            self.root, orient="horizontal", mode="determinate", maximum=100
        )
        self.progress.grid(row=1, column=0, sticky="ew", padx=12)

        table = ttk.Frame(self.root, padding=(12, 8, 12, 12))
        table.grid(row=2, column=0, sticky="nsew")
        self.root.rowconfigure(2, weight=1)

        headings = [
            "Tag",
            "EPC",
            "Baseline RSSI",
            "Baseline rate",
            "Current RSSI",
            "Current rate",
            "RSSI drop",
            "Rate drop",
            "Last seen",
            "Detector",
        ]
        for col, heading in enumerate(headings):
            table.columnconfigure(col, weight=1 if col in (1, 9) else 0)
            ttk.Label(table, text=heading, font=("", 11, "bold")).grid(
                row=0, column=col, sticky="ew", padx=4, pady=(0, 6)
            )

        for row_index, epc in enumerate(SELECTED_EPCS, start=1):
            row_vars = {name: tk.StringVar(value="-") for name in headings}
            row_vars["Tag"].set(f"Tag {row_index}")
            row_vars["EPC"].set(epc)
            self.rows[epc] = row_vars

            for col, heading in enumerate(headings):
                label = ttk.Label(table, textvariable=row_vars[heading])
                if heading == "Detector":
                    label = tk.Label(
                        table,
                        textvariable=row_vars[heading],
                        width=18,
                        padx=8,
                        pady=4,
                        bg="#E5E7EB",
                        fg="#111827",
                    )
                    row_vars["_detector_widget"] = label
                label.grid(row=row_index, column=col, sticky="ew", padx=4, pady=4)

        note = ttk.Label(
            self.root,
            padding=(12, 0, 12, 12),
            text=(
                "Keep tags untouched during calibration. Detection uses each tag's "
                "own baseline RSSI and read rate."
            ),
        )
        note.grid(row=3, column=0, sticky="w")

    def start_reader(self):
        self.reader = TcpReader(self.events, self.stop_event)
        self.reader.start()

    def recalibrate(self):
        self.metrics.start_calibration(time.monotonic())
        self.status_var.set("Recalibrating. Keep all selected tags untouched.")

    def update_ui(self):
        now = time.monotonic()
        self._process_events(now)
        self.metrics.update_stage(now)
        self._refresh_rows(now)
        self._refresh_header(now)

        if not self.stop_event.is_set():
            self.root.after(UI_REFRESH_MS, self.update_ui)

    def _process_events(self, now: float):
        while True:
            try:
                event_type, payload = self.events.get_nowait()
            except queue.Empty:
                break

            if event_type == "record":
                self.metrics.add_record(payload, now)
            elif event_type == "connected":
                self.status_var.set(payload)
                if self.metrics.stage == Stage.WAITING:
                    self.metrics.start_calibration(now)
            elif event_type == "status":
                self.status_var.set(payload)
            elif event_type == "error":
                self.status_var.set(payload)

    def _refresh_header(self, now: float):
        if self.metrics.stage == Stage.WAITING:
            self.stage_var.set("Waiting for reader")
            self.progress["value"] = 0
        elif self.metrics.stage == Stage.CALIBRATING:
            progress = self.metrics.calibration_progress(now)
            remaining = max(0.0, CALIBRATION_SECONDS * (1.0 - progress))
            self.stage_var.set(f"Calibration: {remaining:.1f} s remaining")
            self.progress["value"] = progress * 100.0
        else:
            touched_tags = [
                index + 1
                for index, epc in enumerate(SELECTED_EPCS)
                if self.metrics.touch_state[epc]
            ]
            if touched_tags:
                tag_text = ", ".join(f"Tag {index}" for index in touched_tags)
                self.stage_var.set(f"Detection: {tag_text} touched")
            else:
                self.stage_var.set("Detection: no touch detected")
            self.progress["value"] = 100

    def _refresh_rows(self, now: float):
        for epc in SELECTED_EPCS:
            values = self.metrics.current_values(epc, now)
            row = self.rows[epc]
            row["Baseline RSSI"].set(format_dbm(values["baseline_rssi"]))
            row["Baseline rate"].set(format_rate(values["baseline_rate"]))
            row["Current RSSI"].set(format_dbm(values["current_rssi"]))
            row["Current rate"].set(format_rate(values["current_rate"]))
            row["RSSI drop"].set(format_drop(values["rssi_drop"]))
            row["Rate drop"].set(format_percent(values["rate_drop"]))
            row["Last seen"].set(format_seconds(values["seconds_since_seen"]))
            row["Detector"].set(f"{values['status']}\n{values['reason']}")
            self._style_detector(row["_detector_widget"], values["status"])

    def _style_detector(self, widget: tk.Label, status: str):
        if status == "TOUCHED":
            widget.configure(bg="#991B1B", fg="white")
        elif status == "CLEAR":
            widget.configure(bg="#DCFCE7", fg="#14532D")
        elif status == "NO BASELINE":
            widget.configure(bg="#FEF3C7", fg="#78350F")
        else:
            widget.configure(bg="#E5E7EB", fg="#111827")

    def close(self):
        self.stop_event.set()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

"""


def format_dbm(value):
    if value is None:
        return "-"
    return f"{value:.1f} dBm"


def format_rate(value):
    if value is None:
        return "-"
    return f"{value:.1f} reads/s"


def format_drop(value):
    if value is None:
        return "-"
    return f"{value:.1f} dB"


def format_percent(value):
    if value is None:
        return "-"
    return f"{value:.0%}"


def format_seconds(value):
    if value is None:
        return "never"
    if value < 0.1:
        return "now"
    return f"{value:.1f} s"


def main():
    args = parse_args()
    output_path = Path(args.output_dir).expanduser() / args.filename

    try:
        bytes_written = read_tcp_to_log(
            host=args.host,
            port=args.port,
            duration=args.duration,
            output_path=output_path,
            encoding=args.encoding,
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return
    except socket.timeout:
        print(f"Connection timed out while trying to reach {args.host}:{args.port}.")
        return
    except ConnectionRefusedError:
        print(f"Connection refused by {args.host}:{args.port}.")
        return
    except OSError as exc:
        print(f"Socket error: {exc}")
        return

    print(f"\nSaved {bytes_written} bytes to {output_path.resolve()}")


if __name__ == "__main__":
    main()
