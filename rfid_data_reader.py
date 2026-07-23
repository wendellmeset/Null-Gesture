#!/usr/bin/env -S /home/mattin/Documents/Projects/Null-Gesture/venv/bin/python
"""
SparkFun Simultaneous RFID Reader - M7E Python Interface
========================================================

Uses the official python-mercuryapi library (ThingMagic Mercury API).

Installation:
    See install_mercury.sh — requires manually downloading the Mercury API C library
    from Novanta/Jadak, then building.

Usage:
    python rfid_data_reader.py              # auto-detect + continuous scan
    python rfid_data_reader.py --test       # detect + verify connection only
    python rfid_data_reader.py --port /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

try:
    import mercury
except ImportError:
    print("=" * 60)
    print("  python-mercuryapi not installed.")
    print()
    print("  Run:  bash install_mercury.sh")
    print()
    print("  This requires downloading the Mercury API C library from:")
    print("  https://novanta.com/precision-medicine/product/thingmagic-mercury-api/")
    print("=" * 60)
    sys.exit(1)

try:
    import serial.tools.list_ports
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
#  Port detection
# ═════════════════════════════════════════════════════════════════════════════

KNOWN_VID_PID = [
    (0x10C4, 0xEA60),  # Silicon Labs CP210x
    (0x1A86, 0x7523),  # CH340
    (0x0403, 0x6015),  # FTDI FT231X
    (0x0403, 0x6001),  # FTDI FT232R
]


def _stop_streaming_via_serial(port: str) -> bool:
    """
    Stop a reader that's stuck in streaming mode by sending a raw
    MULTI_PROTOCOL_TAG_OP stop command over serial.
    """
    try:
        import serial as _serial
        import time as _time
        _ser = _serial.Serial(port, 115200, timeout=1)
        _ser.dtr = False
        _ser.rts = False
        _time.sleep(0.5)
        _ser.reset_input_buffer()
        _time.sleep(0.2)
        _ser.write(b'\xff\x03\x2f\x00\x00\x02\x5e\x86')
        _ser.flush()
        _time.sleep(2)
        while _ser.in_waiting:
            _ser.read(_ser.in_waiting)
            _time.sleep(0.1)
        _ser.close()
        _time.sleep(0.5)
        return True
    except OSError:
        return False


_MercuryReader = cast(Any, mercury).Reader


def find_m7e_port() -> str | None:
    """Auto-detect the M7E reader serial port by probing with the Mercury API."""
    print("Scanning for M7E RFID reader...")

    candidates: list[str] = []

    if _SERIAL_AVAILABLE:
        for port_info in sorted(serial.tools.list_ports.comports()):  # type: ignore[possibly-undefined]
            if (
                port_info.device
                and port_info.vid is not None
                and port_info.pid is not None
                and (port_info.vid, port_info.pid) in KNOWN_VID_PID
            ):
                candidates.append(port_info.device)
                print(f"  Found USB serial: {port_info.device}  "
                      f"[VID:{port_info.vid:04X} PID:{port_info.pid:04X}]")

    if not candidates:
        for pattern in ["/dev/ttyUSB*", "/dev/ttyACM*", "/dev/cu.usbserial*"]:
            candidates.extend(sorted(glob.glob(pattern)))
        if candidates:
            print(f"  Found serial devices: {candidates}")

    if not candidates:
        print("  No USB serial devices found.")
        return None

    for port in candidates:
        print(f"  Trying {port}...", end=" ")
        try:
            reader = _MercuryReader(f"tmr://{port}", baudrate=115200)
            model = reader.get_model()
            version = reader.get_software_version()
            reader.stop_reading()
            print(f"✅ {model}  FW: {version}")
            return port
        except TypeError as exc:
            msg = str(exc)
            if "Streaming" not in msg:
                print(f"no response ({exc})")
                continue
            print("streaming (stopping)...", end=" ")
            if not _stop_streaming_via_serial(port):
                print("stop failed (serial)")
                continue
            try:
                reader = _MercuryReader(f"tmr://{port}", baudrate=115200)
                model = reader.get_model()
                version = reader.get_software_version()
                reader.stop_reading()
                print(f"✅ {model}  FW: {version}")
                return port
            except OSError as e2:
                print(f"stop failed ({e2})")
        except OSError as exc:
            print(f"no response ({exc})")

    print("  No M7E reader found on any port.")
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║  TROUBLESHOOTING                                           ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║  1. Check the UART switch on the board — must be in the     ║")
    print("  ║     USB position (not SER)                                 ║")
    print("  ║                                                            ║")
    print("  ║  2. Power: The M7E draws ~700mA at full power. Try:         ║")
    print("  ║     • A powered USB hub                                    ║")
    print("  ║     • A 5V 2A+ USB-C power adapter                         ║")
    print("  ║     • A different USB cable                                ║")
    print("  ║                                                            ║")
    print("  ║  3. Leave the EN pin unconnected (pulled high internally)   ║")
    print("  ║  4. Disconnect and reconnect USB power                     ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    return None


# ═════════════════════════════════════════════════════════════════════════════
#  CSV logging
# ═════════════════════════════════════════════════════════════════════════════

CSV_FILENAME = "rfid_log.csv"
DEDUP_WINDOW = 5.0


class TagLogger:
    """Logs detected tags to CSV, deduplicating within a time window."""

    def __init__(self, filename: str = CSV_FILENAME):
        self.filename = filename
        self._recent: dict[str, float] = {}
        self._fieldnames = ["Timestamp", "EPC", "RSSI"]

        path = Path(filename)
        write_header = not path.exists() or path.stat().st_size == 0
        self._file = open(filename, "a", newline="")  # noqa: SIM115 - kept open for streaming writes
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def should_log(self, epc: str, now: float) -> bool:
        last = self._recent.get(epc)
        if last is not None and (now - last) < DEDUP_WINDOW:
            return False
        self._recent[epc] = now
        expired = [k for k, v in self._recent.items() if (now - v) > DEDUP_WINDOW * 2]
        for k in expired:
            del self._recent[k]
        return True

    def log(self, tag: dict[str, Any]) -> None:
        raw_epc = tag.get("epc", "")
        epc = raw_epc.hex() if isinstance(raw_epc, bytes) else str(raw_epc)
        if not self.should_log(epc, tag.get("time_obj", time.time())):
            return
        ts = tag.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
        self._writer.writerow({
            "Timestamp": ts,
            "EPC": epc,
            "RSSI": tag.get("rssi", ""),
        })
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def _connect_reader(port: str):
    """Create a Mercury reader connection, handling streaming mode recovery."""
    try:
        return _MercuryReader(f"tmr://{port}", baudrate=115200)
    except TypeError as exc:
        if "Streaming" not in str(exc):
            raise
        print("streaming detected, stopping...")
        _stop_streaming_via_serial(port)
        return _MercuryReader(f"tmr://{port}", baudrate=115200)


def do_test(port: str | None = None) -> None:
    """Test mode: find reader, print info, trial scan, then exit."""
    if port is None:
        port = find_m7e_port()
        if port is None:
            sys.exit(1)
    else:
        print(f"Testing reader on port: {port}")

    reader = _connect_reader(port)
    model = reader.get_model()
    version = reader.get_software_version()
    serial = reader.get_serial()

    print("\n✅ Reader connected!")
    print(f"   Port:     {port}")
    print(f"   Model:    {model}")
    print(f"   Firmware: {version}")
    print(f"   Serial:   {serial}")

    regions = reader.get_supported_regions()
    print(f"   Regions:  {regions}")

    reader.set_region("NA")
    print("   Region:   NA ✅")

    print("\n   Testing inventory scan (2 seconds)...")
    reader.set_read_plan([1], "GEN2")
    tags = reader.read(timeout=2000)
    if tags:
        print(f"   Found {len(tags)} tags:")
        for t in tags[:10]:
            print(f"     • EPC: {t.epc.hex().upper()}  RSSI: {t.rssi} dBm")
        if len(tags) > 10:
            print(f"     ... and {len(tags) - 10} more")
    else:
        print("   No tags found (normal if none nearby)")

    reader.stop_reading()


def do_scan(port: str | None = None) -> None:
    """Production mode: scan tags continuously and log to CSV."""
    if port is None:
        port = find_m7e_port()
        if port is None:
            print("\n❌ No reader found. Use --port to specify manually.")
            sys.exit(1)

    print(f"\nConnecting to M7E reader on {port}...")
    logger = TagLogger(CSV_FILENAME)

    try:
        reader = _connect_reader(port)

        model = reader.get_model()
        version = reader.get_software_version()
        print(f"✅ {model} — FW: {version}")
        print(f"📝 Logging to: {CSV_FILENAME}")
        print("🔍 Scanning for tags... Press Ctrl+C to stop.\n")

        reader.set_region("NA")
        reader.set_read_plan([1], "GEN2")

        tag_count = 0

        def on_tag(tag):
            nonlocal tag_count
            tag_count += 1
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            epc = tag.epc.hex().upper()
            rssi = tag.rssi
            print(f"[{ts}]  EPC: {epc}  RSSI: {rssi} dBm")

            logger.log({
                "epc": tag.epc,
                "rssi": tag.rssi,
                "timestamp": ts,
                "time_obj": time.time(),
            })

        reader.start_reading(callback=on_tag, on_time=1000, off_time=0)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping scan...")
        finally:
            reader.stop_reading()
            print(f"\nTotal tags seen: {tag_count}")

    except OSError as exc:
        print(f"\n❌ Error: {exc}")
        sys.exit(1)
    finally:
        logger.close()
        print(f"Log saved to: {Path(CSV_FILENAME).resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SparkFun Simultaneous RFID Reader - M7E Python Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", "-p", help="Serial port (e.g., /dev/ttyUSB0). Auto-detects if omitted.")
    parser.add_argument("--test", action="store_true", help="Test mode: detect reader, verify connection, then exit.")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.test:
        do_test(port=args.port)
    else:
        do_scan(port=args.port)


if __name__ == "__main__":
    main()
