#!/usr/bin/env python3
"""Quick UWB test — starts both controller + controlee, prints distances.

Usage:
    source venv/bin/activate
    python scripts/uwb_test.py /dev/ttyACM1 /dev/ttyACM2
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
TOOLS = PROJ / "uwb-qorvo-tools"
SCRIPT = TOOLS / "scripts" / "fira" / "run_fira_twr" / "run_fira_twr.py"

PYTHONPATH = ":".join([
    str(TOOLS / "lib" / "uwb-uci"),
    str(TOOLS / "lib" / "uqt-utils"),
    str(TOOLS),
])


def main() -> int:
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <controller_port> <controlee_port>")
        return 1

    ctrl_port = sys.argv[1]
    ctrllee_port = sys.argv[2]
    python = sys.executable

    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH + ":" + env.get("PYTHONPATH", "")

    common = [
        python, "-u", str(SCRIPT),
        "--channel", "9",
        "--preamble-idx", "10",
        "--aoa-report", "all-disabled",
        "--slot-span", "2400",
        "--slots-per-rr", "6",
        "--ranging-span", "50",
        "--stats",
    ]

    # 1. Start controlee
    print(f"Starting controlee on {ctrllee_port} ...")
    controlee = subprocess.Popen(
        common + ["-p", ctrllee_port, "-t", "30", "--controlee"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    time.sleep(3)

    # 2. Start controller
    print(f"Starting controller on {ctrl_port} ...")
    controller = subprocess.Popen(
        common + ["-p", ctrl_port, "-t", "10"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    # 3. Parse output
    dist_re = re.compile(r"distance:\s+([\d.]+)\s+cm")
    status_re = re.compile(r"status:\s+(\w+)")
    seq_re = re.compile(r"sequence\s+n?:\s+(\d+)")

    current_dist = None
    current_status = None
    current_seq = -1
    count = 0

    print("\nWaiting for measurements...\n")
    for line in controller.stdout:
        line = line.rstrip()

        sm = seq_re.search(line)
        if sm:
            current_seq = int(sm.group(1))

        dm = dist_re.search(line)
        if dm:
            current_dist = float(dm.group(1))

        stm = status_re.search(line)
        if stm:
            current_status = stm.group(1)

        if current_dist is not None and current_status is not None:
            count += 1
            icon = "✅" if current_status == "Ok" else "❌"
            print(f"  [{count:4d}] {icon} seq={current_seq:4d}  dist={current_dist:7.1f} cm  status={current_status}")
            current_dist = None
            current_status = None

    print(f"\nDone — {count} measurements")
    controlee.terminate()
    controlee.wait(timeout=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
