#!/usr/bin/env python3
"""
Collect IMU training data from the TCP stream (port 9999).
Records exactly 5 seconds per sample and saves to CSV.
Labels are prompted from the terminal.
"""

import socket
import json
import time
import csv
import sys
from datetime import datetime

# Mapping of class names (for display only)
ORIENT_NAMES = ['Flat', 'Up', 'Down', 'Left', 'Right']
MOTION_NAMES = ['Static', 'Forward', 'Backward', 'SwingLeft', 'SwingRight', 'SwingUp', 'SwingDown']

HOST = 'localhost'
PORT = 9999
DURATION = 5.0  # seconds per sample

def get_label(prompt, max_val, name_list):
    while True:
        try:
            val = int(input(prompt))
            if 0 <= val <= max_val:
                return val
            print(f"Enter a number between 0 and {max_val}")
        except ValueError:
            print("Invalid number")

def main():
    print("Connecting to IMU stream...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect((HOST, PORT))
        print(f"Connected to {HOST}:{PORT}\n")
    except Exception as e:
        print(f"Failed to connect: {e}")
        print("Make sure esp32_reader.py is running.")
        return

    print("Recording controls:")
    print("  Enter orientation label (0-4):")
    print(f"    0={ORIENT_NAMES[0]}, 1={ORIENT_NAMES[1]}, ..., 4={ORIENT_NAMES[4]}")
    print("  Enter motion label (0-6):")
    print(f"    0={MOTION_NAMES[0]}, 1={MOTION_NAMES[1]}, ..., 6={MOTION_NAMES[6]}")
    print("  Press 'r' to start recording (will record for 5 seconds)")
    print("  Press 'q' to quit\n")

    # We'll read lines from the socket in a non-blocking way
    sock.settimeout(0.1)

    while True:
        # Ask for labels
        orient_label = get_label("Orientation label (0-4): ", 4, ORIENT_NAMES)
        motion_label = get_label("Motion label (0-6): ", 6, MOTION_NAMES)

        input(f"\nReady to record: Orient={ORIENT_NAMES[orient_label]}, Motion={MOTION_NAMES[motion_label]}")
        print("Press Enter to start recording (5 seconds)...")
        input()

        print(f"Recording for {DURATION} seconds...")
        start_time = time.monotonic()
        end_time = start_time + DURATION
        samples = []

        while time.monotonic() < end_time:
            try:
                raw = sock.recv(4096).decode('utf-8')
                for line in raw.splitlines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        if obj.get('type') == 'sample':
                            # Append: timestamp, ax, ay, az, gx, gy, gz, orient_label, motion_label
                            samples.append([
                                obj['timestamp'],
                                obj['ax'], obj['ay'], obj['az'],
                                obj['gx'], obj['gy'], obj['gz'],
                                orient_label,
                                motion_label
                            ])
                    except json.JSONDecodeError:
                        pass
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Socket error: {e}")
                break

        if not samples:
            print("No samples received. Check IMU stream.")
            continue

        # Save to CSV
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"imu_{ORIENT_NAMES[orient_label]}_{MOTION_NAMES[motion_label]}_{timestamp_str}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'orient_label', 'motion_label'])
            writer.writerows(samples)

        print(f"Saved {len(samples)} samples to {filename}\n")

        # Ask to continue
        cont = input("Record another? (y/n): ").strip().lower()
        if cont != 'y':
            break

    sock.close()
    print("Done.")

if __name__ == "__main__":
    main()