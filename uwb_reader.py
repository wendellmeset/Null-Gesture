import json
import re
import threading
import time
import serial

# ==================== CONFIGURATION ====================
LISTEN_PORT = "/dev/cu.usbmodemF1B6425717661"  # Board A (Responder)
SIGNAL_PORT = "/dev/cu.usbmodemFCEF9F370D4F1"  # Board B (Initiator)
BAUD_RATE = 115200

CMD_LISTEN = "RESPF 9 2400 200 25 2 42 01:02:03:04:05:06:07:08 2 0 0 1"
CMD_SIGNAL = "INITF 9 2400 200 25 2 42 01:02:03:04:05:06:07:08 1 0 0 1 2"
# =======================================================

# Matches a JSON array anywhere in the buffer, even across what were
# originally separate readline() chunks. DOTALL so a stray \n inside
# the buffer doesn't block the match.
JSON_REGEX = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)


def run_uwb_node(port, config_cmd, node_type):
    print(f"[{node_type}] Linking port {port}...")
    try:
        with serial.Serial(port, BAUD_RATE, timeout=0.2) as ser:
            time.sleep(2.0)  # settle time
            ser.reset_input_buffer()

            ser.write(b"\r\n")
            time.sleep(0.3)
            ser.write(b"quit\r\n")
            time.sleep(0.3)
            ser.reset_input_buffer()

            print(f"[{node_type}] Injecting configuration variables...")
            ser.write(f"{config_cmd}\r\n".encode("utf-8"))
            time.sleep(0.8)

            # Read + print whatever the board says immediately after the
            # config command, so you can see an ack or an error explicitly.
            ack = ser.read(ser.in_waiting or 1).decode("utf-8", errors="ignore")
            if ack.strip():
                print(f"[{node_type} Config Response] {ack.strip()}")

            print(f"[{node_type}] Moving to live listener loop...")

            buf = ""  # rolling buffer instead of per-line matching
            while True:
                chunk = ser.read(ser.in_waiting or 1).decode("utf-8", errors="ignore")
                if not chunk:
                    continue

                buf += chunk

                # Echo complete lines as they appear, for visibility,
                # without losing partial lines from the parse buffer.
                while "\n" in buf:
                    line, buf_rest = buf.split("\n", 1)
                    line = line.strip()
                    if line:
                        print(f"[{node_type} Serial Output] {line}")
                    buf = buf_rest

                    # Try to pull a JSON array out of what we've echoed so far
                    match = JSON_REGEX.search(line)
                    if match:
                        try:
                            data = json.loads(match.group(0))
                            for node in data:
                                addr = node.get("Addr", "Unknown")
                                dist_cm = node.get("D_cm", None)
                                if dist_cm is not None:
                                    print(
                                        f"\n🎯 [DISTANCE FOUND] Node {addr} "
                                        f"-> {dist_cm / 100.0:.2f} meters\n"
                                    )
                        except json.JSONDecodeError:
                            pass

                # Safety: also try matching against whatever's left in buf
                # in case a JSON array is split across chunks and never
                # picked up a trailing \n cleanly.
                match = JSON_REGEX.search(buf)
                if match:
                    try:
                        data = json.loads(match.group(0))
                        for node in data:
                            addr = node.get("Addr", "Unknown")
                            dist_cm = node.get("D_cm", None)
                            if dist_cm is not None:
                                print(
                                    f"\n🎯 [DISTANCE FOUND] Node {addr} "
                                    f"-> {dist_cm / 100.0:.2f} meters\n"
                                )
                        # remove the matched portion so we don't re-parse it
                        buf = buf[match.end():]
                    except json.JSONDecodeError:
                        pass

                # keep buffer from growing unbounded if junk accumulates
                if len(buf) > 4096:
                    buf = buf[-2048:]

    except serial.SerialException as e:
        print(f"[{node_type} Fault] Fatal connection loss: {e}")


if __name__ == "__main__":
    print("=== STARTING NATIVE FIRMWARE INTERFACE ===")
    t_listen = threading.Thread(
        target=run_uwb_node, args=(LISTEN_PORT, CMD_LISTEN, "RESPONDER"), daemon=True
    )
    t_signal = threading.Thread(
        target=run_uwb_node, args=(SIGNAL_PORT, CMD_SIGNAL, "INITIATOR"), daemon=True
    )
    t_listen.start()
    time.sleep(1.5)
    t_signal.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting script.")
