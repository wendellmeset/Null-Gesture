import re
import serial
import time
import json
from typing import Pattern
import threading
import queue
from typing import cast
from socketserver import ThreadingTCPServer, StreamRequestHandler

# ---------- Original parsing code ----------
FLOAT_RE: str = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

SAMPLE_RE: Pattern[str] = re.compile(
    rf"accel\[g\]\s+x=\s*({FLOAT_RE})\s+y=\s*({FLOAT_RE})\s+z=\s*({FLOAT_RE})"
    rf"\s+\|\s+gyro\[dps\]\s+x=\s*({FLOAT_RE})\s+y=\s*({FLOAT_RE})\s+z=\s*({FLOAT_RE})"
)

AXES = ("x", "y", "z")
CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")
SEABORN_COLORS = {
    "x": "#4C72B0",
    "y": "#55A868",
    "z": "#C44E52",
}

def parse_sample(line):
    match = SAMPLE_RE.search(line)
    if not match:
        return None
    return tuple(float(value) for value in match.groups())

def serial_worker(port, baud, samples, stop_event, print_unparsed):
    try:
        with serial.Serial(port, baudrate=baud, timeout=0.1) as ser:
            ser.reset_input_buffer()
            while not stop_event.is_set():
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                sample = parse_sample(line)
                if sample is not None:
                    samples.put((time.monotonic(), *sample))
                elif print_unparsed:
                    print(line)
    except serial.SerialException as exc:
        samples.put(("error", str(exc)))

# ---------- Streaming server (broadcaster) ----------
class DataBroadcaster:
    def __init__(self, samples_queue):
        self.samples_queue = samples_queue
        self.clients = []
        self.clients_lock = threading.Lock()
        self.running = True

    def add_client(self, conn, addr):
        with self.clients_lock:
            self.clients.append(conn)
        print(f"New client connected: {addr}")

    def remove_client(self, conn):
        with self.clients_lock:
            if conn in self.clients:
                self.clients.remove(conn)
        try:
            conn.close()
        except OSError:
            pass

    def broadcast(self, data):
        # data is a dict; we send it as a JSON line
        line = json.dumps(data) + '\n'
        to_remove = []
        with self.clients_lock:
            for conn in self.clients:
                try:
                    conn.sendall(line.encode('utf-8'))
                except (BrokenPipeError, ConnectionResetError, OSError):
                    to_remove.append(conn)
            # clean up dead connections
            for conn in to_remove:
                self.remove_client(conn)

    def run(self):
        """Main broadcaster loop: read from queue and broadcast."""
        while self.running:
            try:
                # Wait up to 0.1s for a sample, to allow checking self.running
                item = self.samples_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if isinstance(item, tuple) and item and item[0] == "error":
                # Send error to all clients
                self.broadcast({"type": "error", "message": item[1]})
                continue

            # item is (timestamp, ax, ay, az, gx, gy, gz)
            if len(item) == 7:
                sample = {
                    "type": "sample",
                    "timestamp": item[0],
                    "ax": item[1],
                    "ay": item[2],
                    "az": item[3],
                    "gx": item[4],
                    "gy": item[5],
                    "gz": item[6]
                }
                # print(sample)
                self.broadcast(sample)

    def stop(self):
        self.running = False

# ---------- TCP server ----------
class ClientHandler(StreamRequestHandler):
    def handle(self):
        # This handler runs in a separate thread for each client.
        # We simply add the client to the broadcaster and wait for data
        # (not used for receiving, only for sending).
        # The connection is kept open as long as the client stays connected.
        server = cast(BroadcastingTCPServer, self.server)
        broadcaster = server.broadcaster
        broadcaster.add_client(self.request, self.client_address)
        try:
            # Wait until the client closes the connection
            while True:
                # read something, if client disconnects we get an exception
                data = self.request.recv(1)
                if not data:
                    break
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            broadcaster.remove_client(self.request)

class BroadcastingTCPServer(ThreadingTCPServer):
    broadcaster: DataBroadcaster

    def __init__(self, server_address, RequestHandlerClass, broadcaster):
        self.broadcaster = broadcaster
        super().__init__(server_address, RequestHandlerClass)


def run_tcp_server(host, port, broadcaster):
    server = BroadcastingTCPServer((host, port), ClientHandler, broadcaster)
    print(f"Data streaming server listening on {host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()

# ---------- Main entry point ----------
def main(serial_port, baud_rate, tcp_host='0.0.0.0', tcp_port=9999, print_unparsed=False):
    # Create a thread-safe queue for samples
    samples_queue = queue.Queue()

    # Create stop event for serial worker
    stop_event = threading.Event()

    # Create broadcaster
    broadcaster = DataBroadcaster(samples_queue)

    # Start serial worker thread
    serial_thread = threading.Thread(
        target=serial_worker,
        args=(serial_port, baud_rate, samples_queue, stop_event, print_unparsed),
        daemon=True
    )
    serial_thread.start()

    # Start broadcaster thread
    broadcaster_thread = threading.Thread(
        target=broadcaster.run,
        daemon=True
    )
    broadcaster_thread.start()

    # Start TCP server (this blocks, but we can run it in a separate thread)
    tcp_server_thread = threading.Thread(
        target=run_tcp_server,
        args=(tcp_host, tcp_port, broadcaster),
        daemon=True
    )
    tcp_server_thread.start()

    print("Streaming system started. Press Ctrl+C to stop.")
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_event.set()
        broadcaster.stop()
        # Allow threads to finish
        serial_thread.join(timeout=1)
        broadcaster_thread.join(timeout=1)
        tcp_server_thread.join(timeout=1)
        print("Done.")

if __name__ == "__main__":
    # Example: change these to match your setup
    main(
        serial_port='/dev/ttyUSB0',   # or 'COM3' on Windows
        baud_rate=115200,
        tcp_host='0.0.0.0',           # listen on all interfaces
        tcp_port=9999,
        print_unparsed=True
    )
