# Readers

Minimal USB-device readers for raw data acquisition. Each module wraps one
sensor behind a uniform `connect()` / `read()` / `stream()` / `disconnect()`
interface. No gesture logic, no training, no config dataclasses — just raw
data from the hardware.

## Quick install

```bash
pip install -r Readers/requirements.txt
```

- **RFID** also needs the Mercury API C library. See [RFID setup](#rfid-setup) below.
- **mmWave** needs `numpy` (already in requirements.txt).
- **IMU** needs only `pyserial`.

## Readers at a glance

| Module      | Hardware                  | Returns                                       |
|-------------|---------------------------|-----------------------------------------------|
| `imu.py`    | ESP32 + BMI270            | `{'ax','ay','az','gx','gy','gz','t'}`        |
| `mmwave.py` | TI IWRL6432 60 GHz radar  | `(points (N,3), velocities (N,))` — numpy arrays |
| `rfid.py`   | M7E Hecto UHF RFID        | `{'epc', 'rssi', 'timestamp'}`               |

## Uniform interface

Every reader supports the same patterns:

### Single shot

```python
from Readers import IMUReader

imu = IMUReader("/dev/ttyUSB0")
imu.connect()
sample = imu.read()
print(sample["ax"], sample["ay"], sample["az"])
imu.disconnect()
```

### Continuous streaming

```python
from Readers import MMWaveReader

radar = MMWaveReader("/dev/ttyACM0")
radar.connect(config="Readers/configs/mmwave_hand_50cm.cfg")
for points, velocities in radar.stream():
    print(f"Frame: {len(points)} points")
# Or just:  for pts, vels in radar: ...
```

## Ports & auto-detection

- **RFID** auto-detects the M7E serial port by probing known USB-serial
  VID/PID pairs (CP210x, CH340, FTDI). Pass `port=` to override.
- **IMU** and **mmWave** require an explicit `port=` argument.

### Platform port names

| Platform | Typical port              |
|----------|---------------------------|
| Linux    | `/dev/ttyUSB0`, `/dev/ttyACM0` |
| macOS    | `/dev/cu.usbmodem*`, `/dev/cu.usbserial*` |
| Windows  | `COM3`, `COM4`, …         |

## RFID setup

The M7E RFID reader depends on `python-mercuryapi`, which wraps the
proprietary Mercury API C library from Novanta / Jadak.

### Option A — Pre-built wheel (fastest)

If you're on **Linux x86_64 with Python 3.11**, a pre-compiled wheel ships
in `Readers/mercury/`. Run either:

```bash
python Readers/mercury/install.py
# or
bash Readers/mercury/install.sh
```

### Option B — Build from source (macOS, Windows, other Python versions)

The Mercury API zip is bundled at `Readers/mercury/mercuryapi-BILBO-1.37.3.29-1.zip`.
The installer finds it automatically, so just run:

```bash
python Readers/mercury/install.py
```

No download needed — it builds directly from the bundled zip.

### Platform build requirements

| Platform | Prerequisites |
|----------|--------------|
| **Linux**   | `sudo apt install build-essential git` (or `dnf install make gcc git`) |
| **macOS**   | `xcode-select --install` |
| **Windows** | Visual Studio Build Tools with "Desktop development with C++" workload |

### Verify

```python
import mercury
print("OK")
```

## mmWave config files

mmWave radar needs a `.cfg` file sent at startup. Two presets ship in
`Readers/configs/`:

| File                        | Range | Use case          |
|-----------------------------|-------|-------------------|
| `mmwave_hand_50cm.cfg`      | 50 cm | Hand tracking     |
| `mmwave_point_cloud.cfg`    | 5 m   | Full point cloud  |

Pass the path to `connect(config=...)`:

```python
radar = MMWaveReader("/dev/ttyACM0")
radar.connect(config="Readers/configs/mmwave_hand_50cm.cfg")
```
