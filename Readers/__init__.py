"""Readers: Minimal USB-device readers for raw data acquisition.

Each module provides a single class with a uniform interface:
    connect()    -> bool
    read()       -> latest data point (or None)
    disconnect() -> None

Usage::

    from Readers import IMUReader
    imu = IMUReader(port="/dev/ttyUSB0")
    imu.connect()
    sample = imu.read()  # {'ax': 0.01, 'ay': ..., 'gz': ...}
    imu.disconnect()

All readers support iteration::

    from Readers import MMWaveReader
    radar = MMWaveReader()
    radar.connect()
    for points, velocities in radar:
        print(f"{len(points)} points detected")
"""  # noqa: N999

from Readers.imu import IMUReader
from Readers.mmwave import MMWaveReader
from Readers.rfid import RFIDReader
from Readers.uwb import UWBReader

__all__ = ["IMUReader", "MMWaveReader", "RFIDReader", "UWBReader"]
