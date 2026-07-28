"""Main Gesture Detection Pipeline.

Orchestrates the full pipeline:
  Sensor Multiplexer → Preprocessor → Feature Extractors → Detectors → Fusion → Temporal

Provides both a synchronous `run()` interface and an async generator.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

from src.detectors.hand_gestures import HandGestureDetector
from src.detectors.micro_doppler import MicroDopplerDetector
from src.detectors.motion_gestures import MotionGestureDetector
from src.detectors.posture_gestures import PostureGestureDetector
from src.detectors.proximity_gestures import ProximityGestureDetector
from src.features.uwb_features import RFIDTracker
from src.fusion.dempster_shafer import DempsterShaferFusion
from src.fusion.temporal import GestureEvent, TemporalStateMachine
from src.sensor.multiplexer import SensorMultiplexer
from src.sensor.preprocessor import PreprocessedFrame, SensorPreprocessor

_log = logging.getLogger(__name__)


class GesturePipeline:
    """End-to-end gesture detection pipeline.

    Usage::

        pipeline = GesturePipeline(
            imu_port="/dev/ttyUSB0",
            mmwave_port="/dev/ttyACM0",
            mmwave_config="Readers/configs/mmwave_hand_50cm.cfg",
            uwb_initiator="/dev/ttyACM1",
            uwb_responder="/dev/ttyACM2",
        )

        pipeline.start()

        for event in pipeline.events():
            print(f"{event.gesture}: {event.confidence:.2f}")

        pipeline.stop()
    """

    def __init__(
        self,
        # Sensor ports
        imu_port: str | None = None,
        mmwave_port: str | None = None,
        mmwave_config: str | None = None,
        uwb_initiator: str | None = None,
        uwb_responder: str | None = None,
        rfid_port: str | None = None,
        # RFID tag mapping
        tag_map: dict[str, str] | None = None,
        # Model paths
        models_dir: str | None = None,
        # Fusion parameters
        belief_threshold: float = 0.6,
        conflict_threshold: float = 0.3,
        # Pipeline tuning
        calibration_frames: int = 30,
        imu_window_samples: int = 30,
        mmwave_window_frames: int = 10,
        # Verbose output
        verbose: bool = False,
    ) -> None:
        self._verbose = verbose

        # ── Sensor layer ────────────────────────────────────────────
        self._multiplexer = SensorMultiplexer(
            imu_port=imu_port,
            mmwave_port=mmwave_port,
            mmwave_config=mmwave_config,
            uwb_initiator=uwb_initiator,
            uwb_responder=uwb_responder,
            rfid_port=rfid_port,
        )

        self._preprocessor = SensorPreprocessor(
            tag_map=tag_map,
            calibration_frames=calibration_frames,
        )

        # ── RFID tracker (shared across detectors) ──────────────────
        self._rfid_tracker = RFIDTracker(tag_map=tag_map)

        # ── Detectors ───────────────────────────────────────────────
        self._motion_detector = MotionGestureDetector(
            window_samples=imu_window_samples,
        )
        self._posture_detector = PostureGestureDetector(
            window_frames=mmwave_window_frames,
            rfid_tracker=self._rfid_tracker,
        )
        self._doppler_detector = MicroDopplerDetector(
            window_frames=mmwave_window_frames,
        )
        self._proximity_detector = ProximityGestureDetector(
            window_samples=imu_window_samples,
        )
        self._hand_detector = HandGestureDetector(
            mmwave_window_frames=mmwave_window_frames,
            imu_window_samples=imu_window_samples,
        )

        # ── Fusion ──────────────────────────────────────────────────
        self._fusion = DempsterShaferFusion(
            belief_threshold=belief_threshold,
            conflict_threshold=conflict_threshold,
        )
        self._temporal = TemporalStateMachine(
            onset_threshold=belief_threshold,
            offset_threshold=0.3,
            min_duration_ms=200,
            cooldown_ms=300,
        )

        # ── State ───────────────────────────────────────────────────
        self._running = False
        self._frame_count = 0
        self._start_time = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> dict[str, bool]:
        """Start all sensors and enter the run loop.

        Returns sensor connection status dict.
        """
        status = self._multiplexer.start()
        self._running = True
        self._start_time = time.time()

        if self._verbose:
            _log.info("Sensor status: %s", status)
            _log.info("Calibrating (%d frames)...", self._preprocessor.calibration_frames)

        return status

    def calibrate(self, timeout_s: float = 5.0) -> bool:
        """Drive frames through preprocessor until calibration completes.

        Must be called after start(), before events().
        Returns True if calibration completed within timeout.
        """
        deadline = time.time() + timeout_s
        while not self._preprocessor.is_calibrated():
            if time.time() > deadline:
                _log.warning("Calibration timed out after %.1fs", timeout_s)
                return False
            frame = self._multiplexer.read_frame()
            self._preprocessor.process(frame)
            time.sleep(0.005)
        return True

    def stop(self) -> None:
        """Stop all sensors and clean up."""
        self._running = False
        self._multiplexer.stop()

        if self._verbose:
            elapsed = time.time() - self._start_time
            _log.info(
                "Pipeline stopped. %d frames processed in %.1fs (%.1f Hz avg).",
                self._frame_count,
                elapsed,
                self._frame_count / max(elapsed, 0.001),
            )

    # ── Event stream ─────────────────────────────────────────────────────

    def events(self) -> Iterator[GestureEvent]:
        """Generator yielding GestureEvent objects in real time.

        Yields only when a gesture starts, remains active, or ends.
        """
        if not self._running:
            self.start()

        for frame in self._multiplexer:
            if not self._running:
                break

            self._frame_count += 1
            event = self._process_frame(frame)
            if event is not None:
                yield event

    def run_with_callback(self, callback) -> None:
        """Run pipeline, calling callback(event) for each detected gesture."""
        for event in self.events():
            callback(event)

    # ── Frame processing ─────────────────────────────────────────────────

    def _process_frame(self, raw_frame) -> GestureEvent | None:
        """Process a single raw sensor frame through the full pipeline."""
        # 1. Preprocess
        pf = self._preprocessor.process(raw_frame)

        # 2. Push to detectors
        self._push_to_detectors(pf)

        # 3. Collect evidence from each detector
        evidence_collected = 0

        # Motion (IMU)
        motion_result = self._motion_detector.detect()
        if motion_result.get("unknown", 1.0) < 0.95:
            self._fusion.add_evidence("motion", motion_result)
            evidence_collected += 1

        # Posture (mmWave + RFID)
        posture_result = self._posture_detector.detect()
        if posture_result.get("unknown", 1.0) < 0.95:
            self._fusion.add_evidence("posture", posture_result)
            evidence_collected += 1

        # Micro-Doppler (Soli)
        doppler_result = self._doppler_detector.detect()
        if doppler_result.get("unknown", 1.0) < 0.95:
            self._fusion.add_evidence("micro_doppler", doppler_result)
            evidence_collected += 1

        # Proximity (UWB + IMU)
        proximity_result = self._proximity_detector.detect()
        if proximity_result.get("unknown", 1.0) < 0.95:
            self._fusion.add_evidence("proximity", proximity_result)
            evidence_collected += 1

        # Hand gestures (mmWave + IMU)
        hand_result = self._hand_detector.detect()
        if hand_result.get("unknown", 1.0) < 0.95:
            self._fusion.add_evidence("hand", hand_result)
            evidence_collected += 1

        # ── Cross-detector gate: bye-bye requires UWB distance stability ──
        # Only gate when proximity has an opinion (UWB data is flowing).
        # If proximity is all-unknown (no UWB data), don't suppress motion.
        proximity_has_opinion = proximity_result.get("unknown", 1.0) < 0.9
        if motion_result.get("bye_bye", 0) > 0.3 and proximity_has_opinion:
            proximity_bye = proximity_result.get("bye_bye", 0)
            if proximity_bye < 0.1:
                suppressed = motion_result.get("bye_bye", 0)
                motion_result["bye_bye"] = 0.0
                motion_result["unknown"] = min(1.0, motion_result.get("unknown", 0) + suppressed)

        if evidence_collected == 0:
            self._fusion.reset()
            return None

        # 4. Fuse evidence
        fused = self._fusion.fuse()

        # 5. Temporal filtering
        event = self._temporal.update(fused)

        # Logging
        if self._verbose and event is not None and event.phase in ("start", "end"):
            _log.info(
                "[%s] %s (conf=%.2f, dur=%.2fs, sensors=%s)",
                event.phase.upper(),
                event.gesture,
                event.confidence,
                event.duration,
                event.contributing_sensors,
            )

        return event

    def _push_to_detectors(self, pf: PreprocessedFrame) -> None:
        """Push preprocessed data into all detectors."""

        # Motion detector (IMU)
        if pf.imu_gyro is not None:
            gx, gy, gz = pf.imu_gyro
            ax, ay, az = pf.imu_linear_accel or (0.0, 0.0, 0.0)
            self._motion_detector.push(ax, ay, az, gx, gy, gz)

        # mmWave-dependent detectors
        clusters = pf.mmwave_clusters or []
        velocities = pf.mmwave_raw_velocities
        rfid_hands = pf.rfid_hands or set()

        self._posture_detector.push(clusters, velocities, rfid_hands)
        self._doppler_detector.push(clusters, velocities)

        # Proximity detector (UWB + IMU)
        imu_ax, imu_ay, imu_az = pf.imu_linear_accel or (None, None, None)
        self._proximity_detector.push(
            pf.uwb_distance, pf.uwb_velocity,
            imu_ax, imu_ay, imu_az,
        )

        # Hand gesture detector
        self._hand_detector.push(
            clusters, velocities,
            imu_ax, imu_ay, imu_az,
            pf.imu_gyro[0] if pf.imu_gyro else None,
            pf.imu_gyro[1] if pf.imu_gyro else None,
            pf.imu_gyro[2] if pf.imu_gyro else None,
        )

        # RFID tracker
        if pf.rfid_tags:
            self._rfid_tracker.update(pf.rfid_tags, pf.timestamp)

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def running(self) -> bool:
        return self._running

    @property
    def calibrated(self) -> bool:
        return self._preprocessor.is_calibrated()
