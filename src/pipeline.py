"""Core gesture recognition pipeline.

Orchestrates the full pipeline: sensor acquisition → preprocessing →
feature extraction → classification → fusion → detection output.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

# Resolve paths relative to the project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.features.imu_features import IMUFeatureExtractor
from src.features.mmwave_features import MMWaveFeatureExtractor
from src.features.primitives import PrimitiveDetector, GESTURE_PRIMITIVE_MAP
from src.fusion.bayesian import BayesianFusion
from src.fusion.ensemble import GestureEnsemble
from src.models.calibration import CalibrationRoutine, GESTURE_CALIBRATION_ORDER
from src.models.imu_model import IMUClassifier, GESTURE_CLASSES, GESTURE_TO_INDEX
from src.models.mmwave_model import MMWaveClassifier
from src.preprocessing.imu_processor import IMUProcessor
from src.preprocessing.mmwave_processor import MMWaveProcessor
from src.preprocessing.segmenter import GestureSegmenter
from src.sensors.orchestrator import SensorOrchestrator
from src.utils.config import load_config, load_gestures_config
from src.utils.visualization import LiveVisualizer

_log = logging.getLogger(__name__)


class GesturePipeline:
    """End-to-end gesture recognition pipeline.

    Usage::

        pipeline = GesturePipeline(config, gestures_config)
        pipeline.initialize()
        pipeline.start()

        try:
            for detection in pipeline.run():
                print(f"Detected: {detection['gesture']} "
                      f"({detection['confidence']:.2f})")
        except KeyboardInterrupt:
            pipeline.stop()
    """

    def __init__(
        self,
        config: dict[str, Any],
        gestures_config: dict[str, Any],
    ):
        """
        Args:
            config: Pipeline configuration (from config/pipeline.yaml).
            gestures_config: Gesture definitions (from config/gestures.yaml).
        """
        self._config = config
        self._gestures_config = gestures_config

        # Components (initialized in initialize())
        self._orchestrator: SensorOrchestrator | None = None
        self._imu_processor: IMUProcessor | None = None
        self._mmwave_processor: MMWaveProcessor | None = None
        self._segmenter: GestureSegmenter | None = None
        self._imu_extractor: IMUFeatureExtractor | None = None
        self._mmwave_extractor: MMWaveFeatureExtractor | None = None
        self._primitive_detector: PrimitiveDetector | None = None
        self._imu_classifier: IMUClassifier | None = None
        self._mmwave_classifier: MMWaveClassifier | None = None
        self._fusion: BayesianFusion | None = None
        self._ensemble: GestureEnsemble | None = None
        self._visualizer: LiveVisualizer | None = None
        self._calibration_routine: CalibrationRoutine | None = None

        # State
        self._running = False
        self._calibrated = False

    def initialize(self, with_visualization: bool = False) -> bool:
        """Initialize all pipeline components.

        Args:
            with_visualization: Enable live matplotlib plotting.

        Returns:
            True if initialization succeeded.
        """
        sensor_cfg = self._config["sensors"]
        segmenter_cfg = self._config["segmenter"]
        fusion_cfg = self._config["fusion"]
        output_cfg = self._config["output"]
        model_cfg = self._config["models"]

        # ── Sensor orchestrator ─────────────────────────────────────
        # Resolve mmWave config path relative to project root
        mmwave_cfg_rel = sensor_cfg["mmwave"]["config"]
        mmwave_cfg_abs = str(_PROJECT_ROOT / mmwave_cfg_rel) if mmwave_cfg_rel else None

        self._orchestrator = SensorOrchestrator(
            imu_port=sensor_cfg["imu"]["port"],
            imu_baud=sensor_cfg["imu"]["baud"],
            mmwave_port=sensor_cfg["mmwave"]["port"],
            mmwave_baud=sensor_cfg["mmwave"]["baud"],
            mmwave_config=mmwave_cfg_abs,
            imu_buffer_sec=sensor_cfg["imu"].get("buffer_size_sec", 2.0),
            mmwave_buffer_sec=sensor_cfg["mmwave"].get("buffer_size_sec", 2.0),
        )

        # ── Preprocessing ───────────────────────────────────────────
        self._imu_processor = IMUProcessor(sample_rate=100.0)
        self._mmwave_processor = MMWaveProcessor()

        mw = segmenter_cfg["metric_weights"]
        self._segmenter = GestureSegmenter(
            imu_accel_weight=mw.get("imu_accel", 0.4),
            imu_gyro_weight=mw.get("imu_gyro", 0.3),
            mmwave_range_weight=mw.get("mmwave_range", 0.3),
            noise_ema_alpha=segmenter_cfg.get("noise_ema_alpha", 0.01),
            threshold_on_mult=segmenter_cfg.get("threshold_on_multiplier", 2.5),
            threshold_off_mult=segmenter_cfg.get("threshold_off_multiplier", 1.8),
            min_duration_ms=segmenter_cfg.get("min_gesture_duration_ms", 200),
            max_duration_ms=segmenter_cfg.get("max_gesture_duration_ms", 5000),
            quiet_confirm_ms=segmenter_cfg.get("quiet_confirm_ms", 300),
        )

        # ── Feature extraction ──────────────────────────────────────
        self._imu_extractor = IMUFeatureExtractor(sample_rate=100.0)
        self._mmwave_extractor = MMWaveFeatureExtractor(radar_frame_rate=20.0)

        self._primitive_detector = PrimitiveDetector(
            gestures_config=self._gestures_config,
        )
        self._primitive_detector.set_feature_names(
            self._imu_extractor.feature_names(),
            self._mmwave_extractor.feature_names(),
        )

        # ── Classifiers ─────────────────────────────────────────────
        self._imu_classifier = IMUClassifier(
            feature_names=self._imu_extractor.feature_names(),
        )
        self._imu_classifier.load(
            str(_PROJECT_ROOT / model_cfg.get("imu_model_path", "models/imu_rf.pkl"))
        )

        self._mmwave_classifier = MMWaveClassifier(
            feature_names=self._mmwave_extractor.feature_names(),
            use_cnn=True,
        )
        self._mmwave_classifier.load(
            str(_PROJECT_ROOT / model_cfg.get("mmwave_model_path", "models/mmwave_hybrid.pkl"))
        )

        # ── Fusion ──────────────────────────────────────────────────
        self._fusion = BayesianFusion()

        self._ensemble = GestureEnsemble(
            self_stay_prob=fusion_cfg.get("transition_self_stay", 0.92),
            to_idle_prob=fusion_cfg.get("transition_to_idle", 0.15),
            idle_stay_prob=fusion_cfg.get("idlestay", 0.98),
            detection_threshold=fusion_cfg.get("detection_threshold", 0.7),
            debounce_frames=fusion_cfg.get("debounce_frames", 3),
            cooldown_ms=fusion_cfg.get("cooldown_ms", 600),
            boxing_cooldown_ms=fusion_cfg.get("boxing_cooldown_ms", 300),
        )

        # ── Calibration ─────────────────────────────────────────────
        cal_profile_path = model_cfg.get("calibration_profile", "profiles/default.npz")
        cal_dir = str(_PROJECT_ROOT / Path(cal_profile_path).parent)
        self._calibration_routine = CalibrationRoutine(
            imu_extractor=self._imu_extractor,
            mmwave_extractor=self._mmwave_extractor,
            primitive_detector=self._primitive_detector,
            output_dir=cal_dir,
        )

        # ── Visualization (optional) ────────────────────────────────
        if with_visualization or output_cfg.get("visualization", False):
            self._visualizer = LiveVisualizer()
            self._visualizer.start()

        # ── Load calibration profile if available ──────────────────
        profile = self._calibration_routine.load("default.npz")
        if profile is not None and profile.get("scaling_factors"):
            self._primitive_detector.update_scaling(profile["scaling_factors"])
            self._calibrated = True
            _log.info("Loaded calibration profile: %s", profile["scaling_factors"])

        return True

    def calibrate(self, countdown_sec: float = 2.0, record_duration: float = 2.0) -> bool:
        """Run the few-shot calibration protocol.

        The orchestrator must already be started (call start() first,
        then calibrate()).

        Args:
            countdown_sec: Seconds countdown before each recording.
            record_duration: Seconds to record each gesture.

        Returns:
            True if calibration completed successfully.
        """
        if self._orchestrator is None or not self._orchestrator.running:
            _log.error("Cannot calibrate: orchestrator not running.")
            return False

        if self._calibration_routine is None:
            _log.error("Calibration routine not initialized.")
            return False

        profile = self._calibration_routine.run(
            orchestrator=self._orchestrator,
            countdown_sec=countdown_sec,
            record_duration_sec=record_duration,
        )

        self._calibration_routine.save(profile, "default.npz")

        if self._primitive_detector is not None:
            self._primitive_detector.update_scaling(profile["scaling_factors"])

        self._calibrated = True
        return True

    def start(self) -> bool:
        """Start sensor acquisition.

        Returns:
            True if sensors connected and streaming.
        """
        if self._orchestrator is None:
            _log.error("Pipeline not initialized. Call initialize() first.")
            return False

        self._running = self._orchestrator.start()
        return self._running

    def stop(self) -> None:
        """Stop sensor acquisition and clean up."""
        self._running = False
        if self._orchestrator:
            self._orchestrator.stop()
        if self._visualizer:
            self._visualizer.stop()

    def run(self) -> Any:
        """Run the main detection loop.

        Yields:
            Detection result dicts as gestures are recognized.

        Usage::

            for detection in pipeline.run():
                print(f"{detection['gesture']}: {detection['confidence']:.2f}")
        """
        if not self._running:
            _log.error("Pipeline not started. Call start() first.")
            return

        _log.info("Gesture detection loop starting...")
        print("\nNull-Gesture running. Perform gestures. Press Ctrl+C to stop.\n")

        while self._running:
            try:
                # ── 1. Get latest sensor data ───────────────────────
                imu_sample_raw = self._orchestrator.imu_buffer.latest()
                mmwave_raw = self._orchestrator.mmwave_buffer.latest()

                # Check for fresh data
                imu_ts = self._orchestrator.imu_buffer.latest_timestamp()
                mmwave_ts = self._orchestrator.mmwave_buffer.latest_timestamp()
                now = time.time()

                # Skip if data is too old (> 1 second)
                imu_fresh = imu_ts is not None and (now - imu_ts) < 1.0
                mmwave_fresh = mmwave_ts is not None and (now - mmwave_ts) < 1.0

                if not imu_fresh and not mmwave_fresh:
                    time.sleep(0.01)
                    continue

                # ── 2. Preprocess ───────────────────────────────────
                imu_processed = None
                mmwave_processed = None

                if imu_fresh and imu_sample_raw is not None:
                    imu_processed = self._imu_processor.process(imu_sample_raw)

                if mmwave_fresh and mmwave_raw is not None:
                    points, velocities = mmwave_raw
                    mmwave_processed = self._mmwave_processor.process(points, velocities)

                # ── 3. Segmentation ─────────────────────────────────
                window = self._segmenter.update(imu_processed, mmwave_processed)

                # ── 4. If gesture window detected, run classification ──
                detection = None
                if window is not None:
                    detection = self._process_gesture_window(window)

                # Yield detection
                if detection is not None:
                    detection["latency_ms"] = (time.time() - window.offset_time) * 1000.0
                    yield detection

                    # Print detection
                    if self._config["output"].get("print_gestures", True):
                        print(
                            f"  ➤ {detection['gesture']:>20s}  "
                            f"conf={detection['confidence']:.2f}  "
                            f"latency={detection['latency_ms']:.0f}ms  "
                            f"source={detection.get('source', 'unknown')}"
                        )

                # ── 5. Visualization ────────────────────────────────
                if self._visualizer is not None:
                    gesture_label = detection["gesture"] if detection else None
                    self._visualizer.update(imu_processed, mmwave_processed, gesture_label)

                # Small sleep to prevent busy-waiting
                time.sleep(0.005)

            except KeyboardInterrupt:
                break
            except Exception as e:
                _log.error("Pipeline error: %s", e, exc_info=True)
                time.sleep(0.1)

        _log.info("Gesture detection loop ended.")

    def _process_gesture_window(self, window) -> dict | None:
        """Extract features, classify, and fuse for one gesture window.

        Args:
            window: GestureWindow from the segmenter.

        Returns:
            Detection result dict or None.
        """
        # Get data within the gesture window
        onset = window.onset_time
        offset = window.offset_time

        # Get IMU samples within window
        imu_window_raw = self._orchestrator.imu_buffer.get_window_with_timestamps(
            duration_sec=None  # Get all available, then filter
        )
        imu_samples_in_window = [
            s for ts, s in imu_window_raw
            if onset - 0.15 <= ts <= offset + 0.15
        ]
        # Process IMU samples if not already processed
        imu_processed_list = []
        for sample in imu_samples_in_window:
            processed = self._imu_processor.process(sample)
            if processed is not None:
                imu_processed_list.append(processed)

        # Get mmWave frames within window
        mmwave_raw_all = self._orchestrator.mmwave_buffer.get_window_with_timestamps(
            duration_sec=None
        )
        mmwave_frames_in_window = []
        for ts, (pts, vels) in mmwave_raw_all:
            if onset - 0.15 <= ts <= offset + 0.15:
                processed = self._mmwave_processor.process(pts, vels)
                if processed is not None:
                    mmwave_frames_in_window.append(processed)

        # ── Feature extraction ────────────────────────────────────
        imu_features = None
        mmwave_features = None

        if imu_processed_list:
            imu_features = self._imu_extractor.extract(imu_processed_list)

        if mmwave_frames_in_window:
            mmwave_features = self._mmwave_extractor.extract(mmwave_frames_in_window)

        if imu_features is None and mmwave_features is None:
            return None

        # ── Primitive detection ────────────────────────────────────
        primitive_detections = self._primitive_detector.detect(imu_features, mmwave_features)

        # Build primitive scores for each gesture
        primitive_scores = {}
        for gesture_name, required_primitives in GESTURE_PRIMITIVE_MAP.items():
            primitive_values = [p.value for p in required_primitives]
            detected_values = [d.primitive.value for d in primitive_detections]

            # Score: what fraction of required primitives were detected
            matches = sum(1 for pv in primitive_values if pv in detected_values)
            score = matches / len(primitive_values) if primitive_values else 0.0
            primitive_scores[gesture_name] = score

        # ── Per-sensor classification ──────────────────────────────
        if imu_features is not None and imu_features.any():
            imu_probs, imu_conf = self._imu_classifier.predict(imu_features)
        else:
            imu_probs = np.ones(15) / 15
            imu_conf = 0.0

        if mmwave_features is not None and mmwave_features.any():
            # Build R-D heatmap stack if available
            rd_heatmaps = None
            if mmwave_frames_in_window:
                heatmaps = [
                    f.get("rd_heatmap") for f in mmwave_frames_in_window
                    if f.get("rd_heatmap") is not None
                ]
                if heatmaps:
                    rd_heatmaps = np.stack(heatmaps)
            mmwave_probs, mmwave_conf = self._mmwave_classifier.predict(
                mmwave_features, rd_heatmaps
            )
        else:
            mmwave_probs = np.ones(15) / 15
            mmwave_conf = 0.0

        # ── Fusion ─────────────────────────────────────────────────
        fused_probs = self._fusion.fuse_with_primitive_bias(
            imu_probs, imu_conf,
            mmwave_probs, mmwave_conf,
            primitive_scores=primitive_scores,
            primitive_weight=0.15,
        )

        # ── Ensemble / HMM filter ──────────────────────────────────
        detection = self._ensemble.update(fused_probs)

        if detection is not None:
            # Enrich with sensor contribution info
            detection["source"] = window.source
            detection["sensor_contributions"] = {
                "imu": float(imu_conf),
                "mmwave": float(mmwave_conf),
            }
            detection["primitives"] = [
                {"primitive": d.primitive.value, "confidence": d.confidence}
                for d in primitive_detections[:3]
            ]

        return detection

    @property
    def running(self) -> bool:
        return self._running

    @property
    def calibrated(self) -> bool:
        return self._calibrated
