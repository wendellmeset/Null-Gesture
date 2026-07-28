"""Few-shot calibration routine.

Enables person-specific adaptation with just one example per gesture
(~30 seconds total). Extracts per-person scaling factors and updates
decision thresholds rather than retraining models.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np


GESTURE_CALIBRATION_ORDER = [
    ("pull", "PULL — Pull your hand toward your body"),
    ("push", "PUSH — Push your hand away from your body"),
    ("clockwise", "CLOCKWISE — Rotate your hand clockwise (as if turning a dial right)"),
    ("anti_clockwise", "ANTI-CLOCKWISE — Rotate your hand anti-clockwise"),
    ("left", "LEFT — Move your hand to the left"),
    ("right", "RIGHT — Move your hand to the right"),
    ("bye_bye", "BYE-BYE — Wave your hand side to side"),
    ("one_arm_boxing", "ONE ARM BOXING — Throw a punch forward with one arm"),
    ("clapping", "CLAPPING — Clap your hands together"),
    ("two_arm_boxing", "TWO-ARM BOXING — Alternate punches with both arms"),
    ("t_arms", "T-ARMS — Extend both arms out to the sides like a T"),
    ("raise_arms", "RAISE ARMS — Raise both arms above your head"),
    ("soli", "SOLI — Rub your thumb against your index finger"),
    ("opening_closing_fist", "OPENING & CLOSING FIST — Open and close your hand"),
    ("palm_up_down", "PALM UP & DOWN — Rotate your palm facing up, then down"),
]


class CalibrationRoutine:
    """Runs the few-shot calibration protocol.

    Records one example of each gesture, extracts per-person scaling
    factors for the primitive detector, and saves a profile.

    Usage::

        cal = CalibrationRoutine(imu_extractor, mmwave_extractor,
                                  primitive_detector)
        profile = cal.run(orchestrator)
        cal.save(profile, "profiles/person1.npz")
    """

    def __init__(
        self,
        imu_extractor,
        mmwave_extractor,
        primitive_detector,
        output_dir: str = "profiles",
    ):
        """
        Args:
            imu_extractor: IMUFeatureExtractor instance.
            mmwave_extractor: MMWaveFeatureExtractor instance.
            primitive_detector: PrimitiveDetector instance.
            output_dir: Directory for saving profiles.
        """
        self._imu_extractor = imu_extractor
        self._mmwave_extractor = mmwave_extractor
        self._primitive_detector = primitive_detector
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        orchestrator,
        countdown_sec: float = 2.0,
        record_duration_sec: float = 2.0,
    ) -> dict[str, Any]:
        """Run the full calibration protocol.

        Args:
            orchestrator: SensorOrchestrator instance (must be running).
            countdown_sec: Countdown time before recording each gesture.
            record_duration_sec: How long to record each gesture.

        Returns:
            Profile dict with scaling factors and per-gesture feature vectors.
        """
        print("\n" + "=" * 50)
        print("  Null-Gesture Calibration")
        print("=" * 50)
        print("  Please stand ~50cm in front of the radar")
        print("  Wear the IMU on your dominant wrist")
        print("=" * 50 + "\n")

        profile: dict[str, Any] = {
            "timestamp": time.time(),
            "scaling_factors": {},
            "gesture_prototypes": {},
            "imu_feature_names": [],
            "mmwave_feature_names": [],
        }

        all_imu_features = []
        all_mmwave_features = []
        all_velocities = []
        all_ranges = []
        all_rotation_speeds = []
        all_jerk_values = []

        for i, (gesture_name, instruction) in enumerate(GESTURE_CALIBRATION_ORDER):
            print(f"[{i + 1}/{len(GESTURE_CALIBRATION_ORDER)}] {instruction}")

            # Countdown
            for sec in range(int(countdown_sec), 0, -1):
                print(f"        Starting in {sec}...", end="\r")
                time.sleep(1.0)

            print(f"        Recording...", end="\r")

            # Record gesture
            record_start = time.time()
            imu_samples = []
            mmwave_frames = []

            while time.time() - record_start < record_duration_sec:
                # Get latest data from buffers
                imu_window = orchestrator.imu_buffer.get_window(duration_sec=0.1)
                mmwave_window = orchestrator.mmwave_buffer.get_window(duration_sec=0.1)

                if imu_window:
                    imu_samples.extend(imu_window)
                if mmwave_window:
                    mmwave_frames.extend(mmwave_window)

                time.sleep(0.02)

            print(f"        Recording... ✓ ({len(imu_samples)} IMU, {len(mmwave_frames)} mmWave)")

            # Extract features
            imu_feats = None
            mmwave_feats = None

            if imu_samples:
                imu_feats = self._imu_extractor.extract(imu_samples)
                profile["imu_feature_names"] = self._imu_extractor.feature_names()

            if mmwave_frames:
                mmwave_feats = self._mmwave_extractor.extract(mmwave_frames)
                profile["mmwave_feature_names"] = self._mmwave_extractor.feature_names()

            # Store prototype
            profile["gesture_prototypes"][gesture_name] = {
                "imu_features": imu_feats.tolist() if imu_feats is not None else None,
                "mmwave_features": mmwave_feats.tolist() if mmwave_feats is not None else None,
            }

            # Accumulate for scaling factor computation
            if imu_feats is not None:
                all_imu_features.append(imu_feats)
                # Extract key values for scaling
                accel_norm_idx = self._find_index("imu_accel_norm_mean", profile["imu_feature_names"])
                gyro_norm_idx = self._find_index("imu_gyro_norm_mean", profile["imu_feature_names"])
                jerk_norm_idx = self._find_index("imu_max_jerk", profile["imu_feature_names"])

                if accel_norm_idx is not None and accel_norm_idx < len(imu_feats):
                    all_velocities.append(abs(imu_feats[accel_norm_idx]))
                if gyro_norm_idx is not None and gyro_norm_idx < len(imu_feats):
                    all_rotation_speeds.append(abs(imu_feats[gyro_norm_idx]))
                if jerk_norm_idx is not None and jerk_norm_idx < len(imu_feats):
                    all_jerk_values.append(abs(imu_feats[jerk_norm_idx]))

            if mmwave_feats is not None:
                all_mmwave_features.append(mmwave_feats)
                range_idx = self._find_index("mmw_traj_range_max", profile["mmwave_feature_names"])
                if range_idx is not None and range_idx < len(mmwave_feats):
                    all_ranges.append(abs(mmwave_feats[range_idx]))

            print()

        # Compute scaling factors
        profile["scaling_factors"] = self._compute_scaling_factors(
            all_velocities, all_ranges, all_rotation_speeds, all_jerk_values
        )

        print("=" * 50)
        print("  Calibration complete!")
        print(f"  Scaling factors: {profile['scaling_factors']}")
        print("=" * 50 + "\n")

        return profile

    def _compute_scaling_factors(
        self,
        velocities: list[float],
        ranges: list[float],
        rotation_speeds: list[float],
        jerk_values: list[float],
    ) -> dict[str, float]:
        """Compute person-specific scaling factors from recorded data.

        Scaling factors represent how this person's motion compares to
        the "reference" person that the default thresholds were tuned for:
          - factor < 1.0: this person is slower/smaller/less jerky
          - factor = 1.0: matches reference
          - factor > 1.0: this person is faster/larger/more jerky
        """
        # Reference values (from a typical adult at 50cm range)
        REF_ACCEL = 0.15   # g
        REF_RANGE = 0.4    # m (typical arm reach)
        REF_GYRO = 45.0    # dps
        REF_JERK = 8.0     # m/s³ (arbitrary units)

        factors = {}

        if velocities:
            median_vel = float(np.median(velocities))
            factors["arm_speed"] = round(median_vel / REF_ACCEL, 3)
        else:
            factors["arm_speed"] = 1.0

        if ranges:
            median_range = float(np.median(ranges))
            factors["arm_length"] = round(median_range / REF_RANGE, 3)
        else:
            factors["arm_length"] = 1.0

        if rotation_speeds:
            median_rot = float(np.median(rotation_speeds))
            factors["rotation_speed"] = round(median_rot / REF_GYRO, 3)
        else:
            factors["rotation_speed"] = 1.0

        if jerk_values:
            median_jerk = float(np.median(jerk_values))
            factors["jerk_sensitivity"] = round(median_jerk / REF_JERK, 3)
        else:
            factors["jerk_sensitivity"] = 1.0

        factors["vibration_sensitivity"] = 1.0  # Default; can be adjusted

        return factors

    @staticmethod
    def _find_index(name: str, names: list[str]) -> int | None:
        """Find index of a feature name."""
        try:
            return names.index(name)
        except ValueError:
            return None

    def save(self, profile: dict, filename: str = "default.npz") -> str:
        """Save calibration profile to disk.

        Args:
            profile: Profile dict from run().
            filename: Output filename.

        Returns:
            Full path to saved profile.
        """
        path = self._output_dir / filename

        # Convert to numpy-friendly format
        np.savez_compressed(
            path,
            timestamp=profile["timestamp"],
            scaling_factors=profile["scaling_factors"],
            imu_feature_names=np.array(profile.get("imu_feature_names", []), dtype=object),
            mmwave_feature_names=np.array(profile.get("mmwave_feature_names", []), dtype=object),
        )

        # Save per-gesture prototypes in a separate .npz
        prototypes = profile.get("gesture_prototypes", {})
        proto_path = self._output_dir / f"{Path(filename).stem}_prototypes.npz"
        proto_data = {}
        for gesture, data in prototypes.items():
            if data.get("imu_features") is not None:
                proto_data[f"{gesture}_imu"] = np.array(data["imu_features"])
            if data.get("mmwave_features") is not None:
                proto_data[f"{gesture}_mmwave"] = np.array(data["mmwave_features"])
        if proto_data:
            np.savez_compressed(proto_path, **proto_data)

        return str(path)

    def load(self, filename: str = "default.npz") -> dict[str, Any] | None:
        """Load a saved calibration profile.

        Returns:
            Profile dict or None if not found.
        """
        path = self._output_dir / filename
        if not path.exists():
            return None

        data = np.load(path, allow_pickle=True)

        profile = {
            "timestamp": float(data.get("timestamp", 0)),
            "scaling_factors": data.get("scaling_factors", {}).item()
            if hasattr(data.get("scaling_factors", {}), "item") else {},
            "imu_feature_names": list(data.get("imu_feature_names", [])),
            "mmwave_feature_names": list(data.get("mmwave_feature_names", [])),
            "gesture_prototypes": {},
        }

        # Load prototypes
        proto_path = self._output_dir / f"{Path(filename).stem}_prototypes.npz"
        if proto_path.exists():
            proto_data = np.load(proto_path, allow_pickle=True)
            for gesture in GESTURE_CALIBRATION_ORDER:
                gname = gesture[0]
                imu_key = f"{gname}_imu"
                mmw_key = f"{gname}_mmwave"
                profile["gesture_prototypes"][gname] = {
                    "imu_features": proto_data[imu_key].tolist() if imu_key in proto_data else None,
                    "mmwave_features": proto_data[mmw_key].tolist() if mmw_key in proto_data else None,
                }

        return profile
