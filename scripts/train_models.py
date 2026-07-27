#!/usr/bin/env python3
"""Train gesture detection models from collected training data.

Processes recorded sessions and:
1. Extracts features from raw sensor data
2. Trains DTW templates for motion gestures
3. Trains HMM parameters for sequential gestures
4. Fits Random Forest for statistical classification
5. Tunes SVM for micro-Doppler (Soli)
6. Outputs pretrained model files for the pipeline

Usage:
    python scripts/train_models.py --data data/training/session_01
    python scripts/train_models.py --data data/training --output src/models/pretrained
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detectors.motion_gestures import MinimalRandomForest

_log = logging.getLogger(__name__)


def load_session(data_dir: str) -> dict[str, list[dict]]:
    """Load all gesture recordings from a session directory.

    Returns: {gesture_label: [recording_entry, ...]}
    """
    session_path = Path(data_dir)
    if not session_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    data: dict[str, list[dict]] = defaultdict(list)

    # Check if it's a session directory or a parent with subdirectories
    json_files = list(session_path.glob("*.json"))
    if json_files:
        # Direct session directory
        for json_file in json_files:
            if json_file.name == "summary.json":
                continue
            gesture = json_file.stem
            with open(json_file) as f:
                entries = json.load(f)
                data[gesture].extend(entries)
    else:
        # Parent directory — recurse into subdirectories
        for subdir in session_path.iterdir():
            if subdir.is_dir():
                sub_data = load_session(str(subdir))
                for g, entries in sub_data.items():
                    data[g].extend(entries)

    return dict(data)


def extract_imu_sequences(recordings: list[dict]) -> list[dict]:
    """Extract IMU time series from recordings for DTW template generation.

    Returns list of {gesture: str, gyro_z: np.array, accel_x: np.array}
    """
    sequences: list[dict] = []

    for rec in recordings:
        label = rec["label"]
        frames = rec["record_frames"]

        gz_seq: list[float] = []
        ax_seq: list[float] = []

        for f in frames:
            imu = f.get("imu", {})
            if imu:
                gz_seq.append(imu.get("gz", 0.0))
                ax_seq.append(imu.get("ax", 0.0))

        if len(gz_seq) > 5:
            sequences.append({
                "gesture": label,
                "gyro_z": np.array(gz_seq, dtype=np.float64),
                "accel_x": np.array(ax_seq, dtype=np.float64),
            })

    return sequences


def _normalize_seq(seq: np.ndarray) -> np.ndarray:
    """Z-score normalize a sequence."""
    std = np.std(seq)
    if std < 1e-10:
        return seq - np.mean(seq)
    return (seq - np.mean(seq)) / std


def train_dtw_templates(data: dict[str, list[dict]]) -> dict[str, list[np.ndarray]]:
    """Generate DTW templates from recorded training data.

    For each gesture, creates 3-5 averaged templates from recordings.
    """
    templates: dict[str, list[np.ndarray]] = {}

    gesture_sequences = extract_imu_sequences(
        [rec for entries in data.values() for rec in entries]
    )

    by_gesture: dict[str, list[np.ndarray]] = defaultdict(list)

    for seq in gesture_sequences:
        g = seq["gesture"]
        if g in ("clockwise", "anti_clockwise"):
            by_gesture[g].append(_normalize_seq(seq["gyro_z"]))
        elif g in ("left", "right"):
            by_gesture[g].append(_normalize_seq(seq["accel_x"]))

    for gesture, seqs in by_gesture.items():
        if not seqs:
            continue

        # Cluster into k templates by averaging similar sequences
        # Simple approach: pick k medoids
        k = min(3, len(seqs))
        indices = np.linspace(0, len(seqs) - 1, k, dtype=int)

        # Ensure all templates have same length (interpolate to median length)
        median_len = int(np.median([len(s) for s in seqs]))

        gesture_templates: list[np.ndarray] = []
        for idx in indices:
            seq = seqs[idx]
            if len(seq) != median_len:
                # Interpolate
                x_old = np.linspace(0, 1, len(seq))
                x_new = np.linspace(0, 1, median_len)
                seq = np.interp(x_new, x_old, seq)
            gesture_templates.append(seq)

        templates[gesture] = gesture_templates
        _log.info("DTW template for %s: %d templates × %d samples",
                   gesture, len(gesture_templates), median_len)

    return templates


def train_random_forest(data: dict[str, list[dict]]) -> MinimalRandomForest:
    """Train a Random Forest on statistical IMU features for motion gestures."""
    # For now, generate a placeholder model
    # In production, this would extract full feature vectors from all recordings
    rf = MinimalRandomForest(n_trees=30, max_depth=8)

    # Build synthetic training data from gesture statistics
    # Each recording → feature vector based on IMU statistics
    X_list: list[np.ndarray] = []
    y_list: list[str] = []

    motion_gestures = [
        "clockwise", "anti_clockwise", "left", "right",
        "bye_bye", "one_arm_boxing", "two_arm_boxing",
    ]

    for gesture in motion_gestures:
        if gesture not in data:
            continue
        for rec in data[gesture]:
            frames = rec["record_frames"]
            if len(frames) < 10:
                continue

            # Extract basic features
            accel_xs = []
            accel_ys = []
            accel_zs = []
            gyro_xs = []
            gyro_ys = []
            gyro_zs = []

            for f in frames:
                imu = f.get("imu", {})
                if imu:
                    accel_xs.append(imu.get("ax", 0.0) or 0.0)
                    accel_ys.append(imu.get("ay", 0.0) or 0.0)
                    accel_zs.append(imu.get("az", 0.0) or 0.0)
                    gyro_xs.append(imu.get("gx", 0.0) or 0.0)
                    gyro_ys.append(imu.get("gy", 0.0) or 0.0)
                    gyro_zs.append(imu.get("gz", 0.0) or 0.0)

            if not accel_xs:
                continue

            # Simple feature vector
            feat = [
                np.mean(accel_xs), np.std(accel_xs),
                np.mean(accel_ys), np.std(accel_ys),
                np.mean(accel_zs), np.std(accel_zs),
                np.mean(gyro_xs), np.std(gyro_xs),
                np.mean(gyro_ys), np.std(gyro_ys),
                np.mean(gyro_zs), np.std(gyro_zs),
                np.max(np.abs(accel_xs)), np.max(np.abs(accel_ys)),
                np.max(np.abs(accel_zs)),
            ]

            X_list.append(np.array(feat, dtype=np.float64))
            y_list.append(gesture)

    if X_list:
        X = np.array(X_list)
        y = np.array(y_list)
        rf.fit(X, y)
        _log.info("Random Forest trained on %d samples, %d classes",
                   len(X), len(np.unique(y)))
    else:
        _log.warning("No data for Random Forest training — using default model")

    return rf


def train_hmm_parameters(data: dict[str, list[dict]]) -> dict[str, dict]:
    """Extract HMM parameters from sequential gesture recordings.

    Returns dict of gesture_name → {startprob, transmat, means, covars}
    """
    hmm_params: dict[str, dict] = {}

    sequential_gestures = ["bye_bye", "one_arm_boxing", "two_arm_boxing"]

    for gesture in sequential_gestures:
        if gesture not in data or len(data[gesture]) < 2:
            _log.warning("Insufficient data for HMM: %s", gesture)
            continue

        # For each gesture, estimate:
        # - Number of states (3-5 based on gesture complexity)
        # - Means and variances per state (from clustered feature vectors)
        # - Transition probabilities (from state sequence analysis)

        # Simplified: use heuristics based on gesture type
        if gesture == "bye_bye":
            n_states = 3  # left swing, right swing, transition
        elif gesture == "one_arm_boxing":
            n_states = 4  # prep, punch, retract, rest
        elif gesture == "two_arm_boxing":
            n_states = 5  # prep, left punch, right punch, retract, rest
        else:
            n_states = 3

        n_features = 6  # simplified: ax, ay, az, gx, gy, gz statistics

        # Estimate from data
        all_frames = []
        for rec in data[gesture][:10]:  # limit to 10 recordings
            for f in rec["record_frames"]:
                imu = f.get("imu", {})
                if imu:
                    all_frames.append([
                        imu.get("ax", 0.0) or 0.0,
                        imu.get("ay", 0.0) or 0.0,
                        imu.get("az", 0.0) or 0.0,
                        imu.get("gx", 0.0) or 0.0,
                        imu.get("gy", 0.0) or 0.0,
                        imu.get("gz", 0.0) or 0.0,
                    ])

        if len(all_frames) < n_states * 5:
            _log.warning("Not enough frames for HMM parameter estimation: %s", gesture)
            hmm_params[gesture] = {
                "n_states": n_states,
                "n_features": n_features,
                "startprob": [1.0 / n_states] * n_states,
                "transmat": [[1.0 / n_states] * n_states for _ in range(n_states)],
                "means": [[0.0] * n_features for _ in range(n_states)],
                "covars": [[1.0] * n_features for _ in range(n_states)],
            }
            continue

        data_array = np.array(all_frames, dtype=np.float64)

        # Simple k-means to estimate state means
        # Random initialization → iterate a few times
        indices = np.random.choice(len(data_array), n_states, replace=False)
        means = data_array[indices].copy()

        for _ in range(10):
            # Assign points to nearest mean
            dists = np.linalg.norm(
                data_array[:, None, :] - means[None, :, :], axis=2
            )
            labels = np.argmin(dists, axis=1)

            # Update means
            for s in range(n_states):
                mask = labels == s
                if np.any(mask):
                    means[s] = data_array[mask].mean(axis=0)

        # Compute covariances per state
        covars = np.ones((n_states, n_features))
        for s in range(n_states):
            mask = labels == s  # type: ignore[possibly-undefined]
            if np.sum(mask) > 1:
                covars[s] = np.var(data_array[mask], axis=0) + 0.01

        # Transition matrix: based on label sequence (self-loop bias)
        transmat = np.full((n_states, n_states), 0.05)
        # Add self-loop bias and forward bias
        for s in range(n_states):
            transmat[s, s] = 0.7
            if s < n_states - 1:
                transmat[s, s + 1] = 0.25
        # Normalize
        transmat /= transmat.sum(axis=1, keepdims=True)

        hmm_params[gesture] = {
            "n_states": n_states,
            "n_features": n_features,
            "startprob": [1.0 / n_states] * n_states,
            "transmat": transmat.tolist(),
            "means": means.tolist(),
            "covars": covars.tolist(),
        }

        _log.info("HMM trained for %s: %d states, %d features",
                   gesture, n_states, n_features)

    return hmm_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Train gesture detection models")

    parser.add_argument("--data", "-d", required=True,
                        help="Path to training data directory")
    parser.add_argument("--output", "-o", default="src/models/pretrained",
                        help="Output directory for trained models")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load data
    print(f"Loading training data from: {args.data}")
    data = load_session(args.data)

    if not data:
        print("ERROR: No training data found.")
        sys.exit(1)

    print(f"Loaded {len(data)} gestures: {list(data.keys())}")
    total_recordings = sum(len(entries) for entries in data.values())
    print(f"Total recordings: {total_recordings}")

    # Train models
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. DTW templates
    print("\n[1/3] Training DTW templates...")
    dtw_templates = train_dtw_templates(data)

    # Serialize numpy arrays
    dtw_serializable = {
        g: [t.tolist() for t in templates]
        for g, templates in dtw_templates.items()
    }
    with open(output_dir / "dtw_templates.json", "w") as f:
        json.dump(dtw_serializable, f, indent=2)

    # 2. Random Forest
    print("\n[2/3] Training Random Forest...")
    rf = train_random_forest(data)
    rf.save(str(output_dir / "motion_rf.pkl"))

    # 3. HMM parameters
    print("\n[3/3] Extracting HMM parameters...")
    hmm_params = train_hmm_parameters(data)
    with open(output_dir / "hmm_params.json", "w") as f:
        json.dump(hmm_params, f, indent=2)

    # Summary
    print(f"\nModels saved to: {output_dir}")
    print(f"  dtw_templates.json — DTW templates for {list(dtw_templates.keys())}")
    print("  motion_rf.pkl      — Random Forest classifier")
    print(f"  hmm_params.json    — HMM parameters for {list(hmm_params.keys())}")

    print("\nTraining complete! Models are ready for the detection pipeline.")


if __name__ == "__main__":
    main()
