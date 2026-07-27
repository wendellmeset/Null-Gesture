#!/usr/bin/env python3
"""Gesture diagnostic harness.

Prompts you to perform each gesture, records 3s of sensor data,
runs every detector on it, and prints detailed internal state
so you can see exactly why a gesture fires or misses.

Usage:
    python scripts/diagnose.py                      # all gestures
    python scripts/diagnose.py --gesture bye_bye    # single gesture
    python scripts/diagnose.py --gesture clockwise,left,right
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sensor.multiplexer import SensorMultiplexer, SensorFrame
from src.sensor.preprocessor import SensorPreprocessor, PreprocessedFrame
from src.features.imu_features import IMUWindow
from src.features.mmwave_features import MMWaveWindow
from src.features.uwb_features import UWBWindow
from src.detectors.motion_gestures import MotionGestureDetector, _dtw_distance, _normalize_seq
from src.detectors.posture_gestures import PostureGestureDetector
from src.detectors.micro_doppler import MicroDopplerDetector
from src.detectors.proximity_gestures import ProximityGestureDetector
from src.detectors.hand_gestures import HandGestureDetector
from src.fusion.dempster_shafer import DempsterShaferFusion

ALL_GESTURES = [
    "pull", "push", "clockwise", "anti_clockwise",
    "left", "right", "bye_bye", "one_arm_boxing",
    "clapping", "two_arm_boxing", "t_arms", "raise_arms",
    "soli", "fist_open_close", "palm_up_down",
]

# ── Helpers ───────────────────────────────────────────────────────────────


def _describe_imu(gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z) -> dict:
    """Compute statistics on raw IMU data."""
    gx = np.array(gyro_x)
    gy = np.array(gyro_y)
    gz = np.array(gyro_z)
    ax = np.array(accel_x)
    ay = np.array(accel_y)
    az = np.array(accel_z)

    return {
        "gyro_x": {"mean": float(np.mean(gx)), "std": float(np.std(gx)),
                    "min": float(np.min(gx)), "max": float(np.max(gx)),
                    "range": float(np.ptp(gx))},
        "gyro_y": {"mean": float(np.mean(gy)), "std": float(np.std(gy)),
                    "min": float(np.min(gy)), "max": float(np.max(gy)),
                    "range": float(np.ptp(gy))},
        "gyro_z": {"mean": float(np.mean(gz)), "std": float(np.std(gz)),
                    "min": float(np.min(gz)), "max": float(np.max(gz)),
                    "range": float(np.ptp(gz))},
        "accel_x": {"mean": float(np.mean(ax)), "std": float(np.std(ax)),
                     "min": float(np.min(ax)), "max": float(np.max(ax)),
                     "range": float(np.ptp(ax))},
        "accel_y": {"mean": float(np.mean(ay)), "std": float(np.std(ay)),
                     "min": float(np.min(ay)), "max": float(np.max(ay)),
                     "range": float(np.ptp(ay))},
        "accel_z": {"mean": float(np.mean(az)), "std": float(np.std(az)),
                     "min": float(np.min(az)), "max": float(np.max(az)),
                     "range": float(np.ptp(az))},
        "sample_count": len(gx),
    }


def _describe_mmwave(clusters_list: list, velocities_list: list) -> dict:
    """Stats on mmWave clusters and velocities across frames."""
    point_counts = [len(c) for c in clusters_list]
    n_clusters = [len(c) if isinstance(c, list) else 1 for c in clusters_list]

    all_vels = []
    for v in velocities_list:
        if v is not None and len(v) > 0:
            all_vels.extend(v.tolist() if hasattr(v, 'tolist') else v)

    return {
        "frames_with_clusters": sum(1 for c in clusters_list if c),
        "total_frames": len(clusters_list),
        "point_count_mean": float(np.mean(point_counts)) if point_counts else 0,
        "point_count_std": float(np.std(point_counts)) if point_counts else 0,
        "n_clusters_mean": float(np.mean(n_clusters)) if n_clusters else 0,
        "velocity_std": float(np.std(all_vels)) if all_vels else 0,
        "velocity_mean": float(np.mean(np.abs(all_vels))) if all_vels else 0,
    }


def _describe_uwb(distances: list, velocities: list) -> dict:
    dists = [d for d in distances if d is not None]
    vels = [v for v in velocities if v is not None]
    return {
        "distance_mean": float(np.mean(dists)) if dists else None,
        "distance_std": float(np.std(dists)) if dists else None,
        "distance_range": float(np.ptp(dists)) if dists else None,
        "velocity_mean": float(np.mean(vels)) if vels else None,
        "velocity_std": float(np.std(vels)) if vels else None,
        "sample_count": len(dists),
    }


# ── Main diagnostic ───────────────────────────────────────────────────────


def diagnose_gesture(
    gesture: str,
    mux: SensorMultiplexer,
    preprocessor: SensorPreprocessor,
    record_duration: float = 3.0,
) -> dict:
    """Record and analyze one gesture performance."""

    # ── Build fresh detectors (so windows start clean) ────────────────
    motion = MotionGestureDetector(window_samples=30)
    posture = PostureGestureDetector(window_frames=6)
    doppler = MicroDopplerDetector(window_frames=6)
    proximity = ProximityGestureDetector(window_samples=30)
    hand = HandGestureDetector(mmwave_window_frames=6, imu_window_samples=30)
    fusion = DempsterShaferFusion(belief_threshold=0.5, conflict_threshold=0.4)

    # ── Accumulators ──────────────────────────────────────────────────
    raw_imu_gyro_x: list[float] = []
    raw_imu_gyro_y: list[float] = []
    raw_imu_gyro_z: list[float] = []
    raw_imu_accel_x: list[float] = []
    raw_imu_accel_y: list[float] = []
    raw_imu_accel_z: list[float] = []
    mmwave_clusters_hist: list[list] = []
    mmwave_velocities_hist: list = []
    uwb_distances: list[float | None] = []
    uwb_velocities: list[float | None] = []

    detector_outputs: list[dict] = []
    fusion_outputs: list[dict] = []
    best_gesture_votes: defaultdict[str, int] = defaultdict(int)

    # ── Countdown ─────────────────────────────────────────────────────
    for i in range(3, 0, -1):
        print(f"\r  Get ready: {i}...", end="", flush=True)
        time.sleep(0.8)
    print("\r  PERFORM NOW!        ", flush=True)

    # ── Record ────────────────────────────────────────────────────────
    t_start = time.time()
    frame_count = 0

    while time.time() - t_start < record_duration:
        frame = mux.read_frame()
        pf = preprocessor.process(frame)
        frame_count += 1

        # Collect raw IMU
        if pf.imu_gyro:
            raw_imu_gyro_x.append(pf.imu_gyro[0])
            raw_imu_gyro_y.append(pf.imu_gyro[1])
            raw_imu_gyro_z.append(pf.imu_gyro[2])
        if pf.imu_linear_accel:
            raw_imu_accel_x.append(pf.imu_linear_accel[0])
            raw_imu_accel_y.append(pf.imu_linear_accel[1])
            raw_imu_accel_z.append(pf.imu_linear_accel[2])

        # Collect mmWave
        mmwave_clusters_hist.append(pf.mmwave_clusters or [])
        if pf.mmwave_raw_velocities is not None:
            mmwave_velocities_hist.append(pf.mmwave_raw_velocities)

        # Collect UWB
        uwb_distances.append(pf.uwb_distance)
        uwb_velocities.append(pf.uwb_velocity)

        # Push to detectors
        if pf.imu_gyro and pf.imu_linear_accel:
            gx, gy, gz = pf.imu_gyro
            ax, ay, az = pf.imu_linear_accel
            motion.push(ax, ay, az, gx, gy, gz)
            proximity.push(pf.uwb_distance, pf.uwb_velocity, ax, ay, az)
            hand.push(
                pf.mmwave_clusters or [], pf.mmwave_raw_velocities,
                ax, ay, az, gx, gy, gz,
            )

        posture.push(pf.mmwave_clusters or [], pf.mmwave_raw_velocities, pf.rfid_hands or set())
        doppler.push(pf.mmwave_clusters or [], pf.mmwave_raw_velocities)

        # Detect & fuse
        results = {
            "motion": motion.detect(),
            "posture": posture.detect(),
            "doppler": doppler.detect(),
            "proximity": proximity.detect(),
            "hand": hand.detect(),
        }
        detector_outputs.append({k: dict(v) for k, v in results.items()})

        fusion.reset()
        evidence_count = 0
        for src, res in results.items():
            if res.get("unknown", 1.0) < 0.95:
                fusion.add_evidence(src, res)
                evidence_count += 1

        if evidence_count > 0:
            fused = fusion.fuse()
            fusion_outputs.append(dict(fused))
            if fused["gesture"]:
                best_gesture_votes[fused["gesture"]] += 1

        # Progress bar
        elapsed = time.time() - t_start
        bar_len = int(20 * elapsed / record_duration)
        print(f"\r  [{('#' * bar_len):20s}] {frame_count} frames", end="", flush=True)
        time.sleep(0.002)

    print("\r  Done.                        ", flush=True)

    # ── Analyze ───────────────────────────────────────────────────────
    analysis = {
        "gesture": gesture,
        "frame_count": frame_count,
        "duration_s": time.time() - t_start,
        "raw_imu_stats": _describe_imu(
            raw_imu_gyro_x, raw_imu_gyro_y, raw_imu_gyro_z,
            raw_imu_accel_x, raw_imu_accel_y, raw_imu_accel_z,
        ),
        "mmwave_stats": _describe_mmwave(mmwave_clusters_hist, mmwave_velocities_hist),
        "uwb_stats": _describe_uwb(uwb_distances, uwb_velocities),
        "detector_consensus": dict(best_gesture_votes),
    }

    # Per-detector final state
    analysis["final_detector_outputs"] = detector_outputs[-1] if detector_outputs else {}
    if fusion_outputs:
        last_fused = fusion_outputs[-1]
        analysis["final_fusion"] = {
            "gesture": last_fused.get("gesture"),
            "confidence": last_fused.get("confidence"),
            "conflict": last_fused.get("conflict"),
            "ignorance": last_fused.get("ignorance"),
        }
        # Top 3 beliefs
        beliefs = last_fused.get("beliefs", {})
        top3 = sorted(beliefs.items(), key=lambda x: x[1], reverse=True)[:3]
        analysis["top3_beliefs"] = top3

    # Motion detector internals
    if len(raw_imu_gyro_z) >= 30:
        gz_seq = _normalize_seq(np.array(raw_imu_gyro_z[-30:]))
        ax_seq = _normalize_seq(np.array(raw_imu_accel_x[-30:]))
        analysis["dtw_debug"] = {
            "cw_distance": _dtw_distance(gz_seq, motion._templates.get("clockwise", np.zeros(50))),
            "acw_distance": _dtw_distance(gz_seq, motion._templates.get("anti_clockwise", np.zeros(50))),
            "left_distance": _dtw_distance(ax_seq, motion._templates.get("left", np.zeros(50))),
            "right_distance": _dtw_distance(ax_seq, motion._templates.get("right", np.zeros(50))),
        }

    return analysis


def print_analysis(analysis: dict) -> None:
    """Pretty-print the analysis."""
    print()
    print("=" * 70)
    print(f"  DIAGNOSIS: {analysis['gesture'].upper()}")
    print("=" * 70)

    # IMU stats
    imu = analysis["raw_imu_stats"]
    print(f"\n── IMU ({imu['sample_count']} samples) ──")
    for axis in ["gyro_x", "gyro_y", "gyro_z"]:
        s = imu[axis]
        print(f"  {axis}: mean={s['mean']:+.4f}  std={s['std']:.4f}  range={s['range']:.4f} rad/s")
    for axis in ["accel_x", "accel_y", "accel_z"]:
        s = imu[axis]
        print(f"  {axis}: mean={s['mean']:+.4f}  std={s['std']:.4f}  range={s['range']:.4f} g")

    # mmWave stats
    mmw = analysis["mmwave_stats"]
    print(f"\n── mmWave ({mmw['total_frames']} frames) ──")
    print(f"  frames with clusters: {mmw['frames_with_clusters']}/{mmw['total_frames']}")
    print(f"  point count: μ={mmw['point_count_mean']:.1f} σ={mmw['point_count_std']:.1f}")
    print(f"  cluster count: μ={mmw['n_clusters_mean']:.1f}")
    print(f"  velocity: μ={mmw['velocity_mean']:.4f} σ={mmw['velocity_std']:.4f} m/s")

    # UWB stats
    uwb = analysis["uwb_stats"]
    print(f"\n── UWB ({uwb['sample_count']} samples) ──")
    if uwb["distance_mean"] is not None:
        print(f"  distance: μ={uwb['distance_mean']:.3f}m σ={uwb['distance_std']:.3f}m range={uwb['distance_range']:.3f}m")
        print(f"  velocity: μ={uwb['velocity_mean']:.3f} σ={uwb['velocity_std']:.3f} m/s")

    # Detector outputs
    det = analysis.get("final_detector_outputs", {})
    print(f"\n── Detector Outputs ──")
    for name in ["motion", "posture", "doppler", "proximity", "hand"]:
        out = det.get(name, {})
        if out:
            # Show non-zero beliefs
            active = [(k, v) for k, v in out.items() if v > 0.01 and k != "unknown"]
            active.sort(key=lambda x: x[1], reverse=True)
            if active:
                items = ", ".join(f"{k}={v:.3f}" for k, v in active[:3])
                print(f"  {name:12s}: {items}")
            else:
                print(f"  {name:12s}: (all unknown={out.get('unknown', 1):.3f})")

    # DTW debug
    dtw = analysis.get("dtw_debug", {})
    if dtw:
        print(f"\n── DTW Distances (lower = better match) ──")
        for key in ["cw_distance", "acw_distance", "left_distance", "right_distance"]:
            print(f"  {key}: {dtw[key]:.4f}")

    # Fusion
    fused = analysis.get("final_fusion", {})
    print(f"\n── Fusion Result ──")
    print(f"  detected: {fused.get('gesture')}")
    print(f"  confidence: {fused.get('confidence', 0):.3f}")
    print(f"  conflict: {fused.get('conflict', 0):.3f}")
    print(f"  ignorance: {fused.get('ignorance', 0):.3f}")

    top3 = analysis.get("top3_beliefs", [])
    if top3:
        print(f"  top beliefs: {', '.join(f'{g}={b:.3f}' for g, b in top3)}")

    # Consensus
    consensus = analysis.get("detector_consensus", {})
    if consensus:
        print(f"\n── Frame-by-frame consensus ──")
        total = sum(consensus.values())
        for g, count in sorted(consensus.items(), key=lambda x: x[1], reverse=True):
            pct = count / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {g:20s} [{bar}] {pct:.0f}% ({count}/{total} frames)")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gesture diagnostic harness")
    parser.add_argument("--gesture", "-g", help="Comma-separated gestures to test (default: all)")
    parser.add_argument("--duration", "-d", type=float, default=3.0,
                        help="Recording duration per gesture (default: 3s)")
    args = parser.parse_args()

    gestures = ALL_GESTURES
    if args.gesture:
        gestures = [g.strip() for g in args.gesture.split(",")]
        for g in gestures:
            if g not in ALL_GESTURES:
                print(f"Unknown gesture: {g}")
                print(f"Available: {', '.join(ALL_GESTURES)}")
                sys.exit(1)

    print("=" * 70)
    print("  Null-Gesture — Diagnostic Harness")
    print("=" * 70)

    # Connect sensors
    print("\nConnecting sensors...", flush=True)
    mux = SensorMultiplexer()
    status = mux.start()
    print(f"Sensors: {status}")

    if not any(status.values()):
        print("ERROR: No sensors connected.")
        sys.exit(1)

    preprocessor = SensorPreprocessor(calibration_frames=20)
    print("Calibrating...", flush=True)
    deadline = time.time() + 3.0
    while not preprocessor.is_calibrated() and time.time() < deadline:
        frame = mux.read_frame()
        preprocessor.process(frame)
        time.sleep(0.005)
    print(f"Calibrated: {preprocessor.is_calibrated()}\n")

    # Diagnose each gesture
    all_results = []
    for i, gesture in enumerate(gestures):
        print(f"\n[{i+1}/{len(gestures)}] GESTURE: {gesture.upper()}")
        print("-" * 50)
        resp = input("Press ENTER when ready (or 's' to skip, 'q' to quit)... ")
        if resp.strip().lower() == 'q':
            break
        if resp.strip().lower() == 's':
            print(f"  Skipping {gesture}")
            continue

        analysis = diagnose_gesture(gesture, mux, preprocessor, args.duration)
        print_analysis(analysis)
        all_results.append(analysis)

        # Short rest
        time.sleep(1.0)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY: Detection accuracy per gesture")
    print("=" * 70)

    for r in all_results:
        g = r["gesture"]
        fused = r.get("final_fusion", {})
        detected = fused.get("gesture")
        conf = fused.get("confidence", 0)
        match = "✓" if detected == g else ("~" if detected and detected != "unknown" else "✗")
        print(f"  {match} {g:20s} → detected as: {str(detected):20s} (conf={conf:.3f})")

    mux.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
