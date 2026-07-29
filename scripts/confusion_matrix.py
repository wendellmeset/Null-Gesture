#!/usr/bin/env python3
"""Confusion matrix generator for Null-Gesture gesture detection.

Evaluates recorded training data against the detection pipeline and produces
a confusion matrix with per-class precision, recall, and F1 metrics.

Usage:
    # Evaluate all recorded sessions
    python scripts/confusion_matrix.py

    # Evaluate a specific session
    python scripts/confusion_matrix.py --data data/training/session_01

    # Output to specific files
    python scripts/confusion_matrix.py --output-png results/confusion.png --output-json results/metrics.json

    # Only use IMU-based motion detector (fast, no hardware needed)
    python scripts/confusion_matrix.py --detector motion

    # Use all available detectors (requires recorded mmWave/UWB data)
    python scripts/confusion_matrix.py --detector all

    # Show the plot interactively instead of saving
    python scripts/confusion_matrix.py --show
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Add parent to path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detectors.motion_gestures import MotionGestureDetector
from src.fusion.dempster_shafer import ALL_GESTURES

_log = logging.getLogger(__name__)


# ── Data loading ──────────────────────────────────────────────────────────


def discover_sessions(data_dir: str) -> list[Path]:
    """Find all session directories under data_dir.

    A session directory is one that contains .json gesture files.
    Returns a list of session directories.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    sessions: list[Path] = []

    # Check if data_dir itself is a session
    json_files = list(data_path.glob("*.json"))
    gesture_jsons = [f for f in json_files if f.name != "summary.json"]
    if gesture_jsons:
        sessions.append(data_path)
        return sessions

    # Otherwise recurse one level
    for subdir in sorted(data_path.iterdir()):
        if subdir.is_dir():
            json_files = list(subdir.glob("*.json"))
            gesture_jsons = [f for f in json_files if f.name != "summary.json"]
            if gesture_jsons:
                sessions.append(subdir)

    return sessions


def load_session(session_dir: Path) -> dict[str, list[dict]]:
    """Load all gesture recordings from a session directory.

    Returns: {gesture_label: [recording, ...]}
    """
    data: dict[str, list[dict]] = defaultdict(list)

    for json_file in sorted(session_dir.glob("*.json")):
        if json_file.name == "summary.json":
            continue
        gesture = json_file.stem
        with open(json_file) as f:
            entries = json.load(f)
            data[gesture].extend(entries)

    return dict(data)


def extract_imu_frames(recording: dict) -> list[dict[str, float]]:
    """Extract IMU frames from a recording entry.

    Each frame is a dict with ax, ay, az, gx, gy, gz.
    Uses record_frames (the main gesture segment) for evaluation.
    """
    frames: list[dict[str, float]] = []

    for f in recording.get("record_frames", []):
        imu = f.get("imu", {})
        if imu:
            frames.append({
                "ax": float(imu.get("ax", 0) or 0),
                "ay": float(imu.get("ay", 0) or 0),
                "az": float(imu.get("az", 0) or 0),
                "gx": float(imu.get("gx", 0) or 0),
                "gy": float(imu.get("gy", 0) or 0),
                "gz": float(imu.get("gz", 0) or 0),
            })

    return frames


# ── Prediction ────────────────────────────────────────────────────────────


def predict_motion_gesture(
    frames: list[dict[str, float]],
    window_samples: int = 30,
    skip_initial: int = 0,
) -> dict[str, float]:
    """Run the MotionGestureDetector over a sequence of IMU frames.

    Returns a dict of gesture → aggregate confidence score.
    Uses per-frame voting with confidence accumulation.
    """
    detector = MotionGestureDetector(window_samples=window_samples)
    votes: dict[str, float] = defaultdict(float)
    total_frames = 0

    for i, f in enumerate(frames):
        if i < skip_initial:
            continue

        detector.push(f["ax"], f["ay"], f["az"], f["gx"], f["gy"], f["gz"])

        result = detector.detect()

        # Find the best non-unknown gesture this frame
        best_gesture = None
        best_score = 0.0
        for g, score in result.items():
            if g != "unknown" and score > best_score:
                best_score = score
                best_gesture = g

        if best_gesture and best_score > 0.1:
            votes[best_gesture] += best_score
            total_frames += 1

    # Normalize votes to [0,1]
    if votes:
        max_vote = max(votes.values())
        for g in votes:
            votes[g] /= max(max_vote, 1.0)

    return dict(votes)


def predict_from_recording(
    recording: dict,
    window_samples: int = 30,
) -> tuple[str, dict[str, float]]:
    """Predict the gesture label for a single recording.

    Returns (predicted_label, confidence_dict).
    """
    frames = extract_imu_frames(recording)

    if len(frames) < 10:
        return "unknown", {}

    # Skip a few initial frames to avoid the startle/transition artifact
    skip_initial = min(5, len(frames) // 10)

    votes = predict_motion_gesture(frames, window_samples=window_samples,
                                   skip_initial=skip_initial)

    if not votes:
        return "unknown", {}

    # Majority vote
    predicted = max(votes, key=lambda k: votes[k])
    return predicted, votes


# ── Confusion matrix computation ──────────────────────────────────────────


def build_confusion_matrix(
    all_predictions: list[dict[str, Any]],
    gesture_labels: list[str],
) -> np.ndarray:
    """Build a confusion matrix from prediction results.

    Rows = true labels, Columns = predicted labels.
    """
    label_to_idx = {g: i for i, g in enumerate(gesture_labels)}
    n = len(gesture_labels)
    cm = np.zeros((n, n), dtype=np.int64)

    for entry in all_predictions:
        true_label = entry["true"]
        pred_label = entry["predicted"]

        if true_label in label_to_idx and pred_label in label_to_idx:
            true_idx = label_to_idx[true_label]
            pred_idx = label_to_idx[pred_label]
            cm[true_idx, pred_idx] += 1

    return cm


def compute_metrics(cm: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """Compute precision, recall, F1 per class and overall accuracy."""
    n = len(labels)
    eps = 1e-10

    # Per-class metrics
    per_class: dict[str, dict[str, float]] = {}
    tp_total, fp_total, fn_total = 0, 0, 0

    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        support = int(cm[i, :].sum())

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

        tp_total += tp
        fp_total += fp
        fn_total += fn

    # Overall
    total = int(cm.sum())
    accuracy = tp_total / (total + eps)

    # Macro average
    macro_precision = np.mean([per_class[l]["precision"] for l in labels]).item()
    macro_recall = np.mean([per_class[l]["recall"] for l in labels]).item()
    macro_f1 = np.mean([per_class[l]["f1"] for l in labels]).item()

    # Weighted average (by support)
    supports = np.array([per_class[l]["support"] for l in labels], dtype=np.float64)
    total_support = supports.sum() + eps
    weighted_precision = float(np.sum([per_class[l]["precision"] * per_class[l]["support"] for l in labels]) / total_support)
    weighted_recall = float(np.sum([per_class[l]["recall"] * per_class[l]["support"] for l in labels]) / total_support)
    weighted_f1 = float(np.sum([per_class[l]["f1"] * per_class[l]["support"] for l in labels]) / total_support)

    return {
        "accuracy": round(accuracy, 4),
        "total_samples": total,
        "macro_avg": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4),
        },
        "weighted_avg": {
            "precision": round(weighted_precision, 4),
            "recall": round(weighted_recall, 4),
            "f1": round(weighted_f1, 4),
        },
        "per_class": per_class,
    }


# ── Visualization ─────────────────────────────────────────────────────────


def _ensure_matplotlib():
    """Import matplotlib, with a helpful error if not installed."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "ERROR: matplotlib is required for plotting. Install it with:\n"
            "  pip install matplotlib"
        )
        sys.exit(1)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str = "Null-Gesture Confusion Matrix",
    normalize: bool = True,
    cmap: str = "Blues",
    figsize: tuple[int, int] | None = None,
) -> Any:
    """Plot a confusion matrix using matplotlib.

    Args:
        cm: Raw count matrix (rows=true, cols=predicted).
        labels: Gesture label names.
        title: Plot title.
        normalize: If True, normalize rows to show recall (row-wise %).
        cmap: Matplotlib colormap name.
        figsize: Figure size in inches, auto if None.

    Returns:
        (fig, ax) tuple.
    """
    _ensure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    n = len(labels)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = cm.astype(np.float64) / np.maximum(row_sums, 1)
        fmt = ".2f"
        vmin, vmax = 0, 1
    else:
        cm_display = cm.astype(np.float64)
        fmt = "d"
        vmin, vmax = 0, cm.max()

    if figsize is None:
        figsize = (max(12, n * 0.7), max(10, n * 0.6))

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="auto")

    # Color bar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (row-normalized)" if normalize else "Count",
                   fontsize=11)

    # Tick labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    threshold = (vmax + vmin) / 2
    for i in range(n):
        for j in range(n):
            if normalize:
                text = f"{cm_display[i, j]:.2f}"
            else:
                text = f"{int(cm[i, j])}"
            color = "white" if cm_display[i, j] > threshold else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=8, color=color,
                    fontweight="bold" if i == j else "normal")

    ax.set_xlabel("Predicted Gesture", fontsize=12)
    ax.set_ylabel("True Gesture", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Grid lines between cells
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()

    return fig, ax


def plot_normalized_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    output_path: str,
    title: str = "Null-Gesture Confusion Matrix",
) -> None:
    """Generate and save a beautifully styled confusion matrix figure."""
    fig, ax = plot_confusion_matrix(
        cm, labels, title=title, normalize=True, cmap="Blues",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Confusion matrix saved to: {output_path}")


# ── Report ────────────────────────────────────────────────────────────────


def print_report(
    cm: np.ndarray,
    labels: list[str],
    metrics: dict[str, Any],
    all_predictions: list[dict],
) -> None:
    """Print a formatted evaluation report to stdout."""
    n = len(labels)

    print()
    print("=" * 80)
    print("  NULL-GESTURE CONFUSION MATRIX REPORT")
    print("=" * 80)

    # ── Confusion matrix table ───────────────────────────────────────
    # Determine number of columns that fit
    max_cols = 8
    batches = [labels[i:i + max_cols] for i in range(0, n, max_cols)]

    for batch_idx, batch_labels in enumerate(batches):
        batch_indices = [labels.index(l) for l in batch_labels]

        if batch_idx > 0:
            print(f"\n  (continued — columns {batch_indices[0] + 1}–{batch_indices[-1] + 1})")

        # Header
        header = f"{'True ↓ Pred →':>14s}"
        for l in batch_labels:
            header += f" {l[:8]:>8s}"
        print(header)
        print("-" * len(header))

        for i in range(n):
            row = f"  {labels[i]:<12s}"
            for j in batch_indices:
                row += f" {int(cm[i, j]):>8d}"
            print(row)

    # ── Summary metrics ──────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Overall Accuracy: {metrics['accuracy']:.2%} "
          f"({metrics['total_samples']} samples)")
    print(f"  Macro Avg — P: {metrics['macro_avg']['precision']:.3f}  "
          f"R: {metrics['macro_avg']['recall']:.3f}  "
          f"F1: {metrics['macro_avg']['f1']:.3f}")
    print(f"  Weighted Avg — P: {metrics['weighted_avg']['precision']:.3f}  "
          f"R: {metrics['weighted_avg']['recall']:.3f}  "
          f"F1: {metrics['weighted_avg']['f1']:.3f}")

    # ── Per-class breakdown ──────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print(f"  {'Gesture':<22s} {'Precision':>9s} {'Recall':>9s} {'F1':>9s} {'Support':>8s}")
    print(f"  {'─' * 22} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 8}")

    for label in labels:
        pc = metrics["per_class"][label]
        if pc["support"] > 0:
            print(f"  {label:<22s} {pc['precision']:>9.3f} {pc['recall']:>9.3f} "
                  f"{pc['f1']:>9.3f} {pc['support']:>8d}")

    # ── Common confusions ────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Most Common Confusions:")
    confusions: list[tuple[str, str, int]] = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                confusions.append((labels[i], labels[j], int(cm[i, j])))
    confusions.sort(key=lambda x: x[2], reverse=True)

    for true_label, pred_label, count in confusions[:10]:
        print(f"    {true_label:<22s} → {pred_label:<22s} ({count}×)")


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate confusion matrix for Null-Gesture gesture detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                          # Evaluate all sessions
  %(prog)s --data data/training/session_01          # Single session
  %(prog)s --output-png results/confusion.png       # Custom output path
  %(prog)s --show                                   # Interactive plot window
  %(prog)s --gestures clockwise,left,right,bye_bye  # Subset of gestures
        """,
    )

    parser.add_argument(
        "--data", "-d",
        default="data/training",
        help="Path to training data directory (default: data/training)",
    )
    parser.add_argument(
        "--output-png", "-p",
        default="confusion_matrix.png",
        help="Output path for confusion matrix PNG (default: confusion_matrix.png)",
    )
    parser.add_argument(
        "--output-json", "-j",
        default="confusion_metrics.json",
        help="Output path for metrics JSON (default: confusion_metrics.json)",
    )
    parser.add_argument(
        "--gestures", "-g",
        help="Comma-separated gesture labels to include (default: all found)",
    )
    parser.add_argument(
        "--window", "-w",
        type=int,
        default=30,
        help="IMU window size in samples (default: 30)",
    )
    parser.add_argument(
        "--detector",
        choices=["motion", "all"],
        default="motion",
        help="Which detectors to use (default: motion)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively instead of saving to file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-recording predictions",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    # ── Discover sessions ────────────────────────────────────────────
    print(f"Loading data from: {args.data}")
    sessions = discover_sessions(args.data)

    if not sessions:
        print(f"ERROR: No session data found in {args.data}")
        print("Run scripts/collect_training_data.py first to collect recordings.")
        sys.exit(1)

    print(f"Found {len(sessions)} session(s):")
    for s in sessions:
        print(f"  • {s}")

    # ── Load all data ────────────────────────────────────────────────
    all_data: dict[str, list[dict]] = defaultdict(list)

    for session_dir in sessions:
        session_data = load_session(session_dir)
        for gesture, recordings in session_data.items():
            all_data[gesture].extend(recordings)

    # Filter gestures if requested
    if args.gestures:
        requested = set(g.strip() for g in args.gestures.split(","))
        all_data = {g: recs for g, recs in all_data.items() if g in requested}

    if not all_data:
        print("ERROR: No gesture recordings loaded.")
        sys.exit(1)

    gesture_counts = {g: len(recs) for g, recs in all_data.items()}
    total_recordings = sum(gesture_counts.values())
    print(f"\nLoaded {total_recordings} recordings across {len(all_data)} gestures:")
    for g, count in sorted(gesture_counts.items()):
        print(f"  • {g}: {count} recordings")

    # ── Run predictions ──────────────────────────────────────────────
    print(f"\nRunning MotionGestureDetector (window={args.window} samples)...")
    print("─" * 50)

    all_predictions: list[dict[str, Any]] = []
    gesture_labels_seen: set[str] = set()

    for gesture, recordings in sorted(all_data.items()):
        gesture_labels_seen.add(gesture)
        for rep_idx, rec in enumerate(recordings):
            predicted, confidences = predict_from_recording(
                rec, window_samples=args.window,
            )

            all_predictions.append({
                "true": gesture,
                "predicted": predicted,
                "confidences": confidences,
                "rep": rep_idx + 1,
            })

            status = "✓" if predicted == gesture else "✗"
            if args.verbose or predicted != gesture:
                print(f"  {status} {gesture:>22s} → {predicted:<22s} "
                      f"(rep {rep_idx + 1}/{gesture_counts[gesture]})")
                if predicted != gesture and confidences:
                    top = sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"    confidences: {', '.join(f'{g}={c:.2f}' for g, c in top)}")

    if not all_predictions:
        print("ERROR: No predictions generated. Check data format.")
        sys.exit(1)

    # Order gesture labels by ALL_GESTURES canonical order, falling back
    # to whatever was actually seen
    gesture_labels = [g for g in ALL_GESTURES if g in gesture_labels_seen]

    n_correct = sum(1 for p in all_predictions if p["true"] == p["predicted"])
    raw_accuracy = n_correct / len(all_predictions)
    print(f"\nRaw accuracy: {n_correct}/{len(all_predictions)} = {raw_accuracy:.2%}")

    # ── Build confusion matrix ───────────────────────────────────────
    cm = build_confusion_matrix(all_predictions, gesture_labels)
    metrics = compute_metrics(cm, gesture_labels)

    # ── Print report ─────────────────────────────────────────────────
    print_report(cm, gesture_labels, metrics, all_predictions)

    # ── Save outputs ─────────────────────────────────────────────────
    # JSON metrics
    with open(args.output_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {args.output_json}")

    # Confusion matrix plot
    if args.show:
        _ensure_matplotlib()
        import matplotlib
        matplotlib.use("TkAgg")  # Switch to interactive
        import matplotlib.pyplot as plt
        fig, ax = plot_confusion_matrix(
            cm, gesture_labels,
            title=f"Null-Gesture Confusion Matrix\n"
                  f"({len(all_predictions)} samples, "
                  f"accuracy={metrics['accuracy']:.2%})",
            normalize=True,
        )
        plt.show()
    else:
        plot_normalized_confusion_matrix(
            cm, gesture_labels, args.output_png,
            title=f"Null-Gesture Confusion Matrix\n"
                  f"({len(all_predictions)} samples, "
                  f"accuracy={metrics['accuracy']:.2%})",
        )

    print("Done.")


if __name__ == "__main__":
    main()
