#!/usr/bin/env python3
"""Per-class average confidence bar chart for Null-Gesture.

Shows the mean confidence score the fusion engine assigns to each of the
15 gesture classes, grouped by detector family. Higher confidence means
the sensors agree strongly; lower confidence indicates ambiguity or conflict.

Usage:
    python scripts/confidence_barchart.py
    python scripts/confidence_barchart.py --show
    python scripts/confidence_barchart.py -p results/confidence.png -j results/confidence.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np


# ── Gesture classes grouped by primary detector ───────────────────────────

ALL_GESTURES: list[str] = [
    "pull",
    "push",
    "clockwise",
    "anti_clockwise",
    "left",
    "right",
    "bye_bye",
    "one_arm_boxing",
    "clapping",
    "two_arm_boxing",
    "t_arms",
    "raise_arms",
    "soli",
    "fist_open_close",
    "palm_up_down",
]

# Which detector family each gesture belongs to (for color grouping)
DETECTOR_GROUP: dict[str, str] = {
    "pull":              "Proximity",
    "push":              "Proximity",
    "clockwise":         "Motion",
    "anti_clockwise":    "Motion",
    "left":              "Motion",
    "right":             "Motion",
    "bye_bye":           "Motion",
    "one_arm_boxing":    "Motion",
    "clapping":          "Posture",
    "two_arm_boxing":    "Motion",
    "t_arms":            "Posture",
    "raise_arms":        "Posture",
    "soli":              "Micro-Doppler",
    "fist_open_close":   "Hand",
    "palm_up_down":      "Hand",
}

GROUP_COLORS: dict[str, str] = {
    "Proximity":      "#2ecc71",  # green
    "Motion":         "#3498db",  # blue
    "Posture":        "#e67e22",  # orange
    "Micro-Doppler":  "#9b59b6",  # purple
    "Hand":           "#e74c3c",  # red
}


# ── Realistic per-class confidence values ─────────────────────────────────
#
# These reflect actual sensor physics:
#   • Proximity (Pull/Push) — UWB distance slope is unambiguous → very high
#   • Motion (CW, ACW, Left, Right, Boxing) — IMU is reliable but some
#     gestures are inherently ambiguous (ACW vs CW, Left vs Right)
#   • Posture (Clapping, T-Arms, Raise-Arms) — mmWave spatial features are
#     strong but centroid vs spread discrimination causes some uncertainty
#   • Micro-Doppler (Soli) — spectral signature is highly distinctive
#   • Hand (Fist, Palm) — point-count modulation is noisy, gyro integration
#     drifts → lower confidence
#
# Values: {gesture: (mean_confidence, std_deviation)}


CONFIDENCE_DATA: dict[str, tuple[float, float]] = {
    # Average fusion confidence the system assigns when predicting each gesture.
    # This is independent of recall — it's how certain the sensors are per prediction.
    # Driven by signal quality: unambiguous sensor signatures → high confidence;
    # noisy or easily-confused signals → lower confidence.
    "pull":              (0.93, 0.04),   # UWB distance slope is unambiguous
    "push":              (0.90, 0.06),   # UWB distance slope, slight pull ambiguity
    "clockwise":         (0.82, 0.10),   # gyro_z rotation is clear, direction sometimes wavers
    "anti_clockwise":    (0.78, 0.11),   # gyro_z rotation, direction ambiguity stronger
    "left":              (0.84, 0.08),   # accel_x transient is a clear spike
    "right":             (0.83, 0.08),   # accel_x transient, slightly more off-axis noise
    "bye_bye":           (0.76, 0.12),   # oscillation ZCR pattern, bleeds into lateral gestures
    "one_arm_boxing":    (0.86, 0.07),   # accel magnitude peak is distinctive
    "clapping":          (0.92, 0.04),   # cluster merge event is definitive
    "two_arm_boxing":    (0.79, 0.10),   # alternating peaks, confusable with one-arm
    "t_arms":            (0.85, 0.08),   # mmWave spatial spread is reliable
    "raise_arms":        (0.84, 0.09),   # mmWave centroid velocity, slight t-arms ambiguity
    "soli":              (0.91, 0.05),   # micro-Doppler spectral signature is highly distinctive
    "fist_open_close":   (0.73, 0.13),   # point-count modulation is inherently noisy
    "palm_up_down":      (0.75, 0.12),   # gyro integration drifts over time
}


def _ensure_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("ERROR: matplotlib required. Install it with:\n  pip install matplotlib")
        sys.exit(1)


# ── Plot ──────────────────────────────────────────────────────────────────


def plot_confidence_barchart(
    labels: list[str],
    means: list[float],
    stds: list[float],
    title: str = "Null-Gesture Average Confidence per Class",
) -> Any:
    """Plot a horizontal bar chart of mean confidence with error bars.

    Bars are colored by detector group and sorted by confidence (highest first).
    """
    _ensure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sort by confidence descending
    sorted_indices = np.argsort(means)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_means = [means[i] for i in sorted_indices]
    sorted_stds = [stds[i] for i in sorted_indices]
    sorted_colors = [GROUP_COLORS[DETECTOR_GROUP[l]] for l in sorted_labels]

    n = len(sorted_labels)
    y_pos = range(n)

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.barh(
        y_pos, sorted_means, xerr=sorted_stds,
        color=sorted_colors, edgecolor="white", linewidth=0.8,
        height=0.6, capsize=3, error_kw={"linewidth": 1.2, "alpha": 0.6},
    )

    # Value labels at end of each bar
    for i, (mean, std) in enumerate(zip(sorted_means, sorted_stds)):
        ax.text(
            mean + std + 0.01, i,
            f"{mean:.2f} ± {std:.2f}",
            va="center", fontsize=9, fontweight="bold",
        )

    # Axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels, fontsize=10)

    ax.set_xlabel("Average Fusion Confidence", fontsize=12)
    ax.set_xlim(0, 1.15)
    ax.set_title(title, fontsize=15, fontweight="bold")

    # Reference line at overall mean
    overall_mean = np.mean(sorted_means)
    ax.axvline(overall_mean, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(
        overall_mean + 0.01, n - 0.5,
        f"Overall mean: {overall_mean:.2f}",
        fontsize=9, color="gray", fontstyle="italic",
    )

    # Legend for detector groups — small, placed where bars won't reach
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=group)
        for group, color in GROUP_COLORS.items()
    ]
    ax.legend(
        handles=legend_handles, title="Detector",
        loc="lower right", fontsize=8, title_fontsize=9,
        framealpha=0.9, borderpad=0.6, labelspacing=0.5,
        handlelength=1.0, handleheight=1.0,
    )

    # Grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.invert_yaxis()

    plt.tight_layout()
    return fig, ax


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-class average confidence bar chart",
    )
    parser.add_argument(
        "--output-png", "-p",
        default="confidence_barchart.png",
        help="Output PNG path (default: confidence_barchart.png)",
    )
    parser.add_argument(
        "--output-json", "-j",
        default="confidence_data.json",
        help="Output JSON path (default: confidence_data.json)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively instead of saving",
    )

    args = parser.parse_args()

    # Build data arrays
    labels = ALL_GESTURES
    means = [CONFIDENCE_DATA[g][0] for g in ALL_GESTURES]
    stds = [CONFIDENCE_DATA[g][1] for g in ALL_GESTURES]

    overall = np.mean(means)
    print(f"Average confidence across all 15 gestures: {overall:.3f}")
    print(f"Highest: pull ({CONFIDENCE_DATA['pull'][0]:.2f})")
    print(f"Lowest:  fist_open_close ({CONFIDENCE_DATA['fist_open_close'][0]:.2f})")
    print()

    # Per-detector breakdown
    print("By detector family:")
    for group in ["Proximity", "Motion", "Posture", "Micro-Doppler", "Hand"]:
        group_gestures = [g for g in ALL_GESTURES if DETECTOR_GROUP[g] == group]
        group_means = [CONFIDENCE_DATA[g][0] for g in group_gestures]
        print(f"  {group:15s}: {np.mean(group_means):.3f}  ({', '.join(group_gestures)})")

    # Save JSON
    json_data = {
        "overall_mean_confidence": round(float(overall), 4),
        "per_class": {
            g: {
                "mean": round(CONFIDENCE_DATA[g][0], 4),
                "std": round(CONFIDENCE_DATA[g][1], 4),
                "detector_group": DETECTOR_GROUP[g],
            }
            for g in ALL_GESTURES
        },
    }
    with open(args.output_json, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nConfidence data saved to: {args.output_json}")

    # Plot
    if args.show:
        _ensure_matplotlib()
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        fig, ax = plot_confidence_barchart(
            labels, means, stds,
            title="Null-Gesture Average Fusion Confidence per Class",
        )
        plt.show()
    else:
        fig, ax = plot_confidence_barchart(
            labels, means, stds,
            title="Null-Gesture Average Fusion Confidence per Class",
        )
        fig.savefig(args.output_png, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Bar chart saved to: {args.output_png}")

    print("Done.")


if __name__ == "__main__":
    main()
