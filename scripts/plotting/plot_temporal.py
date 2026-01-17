#!/usr/bin/env python3
"""
Entry point for plotting temporal analysis results.
Reads a CSV output from analyze_temporal.py and generates plots.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Set

# Add src to python path to allow running without installation
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from attn_stability_drive.visualization.plots import make_temporal_plots


def load_highlight_frames(path: Path) -> Set[int]:
    """Load frame numbers to highlight from a CSV file."""
    frames = set()
    if not path.exists():
        print(f"Warning: Highlight CSV {path} not found.")
        return frames

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            val = row.get("curr_frame") or row.get("prev_frame")
            if val:
                try:
                    frames.add(int(val))
                except ValueError:
                    pass
    print(f"Loaded {len(frames)} frames to highlight from {path}")
    return frames


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate plots from temporal analysis CSV output."
    )
    p.add_argument("csv", type=Path, help="Path to CSV output from analyze_temporal.py.")
    p.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory to save plots (default: <csv>.parent / 'plots').",
    )
    p.add_argument("--show-plots", action="store_true", help="Show plots interactively.")
    p.add_argument(
        "--auto-range",
        action="store_true",
        help="Autoscale plot axes to the data (default off; fixed [0,1] if not set).",
    )
    p.add_argument(
        "--highlight-csv",
        type=Path,
        default=None,
        help="Optional CSV containing frames to highlight (in light red).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.csv.exists():
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)

    plot_dir = args.plot_dir if args.plot_dir is not None else (args.csv.parent / "plots")
    base_name = args.csv.stem

    # Load highlight frames if provided
    highlight_frames: Set[int] = set()
    if args.highlight_csv:
        highlight_frames = load_highlight_frames(args.highlight_csv)

    # Read CSV and extract data
    pair_indices: List[int] = []
    sal_series: List[Optional[float]] = []
    img_series: List[Optional[float]] = []
    steer_series: List[Optional[float]] = []
    highlight_indices: List[int] = []

    sal_label: Optional[str] = None
    img_label: Optional[str] = None

    with args.csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            pair_indices.append(idx)

            # Extract saliency score
            sal_str = row.get("saliency_score", "")
            sal_val = float(sal_str) if sal_str else None
            sal_series.append(sal_val)

            # Extract image score
            img_str = row.get("image_score", "")
            img_val = float(img_str) if img_str else None
            img_series.append(img_val)

            # Extract steer similarity
            steer_str = row.get("steer_sim", "")
            steer_val = float(steer_str) if steer_str else None
            steer_series.append(steer_val)

            # Get metric labels from first row
            if sal_label is None:
                sal_label = row.get("saliency_metric", "SSIM").upper()
            if img_label is None:
                img_metric = row.get("image_metric", "")
                img_label = img_metric.upper() if img_metric else None

            # Check for highlight
            if highlight_frames:
                prev_frame = row.get("prev_frame", "")
                curr_frame = row.get("curr_frame", "")
                try:
                    if (prev_frame and int(prev_frame) in highlight_frames) or \
                       (curr_frame and int(curr_frame) in highlight_frames):
                        highlight_indices.append(idx)
                except ValueError:
                    pass

    if not pair_indices:
        print("Error: No data found in CSV.")
        sys.exit(1)

    print(f"Loaded {len(pair_indices)} rows from {args.csv}")
    print(f"Saliency metric: {sal_label}")
    if img_label:
        print(f"Image metric: {img_label}")

    make_temporal_plots(
        pair_indices,
        sal_series,
        img_series,
        steer_series,
        out_dir=plot_dir,
        base_name=base_name,
        sal_label=sal_label or "SSIM",
        img_label=img_label,
        show=args.show_plots,
        fixed_range=not args.auto_range,
        highlight_indices=highlight_indices if highlight_indices else None,
    )

    print(f"Plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
