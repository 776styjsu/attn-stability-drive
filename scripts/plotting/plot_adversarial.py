#!/usr/bin/env python3
"""
Plot adversarial attack trajectories from attack_result.json files.

This script analyzes the "cost" of forcing explanation changes by visualizing:
- L2 perturbation norm vs iterations
- Explanation change vs iterations  
- L2 vs explanation change (cost-benefit trade-off)
- Loss component breakdown

Usage:
    # Single result file
    python scripts/plot_adversarial.py output/adversarial/Town10HD__2383583/attack_result.json

    # Multiple results for comparison
    python scripts/plot_adversarial.py output/adversarial/*/attack_result.json --compare

    # Specify output directory
    python scripts/plot_adversarial.py result.json --out-dir plots/adversarial/

    # Show plots interactively
    python scripts/plot_adversarial.py result.json --show
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to python path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from attn_stability_drive.visualization.plots import (
    plot_adversarial_trajectory,
    plot_adversarial_comparison,
)


def load_result(path: Path) -> Dict[str, Any]:
    """Load attack result from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def get_label_from_path(path: Path) -> str:
    """Extract a meaningful label from the result path."""
    # Try to get parent folder name (usually image stem)
    parent_name = path.parent.name
    if parent_name and parent_name != "adversarial":
        return parent_name
    return path.stem


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot adversarial attack trajectory analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results",
        type=Path,
        nargs="+",
        help="Path(s) to attack_result.json file(s). Supports glob patterns.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots. Default: same directory as input or 'plots/adversarial' for comparison.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison plots across multiple results.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively.",
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default=None,
        help="Base name for output files. Default: derived from input.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics to console.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Expand glob patterns and validate paths
    result_paths: List[Path] = []
    for pattern in args.results:
        if pattern.exists():
            result_paths.append(pattern)
        else:
            # Try as glob pattern
            matches = list(Path(".").glob(str(pattern)))
            if matches:
                result_paths.extend(matches)
            else:
                print(f"Warning: No files found matching '{pattern}'")
    
    if not result_paths:
        print("Error: No valid result files found.")
        sys.exit(1)
    
    # Remove duplicates and sort
    result_paths = sorted(set(result_paths))
    print(f"Found {len(result_paths)} result file(s)")
    
    # Load all results
    results = []
    labels = []
    for path in result_paths:
        try:
            result = load_result(path)
            results.append(result)
            labels.append(get_label_from_path(path))
            print(f"  Loaded: {path}")
        except Exception as e:
            print(f"  Error loading {path}: {e}")
    
    if not results:
        print("Error: No results could be loaded.")
        sys.exit(1)
    
    # Determine output directory
    if args.out_dir:
        out_dir = args.out_dir
    elif args.compare:
        out_dir = Path("plots/adversarial")
    else:
        # Same directory as input file
        out_dir = result_paths[0].parent / "plots"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    # Generate plots
    if args.compare and len(results) > 1:
        # Comparison mode
        base_name = args.base_name or "adversarial_comparison"
        print(f"\nGenerating comparison plots for {len(results)} results...")
        
        comparison_result = plot_adversarial_comparison(
            results=results,
            labels=labels,
            out_dir=out_dir,
            base_name=base_name,
            show=args.show,
        )
        
        print("\nComparison plots saved:")
        for name, path in comparison_result["plots"].items():
            print(f"  {name}: {path}")
        
        # Also generate individual trajectory plots
        print("\nGenerating individual trajectory plots...")
        for result, label, path in zip(results, labels, result_paths):
            loss_history = result.get("loss_history", [])
            if not loss_history:
                print(f"  Skipping {label}: no loss_history")
                continue
            
            individual_out = out_dir / label
            traj_result = plot_adversarial_trajectory(
                loss_history=loss_history,
                out_dir=individual_out,
                base_name="trajectory",
                show=False,  # Don't show individual plots in comparison mode
                title_prefix=label,
            )
            
            if args.summary:
                print(f"\n  === {label} ===")
                for key, value in traj_result["stats"].items():
                    if isinstance(value, float):
                        print(f"    {key}: {value:.6f}")
                    else:
                        print(f"    {key}: {value}")
    
    else:
        # Single result mode (or process each individually)
        for result, label, path in zip(results, labels, result_paths):
            loss_history = result.get("loss_history", [])
            
            if not loss_history:
                print(f"Error: No loss_history found in {path}")
                continue
            
            print(f"\nProcessing: {label}")
            print(f"  Iterations: {len(loss_history)}")
            
            # Output directory for this result
            if args.out_dir:
                individual_out = args.out_dir
                if len(results) > 1:
                    individual_out = args.out_dir / label
            else:
                individual_out = path.parent / "plots"
            
            base_name = args.base_name or "trajectory"
            
            traj_result = plot_adversarial_trajectory(
                loss_history=loss_history,
                out_dir=individual_out,
                base_name=base_name,
                show=args.show,
                title_prefix=label,
            )
            
            print(f"\n  Plots saved to: {individual_out}")
            for name, plot_path in traj_result["plots"].items():
                print(f"    {name}: {plot_path.name}")
            
            if args.summary:
                print(f"\n  Summary statistics:")
                stats = traj_result["stats"]
                print(f"    Total iterations: {stats['total_iterations']}")
                print(f"    Final L2 norm: {stats['final_l2_norm']:.6f}")
                print(f"    Final explanation change: {stats['final_explanation_change']:.6f}")
                print(f"    Final output change: {stats['final_output_change']:.6f}")
                print(f"    Attack efficiency (ΔExp/L2): {stats['efficiency']:.6f}")
                print(f"    Max explanation change: {stats['max_explanation_change']:.6f} (iter {stats['iter_at_max_exp_change']})")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
