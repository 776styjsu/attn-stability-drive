#!/usr/bin/env python3
"""
Visualize attack efficiency across multiple runs.

This script creates comprehensive visualizations for analyzing adversarial
attack efficiency across multiple runs, including:

1. Efficiency Distribution - Histogram/violin plots showing attack efficiency spread
2. Efficiency Convergence - Mean efficiency over iterations with confidence bands
3. Parameter Sensitivity - Heatmaps showing how hyperparameters affect efficiency
4. Per-Run Trajectories - Small multiples of individual attack trajectories
5. Summary Statistics - Tables with aggregate metrics

Usage:
    # From aggregated JSONL file
    python scripts/plot_attack_efficiency.py attack_runs.jsonl

    # From multiple attack_result.json files
    python scripts/plot_attack_efficiency.py output/adversarial/**/attack_result.json

    # Group by town
    python scripts/plot_attack_efficiency.py attack_runs.jsonl --group-by town

    # Custom output directory
    python scripts/plot_attack_efficiency.py attack_runs.jsonl --out-dir plots/efficiency/

    # Generate specific plots only
    python scripts/plot_attack_efficiency.py attack_runs.jsonl --plots distribution convergence
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add src to python path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from attn_stability_drive.visualization.plots import (
    plot_efficiency_distribution,
    plot_efficiency_convergence,
    plot_efficiency_scatter,
    plot_attack_trajectories_grid,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load runs from a JSONL file."""
    runs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def load_attack_result(path: Path) -> Dict[str, Any]:
    """Load a single attack_result.json file."""
    with open(path, "r") as f:
        result = json.load(f)
    
    # Enrich with derived metrics
    loss_history = result.get("loss_history", [])
    if loss_history:
        final_entry = loss_history[-1]
        final_l2 = final_entry.get("loss_perturbation", 0.0)
        exp_sim = final_entry.get("explanation_similarity", 1.0)
        exp_change = final_entry.get("explanation_change", 1.0 - exp_sim)
        efficiency = exp_change / final_l2 if final_l2 > 1e-8 else 0.0
    else:
        final_l2 = result.get("perturbation_norm_l2", 0.0)
        exp_change = 1.0 - result.get("explanation_similarity", 1.0)
        efficiency = exp_change / final_l2 if final_l2 > 1e-8 else 0.0
    
    # Extract label from path
    label = path.parent.name if path.parent.name != "adversarial" else path.stem
    
    return {
        "run_id": label,
        "success": result.get("success", False),
        "iterations": result.get("iterations", len(loss_history)),
        "final_l2": final_l2,
        "final_exp_change": exp_change,
        "efficiency": efficiency,
        "loss_history": loss_history,
        "result_path": str(path),
    }


def load_runs(paths: List[Path]) -> List[Dict[str, Any]]:
    """
    Load runs from various sources (JSONL or individual JSON files).
    
    Args:
        paths: List of paths to JSONL files or attack_result.json files
        
    Returns:
        List of run dictionaries
    """
    runs = []
    
    for path in paths:
        if not path.exists():
            # Try as glob pattern
            matches = list(Path(".").glob(str(path)))
            for match in matches:
                if match.suffix == ".jsonl":
                    runs.extend(load_jsonl(match))
                elif match.name == "attack_result.json":
                    runs.append(load_attack_result(match))
        elif path.suffix == ".jsonl":
            runs.extend(load_jsonl(path))
        elif path.name == "attack_result.json":
            runs.append(load_attack_result(path))
        elif path.is_dir():
            # Search for attack_result.json files
            for result_path in path.glob("**/attack_result.json"):
                runs.append(load_attack_result(result_path))
    
    return runs


def compute_summary_stats(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics across all runs."""
    if not runs:
        return {}
    
    efficiencies = [r["efficiency"] for r in runs]
    l2_norms = [r["final_l2"] for r in runs]
    exp_changes = [r["final_exp_change"] for r in runs]
    successes = [r["success"] for r in runs]
    
    return {
        "num_runs": len(runs),
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
        "efficiency": {
            "mean": float(np.mean(efficiencies)),
            "std": float(np.std(efficiencies)),
            "median": float(np.median(efficiencies)),
            "min": float(np.min(efficiencies)),
            "max": float(np.max(efficiencies)),
        },
        "final_l2": {
            "mean": float(np.mean(l2_norms)),
            "std": float(np.std(l2_norms)),
            "median": float(np.median(l2_norms)),
        },
        "final_exp_change": {
            "mean": float(np.mean(exp_changes)),
            "std": float(np.std(exp_changes)),
            "median": float(np.median(exp_changes)),
        },
    }


def print_summary(runs: List[Dict[str, Any]], group_by: Optional[str] = None):
    """Print summary statistics to console."""
    if group_by and group_by in runs[0]:
        # Group runs
        groups = {}
        for run in runs:
            key = run.get(group_by, "unknown")
            if key not in groups:
                groups[key] = []
            groups[key].append(run)
        
        print(f"\n{'='*60}")
        print(f"Summary by {group_by}")
        print(f"{'='*60}")
        
        for key, group_runs in sorted(groups.items()):
            stats = compute_summary_stats(group_runs)
            print(f"\n{key}:")
            print(f"  Runs: {stats['num_runs']}")
            print(f"  Success rate: {stats['success_rate']*100:.1f}%")
            print(f"  Efficiency: {stats['efficiency']['mean']:.4f} ± {stats['efficiency']['std']:.4f}")
            print(f"  Final L2: {stats['final_l2']['mean']:.4f} ± {stats['final_l2']['std']:.4f}")
            print(f"  Final Δexp: {stats['final_exp_change']['mean']:.4f} ± {stats['final_exp_change']['std']:.4f}")
    else:
        stats = compute_summary_stats(runs)
        print(f"\n{'='*60}")
        print(f"Overall Summary ({stats['num_runs']} runs)")
        print(f"{'='*60}")
        print(f"  Success rate: {stats['success_rate']*100:.1f}%")
        print(f"  Efficiency: {stats['efficiency']['mean']:.4f} ± {stats['efficiency']['std']:.4f}")
        print(f"    Median: {stats['efficiency']['median']:.4f}")
        print(f"    Range: [{stats['efficiency']['min']:.4f}, {stats['efficiency']['max']:.4f}]")
        print(f"  Final L2: {stats['final_l2']['mean']:.4f} ± {stats['final_l2']['std']:.4f}")
        print(f"  Final Δexp: {stats['final_exp_change']['mean']:.4f} ± {stats['final_exp_change']['std']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize attack efficiency across multiple runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Input JSONL file(s) or attack_result.json file(s)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots/attack_efficiency"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        choices=["town", "model", "config", "success"],
        default=None,
        help="Group runs by this field in visualizations",
    )
    parser.add_argument(
        "--plots",
        type=str,
        nargs="+",
        choices=["distribution", "convergence", "scatter", "trajectories", "all"],
        default=["all"],
        help="Which plots to generate",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=16,
        help="Maximum number of trajectories to show in grid plot",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics to console",
    )
    parser.add_argument(
        "--export-stats",
        type=Path,
        default=None,
        help="Export summary statistics to JSON file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load all runs
    print(f"Loading runs from {len(args.inputs)} input(s)...")
    runs = load_runs(args.inputs)
    
    if not runs:
        print("Error: No runs found.")
        return 1
    
    print(f"Loaded {len(runs)} runs")
    
    # Filter out runs with no useful data
    valid_runs = [r for r in runs if r.get("loss_history") or r.get("final_l2", 0) > 0]
    print(f"Valid runs with data: {len(valid_runs)}")
    
    if not valid_runs:
        print("Warning: No runs with valid loss_history or metrics found.")
        valid_runs = runs  # Use all anyway for basic stats
    
    # Print summary if requested
    if args.summary:
        print_summary(valid_runs, args.group_by)
    
    # Export stats if requested
    if args.export_stats:
        stats = compute_summary_stats(valid_runs)
        args.export_stats.parent.mkdir(parents=True, exist_ok=True)
        with open(args.export_stats, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats exported to: {args.export_stats}")
    
    # Determine which plots to generate
    plot_types = set(args.plots)
    if "all" in plot_types:
        plot_types = {"distribution", "convergence", "scatter", "trajectories"}
    
    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating plots in: {args.out_dir}")
    
    saved_plots = {}
    
    # 1. Efficiency Distribution
    if "distribution" in plot_types:
        print("  Generating efficiency distribution...")
        result = plot_efficiency_distribution(
            runs=valid_runs,
            out_dir=args.out_dir,
            group_by=args.group_by,
            show=args.show,
        )
        saved_plots.update(result.get("plots", {}))
    
    # 2. Efficiency Convergence (requires loss_history)
    if "convergence" in plot_types:
        runs_with_history = [r for r in valid_runs if r.get("loss_history")]
        if runs_with_history:
            print("  Generating efficiency convergence...")
            result = plot_efficiency_convergence(
                runs=runs_with_history,
                out_dir=args.out_dir,
                group_by=args.group_by,
                show=args.show,
            )
            saved_plots.update(result.get("plots", {}))
        else:
            print("  Skipping convergence plot (no loss_history data)")
    
    # 3. Scatter plots (L2 vs Exp Change, colored by efficiency)
    if "scatter" in plot_types:
        print("  Generating scatter plots...")
        result = plot_efficiency_scatter(
            runs=valid_runs,
            out_dir=args.out_dir,
            group_by=args.group_by,
            show=args.show,
        )
        saved_plots.update(result.get("plots", {}))
    
    # 4. Trajectory grid
    if "trajectories" in plot_types:
        runs_with_history = [r for r in valid_runs if r.get("loss_history")]
        if runs_with_history:
            print("  Generating trajectory grid...")
            result = plot_attack_trajectories_grid(
                runs=runs_with_history[:args.max_trajectories],
                out_dir=args.out_dir,
                show=args.show,
            )
            saved_plots.update(result.get("plots", {}))
        else:
            print("  Skipping trajectory grid (no loss_history data)")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Generated {len(saved_plots)} plots:")
    for name, path in saved_plots.items():
        print(f"  {name}: {path}")
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
