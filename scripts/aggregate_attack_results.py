#!/usr/bin/env python3
"""
Aggregate multiple adversarial attack results into a single JSONL file.

This script scans directories for attack_result.json files and consolidates
them into a single JSONL file for efficient multi-run analysis.

Usage:
    # Aggregate all results from output/adversarial/
    python scripts/aggregate_attack_results.py output/adversarial/ -o attack_runs.jsonl

    # Recursive search with specific pattern
    python scripts/aggregate_attack_results.py output/adversarial/**/attack_result.json

    # Filter by model or config
    python scripts/aggregate_attack_results.py output/adversarial/ --model dave2v1
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def find_attack_results(search_paths: List[Path], recursive: bool = True) -> Iterator[Path]:
    """
    Find all attack_result.json files in the given paths.
    
    Args:
        search_paths: List of paths to search (files or directories)
        recursive: Whether to search recursively in directories
        
    Yields:
        Paths to attack_result.json files
    """
    for path in search_paths:
        if path.is_file() and path.name == "attack_result.json":
            yield path
        elif path.is_dir():
            pattern = "**/attack_result.json" if recursive else "*/attack_result.json"
            yield from path.glob(pattern)


def extract_run_metadata(result_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from the result path and parent directories.
    
    Tries to infer:
    - run_id: From parent folder name or timestamp
    - image: From parent folder name if it looks like an image stem
    - model: From config if available
    """
    parent = result_path.parent
    grandparent = parent.parent
    
    metadata = {
        "result_path": str(result_path),
        "run_dir": parent.name,
    }
    
    # Try to extract image name from folder (e.g., Town10HD__2383583)
    if "__" in parent.name:
        metadata["image_stem"] = parent.name
        # Extract town from image stem
        parts = parent.name.split("__")
        if parts:
            metadata["town"] = parts[0]
    
    # Try to detect if parent is a timestamp folder (e.g., 20260109_020849)
    try:
        datetime.strptime(parent.name, "%Y%m%d_%H%M%S")
        metadata["run_timestamp"] = parent.name
    except ValueError:
        pass
    
    return metadata


def load_and_enrich_result(
    result_path: Path,
    include_loss_history: bool = True,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load attack result and enrich with metadata.
    
    Args:
        result_path: Path to attack_result.json
        include_loss_history: Whether to include full loss history (can be large)
        run_id: Optional override for run_id
        
    Returns:
        Enriched result dictionary
    """
    with open(result_path, "r") as f:
        result = json.load(f)
    
    # Extract metadata from path
    metadata = extract_run_metadata(result_path)
    
    # Compute derived metrics if not present
    loss_history = result.get("loss_history", [])
    
    if loss_history:
        final_entry = loss_history[-1]
        
        # Final L2 norm
        final_l2 = final_entry.get("loss_perturbation", 0.0)
        
        # Final explanation change
        exp_sim = final_entry.get("explanation_similarity", 1.0)
        exp_change = final_entry.get("explanation_change", 1.0 - exp_sim)
        
        # Efficiency (explanation change per unit L2)
        efficiency = exp_change / final_l2 if final_l2 > 1e-8 else 0.0
        
        # Max explanation change and when it occurred
        exp_changes = [h.get("explanation_change", 1.0 - h.get("explanation_similarity", 1.0)) 
                       for h in loss_history]
        max_exp_change = max(exp_changes) if exp_changes else 0.0
        iter_at_max = exp_changes.index(max_exp_change) if exp_changes else 0
        
    else:
        final_l2 = result.get("perturbation_norm_l2", 0.0)
        exp_change = 1.0 - result.get("explanation_similarity", 1.0)
        efficiency = exp_change / final_l2 if final_l2 > 1e-8 else 0.0
        max_exp_change = exp_change
        iter_at_max = 0
    
    # Build enriched result
    enriched = {
        "run_id": run_id or metadata.get("run_dir", result_path.stem),
        "image": result.get("image") or metadata.get("image_stem"),
        "town": metadata.get("town"),
        "success": result.get("success", False),
        "iterations": result.get("iterations", len(loss_history)),
        "final_l2": final_l2,
        "final_exp_change": exp_change,
        "final_output_change": result.get("output_change", 0.0),
        "efficiency": efficiency,
        "max_exp_change": max_exp_change,
        "iter_at_max_exp_change": iter_at_max,
        "original_output": result.get("original_output"),
        "perturbed_output": result.get("perturbed_output"),
        "perturbation_linf": result.get("perturbation_norm_linf", 0.0),
        "result_path": str(result_path),
    }
    
    # Include loss history if requested
    if include_loss_history:
        enriched["loss_history"] = loss_history
    
    # Include config if it exists in the result
    if "config" in result:
        enriched["config"] = result["config"]
    
    return enriched


def aggregate_results(
    search_paths: List[Path],
    output_path: Path,
    include_loss_history: bool = True,
    filter_model: Optional[str] = None,
    filter_town: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Aggregate all attack results into a JSONL file.
    
    Args:
        search_paths: Paths to search for attack_result.json
        output_path: Output JSONL file path
        include_loss_history: Whether to include full loss history
        filter_model: Optional filter by model name
        filter_town: Optional filter by town name
        verbose: Print progress messages
        
    Returns:
        Summary statistics
    """
    results_found = 0
    results_written = 0
    errors = []
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as out_f:
        for result_path in find_attack_results(search_paths):
            results_found += 1
            
            try:
                enriched = load_and_enrich_result(
                    result_path, 
                    include_loss_history=include_loss_history
                )
                
                # Apply filters
                if filter_town and enriched.get("town") != filter_town:
                    continue
                if filter_model and filter_model not in str(enriched.get("config", {})):
                    continue
                
                # Write as JSONL
                out_f.write(json.dumps(enriched) + "\n")
                results_written += 1
                
                if verbose:
                    status = "✓" if enriched["success"] else "✗"
                    print(f"  [{status}] {enriched['run_id']}: "
                          f"eff={enriched['efficiency']:.4f}, "
                          f"Δexp={enriched['final_exp_change']:.4f}, "
                          f"L2={enriched['final_l2']:.4f}")
                    
            except Exception as e:
                errors.append((result_path, str(e)))
                if verbose:
                    print(f"  [!] Error loading {result_path}: {e}")
    
    summary = {
        "results_found": results_found,
        "results_written": results_written,
        "errors": len(errors),
        "output_path": str(output_path),
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Aggregation complete:")
        print(f"  Found: {results_found}")
        print(f"  Written: {results_written}")
        print(f"  Errors: {len(errors)}")
        print(f"  Output: {output_path}")
    
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate adversarial attack results into JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="Paths to search for attack_result.json files (directories or files)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output/adversarial/attack_runs.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Exclude loss_history from output (smaller files)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter by model name",
    )
    parser.add_argument(
        "--town",
        type=str,
        default=None,
        help="Filter by town name",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Aggregating attack results...")
    print(f"Search paths: {[str(p) for p in args.paths]}")
    print(f"Output: {args.output}")
    print()
    
    summary = aggregate_results(
        search_paths=args.paths,
        output_path=args.output,
        include_loss_history=not args.no_history,
        filter_model=args.model,
        filter_town=args.town,
        verbose=not args.quiet,
    )
    
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
