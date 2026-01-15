"""Temporal analysis visualization functions."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def make_temporal_plots(
    xs: List[int],
    sal_vals: List[Optional[float]],
    img_vals: List[Optional[float]],
    steer_vals: List[Optional[float]],
    out_dir: Path,
    base_name: str,
    sal_label: str,
    img_label: Optional[str],
    show: bool = False,
    fixed_range: bool = True,
    highlight_indices: Optional[List[int]] = None,
) -> None:
    """Generate time-series and scatter plots for the chosen metrics.

    If fixed_range=True (default), all axes are clamped to [0,1].
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    METRIC_MIN, METRIC_MAX = 0.0, 1.0

    # Convert to arrays with NaNs for missing
    x = np.asarray(xs, dtype=np.int64)
    sal = np.array([np.nan if v is None else float(v) for v in sal_vals], dtype=np.float32)
    img = np.array([np.nan if v is None else float(v) for v in img_vals], dtype=np.float32)
    st = np.array([np.nan if v is None else float(v) for v in steer_vals], dtype=np.float32)

    if fixed_range:
        # Keep displayed values within [0,1] so plots/fit lines use the full fixed domain/range
        sal = np.clip(sal, METRIC_MIN, METRIC_MAX)
        img = np.clip(img, METRIC_MIN, METRIC_MAX)
        st = np.clip(st, METRIC_MIN, METRIC_MAX)

    # 1) Time series
    plt.figure(figsize=(10, 4.5), dpi=140)

    # Add highlights
    if highlight_indices:
        plt.vlines(
            highlight_indices,
            METRIC_MIN if fixed_range else 0,
            METRIC_MAX if fixed_range else 1,
            colors="red",
            alpha=0.2,
            linewidth=1,
            zorder=0,
            label="High I/O sim, low saliency sim",
        )

    plt.plot(x, sal, label=f"Saliency {sal_label}")  # NaNs break the line automatically
    if img_label and not np.all(np.isnan(img)):
        plt.plot(x, img, label=f"Image {img_label}")
    if not np.all(np.isnan(st)):
        plt.plot(x, st, label="Steer similarity")
    plt.xlabel("Pair index (consecutive frames within-towns only)")
    plt.ylabel("Metric value (0-1)" if fixed_range else "Metric value")
    plt.title("Metric trends over time")
    if fixed_range:
        plt.ylim(METRIC_MIN, METRIC_MAX)
    plt.legend()
    ts_path = out_dir / f"{base_name}_timeseries.png"
    plt.tight_layout()
    plt.savefig(ts_path)
    if show:
        plt.show()
    plt.close()

    # 2) Relationship scatter: Saliency vs image metric (only points with both)
    if img_label is not None:
        mask_img = ~np.isnan(sal) & ~np.isnan(img)
    else:
        mask_img = np.zeros_like(sal, dtype=bool)

    if img_label is not None and np.count_nonzero(mask_img) >= 2:
        s = sal[mask_img]
        g = img[mask_img]
        r = float(np.corrcoef(s, g)[0, 1])
        if fixed_range:
            xfit = np.linspace(METRIC_MIN, METRIC_MAX, 100, dtype=np.float32)
        else:
            xfit = np.linspace(np.nanmin(s), np.nanmax(s), 100, dtype=np.float32)
        m, b = np.polyfit(s, g, 1)
        yfit = m * xfit + b

        plt.figure(figsize=(6.5, 6.0), dpi=140)
        plt.scatter(s, g, alpha=0.6, s=10)
        plt.plot(xfit, yfit, linewidth=2, label=f"fit: y={m:.3f}x+{b:.3f}")
        plt.xlabel(f"Saliency {sal_label}")
        plt.ylabel(f"Image {img_label}")
        plt.title(f"Relationship (Pearson r = {r:.3f})")
        if fixed_range:
            plt.xlim(METRIC_MIN, METRIC_MAX)
            plt.ylim(METRIC_MIN, METRIC_MAX)
        plt.legend()
        sc_path = out_dir / f"{base_name}_scatter.png"
        plt.tight_layout()
        plt.savefig(sc_path)
        if show:
            plt.show()
        plt.close()

    # 3) Relationship scatter: Saliency metric vs steer similarity
    mask_st = ~np.isnan(sal) & ~np.isnan(st)
    if np.count_nonzero(mask_st) >= 2:
        s = sal[mask_st]
        g = st[mask_st]
        r = float(np.corrcoef(s, g)[0, 1])
        if fixed_range:
            xfit = np.linspace(METRIC_MIN, METRIC_MAX, 100, dtype=np.float32)
        else:
            xfit = np.linspace(np.nanmin(s), np.nanmax(s), 100, dtype=np.float32)
        m, b = np.polyfit(s, g, 1)
        yfit = m * xfit + b

        plt.figure(figsize=(6.5, 6.0), dpi=140)
        plt.scatter(s, g, alpha=0.6, s=10)
        plt.plot(xfit, yfit, linewidth=2, label=f"fit: y={m:.3f}x+{b:.3f}")
        plt.xlabel(f"Saliency {sal_label}")
        plt.ylabel("Steer similarity")
        plt.title(f"Relationship (Pearson r = {r:.3f})")
        if fixed_range:
            plt.xlim(METRIC_MIN, METRIC_MAX)
            plt.ylim(METRIC_MIN, METRIC_MAX)
        plt.legend()
        sc_path = out_dir / f"{base_name}_scatter_steer.png"
        plt.tight_layout()
        plt.savefig(sc_path)
        if show:
            plt.show()
        plt.close()


def plot_adversarial_trajectory(
    loss_history: List[dict],
    out_dir: Path,
    base_name: str = "adversarial",
    show: bool = False,
    title_prefix: str = "",
) -> dict:
    """
    Generate plots for adversarial attack trajectory analysis.
    
    Visualizes the "cost" of forcing explanation changes by plotting:
    1. L2 perturbation norm vs iteration
    2. Explanation change vs iteration
    3. L2 norm vs explanation change (Pareto/cost trade-off)
    4. All loss components over iterations
    
    Args:
        loss_history: List of dicts from attack_result.json, each containing:
            - loss_perturbation: L2 norm of perturbation
            - explanation_similarity: Cosine/other similarity (1 = identical)
            - explanation_change: 1 - similarity
            - output_change: Change in model output
            - loss_explanation, loss_output, total_loss: Loss components
        out_dir: Directory to save plots
        base_name: Prefix for output filenames
        show: Whether to display plots interactively
        title_prefix: Optional prefix for plot titles (e.g., image name)
        
    Returns:
        Dictionary with paths to generated plots and summary statistics
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data from loss history
    iterations = np.arange(len(loss_history))
    
    # L2 norm (from loss_perturbation which is the L2 norm)
    l2_norms = np.array([h.get("loss_perturbation", 0.0) for h in loss_history])
    
    # Explanation metrics
    exp_similarities = np.array([h.get("explanation_similarity", 1.0) for h in loss_history])
    exp_changes = np.array([h.get("explanation_change", 0.0) for h in loss_history])
    
    # If explanation_change not in history, compute from similarity
    if np.all(exp_changes == 0) and not np.all(exp_similarities == 1.0):
        exp_changes = 1.0 - exp_similarities
    
    # Output changes
    output_changes = np.array([h.get("output_change", 0.0) for h in loss_history])
    
    # Loss components
    loss_exp = np.array([h.get("loss_explanation", 0.0) for h in loss_history])
    loss_out = np.array([h.get("loss_output", 0.0) for h in loss_history])
    total_loss = np.array([h.get("total_loss", 0.0) for h in loss_history])
    
    saved_plots = {}
    title_base = f"{title_prefix} " if title_prefix else ""
    
    # Color scheme
    COLOR_L2 = "#2ecc71"       # Green for perturbation
    COLOR_EXP = "#e74c3c"      # Red for explanation change
    COLOR_OUT = "#3498db"      # Blue for output change
    COLOR_LOSS = "#9b59b6"     # Purple for total loss
    
    # =========================================================================
    # Plot 1: L2 Perturbation Norm vs Iteration
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    ax.plot(iterations, l2_norms, color=COLOR_L2, linewidth=2, marker='o', 
            markersize=3, label="L2 Norm")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("L2 Perturbation Norm", fontsize=12)
    ax.set_title(f"{title_base}Perturbation Magnitude Over Iterations", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add final value annotation
    ax.annotate(f"Final: {l2_norms[-1]:.4f}", 
                xy=(iterations[-1], l2_norms[-1]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color=COLOR_L2)
    
    plt.tight_layout()
    path_l2 = out_dir / f"{base_name}_l2_vs_iter.png"
    plt.savefig(path_l2)
    saved_plots["l2_vs_iter"] = path_l2
    if show:
        plt.show()
    plt.close()
    
    # =========================================================================
    # Plot 2: Explanation Change vs Iteration
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    ax.plot(iterations, exp_changes, color=COLOR_EXP, linewidth=2, marker='o',
            markersize=3, label="Explanation Change (1 - similarity)")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Explanation Change", fontsize=12)
    ax.set_title(f"{title_base}Explanation Change Over Iterations", fontsize=14)
    ax.set_ylim(0, max(0.1, exp_changes.max() * 1.1))  # Start from 0
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add final value annotation
    ax.annotate(f"Final: {exp_changes[-1]:.6f}", 
                xy=(iterations[-1], exp_changes[-1]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color=COLOR_EXP)
    
    plt.tight_layout()
    path_exp = out_dir / f"{base_name}_explanation_vs_iter.png"
    plt.savefig(path_exp)
    saved_plots["explanation_vs_iter"] = path_exp
    if show:
        plt.show()
    plt.close()
    
    # =========================================================================
    # Plot 3: L2 Norm vs Explanation Change (Cost Trade-off / Pareto)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 8), dpi=140)
    
    # Color points by iteration (early = light, late = dark)
    scatter = ax.scatter(l2_norms, exp_changes, c=iterations, cmap='viridis',
                         s=50, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Iteration", fontsize=11)
    
    # Connect points with a line to show trajectory
    ax.plot(l2_norms, exp_changes, color='gray', alpha=0.3, linewidth=1, zorder=0)
    
    # Mark start and end points
    ax.scatter([l2_norms[0]], [exp_changes[0]], color='green', s=150, 
               marker='^', label='Start', zorder=5, edgecolors='white', linewidth=2)
    ax.scatter([l2_norms[-1]], [exp_changes[-1]], color='red', s=150,
               marker='s', label='End', zorder=5, edgecolors='white', linewidth=2)
    
    ax.set_xlabel("L2 Perturbation Norm (Cost)", fontsize=12)
    ax.set_ylabel("Explanation Change (Benefit)", fontsize=12)
    ax.set_title(f"{title_base}Attack Cost-Benefit Trade-off", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Compute efficiency metric (explanation change per unit L2)
    if l2_norms[-1] > 1e-8:
        efficiency = exp_changes[-1] / l2_norms[-1]
        ax.text(0.95, 0.05, f"Efficiency: {efficiency:.6f}\n(Δexp / L2)",
                transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    path_tradeoff = out_dir / f"{base_name}_cost_tradeoff.png"
    plt.savefig(path_tradeoff)
    saved_plots["cost_tradeoff"] = path_tradeoff
    if show:
        plt.show()
    plt.close()
    
    # =========================================================================
    # Plot 4: All Metrics Combined (Dual Y-axis)
    # =========================================================================
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=140)
    
    # Left Y-axis: L2 norm and output change
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("L2 Norm / Output Change", fontsize=12, color='black')
    
    ln1 = ax1.plot(iterations, l2_norms, color=COLOR_L2, linewidth=2, 
                   label="L2 Perturbation Norm")
    ln2 = ax1.plot(iterations, output_changes, color=COLOR_OUT, linewidth=2,
                   linestyle='--', label="Output Change")
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3)
    
    # Right Y-axis: Explanation change
    ax2 = ax1.twinx()
    ax2.set_ylabel("Explanation Change", fontsize=12, color=COLOR_EXP)
    ln3 = ax2.plot(iterations, exp_changes, color=COLOR_EXP, linewidth=2,
                   label="Explanation Change")
    ax2.tick_params(axis='y', labelcolor=COLOR_EXP)
    ax2.set_ylim(0, max(0.1, exp_changes.max() * 1.1))
    
    # Combine legends
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')
    
    ax1.set_title(f"{title_base}Attack Trajectory: All Metrics", fontsize=14)
    
    plt.tight_layout()
    path_combined = out_dir / f"{base_name}_combined_metrics.png"
    plt.savefig(path_combined)
    saved_plots["combined_metrics"] = path_combined
    if show:
        plt.show()
    plt.close()
    
    # =========================================================================
    # Plot 5: Loss Components Over Iterations
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    
    ax.plot(iterations, loss_exp, label="Loss (Explanation)", linewidth=2, alpha=0.8)
    ax.plot(iterations, loss_out, label="Loss (Output)", linewidth=2, alpha=0.8)
    ax.plot(iterations, l2_norms, label="Loss (Perturbation/L2)", linewidth=2, alpha=0.8)
    ax.plot(iterations, total_loss, label="Total Loss", linewidth=2.5, 
            color=COLOR_LOSS, linestyle='--')
    
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Loss Value", fontsize=12)
    ax.set_title(f"{title_base}Loss Components Over Iterations", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    path_loss = out_dir / f"{base_name}_loss_components.png"
    plt.savefig(path_loss)
    saved_plots["loss_components"] = path_loss
    if show:
        plt.show()
    plt.close()
    
    # =========================================================================
    # Compute summary statistics
    # =========================================================================
    stats = {
        "total_iterations": len(loss_history),
        "final_l2_norm": float(l2_norms[-1]),
        "final_explanation_change": float(exp_changes[-1]),
        "final_output_change": float(output_changes[-1]),
        "max_l2_norm": float(l2_norms.max()),
        "max_explanation_change": float(exp_changes.max()),
        "efficiency": float(exp_changes[-1] / l2_norms[-1]) if l2_norms[-1] > 1e-8 else 0.0,
        "l2_at_max_exp_change": float(l2_norms[np.argmax(exp_changes)]),
        "iter_at_max_exp_change": int(np.argmax(exp_changes)),
    }
    
    return {
        "plots": saved_plots,
        "stats": stats,
    }


def plot_adversarial_comparison(
    results: List[dict],
    labels: List[str],
    out_dir: Path,
    base_name: str = "comparison",
    show: bool = False,
) -> dict:
    """
    Compare multiple adversarial attack trajectories.
    
    Args:
        results: List of attack_result dicts (each with 'loss_history')
        labels: Labels for each result (e.g., image names or config descriptions)
        out_dir: Directory to save plots
        base_name: Prefix for output filenames
        show: Whether to display plots interactively
        
    Returns:
        Dictionary with paths to generated plots
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = {}
    
    # =========================================================================
    # Plot 1: L2 vs Explanation Change for all attacks
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8), dpi=140)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, (result, label) in enumerate(zip(results, labels)):
        loss_history = result.get("loss_history", [])
        if not loss_history:
            continue
            
        l2_norms = np.array([h.get("loss_perturbation", 0.0) for h in loss_history])
        exp_changes = np.array([h.get("explanation_change", 0.0) for h in loss_history])
        
        # If explanation_change not computed, derive from similarity
        if np.all(exp_changes == 0):
            exp_similarities = np.array([h.get("explanation_similarity", 1.0) for h in loss_history])
            exp_changes = 1.0 - exp_similarities
        
        ax.plot(l2_norms, exp_changes, color=colors[i], linewidth=2, 
                label=label, alpha=0.8)
        ax.scatter([l2_norms[-1]], [exp_changes[-1]], color=colors[i], 
                   s=100, marker='o', edgecolors='white', linewidth=2, zorder=5)
    
    ax.set_xlabel("L2 Perturbation Norm (Cost)", fontsize=12)
    ax.set_ylabel("Explanation Change (Benefit)", fontsize=12)
    ax.set_title("Attack Cost-Benefit Comparison", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    path_compare = out_dir / f"{base_name}_cost_comparison.png"
    plt.savefig(path_compare)
    saved_plots["cost_comparison"] = path_compare
    if show:
        plt.show()
    plt.close()
    
    # =========================================================================
    # Plot 2: Final metrics bar chart
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=140)
    
    final_l2 = []
    final_exp_change = []
    final_efficiency = []
    
    for result in results:
        loss_history = result.get("loss_history", [])
        if loss_history:
            l2 = loss_history[-1].get("loss_perturbation", 0.0)
            exp_sim = loss_history[-1].get("explanation_similarity", 1.0)
            exp_change = 1.0 - exp_sim
            efficiency = exp_change / l2 if l2 > 1e-8 else 0.0
        else:
            l2 = exp_change = efficiency = 0.0
        
        final_l2.append(l2)
        final_exp_change.append(exp_change)
        final_efficiency.append(efficiency)
    
    x = np.arange(len(labels))
    
    axes[0].bar(x, final_l2, color=colors[:len(labels)])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_ylabel("L2 Norm")
    axes[0].set_title("Final Perturbation Magnitude")
    
    axes[1].bar(x, final_exp_change, color=colors[:len(labels)])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_ylabel("Explanation Change")
    axes[1].set_title("Final Explanation Change")
    
    axes[2].bar(x, final_efficiency, color=colors[:len(labels)])
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha='right')
    axes[2].set_ylabel("Efficiency (ΔExp / L2)")
    axes[2].set_title("Attack Efficiency")
    
    plt.tight_layout()
    path_bars = out_dir / f"{base_name}_final_metrics.png"
    plt.savefig(path_bars)
    saved_plots["final_metrics"] = path_bars
    if show:
        plt.show()
    plt.close()
    
    return {"plots": saved_plots}


# =============================================================================
# Multi-Run Attack Efficiency Visualization Functions
# =============================================================================


def plot_efficiency_distribution(
    runs: List[Dict[str, Any]],
    out_dir: Path,
    group_by: Optional[str] = None,
    base_name: str = "efficiency",
    show: bool = False,
) -> Dict[str, Any]:
    """
    Plot efficiency distribution across multiple attack runs.
    
    Creates histogram and violin plots showing the distribution of attack
    efficiency (explanation change per unit L2 perturbation).
    
    Args:
        runs: List of run dictionaries with 'efficiency', 'final_l2', 'final_exp_change'
        out_dir: Directory to save plots
        group_by: Optional field to group runs by (e.g., 'town', 'model')
        base_name: Prefix for output filenames
        show: Whether to display plots interactively
        
    Returns:
        Dictionary with paths to generated plots
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = {}
    
    # Extract data
    efficiencies = np.array([r.get("efficiency", 0.0) for r in runs])
    l2_norms = np.array([r.get("final_l2", 0.0) for r in runs])
    exp_changes = np.array([r.get("final_exp_change", 0.0) for r in runs])
    successes = np.array([r.get("success", False) for r in runs])
    
    # Filter out invalid data (efficiency = 0 means no perturbation)
    valid_mask = efficiencies > 0
    if not np.any(valid_mask):
        print("Warning: No runs with positive efficiency")
        valid_mask = np.ones(len(efficiencies), dtype=bool)
    
    # =========================================================================
    # Plot 1: Efficiency Histogram
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    
    if group_by and group_by in runs[0]:
        # Grouped histogram
        groups = {}
        for i, run in enumerate(runs):
            key = run.get(group_by, "unknown")
            if key not in groups:
                groups[key] = []
            groups[key].append(efficiencies[i])
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        for (key, values), color in zip(sorted(groups.items()), colors):
            values = np.array(values)
            values = values[values > 0]  # Filter zeros
            if len(values) > 0:
                ax.hist(values, bins=20, alpha=0.6, label=f"{key} (n={len(values)})", 
                        color=color, edgecolor='white')
        ax.legend()
    else:
        # Single histogram
        valid_eff = efficiencies[valid_mask]
        ax.hist(valid_eff, bins=30, color='#3498db', alpha=0.7, edgecolor='white')
        
        # Add statistics
        mean_eff = np.mean(valid_eff)
        median_eff = np.median(valid_eff)
        ax.axvline(mean_eff, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_eff:.4f}')
        ax.axvline(median_eff, color='green', linestyle=':', linewidth=2, label=f'Median: {median_eff:.4f}')
        ax.legend()
    
    ax.set_xlabel("Attack Efficiency (Δexp / L2)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Attack Efficiency Distribution (n={np.sum(valid_mask)})", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path_hist = out_dir / f"{base_name}_histogram.png"
    plt.savefig(path_hist)
    saved_plots["histogram"] = path_hist
    if show:
        plt.show()
    plt.close()
    
    # =========================================================================
    # Plot 2: Box/Violin Plot (if grouped)
    # =========================================================================
    if group_by and group_by in runs[0]:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
        
        groups = {}
        for i, run in enumerate(runs):
            key = run.get(group_by, "unknown")
            if key not in groups:
                groups[key] = []
            if efficiencies[i] > 0:
                groups[key].append(efficiencies[i])
        
        # Filter empty groups
        groups = {k: v for k, v in groups.items() if len(v) > 0}
        
        if groups:
            labels = sorted(groups.keys())
            data = [groups[k] for k in labels]
            
            # Box plot with individual points
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            # Overlay individual points
            for i, (d, color) in enumerate(zip(data, colors)):
                x = np.random.normal(i + 1, 0.04, size=len(d))
                ax.scatter(x, d, alpha=0.4, color=color, s=20, zorder=3)
            
            ax.set_xlabel(group_by.capitalize(), fontsize=12)
            ax.set_ylabel("Attack Efficiency (Δexp / L2)", fontsize=12)
            ax.set_title(f"Efficiency by {group_by.capitalize()}", fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            path_box = out_dir / f"{base_name}_boxplot.png"
            plt.savefig(path_box)
            saved_plots["boxplot"] = path_box
            if show:
                plt.show()
            plt.close()
    
    # =========================================================================
    # Plot 3: Multi-metric distribution (3 subplots)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=140)
    
    # Efficiency
    valid_eff = efficiencies[valid_mask]
    axes[0].hist(valid_eff, bins=25, color='#9b59b6', alpha=0.7, edgecolor='white')
    axes[0].axvline(np.mean(valid_eff), color='red', linestyle='--', linewidth=1.5)
    axes[0].set_xlabel("Efficiency (Δexp / L2)")
    axes[0].set_title("Attack Efficiency")
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # L2 Norm
    valid_l2 = l2_norms[l2_norms > 0]
    if len(valid_l2) > 0:
        axes[1].hist(valid_l2, bins=25, color='#2ecc71', alpha=0.7, edgecolor='white')
        axes[1].axvline(np.mean(valid_l2), color='red', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel("L2 Perturbation Norm")
    axes[1].set_title("Perturbation Magnitude")
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Explanation Change
    valid_exp = exp_changes[exp_changes > 0]
    if len(valid_exp) > 0:
        axes[2].hist(valid_exp, bins=25, color='#e74c3c', alpha=0.7, edgecolor='white')
        axes[2].axvline(np.mean(valid_exp), color='red', linestyle='--', linewidth=1.5)
    axes[2].set_xlabel("Explanation Change (1 - similarity)")
    axes[2].set_title("Explanation Change")
    axes[2].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f"Attack Metrics Distribution (n={len(runs)})", fontsize=14, y=1.02)
    plt.tight_layout()
    path_multi = out_dir / f"{base_name}_multi_distribution.png"
    plt.savefig(path_multi, bbox_inches='tight')
    saved_plots["multi_distribution"] = path_multi
    if show:
        plt.show()
    plt.close()
    
    return {"plots": saved_plots}


def plot_efficiency_convergence(
    runs: List[Dict[str, Any]],
    out_dir: Path,
    group_by: Optional[str] = None,
    base_name: str = "convergence",
    show: bool = False,
    max_iterations: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Plot efficiency convergence over iterations with confidence bands.
    
    Shows mean efficiency trajectory with standard deviation bands,
    allowing comparison of how quickly attacks achieve their goals.
    
    Args:
        runs: List of run dictionaries with 'loss_history'
        out_dir: Directory to save plots
        group_by: Optional field to group runs by
        base_name: Prefix for output filenames
        show: Whether to display plots interactively
        max_iterations: Maximum iterations to plot (None = auto)
        
    Returns:
        Dictionary with paths to generated plots
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = {}
    
    # Filter runs with loss_history
    runs_with_history = [r for r in runs if r.get("loss_history")]
    if not runs_with_history:
        print("Warning: No runs with loss_history")
        return {"plots": {}}
    
    # =========================================================================
    # Plot 1: Efficiency Over Iterations (mean ± std)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    
    def compute_efficiency_trajectory(loss_history):
        """Compute efficiency at each iteration."""
        efficiencies = []
        for h in loss_history:
            l2 = h.get("loss_perturbation", 0.0)
            exp_sim = h.get("explanation_similarity", 1.0)
            exp_change = h.get("explanation_change", 1.0 - exp_sim)
            eff = exp_change / l2 if l2 > 1e-8 else 0.0
            efficiencies.append(eff)
        return np.array(efficiencies)
    
    if group_by and group_by in runs_with_history[0]:
        # Grouped plot
        groups = {}
        for run in runs_with_history:
            key = run.get(group_by, "unknown")
            if key not in groups:
                groups[key] = []
            groups[key].append(compute_efficiency_trajectory(run["loss_history"]))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        
        for (key, trajectories), color in zip(sorted(groups.items()), colors):
            # Pad trajectories to same length
            max_len = max(len(t) for t in trajectories)
            if max_iterations:
                max_len = min(max_len, max_iterations)
            
            padded = np.full((len(trajectories), max_len), np.nan)
            for i, t in enumerate(trajectories):
                length = min(len(t), max_len)
                padded[i, :length] = t[:length]
            
            mean_traj = np.nanmean(padded, axis=0)
            std_traj = np.nanstd(padded, axis=0)
            iterations = np.arange(max_len)
            
            ax.plot(iterations, mean_traj, color=color, linewidth=2, 
                    label=f"{key} (n={len(trajectories)})")
            ax.fill_between(iterations, mean_traj - std_traj, mean_traj + std_traj,
                           color=color, alpha=0.2)
    else:
        # Single group
        trajectories = [compute_efficiency_trajectory(r["loss_history"]) 
                       for r in runs_with_history]
        
        max_len = max(len(t) for t in trajectories)
        if max_iterations:
            max_len = min(max_len, max_iterations)
        
        padded = np.full((len(trajectories), max_len), np.nan)
        for i, t in enumerate(trajectories):
            length = min(len(t), max_len)
            padded[i, :length] = t[:length]
        
        mean_traj = np.nanmean(padded, axis=0)
        std_traj = np.nanstd(padded, axis=0)
        iterations = np.arange(max_len)
        
        ax.plot(iterations, mean_traj, color='#3498db', linewidth=2.5, 
                label=f"Mean (n={len(trajectories)})")
        ax.fill_between(iterations, mean_traj - std_traj, mean_traj + std_traj,
                       color='#3498db', alpha=0.3, label='±1 std')
        
        # Also plot individual trajectories faintly
        for t in trajectories[:20]:  # Limit to 20 for clarity
            ax.plot(np.arange(len(t)), t, color='gray', alpha=0.1, linewidth=0.5)
    
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Attack Efficiency (Δexp / L2)", fontsize=12)
    ax.set_title("Efficiency Convergence Over Iterations", fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    
    plt.tight_layout()
    path_conv = out_dir / f"{base_name}_efficiency.png"
    plt.savefig(path_conv)
    saved_plots["efficiency_convergence"] = path_conv
    if show:
        plt.show()
    plt.close()
    
    # =========================================================================
    # Plot 2: Explanation Change Over Iterations
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    
    def get_exp_change_trajectory(loss_history):
        exp_changes = []
        for h in loss_history:
            exp_sim = h.get("explanation_similarity", 1.0)
            exp_change = h.get("explanation_change", 1.0 - exp_sim)
            exp_changes.append(exp_change)
        return np.array(exp_changes)
    
    trajectories = [get_exp_change_trajectory(r["loss_history"]) 
                   for r in runs_with_history]
    
    max_len = max(len(t) for t in trajectories)
    if max_iterations:
        max_len = min(max_len, max_iterations)
    
    padded = np.full((len(trajectories), max_len), np.nan)
    for i, t in enumerate(trajectories):
        length = min(len(t), max_len)
        padded[i, :length] = t[:length]
    
    mean_traj = np.nanmean(padded, axis=0)
    std_traj = np.nanstd(padded, axis=0)
    iterations = np.arange(max_len)
    
    ax.plot(iterations, mean_traj, color='#e74c3c', linewidth=2.5, label='Mean')
    ax.fill_between(iterations, mean_traj - std_traj, mean_traj + std_traj,
                   color='#e74c3c', alpha=0.3, label='±1 std')
    
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Explanation Change (1 - similarity)", fontsize=12)
    ax.set_title(f"Explanation Change Convergence (n={len(trajectories)})", fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    path_exp = out_dir / f"{base_name}_exp_change.png"
    plt.savefig(path_exp)
    saved_plots["exp_change_convergence"] = path_exp
    if show:
        plt.show()
    plt.close()
    
    # =========================================================================
    # Plot 3: L2 Norm Over Iterations
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    
    def get_l2_trajectory(loss_history):
        return np.array([h.get("loss_perturbation", 0.0) for h in loss_history])
    
    trajectories = [get_l2_trajectory(r["loss_history"]) for r in runs_with_history]
    
    padded = np.full((len(trajectories), max_len), np.nan)
    for i, t in enumerate(trajectories):
        length = min(len(t), max_len)
        padded[i, :length] = t[:length]
    
    mean_traj = np.nanmean(padded, axis=0)
    std_traj = np.nanstd(padded, axis=0)
    
    ax.plot(iterations, mean_traj, color='#2ecc71', linewidth=2.5, label='Mean')
    ax.fill_between(iterations, mean_traj - std_traj, mean_traj + std_traj,
                   color='#2ecc71', alpha=0.3, label='±1 std')
    
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("L2 Perturbation Norm", fontsize=12)
    ax.set_title(f"Perturbation Growth Over Iterations (n={len(trajectories)})", fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    path_l2 = out_dir / f"{base_name}_l2_norm.png"
    plt.savefig(path_l2)
    saved_plots["l2_convergence"] = path_l2
    if show:
        plt.show()
    plt.close()
    
    return {"plots": saved_plots}


def plot_efficiency_scatter(
    runs: List[Dict[str, Any]],
    out_dir: Path,
    group_by: Optional[str] = None,
    base_name: str = "scatter",
    show: bool = False,
) -> Dict[str, Any]:
    """
    Create scatter plots showing relationships between attack metrics.
    
    Plots:
    - L2 vs Explanation Change (colored by success or group)
    - Efficiency vs L2 norm
    
    Args:
        runs: List of run dictionaries
        out_dir: Directory to save plots
        group_by: Optional field to color points by
        base_name: Prefix for output filenames
        show: Whether to display plots interactively
        
    Returns:
        Dictionary with paths to generated plots
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = {}
    
    # Extract data
    l2_norms = np.array([r.get("final_l2", 0.0) for r in runs])
    exp_changes = np.array([r.get("final_exp_change", 0.0) for r in runs])
    efficiencies = np.array([r.get("efficiency", 0.0) for r in runs])
    successes = np.array([r.get("success", False) for r in runs])
    
    # =========================================================================
    # Plot 1: L2 vs Explanation Change (Cost-Benefit)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8), dpi=140)
    
    if group_by and group_by in runs[0]:
        # Color by group
        groups = {}
        for i, run in enumerate(runs):
            key = run.get(group_by, "unknown")
            if key not in groups:
                groups[key] = {"l2": [], "exp": [], "eff": []}
            groups[key]["l2"].append(l2_norms[i])
            groups[key]["exp"].append(exp_changes[i])
            groups[key]["eff"].append(efficiencies[i])
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        for (key, data), color in zip(sorted(groups.items()), colors):
            ax.scatter(data["l2"], data["exp"], c=[color], s=50, alpha=0.7,
                      label=f"{key} (n={len(data['l2'])})", edgecolors='white', linewidth=0.5)
    else:
        # Color by efficiency
        scatter = ax.scatter(l2_norms, exp_changes, c=efficiencies, cmap='viridis',
                            s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Efficiency (Δexp / L2)", fontsize=11)
        
        # Mark successful vs failed
        success_mask = successes == True
        if np.any(success_mask):
            ax.scatter(l2_norms[success_mask], exp_changes[success_mask], 
                      facecolors='none', edgecolors='green', s=100, linewidth=2,
                      label='Successful', zorder=5)
        fail_mask = successes == False
        if np.any(fail_mask) and np.any(l2_norms[fail_mask] > 0):
            ax.scatter(l2_norms[fail_mask], exp_changes[fail_mask],
                      facecolors='none', edgecolors='red', s=100, linewidth=2,
                      label='Failed', zorder=5)
    
    ax.set_xlabel("L2 Perturbation Norm (Cost)", fontsize=12)
    ax.set_ylabel("Explanation Change (Benefit)", fontsize=12)
    ax.set_title(f"Attack Cost-Benefit Analysis (n={len(runs)})", fontsize=14)
    ax.grid(True, alpha=0.3)
    if group_by or np.any(successes):
        ax.legend(loc='best')
    
    plt.tight_layout()
    path_cost = out_dir / f"{base_name}_cost_benefit.png"
    plt.savefig(path_cost)
    saved_plots["cost_benefit"] = path_cost
    if show:
        plt.show()
    plt.close()
    
    # =========================================================================
    # Plot 2: Efficiency vs L2 Norm
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    
    # Filter to valid points
    valid_mask = (l2_norms > 0) & (efficiencies > 0)
    
    if np.any(valid_mask):
        ax.scatter(l2_norms[valid_mask], efficiencies[valid_mask], 
                  c=exp_changes[valid_mask], cmap='Reds', s=50, alpha=0.7,
                  edgecolors='white', linewidth=0.5)
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("Explanation Change", fontsize=11)
    
    ax.set_xlabel("L2 Perturbation Norm", fontsize=12)
    ax.set_ylabel("Attack Efficiency (Δexp / L2)", fontsize=12)
    ax.set_title(f"Efficiency vs Perturbation Magnitude (n={np.sum(valid_mask)})", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path_eff = out_dir / f"{base_name}_efficiency_vs_l2.png"
    plt.savefig(path_eff)
    saved_plots["efficiency_vs_l2"] = path_eff
    if show:
        plt.show()
    plt.close()
    
    return {"plots": saved_plots}


def plot_attack_trajectories_grid(
    runs: List[Dict[str, Any]],
    out_dir: Path,
    base_name: str = "trajectories",
    show: bool = False,
    ncols: int = 4,
    sort_by: str = "efficiency",
    fixed_range: bool = True,
) -> Dict[str, Any]:
    """
    Create a grid of small multiples showing individual attack trajectories.
    
    Each subplot shows L2 vs Explanation Change for one run, allowing
    visual comparison of attack behavior across different images.
    
    Args:
        runs: List of run dictionaries with 'loss_history'
        out_dir: Directory to save plots
        base_name: Prefix for output filenames
        show: Whether to display plots interactively
        ncols: Number of columns in grid
        sort_by: How to sort runs ('efficiency', 'exp_change', 'l2')
        fixed_range: If True, use same axis limits for all subplots
        
    Returns:
        Dictionary with paths to generated plots
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = {}
    
    # Filter runs with loss_history
    runs_with_history = [r for r in runs if r.get("loss_history")]
    if not runs_with_history:
        print("Warning: No runs with loss_history")
        return {"plots": {}}
    
    # Sort runs
    if sort_by == "efficiency":
        runs_with_history = sorted(runs_with_history, 
                                   key=lambda r: r.get("efficiency", 0), reverse=True)
    elif sort_by == "exp_change":
        runs_with_history = sorted(runs_with_history,
                                   key=lambda r: r.get("final_exp_change", 0), reverse=True)
    elif sort_by == "l2":
        runs_with_history = sorted(runs_with_history,
                                   key=lambda r: r.get("final_l2", 0), reverse=True)
    
    # Calculate global limits if fixed_range is requested
    max_l2 = 0.0
    max_exp = 0.0
    if fixed_range:
        for run in runs_with_history:
            lh = run["loss_history"]
            l2s = [h.get("loss_perturbation", 0.0) for h in lh]
            exps = [h.get("explanation_change", 1.0 - h.get("explanation_similarity", 1.0)) for h in lh]
            if l2s:
                max_l2 = max(max_l2, max(l2s))
            if exps:
                max_exp = max(max_exp, max(exps))
        
        # Add slight margin
        max_l2 *= 1.05
        max_exp *= 1.05

    n_runs = len(runs_with_history)
    nrows = (n_runs + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3 * nrows), dpi=120)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, run in enumerate(runs_with_history):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        loss_history = run["loss_history"]
        
        l2_norms = np.array([h.get("loss_perturbation", 0.0) for h in loss_history])
        exp_changes = np.array([h.get("explanation_change", 
                                      1.0 - h.get("explanation_similarity", 1.0)) 
                               for h in loss_history])
        iterations = np.arange(len(loss_history))
        
        # Plot trajectory with color gradient
        scatter = ax.scatter(l2_norms, exp_changes, c=iterations, cmap='viridis',
                            s=15, alpha=0.8)
        ax.plot(l2_norms, exp_changes, color='gray', alpha=0.3, linewidth=0.5)
        
        # Mark start and end
        ax.scatter([l2_norms[0]], [exp_changes[0]], color='green', s=40, 
                   marker='^', zorder=5)
        ax.scatter([l2_norms[-1]], [exp_changes[-1]], color='red', s=40,
                   marker='s', zorder=5)
        
        # Title with run info
        run_id = run.get("run_id", f"Run {idx+1}")
        if len(run_id) > 20:
            run_id = run_id[:17] + "..."
        efficiency = run.get("efficiency", 0.0)
        ax.set_title(f"{run_id}\neff={efficiency:.3f}", fontsize=9)
        
        ax.set_xlabel("L2", fontsize=8)
        ax.set_ylabel("Δexp", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        
        if fixed_range:
            ax.set_xlim(0, max_l2)
            ax.set_ylim(0, max_exp)
    
    # Hide empty subplots
    for idx in range(n_runs, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    title_suffix = " (Fixed Scale)" if fixed_range else ""
    fig.suptitle(f"Attack Trajectories (n={n_runs}, sorted by {sort_by}){title_suffix}", 
                 fontsize=14, y=1.01)
    plt.tight_layout()
    
    path_grid = out_dir / f"{base_name}_grid.png"
    plt.savefig(path_grid, bbox_inches='tight')
    saved_plots["trajectories_grid"] = path_grid
    if show:
        plt.show()
    plt.close()
    
    return {"plots": saved_plots}

