"""Visualization module for attention stability analysis."""

from .plots import (
    make_temporal_plots,
    plot_adversarial_trajectory,
    plot_adversarial_comparison,
    plot_efficiency_distribution,
    plot_efficiency_convergence,
    plot_efficiency_scatter,
    plot_attack_trajectories_grid,
)

__all__ = [
    "make_temporal_plots",
    "plot_adversarial_trajectory",
    "plot_adversarial_comparison",
    "plot_efficiency_distribution",
    "plot_efficiency_convergence",
    "plot_efficiency_scatter",
    "plot_attack_trajectories_grid",
]
