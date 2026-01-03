"""Visualization module for attention stability analysis."""

from .plots import (
    make_temporal_plots,
    plot_adversarial_trajectory,
    plot_adversarial_comparison,
)

__all__ = [
    "make_temporal_plots",
    "plot_adversarial_trajectory",
    "plot_adversarial_comparison",
]
