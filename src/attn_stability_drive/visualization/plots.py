"""Temporal analysis visualization functions."""

from pathlib import Path
from typing import List, Optional

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
