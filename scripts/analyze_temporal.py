#!/usr/bin/env python3
"""
Entry point for temporal stability analysis.
Wraps src/attn_stability_drive/analysis/temporal.py
"""

import argparse
from pathlib import Path

from attn_stability_drive.analysis.temporal import run_analysis


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute similarity metrics between consecutive frames' saliency maps."
    )
    p.add_argument("jsonl", type=Path, help="Path to measurements.jsonl (may include event lines).")
    p.add_argument("--out-shap", type=Path, required=True, help="Root dir of saliency outputs.")
    p.add_argument("--out-csv", type=Path, default=Path("saliency_sim.csv"), help="Where to write CSV.")
    p.add_argument("--skip-after-respawn", type=int, default=4, help="Skip next K frames after respawn.")
    p.add_argument("--pad", type=int, default=6, help="Zero-pad width for frame directories (default 6).")
    p.add_argument("--saliency-metric", type=str, choices=["ssim", "fsim", "lpips"],
                   default="ssim", help="Similarity metric for saliency maps.")
    p.add_argument("--image-metric", type=str, choices=["ssim", "fsim", "lpips"],
                   default="ssim", help="Similarity metric for original images.")
    p.add_argument("--with-image-sim", action="store_true",
                   help="Also compute similarity on original image pairs.")
    p.add_argument("--img-root", type=Path, default=None,
                   help="Root for resolving relative image_path; default = jsonl parent.")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar output.")
    return p.parse_args()


def main():
    args = parse_args()
    run_analysis(
        jsonl=args.jsonl,
        out_shap=args.out_shap,
        out_csv=args.out_csv,
        saliency_metric=args.saliency_metric,
        image_metric=args.image_metric,
        with_image_sim=args.with_image_sim,
        skip_after_respawn=args.skip_after_respawn,
        pad=args.pad,
        img_root=args.img_root,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
