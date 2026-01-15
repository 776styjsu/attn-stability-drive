#!/usr/bin/env python3
"""
adversarial_explanation.py

Adversarial perturbation generation for explanation instability testing.

This script finds minimal perturbations to input images such that:
1. The model's output (steering angle) remains similar
2. The explanation (saliency map) changes drastically

The attack uses differentiable explanation approximations (gradient, smooth_grad,
integrated_grad, or gradient_shap) to enable backpropagation through the explanation.

Usage:
  # Basic usage with smooth_grad (default)
  python adversarial_explanation.py \
      --model-class attn_stability_drive.models.dave2.DAVE2v1 \
      --model-args '{"input_shape": [180, 320]}' \
      --ckpt model.pth \
      --image input.jpg \
      --out output/adversarial/

  # With GradientSHAP (requires background dataset)
  python adversarial_explanation.py \
      --model-class attn_stability_drive.models.dave2.DAVE2v1 \
      --ckpt model.pth \
      --image input.jpg \
      --diff-method gradient_shap \
      --data-jsonl data.jsonl \
      --img-root /path/to/images \
      --bg-count 16 \
      --out output/adversarial/

  # With config file
  python adversarial_explanation.py --config configs/experiments/adversarial.yaml
"""

import argparse
import datetime
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attn_stability_drive.adversarial import ExplanationAdversarialAttack
from attn_stability_drive.visualization import plot_adversarial_trajectory


def import_class(class_path: str) -> Type:
    """Dynamically import a class from a module path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_model(
    model_class: str,
    model_args: Dict[str, Any],
    checkpoint_path: str,
    device: str = "auto"
) -> tuple:
    """Load a model from a class path and checkpoint."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ModelClass = import_class(model_class)
    model = ModelClass(**model_args).to(device)
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    
    # Handle DataParallel prefix if present
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model_keys = set(model.state_dict().keys())
    
    # Filter to only keys that exist in model
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    
    # Check for missing keys
    missing_keys = model_keys - set(filtered_state_dict.keys())
    if missing_keys:
        print(f"[adversarial] WARNING: {len(missing_keys)} keys missing from checkpoint!")
        print(f"[adversarial] Missing keys: {list(missing_keys)[:5]} ...")
        
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    print(f"[adversarial] Loaded model: {model_class}")
    print(f"[adversarial] Device: {device}")
    
    return model, device


def build_preprocess(
    input_shape: tuple,
    mean: list = None,
    std: list = None
) -> Compose:
    """Build preprocessing pipeline."""
    mean = mean or [0.5, 0.5, 0.5]
    std = std or [0.5, 0.5, 0.5]
    H, W = input_shape
    return Compose([
        Resize((H, W)),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])


def build_background(
    data_jsonl: Optional[str],
    img_root: Optional[str],
    preprocess_fn: Compose,
    device: str,
    bg_count: int = 16,
    sample_size: int = 1000,
    seed: int = 42,
) -> Optional[torch.Tensor]:
    """
    Build background tensor for GradientSHAP from a dataset.
    
    Args:
        data_jsonl: Path to JSONL file with image paths
        img_root: Root directory for resolving relative image paths
        preprocess_fn: Preprocessing transform
        device: Device to place tensor on
        bg_count: Number of background samples to select
        sample_size: Number of images to sample from dataset
        seed: Random seed for reproducibility
        
    Returns:
        Background tensor (N, C, H, W) or None if data not available
    """
    import numpy as np
    
    if not data_jsonl:
        return None
    
    # Read JSONL
    with open(data_jsonl) as f:
        lines = f.readlines()
    
    if not lines:
        return None
    
    # Parse image paths
    img_root = Path(img_root) if img_root else Path(".")
    paths = []
    for line in lines:
        try:
            entry = json.loads(line)
            rel_path = entry.get("image_path", "")
            if rel_path:
                p = img_root / rel_path
                if p.exists():
                    paths.append(p)
        except:
            continue
    
    if not paths:
        print("[adversarial] Warning: No valid image paths found in JSONL")
        return None
    
    # Sample subset
    rng = np.random.RandomState(seed)
    if len(paths) > sample_size:
        indices = rng.choice(len(paths), sample_size, replace=False)
        paths = [paths[i] for i in indices]
    
    # Uniformly sample bg_count images from the paths
    if len(paths) > bg_count:
        indices = np.linspace(0, len(paths) - 1, bg_count, dtype=int)
        paths = [paths[i] for i in indices]
    
    # Load and preprocess
    backgrounds = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            tensor = preprocess_fn(img)
            backgrounds.append(tensor)
        except Exception as e:
            print(f"[adversarial] Warning: Failed to load {p}: {e}")
            continue
    
    if not backgrounds:
        return None
    
    return torch.stack(backgrounds).to(device)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate adversarial perturbations that break explanation stability"
    )
    
    # Config file (optional, overrides other args)
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    
    # Model arguments
    parser.add_argument(
        "--model-class", type=str,
        default="attn_stability_drive.models.dave2.DAVE2v1",
        help="Full class path for the model"
    )
    parser.add_argument(
        "--model-args", type=str, default='{"input_shape": [180, 320]}',
        help="JSON string of model constructor arguments"
    )
    parser.add_argument("--ckpt", type=str, required=False, help="Path to model checkpoint")
    
    # Input
    parser.add_argument("--image", type=str, required=False, help="Path to input image")
    parser.add_argument("--image-dir", type=str, help="Directory of images to process")
    
    # Output
    parser.add_argument("--out", type=str, default="output/adversarial", help="Output directory")
    
    # Attack parameters
    parser.add_argument("--epsilon", type=float, default=0.1, help="Max perturbation magnitude (L∞)")
    parser.add_argument("--step-size", type=float, default=0.01, help="Optimization step size")
    parser.add_argument("--max-iter", type=int, default=100, help="Max optimization iterations")
    parser.add_argument("--output-threshold", type=float, default=0.05, help="Max allowed output change")
    parser.add_argument("--explanation-threshold", type=float, default=0.1, help="Min required explanation change")
    
    # Loss weights
    parser.add_argument("--lambda-exp", type=float, default=1.0, help="Weight for explanation dissimilarity")
    parser.add_argument("--lambda-out", type=float, default=10.0, help="Weight for output similarity")
    parser.add_argument("--lambda-pert", type=float, default=0.01, help="Weight for perturbation norm")
    
    # Loss metrics
    parser.add_argument("--explanation-metric", type=str, default="mse", choices=["mse", "cosine", "correlation", "l1", "ssim", "topk"], help="Metric for explanation dissimilarity")
    parser.add_argument("--topk-percent", type=float, default=0.1, help="For 'topk' metric, fraction of top pixels to consider (default 10%%)")
    parser.add_argument("--output-metric", type=str, default="mse", choices=["mse", "l1"], help="Metric for output similarity")
    parser.add_argument("--perturbation-norm", type=str, default="l2", choices=["l2", "linf"], help="Norm for perturbation penalty")

    # Explanation method
    parser.add_argument(
        "--diff-method", type=str, default="smooth_grad",
        choices=["gradient", "smooth_grad", "integrated_grad", "gradient_shap"],
        help="Differentiable explanation method for optimization"
    )
    parser.add_argument("--smooth-samples", type=int, default=10, help="Samples for smooth_grad")
    parser.add_argument("--ig-steps", type=int, default=50, help="Interpolation steps for integrated_grad/gradient_shap (default 50 matches SHAP library)")
    parser.add_argument("--gradient-shap-samples", type=int, default=None, help="Background samples per iteration for gradient_shap. Default (None) uses all samples. Set to 4-8 for faster optimization.")
    
    # Background dataset (for gradient_shap)
    parser.add_argument("--data-jsonl", type=str, help="JSONL file for background sampling (required for gradient_shap)")
    parser.add_argument("--img-root", type=str, help="Root directory for resolving relative image paths in data-jsonl")
    parser.add_argument("--bg-count", type=int, default=16, help="Number of background samples to use")
    parser.add_argument("--bg-sample-size", type=int, default=1000, help="Number of images to sample from dataset for background selection")
    
    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cuda', 'cpu', or 'auto'")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility. If set, enables deterministic mode.")
    
    # Perturbation mask
    parser.add_argument(
        "--mask", type=str, default=None,
        help="Path to perturbation mask image (grayscale). White pixels (255) are fully attackable, "
             "black pixels (0) are protected. Mask will be resized to match input dimensions."
    )

    # Save options
    parser.add_argument("--save-last", action="store_true", help="Save the result from the last iteration instead of the best iteration")

    # Plotting
    parser.add_argument("--plot", action="store_true", help="Generate trajectory plots after attack")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    # Force CUDA initialization to avoid cuBLAS warnings
    # The warning "Attempting to run cuBLAS, but there was no current CUDA context!"
    # happens when cuBLAS is first used. We need to do an actual matmul to warm it up.
    if torch.cuda.is_available():
        try:
            # Initialize CUDA context
            torch.cuda.init()
            # Perform a small matmul to initialize cuBLAS
            _warmup = torch.randn(2, 2, device='cuda') @ torch.randn(2, 2, device='cuda')
            del _warmup
            torch.cuda.empty_cache()
        except Exception:
            pass

    args = parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Merge config with args (args take precedence for explicitly set values)
    model_class = args.model_class
    model_args = json.loads(args.model_args)
    checkpoint = args.ckpt or config.get("model", {}).get("checkpoint")
    
    if not checkpoint:
        print("Error: --ckpt or config.model.checkpoint is required")
        sys.exit(1)
    
    # Get input shape from model args
    input_shape = model_args.get("input_shape", [180, 320])
    if isinstance(input_shape, list):
        input_shape = tuple(input_shape)
    
    # Load model
    model, device = load_model(model_class, model_args, checkpoint, args.device)
    
    # Build preprocessing
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    preprocess_fn = build_preprocess(input_shape, mean=norm_mean, std=norm_std)
    
    # Build background for GradientSHAP if needed
    background = None
    if args.diff_method == "gradient_shap":
        background = build_background(
            data_jsonl=args.data_jsonl,
            img_root=args.img_root,
            preprocess_fn=preprocess_fn,
            device=device,
            bg_count=args.bg_count,
            sample_size=args.bg_sample_size,
        )
        if background is not None:
            print(f"[adversarial] Built background with {len(background)} samples")
        else:
            print("[adversarial] Warning: GradientSHAP requested but no background built. Using zero baseline.")
            background = torch.zeros(1, 3, *input_shape).to(device)

    # Load perturbation mask if provided
    perturbation_mask = None
    if args.mask:
        mask_path = Path(args.mask)
        if not mask_path.exists():
            print(f"Error: Perturbation mask not found: {mask_path}")
            sys.exit(1)
        print(f"[adversarial] Using perturbation mask: {mask_path}")
        perturbation_mask = str(mask_path)

    # Create adversarial attack
    attack = ExplanationAdversarialAttack(
        model=model,
        preprocess_fn=preprocess_fn,
        device=device,
        lambda_explanation=args.lambda_exp,
        lambda_output=args.lambda_out,
        lambda_perturbation=args.lambda_pert,
        explanation_metric=args.explanation_metric,
        output_metric=args.output_metric,
        perturbation_norm=args.perturbation_norm,
        topk_percent=args.topk_percent,
        diff_explanation_method=args.diff_method,
        smooth_samples=args.smooth_samples,
        ig_steps=args.ig_steps,
        gradient_shap_samples=args.gradient_shap_samples,
        epsilon=args.epsilon,
        step_size=args.step_size,
        max_iterations=args.max_iter,
        output_threshold=args.output_threshold,
        explanation_threshold=args.explanation_threshold,
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        background=background,
        perturbation_mask=perturbation_mask,
        seed=args.seed,
        save_last=args.save_last,
    )
    
    # Collect images to process
    images = []
    if args.image:
        images.append(Path(args.image))
    elif args.image_dir:
        img_dir = Path(args.image_dir)
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    else:
        print("Error: --image or --image-dir is required")
        sys.exit(1)
    
    print(f"[adversarial] Processing {len(images)} image(s)")
    
    # Create timestamped output directory to avoid overwriting
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    results_summary = []
    
    for img_path in images:
        print(f"\n[adversarial] Processing: {img_path.name}")
        
        # Run attack
        result = attack.attack(
            image=img_path,
            verbose=True,
        )
        
        # Create output directory for this image
        img_out_dir = out_root / img_path.stem
        
        # Save results
        attack.save_result(
            result=result,
            output_dir=img_out_dir,
            original_image=img_path,
        )
        
        # Generate plots if requested
        if args.plot:
            print(f"[adversarial] Generating trajectory plots for {img_path.name}...")
            plot_adversarial_trajectory(
                loss_history=result.loss_history,
                out_dir=img_out_dir / "plots",
                base_name="trajectory",
                show=False,
                title_prefix=img_path.stem,
            )
        
        # Track summary
        results_summary.append({
            "image": str(img_path),
            "success": result.success,
            "iterations": result.iterations,
            "output_change": result.output_change,
            "explanation_change": 1.0 - result.explanation_similarity,
            "perturbation_l2": result.perturbation_norm_l2,
            "perturbation_linf": result.perturbation_norm_linf,
        })
        
        status = "✓ SUCCESS" if result.success else "✗ FAILED"
        print(f"[adversarial] {status}: output_change={result.output_change:.4f}, "
              f"explanation_change={1.0 - result.explanation_similarity:.4f}")
    
    # Save summary
    summary_path = out_root / "attack_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n[adversarial] Results saved to: {out_root}")
    print(f"[adversarial] Summary: {summary_path}")
    
    # Print overall statistics
    n_success = sum(1 for r in results_summary if r["success"])
    print(f"\n[adversarial] Overall: {n_success}/{len(results_summary)} attacks successful")


if __name__ == "__main__":
    main()
