#!/usr/bin/env python3
"""
Adversarial Explanation Attack Demo

This demo shows how to generate adversarial perturbations that change a model's
explanation (saliency map) while keeping its output (steering angle) stable.

This is a simplified demo. For full configurability, use:
    python scripts/adversarial_attack.py --help

Demo Modes:
    1. Basic attack (no mask):
       python examples/adversarial_demo.py

    2. Masked attack (only perturb road region):
       python examples/adversarial_demo.py --use-mask

Requirements:
    - Model checkpoint: Place at examples/data/adversarial_demo/dave2v1.pt
      (Checkpoint coming soon - contact authors for early access)
    - Sample image: examples/data/adversarial_demo/sample_image.png (included)
    - Road mask: examples/data/adversarial_demo/road_mask.png (included)
"""

import argparse
import sys
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Optional
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Demo configuration
DEMO_DIR = PROJECT_ROOT / "examples" / "data" / "adversarial_demo"
DEFAULT_CHECKPOINT = DEMO_DIR / "dave2v1.pt"
SAMPLE_IMAGE = DEMO_DIR / "sample_image.png"
ROAD_MASK = DEMO_DIR / "road_mask.png"
OUTPUT_DIR = PROJECT_ROOT / "output" / "adversarial_demo"

# Attack configuration (based on recommended settings)
DEMO_CONFIG = {
    "model_class": "attn_stability_drive.models.dave2.DAVE2v1",
    "model_args": {"input_shape": [180, 320]},
    "diff_method": "gradient_shap",  # Use gradient_shap for demo (no background needed)
    "lambda_exp": 1000.0,             # Weight for explanation change
    "lambda_out": 1.0,           # Negative = penalize output change
    "lambda_pert": 1.0,           # Small penalty on perturbation size
    "explanation_metric": "cosine",
    "max_iter": 50,                # Reduced for demo
    "epsilon": 0.1,
    "step_size": 0.01,
    "ig_steps": 50,
    "seed": 42,
}


def check_requirements():
    """Check that required files exist."""
    missing = []
    
    if not DEFAULT_CHECKPOINT.exists():
        missing.append(f"Checkpoint: {DEFAULT_CHECKPOINT}")
    
    if not SAMPLE_IMAGE.exists():
        missing.append(f"Sample image: {SAMPLE_IMAGE}")
    
    return missing


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
        print("[demo] Warning: No valid image paths found in JSONL")
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
            print(f"[demo] Warning: Failed to load {p}: {e}")
            continue
    
    if not backgrounds:
        return None
    
    return torch.stack(backgrounds).to(device)


def run_demo(
    use_mask: bool = False, 
    checkpoint: Path = None, 
    data_jsonl: str = None,
    img_root: str = None,
    bg_count: int = 16,
    verbose: bool = True,
    seed: int = 42,
):
    """
    Run the adversarial attack demo.
    
    Args:
        use_mask: If True, only perturb road region using mask
        checkpoint: Optional path to model checkpoint
        data_jsonl: Optional path to JSONL file for background sampling
        img_root: Optional root directory for resolving relative image paths
        bg_count: Number of background samples to use (default 16)
        verbose: Print progress messages
    """
    from attn_stability_drive.adversarial import ExplanationAdversarialAttack
    from attn_stability_drive.visualization import plot_adversarial_trajectory
    import importlib
    
    checkpoint = checkpoint or DEFAULT_CHECKPOINT
    
    # Check requirements
    missing = check_requirements() if checkpoint == DEFAULT_CHECKPOINT else []
    if checkpoint != DEFAULT_CHECKPOINT and not checkpoint.exists():
        missing.append(f"Checkpoint: {checkpoint}")
    
    if missing:
        print("=" * 60)
        print("MISSING REQUIRED FILES")
        print("=" * 60)
        for m in missing:
            print(f"  ✗ {m}")
        print()
        if "Checkpoint" in str(missing):
            print("The model checkpoint is not included in the repository.")
            print("Options:")
            print("  1. Contact authors for checkpoint access")
            print("  2. Train your own DAVE2 model")
            print("  3. Use --ckpt to specify your own checkpoint path")
            print()
            print("Expected checkpoint location:")
            print(f"  {DEFAULT_CHECKPOINT}")
        print("=" * 60)
        return None
    
    if verbose:
        print("=" * 60)
        print("ADVERSARIAL EXPLANATION ATTACK DEMO")
        print("=" * 60)
        print(f"Mode: {'Masked (road only)' if use_mask else 'Full image'}")
        print(f"Checkpoint: {checkpoint}")
        print(f"Image: {SAMPLE_IMAGE}")
        if use_mask:
            print(f"Mask: {ROAD_MASK}")
        print(f"Output: {OUTPUT_DIR}")
        if data_jsonl:
            print(f"Background: {data_jsonl} ({bg_count} samples)")
        print("=" * 60)
        print()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"[demo] Using device: {device}")
    
    # Load model
    if verbose:
        print("[demo] Loading model...")
    
    module_path, class_name = DEMO_CONFIG["model_class"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)
    
    model = ModelClass(**DEMO_CONFIG["model_args"]).to(device)
    
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    
    # Handle DataParallel prefix
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    if verbose:
        print("[demo] Model loaded successfully")
    
    # Build preprocessing
    input_shape = DEMO_CONFIG["model_args"]["input_shape"]
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    preprocess_fn = Compose([
        Resize(tuple(input_shape)),
        ToTensor(),
        Normalize(mean=norm_mean, std=norm_std),
    ])
    
    # Build background if requested
    background = None
    if data_jsonl:
        if verbose:
            print(f"[demo] Building background from {data_jsonl}...")
        background = build_background(
            data_jsonl=data_jsonl,
            img_root=img_root,
            preprocess_fn=preprocess_fn,
            device=device,
            bg_count=bg_count,
            seed=seed,
        )
        if background is not None and verbose:
            print(f"[demo] Built background with {len(background)} samples")
    
    # Create attack
    if verbose:
        print("[demo] Initializing attack...")
    
    attack = ExplanationAdversarialAttack(
        model=model,
        preprocess_fn=preprocess_fn,
        device=device,
        lambda_explanation=DEMO_CONFIG["lambda_exp"],
        lambda_output=DEMO_CONFIG["lambda_out"],
        lambda_perturbation=DEMO_CONFIG["lambda_pert"],
        explanation_metric=DEMO_CONFIG["explanation_metric"],
        diff_explanation_method=DEMO_CONFIG["diff_method"],
        ig_steps=DEMO_CONFIG["ig_steps"],
        epsilon=DEMO_CONFIG["epsilon"],
        step_size=DEMO_CONFIG["step_size"],
        max_iterations=DEMO_CONFIG["max_iter"],
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        perturbation_mask=str(ROAD_MASK) if use_mask else None,
        background=background,
        seed=seed,
    )
    
    # Run attack
    if verbose:
        print("[demo] Running attack...")
        print()
    
    result = attack.attack(
        image=SAMPLE_IMAGE,
        verbose=verbose,
    )
    
    # Create output directory
    output_subdir = OUTPUT_DIR / ("masked" if use_mask else "full")
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    attack.save_result(
        result=result,
        output_dir=output_subdir,
        original_image=SAMPLE_IMAGE,
    )
    
    # Generate plots
    if verbose:
        print("[demo] Generating visualization plots...")
    
    plot_adversarial_trajectory(
        loss_history=result.loss_history,
        out_dir=output_subdir / "plots",
        base_name="trajectory",
        show=False,
        title_prefix="Demo",
    )
    
    # Print summary
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    status = "✓ SUCCESS" if result.success else "✗ Attack did not converge"
    print(f"Status: {status}")
    print(f"Iterations: {result.iterations}")
    print(f"Output change: {result.output_change:.6f}")
    print(f"Explanation change: {1.0 - result.explanation_similarity:.6f}")
    print(f"Perturbation L2: {result.perturbation_norm_l2:.6f}")
    print(f"Perturbation L∞: {result.perturbation_norm_linf:.6f}")
    print()
    print(f"Results saved to: {output_subdir}")
    print("=" * 60)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial Explanation Attack Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--use-mask",
        action="store_true",
        help="Use road mask to constrain perturbations to road region only",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    
    # Background dataset arguments
    parser.add_argument("--data-jsonl", type=str, help="JSONL file for background sampling")
    parser.add_argument("--img-root", type=str, help="Root directory for background images")
    parser.add_argument("--bg-count", type=int, default=16, help="Number of background samples")

    # Seed arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    result = run_demo(
        use_mask=args.use_mask,
        checkpoint=args.ckpt,
        data_jsonl=args.data_jsonl,
        img_root=args.img_root,
        bg_count=args.bg_count,
        verbose=not args.quiet,
        seed=args.seed,
    )
    
    return 0 if result is not None else 1


if __name__ == "__main__":
    sys.exit(main())
