#!/usr/bin/env python3
"""
explain.py

Explanation script that orchestrates model loading, data I/O, and explanation methods.

All components are specified via CLI or config:
- Model: architecture class + checkpoint
- Explainer: method (shap, gradcam, etc.) + parameters
- Data: single image, JSONL batch, or directory

Usage:
  # Single image with SHAP
  python explain.py \
      --model-class attn_stability_drive.models.dave2.DAVE2v1 \
      --model-args '{"input_shape": [180, 320]}' \
      --ckpt model.pth \
      --explainer-class attn_stability_drive.explainers.shap_explainer.ShapExplainer \
      --explainer-args '{"bg_mode": "fixed_dataset_kmeans", "bg_count": 16}' \
      --image input.jpg \
      --out output/

  # Batch with config file
  python explain.py --config configs/experiments/dave2_shap.yaml
"""

import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Type, Union

import yaml
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ============================================================================
# Dynamic Class Loading
# ============================================================================

def import_class(class_path: str) -> Type:
    """
    Dynamically import a class from a module path.
    
    Args:
        class_path: Full path like 'attn_stability_drive.models.dave2.DAVE2v1'
    
    Returns:
        The class object.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_model(
    model_class: str,
    model_args: Dict[str, Any],
    checkpoint_path: str,
    device: str = "auto"
) -> tuple:
    """
    Load a model from a class path and checkpoint.
    
    Args:
        model_class: Full class path (e.g., 'attn_stability_drive.models.dave2.DAVE2v1')
        model_args: Arguments to pass to the model constructor
        checkpoint_path: Path to the checkpoint file
        device: 'cuda', 'cpu', or 'auto'
    
    Returns:
        Tuple of (model, device_string)
    """
    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Import and instantiate model
    ModelClass = import_class(model_class)
    model = ModelClass(**model_args).to(device)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    state_dict = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    
    # Handle DataParallel prefix if present
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    # Filter out non-parameter keys
    model_keys = set(model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    
    # Check for missing keys
    missing_keys = model_keys - set(filtered_state_dict.keys())
    if missing_keys:
        print(f"[explain] WARNING: {len(missing_keys)} keys missing from checkpoint!")
        print(f"[explain] Missing keys: {list(missing_keys)[:5]} ...")
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    print(f"[explain] Loaded model: {model_class}")
    print(f"[explain] Checkpoint: {checkpoint_path}")
    print(f"[explain] Device: {device}")
    
    return model, device


def create_explainer(
    explainer_class: str,
    model: nn.Module,
    preprocess_fn: Compose,
    device: str,
    explainer_args: Dict[str, Any],
    data_source: Optional[list] = None,
    img_root: Optional[Path] = None
):
    """
    Create an explainer instance.
    
    Args:
        explainer_class: Full class path
        model: The model to explain
        preprocess_fn: Preprocessing transform
        device: Device string
        explainer_args: Additional arguments for the explainer
        data_source: Optional list of JSONL lines (for background sampling)
        img_root: Root path for images
    
    Returns:
        Explainer instance
    """
    ExplainerClass = import_class(explainer_class)
    
    # Build kwargs
    kwargs = {
        "model": model,
        "preprocess_fn": preprocess_fn,
        "device": device,
        **explainer_args
    }
    
    # Add data source if needed for background modes
    if data_source is not None:
        kwargs["data_source"] = data_source
    if img_root is not None:
        kwargs["img_root"] = img_root
    
    explainer = ExplainerClass(**kwargs)
    print(f"[explain] Created explainer: {explainer_class}")
    
    return explainer


# ============================================================================
# Preprocessing Builder
# ============================================================================

def build_preprocess(
    input_shape: tuple,
    mean: list = None,
    std: list = None
) -> Compose:
    """Build a standard preprocessing pipeline."""
    if mean is None:
        mean = [0.5, 0.5, 0.5]
    if std is None:
        std = [0.5, 0.5, 0.5]
    H, W = input_shape
    return Compose([
        Resize((H, W)),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])


# ============================================================================
# Output Directory Logic
# ============================================================================

_TOWN_RE = re.compile(r"(Town\d+HD|Town\d+)", re.IGNORECASE)

def infer_town(path_like: Union[str, Path]) -> Optional[str]:
    m = _TOWN_RE.search(str(path_like))
    return m.group(1) if m else None


def compute_output_dir(
    entry: Dict[str, Any],
    img_rel: Union[str, Path],
    out_root: Path,
    output_structure: str = "town"
) -> Path:
    """
    Compute output directory based on structure mode.
    
    Args:
        entry: JSONL entry dict
        img_rel: Relative image path
        out_root: Root output directory
        output_structure: 'flat', 'town', or 'mirror'
    
    Returns:
        Output directory path
    """
    stem = Path(str(img_rel)).stem
    
    if output_structure == "flat":
        return out_root / stem
    elif output_structure == "town":
        town = entry.get("town") or infer_town(str(img_rel)) or "UnknownTown"
        return out_root / town / stem
    elif output_structure == "mirror":
        # Mirror the input directory structure
        return out_root / Path(str(img_rel)).parent / stem
    else:
        return out_root / stem


# ============================================================================
# Config Loading
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge config file with CLI arguments.
    CLI arguments take precedence over config file.
    """
    result = config.copy()
    
    # Override with non-None CLI args
    for key, value in vars(args).items():
        if value is not None and key != "config":
            # Convert dashes to underscores for consistency
            result[key] = value
    
    return result


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Generic explanation script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Config file (can override everything)
    ap.add_argument("--config", help="Path to YAML config file")
    
    # Model
    ap.add_argument("--model-class", dest="model_class",
                    help="Full class path (e.g., attn_stability_drive.models.dave2.DAVE2v1)")
    ap.add_argument("--model-args", dest="model_args", type=str, default="{}",
                    help="JSON string of model constructor args")
    ap.add_argument("--ckpt", help="Path to model checkpoint")
    
    # Explainer
    ap.add_argument("--explainer-class", dest="explainer_class",
                    help="Full class path for explainer")
    ap.add_argument("--explainer-args", dest="explainer_args", type=str, default="{}",
                    help="JSON string of explainer args")
    
    # Preprocessing
    ap.add_argument("--input-h", dest="input_h", type=int, default=180)
    ap.add_argument("--input-w", dest="input_w", type=int, default=320)
    ap.add_argument("--norm-mean", dest="norm_mean", type=str, default="[0.5, 0.5, 0.5]")
    ap.add_argument("--norm-std", dest="norm_std", type=str, default="[0.5, 0.5, 0.5]")
    
    # Data Input
    ap.add_argument("--image", help="Single image path")
    ap.add_argument("--jsonl", help="JSONL file for batch processing")
    ap.add_argument("--image-dir", dest="image_dir", help="Directory of images")
    ap.add_argument("--img-root", dest="img_root", type=str, default=None, help="Root dir for relative paths")
    
    # Output
    ap.add_argument("--out", help="Output directory (single image)")
    ap.add_argument("--out-root", dest="out_root", type=str, default="out_explain", help="Output root (batch)")
    ap.add_argument("--output-structure", dest="output_structure", default="town", 
                    choices=["flat", "town", "mirror"])
    
    # Processing Options
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--max", type=int, default=0, help="Max samples (0=all)")
    ap.add_argument("--start", type=int, default=0, help="Start index")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    ap.add_argument("--progress-interval", dest="progress_interval", type=int, default=10)
    
    return ap.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    
    # Load and merge config
    config = {}
    if args.config:
        config = load_config(args.config)
    config = merge_config_with_args(config, args)
    
    # Validate required fields
    required = ["model_class", "ckpt", "explainer_class"]
    for field in required:
        if field not in config or config[field] is None:
            print(f"Error: --{field.replace('_', '-')} is required")
            return 1
    
    # Parse JSON args
    model_args = json.loads(config.get("model_args", "{}"))
    explainer_args = json.loads(config.get("explainer_args", "{}"))
    norm_mean = json.loads(config.get("norm_mean", "[0.5, 0.5, 0.5]"))
    norm_std = json.loads(config.get("norm_std", "[0.5, 0.5, 0.5]"))
    
    input_h = config.get("input_h", 180)
    input_w = config.get("input_w", 320)
    
    # Ensure input_shape is in model_args if not present
    if "input_shape" not in model_args:
        model_args["input_shape"] = (input_h, input_w)
    
    # Ensure input_shape is in explainer_args if not present
    if "input_shape" not in explainer_args:
        explainer_args["input_shape"] = (3, input_h, input_w)
    
    # Load model
    model, device = load_model(
        config["model_class"],
        model_args,
        config["ckpt"],
        config.get("device", "auto")
    )
    
    # Build preprocessing
    preprocess = build_preprocess(
        (input_h, input_w),
        mean=norm_mean,
        std=norm_std
    )
    
    # Also pass normalization to explainer for correct denormalization
    explainer_args["normalization_mean"] = norm_mean
    explainer_args["normalization_std"] = norm_std
    
    # Load data source if JSONL provided
    data_source = None
    # Resolve img_root to absolute path to avoid ambiguity
    img_root = Path(config.get("img_root", ".")).resolve()
    print(f"[explain] Using image root: {img_root}")
    
    if config.get("jsonl"):
        with open(config["jsonl"], "r") as f:
            data_source = f.readlines()
    
    # Create explainer
    explainer = create_explainer(
        config["explainer_class"],
        model,
        preprocess,
        device,
        explainer_args,
        data_source=data_source,
        img_root=img_root
    )
    
    # ========== Single Image Mode ==========
    if config.get("image"):
        out_dir = Path(config.get("out", "out_explain"))
        img_path = Path(config["image"])
        
        print(f"[explain] Processing single image: {img_path}")
        explainer.explain(img_path, model)
        explainer.save(out_dir)
        print(f"[explain] Saved to: {out_dir}")
        return 0
    
    # ========== Directory Mode ==========
    if config.get("image_dir"):
        img_dir = Path(config["image_dir"])
        out_root = Path(config.get("out_root", "out_explain"))
        
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        print(f"[explain] Found {len(images)} images in {img_dir}")
        
        for i, img_path in enumerate(images):
            out_dir = out_root / img_path.stem
            
            if config.get("skip_existing") and (out_dir / "saliency.png").exists():
                continue
            
            explainer.explain(img_path, model)
            explainer.save(out_dir)
            
            if i % config.get("progress_interval", 10) == 0:
                print(f"[explain] Processed {i}/{len(images)}")
        
        return 0
    
    # ========== Batch JSONL Mode ==========
    if not data_source:
        print("Error: Provide --image, --image-dir, or --jsonl")
        return 1
    
    out_root = Path(config.get("out_root", "out_explain"))
    output_structure = config.get("output_structure", "town")
    
    lines = data_source[config.get("start", 0):]
    if config.get("max", 0) > 0:
        lines = lines[:config["max"]]
    
    print(f"[explain] Processing {len(lines)} entries from JSONL")
    
    processed = 0
    skipped = 0
    errors = 0
    
    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
            img_rel = entry.get("image_path")
            if not img_rel:
                continue
            
            img_abs = img_root / img_rel
            if not img_abs.exists():
                skipped += 1
                continue
            
            out_dir = compute_output_dir(entry, img_rel, out_root, output_structure)
            
            if config.get("skip_existing") and (out_dir / "saliency.png").exists():
                skipped += 1
                continue
            
            explainer.explain(img_abs, model)
            explainer.save(out_dir)
            processed += 1
            
            if i % config.get("progress_interval", 10) == 0:
                print(f"[explain] Progress: {i}/{len(lines)} (processed={processed}, skipped={skipped})")
                
        except Exception as e:
            print(f"[explain] Error at line {i}: {e}")
            errors += 1
            continue
    
    print(f"[explain] Done: processed={processed}, skipped={skipped}, errors={errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
