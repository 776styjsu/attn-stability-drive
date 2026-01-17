#!/usr/bin/env python3
"""
Simple inference script to test model loading and forward pass.

Usage:
    python scripts/inference.py --image path/to/image.png --ckpt path/to/model.pt
"""

import argparse
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


def load_model(checkpoint_path: str, device: str = "auto"):
    """Load DAVE2v1 model from checkpoint."""
    from attn_stability_drive.models.dave2 import DAVE2v1
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = DAVE2v1(input_shape=(180, 320)).to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    
    # Handle DataParallel prefix
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, device


def build_preprocess(input_shape=(180, 320)):
    """Build preprocessing pipeline."""
    return Compose([
        Resize(input_shape),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def run_inference(image_path: str, checkpoint_path: str, device: str = "auto"):
    """Run inference on a single image."""
    print(f"[inference] Loading model from: {checkpoint_path}")
    model, device = load_model(checkpoint_path, device)
    print(f"[inference] Device: {device}")
    
    print(f"[inference] Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    print(f"[inference] Original image size: {img.size}")
    
    preprocess = build_preprocess()
    x = preprocess(img).unsqueeze(0).to(device)
    print(f"[inference] Preprocessed tensor shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    steering = output.item()
    print(f"\n[RESULT] Steering angle prediction: {steering:.6f}")
    
    return steering


def main():
    parser = argparse.ArgumentParser(description="Simple inference script")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        return 1
    
    if not Path(args.ckpt).exists():
        print(f"ERROR: Checkpoint not found: {args.ckpt}")
        return 1
    
    run_inference(args.image, args.ckpt, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
