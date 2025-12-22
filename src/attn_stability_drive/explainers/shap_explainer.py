import json
import random
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union, Callable
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from .base import BaseExplainer

# Generic SHAP Explainer
class ShapExplainer(BaseExplainer):
    def __init__(self, 
                 model: nn.Module, 
                 preprocess_fn: Compose,
                 device: str,
                 input_shape: Tuple[int, int, int] = (3, 180, 320),
                 feature_method: str = "get_flattened_features",
                 normalization_mean: List[float] = [0.5, 0.5, 0.5],
                 normalization_std: List[float] = [0.5, 0.5, 0.5],
                 bg_mode: str = "fixed_dataset_kmeans",
                 bg_count: int = 16,
                 bg_seed: int = 42,
                 data_source: Optional[List[str]] = None,
                 img_root: Optional[Path] = None,
                 kmeans_sample_size: int = 5000):
        """
        Args:
            model: The model instance.
            preprocess_fn: Torchvision transform pipeline.
            device: 'cuda' or 'cpu'.
            input_shape: (C, H, W) of the input tensor.
            feature_method: Name of the method in model to extract features for K-Means.
            normalization_mean: Mean used in preprocessing (for visualization denormalization).
            normalization_std: Std used in preprocessing (for visualization denormalization).
            bg_mode: Strategy for background ('fixed_zero', 'fixed_dataset_blur', 'fixed_dataset_kmeans').
            bg_count: Number of background samples.
            data_source: List of JSONL strings (lines) for dataset sampling (required for dataset modes).
            img_root: Root path for images in data_source.
        """
        self.device = device
        self.preprocess = preprocess_fn
        self.input_shape = input_shape
        self.feature_method = feature_method
        self.norm_mean = np.array(normalization_mean).reshape(1, 1, -1) # (1, 1, C)
        self.norm_std = np.array(normalization_std).reshape(1, 1, -1)   # (1, 1, C)
        self.bg_mode = bg_mode
        self.bg_count = bg_count
        self.bg_seed = bg_seed
        
        # State to store results between .explain() and .save()
        self.last_results = {} 

        # Setup Shared Explainer (if mode allows)
        self.shared_explainer = None
        
        # Ensure determinism for initialization
        self._set_seed(self.bg_seed)

        bg_tensor = self._build_background(model, data_source, img_root, kmeans_sample_size)
        self.shared_explainer = shap.GradientExplainer(model, bg_tensor)
        print(f"[ShapExplainer] Initialized shared explainer with mode: {bg_mode}")

    def explain(self, image: Union[Image.Image, Path], model: nn.Module) -> np.ndarray:
        """
        Compute SHAP values.
        
        Args:
            image: PIL Image or Path to image.
            model: The model to explain (should be same structure as init model).
        
        Returns:
            np.ndarray: The saliency map (Sum of Absolute SHAP values), shape (H, W).
        """
        # 1. Load and Preprocess
        if isinstance(image, (str, Path)):
            img_pil = Image.open(image).convert("RGB")
            img_path_str = str(image)
        else:
            img_pil = image
            img_path_str = "memory_image"

        x_tensor = self._pil_to_tensor(img_pil) # (1, C, H, W)
        
        # 2. Get Model Prediction
        model.eval()
        with torch.no_grad():
            output = model(x_tensor)
            # Handle scalar or 1D output
            if output.numel() == 1:
                pred_val = output.item()
            else:
                # Default to first output for multi-output models if not specified
                pred_val = output.view(-1)[0].item()

        # 3. Determine Explainer (Shared vs Local)
        self._set_seed(self.bg_seed) # Reset seed for consistent jitter/randomness
        
        if self.shared_explainer:
            explainer = self.shared_explainer
            shap_vals_list = explainer.shap_values(x_tensor)
        else:
            raise ValueError(f"Unknown bg_mode: {self.bg_mode}")

        # 4. Process Output
        # shap_values returns a list (one per output node), we have 1 output
        shap_vals = shap_vals_list[0] if isinstance(shap_vals_list, list) else shap_vals_list
        if torch.is_tensor(shap_vals):
            shap_vals = shap_vals.detach().cpu().numpy()
        
        shap_chw = shap_vals[0]  # (C, H, W)
        shap_sum_abs = np.abs(shap_chw).sum(axis=0)  # (H, W) -> Saliency Map

        # 5. Store state for save()
        self.last_results = {
            "image_path": img_path_str,
            "input_tensor": x_tensor.detach().cpu(), # Keep on CPU
            "shap_values": shap_chw,
            "shap_sum_abs": shap_sum_abs,
            "prediction": pred_val,
            "img_pil_raw": img_pil # Store for potential blurring if needed later
        }

        return shap_sum_abs

    def save(self, output_dir: Path) -> None:
        """
        Save the artifacts from the last executed explain() call.
        """
        if not self.last_results:
            raise RuntimeError("Call explain() before calling save().")

        output_dir.mkdir(parents=True, exist_ok=True)
        res = self.last_results

        # Unpack
        shap_chw = res["shap_values"]
        shap_sum_abs = res["shap_sum_abs"]
        x_tensor = res["input_tensor"]
        
        # Save Numpy Arrays
        np.save(output_dir / "shap_values.npy", shap_chw)
        np.save(output_dir / "shap_sum_abs.npy", shap_sum_abs)
        np.save(output_dir / "preprocessed_input.npy", x_tensor.numpy())

        # Save Visualizations
        img_uint8 = self._tensor_to_vis_uint8(x_tensor) # (H, W, 3)
        
        self._save_overlay(
            img_uint8, shap_chw, 
            str(output_dir / "shap_overlay.png"), 
            title=f"SHAP — steer={res['prediction']:.3f}"
        )
        
        self._save_saliency_grayscale(shap_sum_abs, str(output_dir / "saliency.png"))

        # Save Metadata
        report = {
            "image": res["image_path"],
            "predicted_steer": float(res["prediction"]),
            "bg_mode": self.bg_mode,
            "bg_samples": self.bg_count,
            "artifacts": {
                "overlay": "shap_overlay.png",
                "saliency": "saliency.png",
                "shap_raw": "shap_values.npy"
            }
        }
        with open(output_dir / "pred.json", "w") as f:
            json.dump(report, f, indent=2)

    # -------------------------------------------------------------------------
    # Internal Helper Methods
    # -------------------------------------------------------------------------

    def _pil_to_tensor(self, img_pil):
        return self.preprocess(img_pil).unsqueeze(0).to(self.device)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _tensor_to_vis_uint8(self, x_1chw) -> np.ndarray:
        """Inverse normalize to get back to 0-255 RGB."""
        v = x_1chw[0].numpy().transpose(1, 2, 0) # (H, W, C)
        # Denormalize: x = (x * std) + mean
        v = (v * self.norm_std) + self.norm_mean
        v = np.clip(v, 0.0, 1.0)
        return (v * 255.0).astype(np.uint8)

    def _save_overlay(self, img_hwc, shap_chw, out_path, title):
        shap_hwc = shap_chw.transpose(1, 2, 0)[None, ...]
        x_vis = img_hwc.astype(np.float32) / 255.0
        x_vis = x_vis[None, ...]

        plt.figure(figsize=(8, 4.5), dpi=160)
        shap.image_plot([shap_hwc], x_vis, show=False)
        plt.suptitle(title, y=0.95)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

    def _save_saliency_grayscale(self, saliency, out_path):
        v = saliency.astype(np.float32)
        vmax = np.percentile(v, 99) if np.any(v > 0) else v.max()
        if vmax <= 1e-12: vmax = 1.0
        v = np.clip(v / vmax, 0.0, 1.0)
        Image.fromarray((v * 255.0).astype(np.uint8)).save(out_path)

    # --- Background Builders ---

    def _build_background(self, model, data_source, img_root, kmeans_sample_size):
        """Dispatches to specific background builders based on mode."""
        C, H, W = self.input_shape
        
        if self.bg_mode == "fixed_zero":
            return torch.zeros((max(1, self.bg_count), C, H, W), device=self.device)
        
        if not data_source:
            raise ValueError(f"Mode {self.bg_mode} requires 'data_source' (JSONL lines).")
            
        if self.bg_mode == "fixed_dataset_blur":
            return self._build_fixed_dataset_blur(data_source, img_root)
            
        if self.bg_mode == "fixed_dataset_kmeans":
            return self._build_fixed_dataset_kmeans(model, data_source, img_root, kmeans_sample_size)
            
        return None

    def _collect_paths(self, lines, img_root):
        paths = []
        missing_count = 0
        for i, line in enumerate(lines):
            try:
                entry = json.loads(line)
                rel_path = entry.get("image_path", "")
                if not rel_path: continue
                
                p = Path(img_root) / rel_path
                if p.exists(): 
                    paths.append(p)
                else:
                    missing_count += 1
                    if missing_count <= 5:
                        print(f"[ShapExplainer] Warning: Image not found: {p.resolve()}")
            except Exception as e: 
                if i <= 5: print(f"[ShapExplainer] JSON parse error: {e}")
                continue
        
        if missing_count > 0:
            print(f"[ShapExplainer] Total missing images: {missing_count}")
            
        return paths

    def _build_fixed_dataset_blur(self, lines, img_root):
        paths = self._collect_paths(lines, img_root)
        indices = np.linspace(0, len(paths)-1, self.bg_count + 2, dtype=int)[1:-1]
        
        bgs = []
        rng = np.random.RandomState(self.bg_seed)
        
        for i in indices:
            img = Image.open(paths[i]).convert("RGB")
            # Blur
            img = img.filter(ImageFilter.GaussianBlur(1.0 + 2.0 * rng.rand()))
            # Brightness jitter
            arr = np.array(img, dtype=np.float32) * (0.9 + 0.2 * rng.rand())
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            bgs.append(self._pil_to_tensor(img).squeeze(0))
            
        return torch.stack(bgs).to(self.device)

    def _build_fixed_dataset_kmeans(self, model, lines, img_root, sample_size):
        paths = self._collect_paths(lines, img_root)
        rng = np.random.RandomState(self.bg_seed)
        
        # Subsample for K-means
        if len(paths) > sample_size:
            paths = [paths[i] for i in rng.choice(len(paths), sample_size, replace=False)]
            
        # Extract features
        features_list = []
        model.eval()
        print(f"[ShapExplainer] Extracting features from {len(paths)} images for K-Means...")
        
        # Ensure model has the feature extractor hook
        if not hasattr(model, self.feature_method):
             raise AttributeError(f"Model must have '{self.feature_method}' method.")

        extractor = getattr(model, self.feature_method)

        with torch.no_grad():
            for p in paths:
                try:
                    x = self._pil_to_tensor(Image.open(p).convert("RGB"))
                    feats = extractor(x)
                    features_list.append(feats.cpu().numpy())
                except Exception: continue
        
        if not features_list: raise RuntimeError("K-Means feature extraction failed.")
        
        features = np.vstack(features_list)
        
        # Run K-Means
        kmeans = KMeans(n_clusters=self.bg_count, random_state=self.bg_seed, n_init=10)
        kmeans.fit(features)
        
        # Find exemplars
        distances = kmeans.transform(features)
        exemplar_indices = np.argmin(distances, axis=0)
        exemplar_paths = sorted(list(set([paths[i] for i in exemplar_indices])))
        
        # Load Exemplars (No Blur)
        bgs = []
        for p in exemplar_paths:
            img = Image.open(p).convert("RGB")
            bgs.append(self._pil_to_tensor(img).squeeze(0))
            
        return torch.stack(bgs).to(self.device)