"""
Adversarial attack for explanation instability.

This module implements optimization-based attacks that find minimal perturbations
to input images such that:
1. The model output (steering angle) remains similar
2. The explanation (saliency map) changes drastically
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .objectives import (
    CombinedAdversarialLoss,
    DifferentiableSHAP,
)


@dataclass
class AttackResult:
    """Results from an adversarial explanation attack."""
    
    # Attack success metrics
    success: bool
    iterations: int
    
    # Original state
    original_output: float
    original_explanation: np.ndarray  # (H, W)
    original_image: Optional[np.ndarray]  # (H, W, C) uint8
    
    # Perturbed state
    perturbed_output: float
    perturbed_explanation: np.ndarray  # (H, W)
    perturbation: np.ndarray  # (C, H, W)
    perturbed_image: np.ndarray  # (H, W, C) uint8
    
    # Metrics
    output_change: float  # |y - y'|
    explanation_similarity: float  # Similarity metric value
    perturbation_norm_l2: float
    perturbation_norm_linf: float
    
    # Loss history
    loss_history: List[Dict[str, float]] = field(default_factory=list)
    
    # Configuration used for reproducibility
    config: Dict[str, any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        d['original_explanation'] = None  # Too large, save separately
        d['original_image'] = None
        d['perturbed_explanation'] = None
        d['perturbation'] = None
        d['perturbed_image'] = None
        return d


class ExplanationAdversarialAttack:
    """
    Adversarial attack that maximizes explanation change while preserving model output.
    
    Given an input image x, finds perturbation δ such that:
    - ||Model(x) - Model(x + δ)|| is small (output preserved)
    - Sim(Explain(x), Explain(x + δ)) is small (explanation changed)
    - ||δ|| is small (perturbation is minimal)
    
    Uses Projected Gradient Descent (PGD) style optimization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        preprocess_fn: Callable,
        device: str = "cuda",
        # Loss weights
        lambda_explanation: float = 1.0,
        lambda_output: float = 10.0,
        lambda_perturbation: float = 0.01,
        # Loss metrics
        explanation_metric: str = "mse",
        output_metric: str = "mse",
        perturbation_norm: str = "l2",
        topk_percent: float = 0.1,
        # Differentiable explanation method
        diff_explanation_method: str = "smooth_grad",
        smooth_samples: int = 10,
        ig_steps: int = 50,
        gradient_shap_samples: Optional[int] = None,
        # Optimization params
        epsilon: float = 0.1,  # Max perturbation in normalized space
        step_size: float = 0.01,
        max_iterations: int = 100,
        # Convergence criteria
        output_threshold: float = 0.05,  # Max allowed output change
        explanation_threshold: float = 0.3,  # Min required explanation change (1 - similarity)
        early_stop_patience: int = 100,
        # Normalization params (for visualization)
        normalization_mean: List[float] = None,
        normalization_std: List[float] = None,
        # Background for GradientSHAP
        background: Optional[torch.Tensor] = None,
        # Perturbation mask (restrict which pixels can be attacked)
        perturbation_mask: Optional[Union[torch.Tensor, np.ndarray, str, Path]] = None,
        # Reproducibility
        seed: Optional[int] = None,
        # Save options
        save_last: bool = False,
    ):
        """
        Initialize the adversarial attack.
        
        Args:
            seed: Random seed for reproducibility. If provided, sets global seeds
                  and enables deterministic CUDA operations.
            topk_percent: For 'topk' explanation metric, fraction of top pixels (default 10%)
            perturbation_mask: Optional mask to restrict which pixels can be perturbed.
                Can be one of:
                - torch.Tensor or np.ndarray: Shape (H, W), (1, H, W), (C, H, W), or (1, C, H, W)
                  Values should be in [0, 1] where 1 = fully attackable, 0 = protected
                - str or Path: Path to a grayscale image file (white=attackable, black=protected)
                The mask will be resized to match input dimensions if needed.
            save_last: If True, saves the result from the last iteration instead of the best iteration.
        """
        # Set seed for reproducibility (must be done before any random operations)
        self.seed = seed
        if seed is not None:
            from ..utils import set_seed
            set_seed(seed, deterministic=True)
        
        self.device = device
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.save_last = save_last
        self._call_count = 0  # Track calls for reproducible per-call seeding
        
        # Validate and store perturbation mask (will be finalized when input shape is known)
        self._perturbation_mask_input = perturbation_mask
        self._perturbation_mask = None  # Will be set during attack() when input shape is known
        
        self.epsilon = epsilon
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.output_threshold = output_threshold
        self.explanation_threshold = explanation_threshold
        self.early_stop_patience = early_stop_patience
        
        # Store config for reproducibility
        self.config = {
            "lambda_explanation": lambda_explanation,
            "lambda_output": lambda_output,
            "lambda_perturbation": lambda_perturbation,
            "explanation_metric": explanation_metric,
            "output_metric": output_metric,
            "perturbation_norm": perturbation_norm,
            "topk_percent": topk_percent,
            "diff_explanation_method": diff_explanation_method,
            "smooth_samples": smooth_samples,
            "ig_steps": ig_steps,
            "gradient_shap_samples": gradient_shap_samples,
            "epsilon": epsilon,
            "step_size": step_size,
            "max_iterations": max_iterations,
            "output_threshold": output_threshold,
            "explanation_threshold": explanation_threshold,
            "early_stop_patience": early_stop_patience,
            "normalization_mean": list(normalization_mean) if normalization_mean else [0.5, 0.5, 0.5],
            "normalization_std": list(normalization_std) if normalization_std else [0.5, 0.5, 0.5],
            "has_background": background is not None,
            "background_size": background.shape[0] if background is not None else 0,
            "seed": seed,
            "has_perturbation_mask": perturbation_mask is not None,
            "save_last": save_last,
        }
        
        # Normalization for visualization
        self.norm_mean = np.array(normalization_mean or [0.5, 0.5, 0.5]).reshape(1, 1, -1)
        self.norm_std = np.array(normalization_std or [0.5, 0.5, 0.5]).reshape(1, 1, -1)
        
        # Initialize loss function
        # The warning was likely triggering here when moving to device
        self.loss_fn = CombinedAdversarialLoss(
            lambda_explanation=lambda_explanation,
            lambda_output=lambda_output,
            lambda_perturbation=lambda_perturbation,
            explanation_metric=explanation_metric,
            output_metric=output_metric,
            perturbation_norm=perturbation_norm,
            topk_percent=topk_percent,
        ).to(device)
        
        # Initialize differentiable explanation
        self.diff_explainer = DifferentiableSHAP(
            model=model,
            method=diff_explanation_method,
            smooth_samples=smooth_samples,
            ig_steps=ig_steps,
            background=background,
            gradient_shap_samples=gradient_shap_samples,
            seed=seed,
        )
    
    def _get_generator(self) -> Optional[torch.Generator]:
        """
        Get a generator seeded deterministically based on the base seed and call count.
        """
        if self.seed is None:
            return None
        
        # Generator device must match the device of tensors used with it
        gen_device = "cuda" if "cuda" in str(self.device).lower() else "cpu"
        generator = torch.Generator(device=gen_device)
        
        # Use prime to avoid collisions
        generator.manual_seed(self.seed + self._call_count * 1000003)
        return generator

    def _validate_and_prepare_mask(
        self,
        mask_input: Optional[Union[torch.Tensor, np.ndarray, str, Path]],
        target_shape: Tuple[int, int, int, int],  # (N, C, H, W)
    ) -> Optional[torch.Tensor]:
        """
        Validate and prepare perturbation mask to match input tensor shape.
        
        Args:
            mask_input: Raw mask input (tensor, array, or path to image)
            target_shape: Target tensor shape (N, C, H, W)
            
        Returns:
            Validated mask tensor of shape (1, C, H, W) with values in [0, 1],
            or None if no mask provided.
            
        Raises:
            ValueError: If mask is invalid (wrong shape, values out of range, etc.)
            FileNotFoundError: If mask path doesn't exist
        """
        if mask_input is None:
            return None
        
        N, C, H, W = target_shape
        
        # Load from file if path provided
        if isinstance(mask_input, (str, Path)):
            mask_path = Path(mask_input)
            if not mask_path.exists():
                raise FileNotFoundError(f"Perturbation mask file not found: {mask_path}")
            
            # Load as grayscale image
            try:
                mask_img = Image.open(mask_path).convert("L")  # Grayscale
                mask_array = np.array(mask_img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            except Exception as e:
                raise ValueError(f"Failed to load perturbation mask from {mask_path}: {e}")
            
            mask_input = mask_array
        
        # Convert to tensor if numpy array
        if isinstance(mask_input, np.ndarray):
            mask_tensor = torch.from_numpy(mask_input.astype(np.float32))
        elif isinstance(mask_input, torch.Tensor):
            mask_tensor = mask_input.float()
        else:
            raise TypeError(
                f"Perturbation mask must be torch.Tensor, np.ndarray, or path to image. "
                f"Got: {type(mask_input)}"
            )
        
        # Validate and reshape
        original_shape = mask_tensor.shape
        
        # Handle different input shapes
        if mask_tensor.ndim == 2:
            # (H, W) -> (1, 1, H, W) then expand to (1, C, H, W)
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            mask_tensor = mask_tensor.expand(1, C, -1, -1)
        elif mask_tensor.ndim == 3:
            if mask_tensor.shape[0] == 1:
                # (1, H, W) -> (1, C, H, W)
                mask_tensor = mask_tensor.unsqueeze(0).expand(1, C, -1, -1)
            elif mask_tensor.shape[0] == C:
                # (C, H, W) -> (1, C, H, W)
                mask_tensor = mask_tensor.unsqueeze(0)
            else:
                raise ValueError(
                    f"3D mask must have shape (1, H, W) or ({C}, H, W). "
                    f"Got: {original_shape}"
                )
        elif mask_tensor.ndim == 4:
            if mask_tensor.shape[0] != 1:
                raise ValueError(
                    f"4D mask must have batch size 1. Got shape: {original_shape}"
                )
            if mask_tensor.shape[1] == 1:
                # (1, 1, H, W) -> (1, C, H, W)
                mask_tensor = mask_tensor.expand(1, C, -1, -1)
            elif mask_tensor.shape[1] != C:
                raise ValueError(
                    f"4D mask must have 1 or {C} channels. Got shape: {original_shape}"
                )
        else:
            raise ValueError(
                f"Perturbation mask must be 2D, 3D, or 4D. Got {mask_tensor.ndim}D with shape: {original_shape}"
            )
        
        # Resize if spatial dimensions don't match
        mask_H, mask_W = mask_tensor.shape[2], mask_tensor.shape[3]
        if mask_H != H or mask_W != W:
            mask_tensor = F.interpolate(
                mask_tensor,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        
        # Validate values are in [0, 1]
        min_val, max_val = mask_tensor.min().item(), mask_tensor.max().item()
        if min_val < -1e-6 or max_val > 1.0 + 1e-6:
            raise ValueError(
                f"Perturbation mask values must be in [0, 1]. "
                f"Got range: [{min_val:.4f}, {max_val:.4f}]"
            )
        
        # Clamp to ensure exact [0, 1] range
        mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)
        
        # Move to device
        mask_tensor = mask_tensor.to(self.device)
        
        return mask_tensor
    
    def attack(
        self,
        image: Union[Image.Image, Path, str],
        verbose: bool = True,
    ) -> AttackResult:
        """
        Run adversarial attack on an image.
        
        Args:
            image: Input image (PIL Image or path)
            verbose: Print progress
            
        Returns:
            AttackResult with attack outcomes
        """
        # Reset the diff_explainer call count for reproducibility
        # This ensures that running attack() twice with the same seed produces identical results
        self.diff_explainer.reset_call_count()
        generator = self._get_generator()
        self._call_count += 1
        
        # Load image
        if isinstance(image, (str, Path)):
            img_pil = Image.open(image).convert("RGB")
        else:
            img_pil = image
        
        # Preprocess
        x_original = self.preprocess_fn(img_pil).unsqueeze(0).to(self.device)
        
        # Validate and prepare perturbation mask now that we know input shape
        self._perturbation_mask = self._validate_and_prepare_mask(
            self._perturbation_mask_input,
            x_original.shape
        )
        if self._perturbation_mask is not None:
            mask_coverage = self._perturbation_mask.mean().item() * 100
            if verbose:
                print(f"[Attack] Perturbation mask active: {mask_coverage:.1f}% of pixels attackable")
        
        # Get original output
        self.model.eval()
        with torch.no_grad():
            output_original = self.model(x_original)
            if output_original.numel() > 1:
                output_original = output_original.view(-1)[0:1]
        
        # Get original explanation using differentiable explainer
        with torch.enable_grad():
            x_temp = x_original.clone().requires_grad_(True)
            explanation_original = self.diff_explainer(x_temp).detach()
        
        # Initialize perturbation
        # Start with random perturbation to avoid zero gradients (e.g. when maximizing MSE from 0)
        if self.epsilon > 0:
            # Generate random values in [-epsilon, epsilon]
            # Use generator for determinism if provided
            delta = (torch.rand(x_original.shape, generator=generator, device=self.device) * 2 - 1) * self.epsilon
            # Apply mask if present
            if self._perturbation_mask is not None:
                delta = delta * self._perturbation_mask
        else:
            delta = torch.zeros_like(x_original, device=self.device)
        
        delta.requires_grad_(True)
        optimizer = torch.optim.Adam([delta], lr=self.step_size)
        
        # Track best result
        best_loss = float('inf')
        best_delta = None
        best_explanation_change = 0.0
        patience_counter = 0
        loss_history = []
        
        if verbose:
            print(f"[Attack] Starting adversarial optimization (max {self.max_iterations} iterations)")
            print(f"[Attack] Original output: {output_original.item():.4f}")
        
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            # Perturbed input
            x_perturbed = x_original + delta
            
            # Clamp to valid range (assuming normalized to [-1, 1] or [0, 1])
            x_perturbed = torch.clamp(x_perturbed, -1.0, 1.0)
            
            # Get perturbed output
            output_perturbed = self.model(x_perturbed)
            if output_perturbed.numel() > 1:
                output_perturbed = output_perturbed.view(-1)[0:1]
            
            # Get perturbed explanation (differentiable)
            explanation_perturbed = self.diff_explainer(x_perturbed)
            
            # Compute loss
            loss, components = self.loss_fn(
                explanation_original=explanation_original,
                explanation_perturbed=explanation_perturbed,
                output_original=output_original.detach(),
                output_perturbed=output_perturbed,
                perturbation=delta,
            )
            
            # Track metrics
            output_change = abs(output_original.item() - output_perturbed.item())
            explanation_similarity = self._compute_similarity(
                explanation_original, explanation_perturbed
            )
            explanation_change = 1.0 - explanation_similarity
            
            components['output_change'] = output_change
            components['explanation_similarity'] = explanation_similarity
            components['explanation_change'] = explanation_change
            loss_history.append(components)
            
            # Check if this is the best result (minimize loss)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_delta = delta.detach().clone()
                best_explanation_change = explanation_change
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Backward pass
            loss.backward()
            
            # Apply mask to gradients before optimizer step
            # Note: We rely on the final projection step to enforce the mask strictly.
            # While masking gradients could help the optimizer's momentum state in Adam,
            # it is not strictly necessary for correctness or performance.
            # if self._perturbation_mask is not None:
            #     delta.grad.data *= self._perturbation_mask
            
            optimizer.step()
            
            # Project delta to epsilon ball (L∞ constraint)
            # We must use the mask here to enforce bounds for non-binary masks
            with torch.no_grad():
                if self._perturbation_mask is not None:
                    limit = self.epsilon * self._perturbation_mask
                    delta.data = torch.clamp(delta.data, -limit, limit)
                else:
                    delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(
                    f"[Attack] Iter {iteration + 1}: "
                    f"loss={loss.item():.4f}, "
                    f"out_change={output_change:.4f}, "
                    f"exp_change={explanation_change:.4f}, "
                    f"δ_norm={torch.norm(delta).item():.4f}"
                )
            
            # Early stopping
            if patience_counter >= self.early_stop_patience:
                if verbose:
                    print(f"[Attack] Early stopping at iteration {iteration + 1}")
                break
            
            # Success check
            if output_change < self.output_threshold and explanation_change > self.explanation_threshold:
                if verbose:
                    print(f"[Attack] Success at iteration {iteration + 1}!")
                break
        
        # Use best delta for final result
        if best_delta is not None and not self.save_last:
            delta = best_delta
        
        # Final evaluation with accurate SHAP if available
        with torch.no_grad():
            x_perturbed_final = torch.clamp(x_original + delta, -1.0, 1.0)
            output_perturbed_final = self.model(x_perturbed_final)
            if output_perturbed_final.numel() > 1:
                output_perturbed_final = output_perturbed_final.view(-1)[0]
        
        # Get final perturbed explanation
        with torch.enable_grad():
            x_temp = x_perturbed_final.clone().requires_grad_(True)
            explanation_perturbed_np = self.diff_explainer(x_temp).detach().cpu().numpy()
        
        # Compute final metrics
        final_output_change = abs(output_original.item() - output_perturbed_final.item())
        final_explanation_similarity = self._compute_similarity_np(
            explanation_original.cpu().numpy(),
            explanation_perturbed_np
        )
        
        # Check success
        success = (
            final_output_change < self.output_threshold and 
            (1.0 - final_explanation_similarity) > self.explanation_threshold
        )
        
        # Prepare result
        delta_np = delta.detach().cpu().numpy()[0]  # (C, H, W)
        perturbed_image_np = self._tensor_to_vis_uint8(x_perturbed_final.cpu())
        original_image_np = self._tensor_to_vis_uint8(x_original.cpu())
        
        result = AttackResult(
            success=success,
            iterations=iteration + 1,
            original_output=output_original.item(),
            original_explanation=explanation_original.cpu().numpy(),
            original_image=original_image_np,
            perturbed_output=output_perturbed_final.item(),
            perturbed_explanation=explanation_perturbed_np,
            perturbation=delta_np,
            perturbed_image=perturbed_image_np,
            output_change=final_output_change,
            explanation_similarity=final_explanation_similarity,
            perturbation_norm_l2=float(np.linalg.norm(delta_np)),
            perturbation_norm_linf=float(np.max(np.abs(delta_np))),
            loss_history=loss_history,
            config=self.config,
        )
        
        if verbose:
            print(f"[Attack] Final: success={success}, "
                  f"output_change={final_output_change:.4f}, "
                  f"explanation_change={1.0 - final_explanation_similarity:.4f}")
        
        return result
    
    def _compute_similarity(
        self, 
        exp1: torch.Tensor, 
        exp2: torch.Tensor
    ) -> float:
        """Compute cosine similarity between two explanation tensors."""
        e1 = exp1.flatten()
        e2 = exp2.flatten()
        similarity = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0))
        return float(similarity.item())
    
    def _compute_similarity_np(self, exp1: np.ndarray, exp2: np.ndarray) -> float:
        """Compute cosine similarity between two numpy explanation arrays."""
        e1 = exp1.flatten()
        e2 = exp2.flatten()
        dot = np.dot(e1, e2)
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(dot / (norm1 * norm2))
    
    def _tensor_to_pil(self, x_tensor: torch.Tensor) -> Image.Image:
        """Convert normalized tensor back to PIL Image."""
        x = x_tensor[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        x = (x * self.norm_std) + self.norm_mean
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0).astype(np.uint8)
        return Image.fromarray(x)
    
    def _tensor_to_vis_uint8(self, x_tensor: torch.Tensor) -> np.ndarray:
        """Convert normalized tensor to uint8 numpy array for visualization."""
        x = x_tensor[0].numpy().transpose(1, 2, 0)  # (H, W, C)
        x = (x * self.norm_std) + self.norm_mean
        x = np.clip(x, 0.0, 1.0)
        return (x * 255.0).astype(np.uint8)
    
    def save_result(
        self,
        result: AttackResult,
        output_dir: Path,
        original_image: Union[Image.Image, Path, str] = None,
    ) -> None:
        """
        Save attack results to disk.
        
        Args:
            result: AttackResult from attack()
            output_dir: Directory to save results
            original_image: Original image for side-by-side visualization
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save numpy arrays
        np.save(output_dir / "perturbation.npy", result.perturbation)
        np.save(output_dir / "original_explanation.npy", result.original_explanation)
        np.save(output_dir / "perturbed_explanation.npy", result.perturbed_explanation)
        
        # Save perturbation mask if used
        if self._perturbation_mask is not None:
            mask_np = self._perturbation_mask[0, 0].cpu().numpy()  # (H, W)
            np.save(output_dir / "perturbation_mask.npy", mask_np)
            self._save_mask_vis(mask_np, output_dir / "perturbation_mask_vis.png")
        
        # Save perturbed image
        Image.fromarray(result.perturbed_image).save(output_dir / "perturbed_image.png")
        
        # Save perturbation visualization
        self._save_perturbation_vis(result.perturbation, output_dir / "perturbation_vis.png")
        
        # Save explanation comparison
        self._save_explanation_comparison(
            result.original_explanation,
            result.perturbed_explanation,
            output_dir / "explanation_comparison.png",
            result.original_output,
            result.perturbed_output,
        )
        
        # Save original if provided
        if original_image is not None:
            if isinstance(original_image, (str, Path)):
                original_image = Image.open(original_image).convert("RGB")
            original_image.save(output_dir / "original_image.png")
            
            # Side-by-side comparison
            # Use processed original from result if available for pixel-perfect comparison
            comparison_original = result.original_image if result.original_image is not None else np.array(original_image)
            
            self._save_image_comparison(
                comparison_original,
                result.perturbed_image,
                output_dir / "image_comparison.png",
            )
        
        # Save loss history plot
        self._save_loss_history(result.loss_history, output_dir / "loss_history.png")
        
        # Save metadata
        metadata = result.to_dict()
        with open(output_dir / "attack_result.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _save_mask_vis(self, mask: np.ndarray, path: Path) -> None:
        """Visualize perturbation mask."""
        plt.figure(figsize=(8, 6))
        plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
        coverage = mask.mean() * 100
        plt.title(f"Perturbation Mask\n{coverage:.1f}% attackable (white=attackable, black=protected)")
        plt.colorbar(label='Attackable')
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _save_perturbation_vis(self, perturbation: np.ndarray, path: Path) -> None:
        """Visualize perturbation (amplified for visibility)."""
        # perturbation is (C, H, W)
        delta = perturbation.transpose(1, 2, 0)  # (H, W, C)
        
        # Amplify for visualization
        amp = 10.0
        delta_vis = 0.5 + delta * amp
        delta_vis = np.clip(delta_vis, 0, 1)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(delta_vis)
        plt.title(f"Perturbation (amplified {amp}x)\nL2={np.linalg.norm(perturbation):.4f}, L∞={np.max(np.abs(perturbation)):.4f}")
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _save_explanation_comparison(
        self,
        exp_original: np.ndarray,
        exp_perturbed: np.ndarray,
        path: Path,
        out_original: float,
        out_perturbed: float,
    ) -> None:
        """Save side-by-side explanation comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        vmax = max(np.percentile(exp_original, 99), np.percentile(exp_perturbed, 99))
        axes[0].imshow(exp_original, cmap='hot', vmin=0, vmax=vmax)
        axes[0].set_title(f"Original Explanation\nOutput: {out_original:.4f}")
        axes[0].axis('off')
        
        # Perturbed
        axes[1].imshow(exp_perturbed, cmap='hot', vmin=0, vmax=vmax)
        axes[1].set_title(f"Perturbed Explanation\nOutput: {out_perturbed:.4f}")
        axes[1].axis('off')
        
        # Difference
        diff = np.abs(exp_original - exp_perturbed)
        axes[2].imshow(diff, cmap='coolwarm')
        axes[2].set_title(f"Absolute Difference\nΔOutput: {abs(out_original - out_perturbed):.4f}")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _save_image_comparison(
        self,
        img_original: np.ndarray,
        img_perturbed: np.ndarray,
        path: Path,
    ) -> None:
        """Save side-by-side image comparison."""
        # Resize original if needed to match perturbed
        if img_original.shape != img_perturbed.shape:
            # Assuming (H, W, C)
            H, W = img_perturbed.shape[:2]
            img_original_pil = Image.fromarray(img_original)
            img_original_pil = img_original_pil.resize((W, H), Image.BILINEAR)
            img_original = np.array(img_original_pil)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_original)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(img_perturbed)
        axes[1].set_title("Perturbed Image")
        axes[1].axis('off')
        
        # Difference (amplified)
        diff = np.abs(img_original.astype(float) - img_perturbed.astype(float))
        diff = np.clip(diff * 10, 0, 255).astype(np.uint8)
        axes[2].imshow(diff)
        axes[2].set_title("Difference (10x amplified)")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _save_loss_history(self, history: List[Dict], path: Path) -> None:
        """Plot loss history over iterations."""
        if not history:
            return
        
        iterations = range(1, len(history) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total loss
        axes[0, 0].plot(iterations, [h['total_loss'] for h in history])
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Output change
        axes[0, 1].plot(iterations, [h.get('output_change', 0) for h in history])
        axes[0, 1].axhline(y=self.output_threshold, color='r', linestyle='--', label='Threshold')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Output Change')
        axes[0, 1].set_title('Output Change (want < threshold)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Explanation change
        axes[1, 0].plot(iterations, [h.get('explanation_change', 0) for h in history])
        axes[1, 0].axhline(y=self.explanation_threshold, color='g', linestyle='--', label='Threshold')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Explanation Change')
        axes[1, 0].set_title('Explanation Change (want > threshold)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Individual losses
        axes[1, 1].plot(iterations, [h['loss_explanation'] for h in history], label='Explanation Loss')
        axes[1, 1].plot(iterations, [h['loss_output'] for h in history], label='Output Loss')
        axes[1, 1].plot(iterations, [h['loss_perturbation'] for h in history], label='Perturbation Loss')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Weighted Loss')
        axes[1, 1].set_title('Weighted Loss Components')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
