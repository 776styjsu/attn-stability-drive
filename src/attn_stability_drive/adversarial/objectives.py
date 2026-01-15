"""
Objective/Loss functions for adversarial explanation attacks.

These loss functions are designed to find perturbations that:
1. Maximize explanation change (dissimilarity)
2. Minimize model output change (keep predictions similar)
3. Minimize perturbation magnitude (stay close to original)
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np
import piq
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplanationDissimilarityLoss(nn.Module):
    """
    Loss that measures how similar two explanations are.
    
    We want to MAXIMIZE dissimilarity, so this returns negative similarity
    (or positive distance) for gradient descent.
    
    Supports multiple distance metrics:
    - 'mse': Mean Squared Error between saliency maps
    - 'cosine': 1 - Cosine similarity (flattened)
    - 'ssim': 1 - Structural Similarity (differentiable approximation)
    - 'correlation': 1 - Pearson correlation
    - 'topk': Top-k intersection - measures overlap of top-k% most important pixels
    """
    
    def __init__(self, metric: str = "mse", topk_percent: float = 0.1):
        """
        Initialize the dissimilarity loss.
        
        Args:
            metric: Distance metric to use
            topk_percent: For 'topk' metric, the fraction of top pixels to consider (default 10%)
        """
        super().__init__()
        self.metric = metric.lower()
        self.topk_percent = topk_percent
        if self.metric not in {"mse", "cosine", "correlation", "l1", "ssim", "topk"}:
            raise ValueError(f"Unsupported metric: {metric}. Use 'mse', 'cosine', 'correlation', 'l1', 'ssim', or 'topk'.")
    
    def forward(
        self, 
        explanation_original: torch.Tensor, 
        explanation_perturbed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dissimilarity between two explanations.
        
        Args:
            explanation_original: Original saliency map, shape (H, W) or (1, H, W) or (C, H, W)
            explanation_perturbed: Perturbed saliency map, same shape
            
        Returns:
            Negative dissimilarity (lower = more different explanations)
        """
        # Flatten for metrics that need it
        e1 = explanation_original.flatten()
        e2 = explanation_perturbed.flatten()
        
        if self.metric == "mse":
            # MSE distance - we want to maximize this, so return negative
            distance = F.mse_loss(e1, e2)
            return -distance
        
        elif self.metric == "l1":
            # L1 distance
            distance = F.l1_loss(e1, e2)
            return -distance
        
        elif self.metric == "cosine":
            # Cosine similarity in [−1, 1], we want to minimize similarity
            # Return similarity directly (minimizing it maximizes dissimilarity)
            similarity = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0))
            return similarity  # Minimize this
        
        elif self.metric == "correlation":
            # Pearson correlation
            e1_centered = e1 - e1.mean()
            e2_centered = e2 - e2.mean()
            
            num = (e1_centered * e2_centered).sum()
            denom = torch.sqrt((e1_centered ** 2).sum() * (e2_centered ** 2).sum() + 1e-8)
            correlation = num / denom
            return correlation  # Minimize this
        
        elif self.metric == "ssim":
            # Structural Similarity Index (differentiable via piq)
            # piq.ssim expects (N, C, H, W) tensors
            e1_4d = self._to_4d(explanation_original)
            e2_4d = self._to_4d(explanation_perturbed)
            similarity = piq.ssim(e1_4d, e2_4d, data_range=1.0)
            return similarity  # Minimize SSIM = maximize dissimilarity
        
        elif self.metric == "topk":
            # Top-k intersection: measures overlap of top-k% most important pixels
            # Uses soft top-k via sigmoid approximation for differentiability
            intersection = self._soft_topk_intersection(e1, e2, self.topk_percent)
            return intersection  # Minimize intersection = maximize dissimilarity
    
    def _to_4d(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to (N, C, H, W) format for piq metrics."""
        if tensor.ndim == 2:
            # (H, W) -> (1, 1, H, W)
            return tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            # (C, H, W) -> (1, C, H, W)
            return tensor.unsqueeze(0)
        elif tensor.ndim == 4:
            return tensor
        else:
            raise ValueError(f"Expected 2D, 3D, or 4D tensor, got {tensor.ndim}D")
    
    def _soft_topk_intersection(
        self, 
        e1: torch.Tensor, 
        e2: torch.Tensor, 
        k_percent: float,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute differentiable top-k intersection between two saliency maps.
        
        Uses a soft approximation of top-k selection via sigmoid functions
        to maintain differentiability for gradient-based optimization.
        
        The intersection measures what fraction of the top-k% most important
        pixels in each explanation overlap. High intersection means similar
        explanations focus on the same regions.
        
        Args:
            e1: First explanation (flattened)
            e2: Second explanation (flattened)  
            k_percent: Fraction of top pixels to consider (0.1 = top 10%)
            temperature: Softmax temperature for soft thresholding (lower = sharper)
            
        Returns:
            Soft intersection score in [0, 1], where 1 = perfect overlap
        """
        n = e1.numel()
        k = max(1, int(n * k_percent))
        
        # Use absolute values for saliency (importance can be positive or negative)
        e1_abs = torch.abs(e1)
        e2_abs = torch.abs(e2)
        
        # Find the k-th largest value as threshold
        # topk returns (values, indices), we need the k-th value as threshold
        threshold1 = torch.topk(e1_abs, k, largest=True).values[-1]
        threshold2 = torch.topk(e2_abs, k, largest=True).values[-1]
        
        # Soft top-k membership using sigmoid
        # Pixels above threshold get weight ~1, below get weight ~0
        # The temperature controls the sharpness of the transition
        mask1 = torch.sigmoid((e1_abs - threshold1) / temperature)
        mask2 = torch.sigmoid((e2_abs - threshold2) / temperature)
        
        # Soft intersection: product of masks (both need to be "selected")
        intersection = (mask1 * mask2).sum()
        
        # Normalize by k to get intersection ratio
        # This gives us IoU-style metric in [0, 1]
        intersection_ratio = intersection / k
        
        return intersection_ratio


def topk_intersection(
    explanation1: torch.Tensor,
    explanation2: torch.Tensor,
    k_percent: float = 0.1,
) -> float:
    """
    Compute top-k intersection between two explanations (non-differentiable).
    
    This is the "hard" version of top-k intersection for evaluation purposes.
    It finds the indices of the top-k% pixels in each explanation and computes
    the Jaccard index (intersection over union) of these sets.
    
    Args:
        explanation1: First saliency map, any shape (will be flattened)
        explanation2: Second saliency map, same shape as explanation1
        k_percent: Fraction of top pixels to consider (default 0.1 = top 10%)
        
    Returns:
        Intersection over union (IoU) of top-k pixel sets, in [0, 1]
        
    Example:
        >>> e1 = torch.randn(64, 64)
        >>> e2 = e1.clone()  # Identical
        >>> topk_intersection(e1, e2)  # Returns 1.0
        
        >>> e2 = torch.randn(64, 64)  # Different
        >>> topk_intersection(e1, e2)  # Returns ~0.1 (random overlap)
    """
    e1 = explanation1.flatten()
    e2 = explanation2.flatten()
    
    n = e1.numel()
    k = max(1, int(n * k_percent))
    
    # Get indices of top-k pixels by absolute value
    _, topk_idx1 = torch.topk(torch.abs(e1), k, largest=True)
    _, topk_idx2 = torch.topk(torch.abs(e2), k, largest=True)
    
    # Convert to sets for intersection/union
    set1 = set(topk_idx1.cpu().numpy().tolist())
    set2 = set(topk_idx2.cpu().numpy().tolist())
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


class OutputSimilarityLoss(nn.Module):
    """
    Loss that penalizes changes in model output.
    
    We want the perturbed input to produce a similar output to the original,
    so we minimize the difference between outputs.
    """
    
    def __init__(self, metric: str = "mse"):
        super().__init__()
        self.metric = metric.lower()
    
    def forward(
        self, 
        output_original: torch.Tensor, 
        output_perturbed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity loss between model outputs.
        
        Args:
            output_original: Original model output
            output_perturbed: Perturbed model output
            
        Returns:
            Distance between outputs (minimize to keep outputs similar)
        """
        if self.metric == "mse":
            return F.mse_loss(output_original, output_perturbed)
        elif self.metric == "l1":
            return F.l1_loss(output_original, output_perturbed)
        elif self.metric == "huber":
            return F.smooth_l1_loss(output_original, output_perturbed)
        else:
            return F.mse_loss(output_original, output_perturbed)


class PerturbationNormLoss(nn.Module):
    """
    Loss that penalizes the magnitude of the perturbation.
    
    Encourages finding minimal perturbations that achieve the attack goal.
    """
    
    def __init__(self, norm: str = "l2", epsilon: float = 0.0):
        """
        Args:
            norm: Type of norm ('l2', 'l1', 'linf')
            epsilon: Threshold below which perturbation is "free" (no penalty)
        """
        super().__init__()
        self.norm = norm.lower()
        self.epsilon = epsilon
    
    def forward(self, perturbation: torch.Tensor) -> torch.Tensor:
        """
        Compute norm of perturbation.
        
        Args:
            perturbation: The delta added to original image, shape (1, C, H, W) or (C, H, W)
            
        Returns:
            Norm of perturbation
        """
        p = perturbation.flatten()
        
        if self.norm == "l2":
            norm_val = torch.norm(p, p=2)
        elif self.norm == "l1":
            norm_val = torch.norm(p, p=1)
        elif self.norm == "linf":
            norm_val = torch.max(torch.abs(p))
        else:
            norm_val = torch.norm(p, p=2)
        
        # Apply epsilon threshold
        if self.epsilon > 0:
            norm_val = F.relu(norm_val - self.epsilon)
        
        return norm_val


class CombinedAdversarialLoss(nn.Module):
    """
    Combined loss for explanation adversarial attack.
    
    L = -λ_exp * Sim(E(x), E(x+δ)) 
        + λ_out * ||Model(x) - Model(x+δ)||
        + λ_pert * ||δ||
    
    Where:
    - First term: Maximize explanation dissimilarity (negative because we minimize loss)
    - Second term: Keep model outputs similar
    - Third term: Minimize perturbation magnitude
    """
    
    def __init__(
        self,
        lambda_explanation: float = 1.0,
        lambda_output: float = 10.0,
        lambda_perturbation: float = 0.01,
        explanation_metric: str = "mse",
        output_metric: str = "mse",
        perturbation_norm: str = "l2",
        topk_percent: float = 0.1,
    ):
        """
        Args:
            lambda_explanation: Weight for explanation dissimilarity (maximize)
            lambda_output: Weight for output similarity (minimize change)
            lambda_perturbation: Weight for perturbation norm (minimize)
            explanation_metric: Metric for explanation comparison
            output_metric: Metric for output comparison
            perturbation_norm: Norm type for perturbation
            topk_percent: For 'topk' metric, the fraction of top pixels (default 10%)
        """
        super().__init__()
        self.lambda_exp = lambda_explanation
        self.lambda_out = lambda_output
        self.lambda_pert = lambda_perturbation
        
        self.explanation_loss = ExplanationDissimilarityLoss(
            metric=explanation_metric, topk_percent=topk_percent
        )
        self.output_loss = OutputSimilarityLoss(metric=output_metric)
        self.perturbation_loss = PerturbationNormLoss(norm=perturbation_norm)
    
    def forward(
        self,
        explanation_original: torch.Tensor,
        explanation_perturbed: torch.Tensor,
        output_original: torch.Tensor,
        output_perturbed: torch.Tensor,
        perturbation: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined adversarial loss.
        
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Explanation dissimilarity (already returns negative for maximization)
        loss_exp = self.explanation_loss(explanation_original, explanation_perturbed)
        
        # Output similarity
        loss_out = self.output_loss(output_original, output_perturbed)
        
        # Perturbation norm
        loss_pert = self.perturbation_loss(perturbation)
        
        # Combined loss
        total_loss = (
            self.lambda_exp * loss_exp +
            self.lambda_out * loss_out +
            self.lambda_pert * loss_pert
        )
        
        components = {
            "loss_explanation": (self.lambda_exp * loss_exp).item(),
            "loss_output": (self.lambda_out * loss_out).item(),
            "loss_perturbation": (self.lambda_pert * loss_pert).item(),
            "total_loss": total_loss.item(),
        }
        
        return total_loss, components


class DifferentiableSHAP(nn.Module):
    """
    Differentiable approximation of SHAP values for gradient-based optimization.
    
    This uses a simplified gradient-based saliency as a differentiable proxy
    for SHAP values during the optimization loop. The actual SHAP values
    are computed at the start and end for accurate comparison.
    
    Methods:
    - 'gradient': Simple input gradient (Vanilla Gradient)
    - 'smooth_grad': Averaged gradients with noise (SmoothGrad)
    - 'integrated_grad': Integrated Gradients approximation
    - 'gradient_shap': Expected Gradients (GradientSHAP) - averages IG over background distribution
    
    Note on Reproducibility:
        When a seed is provided, this class ensures deterministic behavior by
        resetting the random generator state at the start of each forward pass.
        This means calling forward() multiple times with the same input will
        produce identical results, which is essential for reproducible adversarial
        attacks.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        method: str = "gradient",
        smooth_samples: int = 20,
        smooth_noise: float = 0.1,
        ig_steps: int = 50,
        background: Optional[torch.Tensor] = None,
        gradient_shap_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            model: The model to explain.
            method: Explanation method ('gradient', 'smooth_grad', 'integrated_grad', 'gradient_shap').
            smooth_samples: Number of noise samples for smooth_grad.
            smooth_noise: Noise std for smooth_grad.
            ig_steps: Number of interpolation steps for integrated gradients (default 50 to match SHAP library).
            background: Background tensor for gradient_shap (N, C, H, W).
            gradient_shap_samples: Number of background samples to use per forward pass for gradient_shap.
                                   If None, uses all background samples (matches SHAP library behavior).
                                   Set to a smaller value (e.g., 4-8) for faster optimization.
            seed: Random seed for reproducibility. If provided, random operations in
                  smooth_grad and gradient_shap will be deterministic per call.
        """
        super().__init__()
        self.model = model
        self.method = method.lower()
        self.smooth_samples = smooth_samples
        self.smooth_noise = smooth_noise
        self.ig_steps = ig_steps
        self.gradient_shap_samples = gradient_shap_samples
        self.seed = seed
        self._call_count = 0  # Track calls for reproducible per-call seeding
        self.register_buffer("background", background if background is not None else None)
    
    def reset_call_count(self) -> None:
        """Reset the call counter. Call this at the start of a new attack for reproducibility."""
        self._call_count = 0
    
    def _get_generator(self, device: torch.device = torch.device("cpu")) -> Optional[torch.Generator]:
        """
        Get a generator seeded deterministically based on the base seed and call count.
        
        This ensures that each call to forward() produces the same random sequence
        when called with the same call_count, regardless of what happened in between.
        """
        if self.seed is None:
            return None
        
        # Generator device must match the device of tensors used with it
        gen_device = "cuda" if "cuda" in str(device).lower() else "cpu"
        generator = torch.Generator(device=gen_device)
        
        # Combine base seed with call count for unique but reproducible per-call seed
        generator.manual_seed(self.seed + self._call_count * 1000003)  # Large prime to avoid collisions
        return generator
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable saliency approximation.
        
        Args:
            x: Input tensor (1, C, H, W), requires_grad should be True
            
        Returns:
            Saliency map (H, W) as tensor
        
        Note:
            When a seed is provided, each call uses a deterministic random sequence
            based on the call count. Use reset_call_count() at the start of each
            attack for full reproducibility.
        """
        # Get a fresh generator for this call (deterministic based on call count)
        generator = self._get_generator(device=x.device)
        self._call_count += 1
        
        if self.method == "gradient":
            return self._vanilla_gradient(x)
        elif self.method == "smooth_grad":
            return self._smooth_grad(x, generator)
        elif self.method == "integrated_grad":
            return self._integrated_gradients(x)
        elif self.method == "gradient_shap":
            return self._gradient_shap(x, generator)
        else:
            return self._vanilla_gradient(x)
    
    def _vanilla_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Simple input gradient."""
        x = x.clone().requires_grad_(True)
        output = self.model(x)
        
        # Handle multi-output (take first/steering)
        if output.numel() > 1:
            output = output.view(-1)[0]
        
        grad = torch.autograd.grad(
            outputs=output,
            inputs=x,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Sum absolute values across channels
        saliency = grad.abs().sum(dim=1).squeeze(0)  # (H, W)
        return saliency
    
    def _smooth_grad(self, x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """SmoothGrad: average gradients over noisy samples."""
        saliency_sum = torch.zeros_like(x[:, 0, :, :]).squeeze(0)  # (H, W)
        
        for _ in range(self.smooth_samples):
            if generator is not None:
                noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) * self.smooth_noise
            else:
                noise = torch.randn_like(x) * self.smooth_noise
            x_noisy = x + noise
            x_noisy.requires_grad_(True)
            
            output = self.model(x_noisy)
            if output.numel() > 1:
                output = output.view(-1)[0]
            
            grad = torch.autograd.grad(
                outputs=output,
                inputs=x_noisy,
                create_graph=True,
                retain_graph=True,
            )[0]
            
            saliency_sum = saliency_sum + grad.abs().sum(dim=1).squeeze(0)
        
        return saliency_sum / self.smooth_samples
    
    def _integrated_gradients(self, x: torch.Tensor) -> torch.Tensor:
        """Integrated Gradients approximation."""
        baseline = torch.zeros_like(x)
        saliency_sum = torch.zeros_like(x[:, 0, :, :]).squeeze(0)  # (H, W)
        
        for i in range(self.ig_steps):
            alpha = (i + 0.5) / self.ig_steps
            x_interp = baseline + alpha * (x - baseline)
            x_interp.requires_grad_(True)
            
            output = self.model(x_interp)
            if output.numel() > 1:
                output = output.view(-1)[0]
            
            grad = torch.autograd.grad(
                outputs=output,
                inputs=x_interp,
                create_graph=True,
                retain_graph=True,
            )[0]
            
            saliency_sum = saliency_sum + grad.abs().sum(dim=1).squeeze(0)
        
        # Scale by (x - baseline)
        delta = (x - baseline).abs().sum(dim=1).squeeze(0)
        integrated = saliency_sum * delta / self.ig_steps
        
        return integrated

    def _gradient_shap(self, x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Differentiable GradientSHAP approximation.
        
        Computes Expected Gradients by averaging Integrated Gradients 
        over the background distribution.
        
        If background is large, we sample a subset for efficiency.
        """
        if self.background is None:
            # Fallback to IG with zero baseline if no background provided
            return self._integrated_gradients(x)
            
        n_bg = self.background.size(0)
        
        # Determine number of background samples to use
        if self.gradient_shap_samples is None:
            # Use all background samples (matches SHAP library behavior)
            n_samples = n_bg
        else:
            # Use specified number of samples (for faster optimization)
            n_samples = min(n_bg, self.gradient_shap_samples)
        
        if n_samples < n_bg:
            if generator is not None:
                indices = torch.randperm(n_bg, generator=generator)[:n_samples].to(self.background.device)
            else:
                indices = torch.randperm(n_bg, device=self.background.device)[:n_samples]
            bg_samples = self.background[indices]
        else:
            bg_samples = self.background
            
        saliency_sum = torch.zeros_like(x[:, 0, :, :]).squeeze(0)
        
        for i in range(n_samples):
            baseline = bg_samples[i:i+1]
            # Expand baseline to match x if needed
            if baseline.ndim == 3:
                baseline = baseline.unsqueeze(0)
                
            # Compute IG for this baseline
            # We inline the IG logic here to avoid re-implementing or complex calls
            ig_sum = torch.zeros_like(saliency_sum)
            
            for step in range(self.ig_steps):
                alpha = (step + 0.5) / self.ig_steps
                x_interp = baseline + alpha * (x - baseline)
                x_interp.requires_grad_(True)
                
                output = self.model(x_interp)
                if output.numel() > 1:
                    output = output.view(-1)[0]
                
                grad = torch.autograd.grad(
                    outputs=output,
                    inputs=x_interp,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                
                ig_sum = ig_sum + grad.abs().sum(dim=1).squeeze(0)
            
            # Scale by (x - baseline)
            delta = (x - baseline).abs().sum(dim=1).squeeze(0)
            saliency_sum = saliency_sum + (ig_sum * delta / self.ig_steps)
            
        return saliency_sum / n_samples
