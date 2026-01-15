"""
Tests for adversarial objectives (loss functions).
"""

import pytest
import torch
import numpy as np

from attn_stability_drive.adversarial.objectives import (
    ExplanationDissimilarityLoss,
    OutputSimilarityLoss,
    PerturbationNormLoss,
    CombinedAdversarialLoss,
    DifferentiableSHAP,
    topk_intersection,
)


class TestExplanationDissimilarityLoss:
    """Tests for ExplanationDissimilarityLoss."""
    
    def test_identical_explanations_mse(self):
        """Identical explanations should have zero distance (negative loss for maximization)."""
        loss_fn = ExplanationDissimilarityLoss(metric="mse")
        exp = torch.randn(10, 10)
        loss = loss_fn(exp, exp.clone())
        # MSE of identical tensors is 0, so -0 = 0
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
    
    def test_different_explanations_mse(self):
        """Different explanations should have negative loss (distance is positive)."""
        loss_fn = ExplanationDissimilarityLoss(metric="mse")
        exp1 = torch.zeros(10, 10)
        exp2 = torch.ones(10, 10)
        loss = loss_fn(exp1, exp2)
        # MSE is 1.0, so loss is -1.0
        assert loss.item() < 0
    
    def test_cosine_identical(self):
        """Identical explanations should have cosine similarity of 1."""
        loss_fn = ExplanationDissimilarityLoss(metric="cosine")
        exp = torch.randn(10, 10) + 1  # Avoid zero mean
        loss = loss_fn(exp, exp.clone())
        assert loss.item() == pytest.approx(1.0, abs=1e-4)
    
    def test_cosine_orthogonal(self):
        """Orthogonal explanations should have cosine similarity near 0."""
        loss_fn = ExplanationDissimilarityLoss(metric="cosine")
        exp1 = torch.tensor([[1.0, 0.0]])
        exp2 = torch.tensor([[0.0, 1.0]])
        loss = loss_fn(exp1, exp2)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)
    
    def test_correlation_identical(self):
        """Identical explanations should have correlation of 1."""
        loss_fn = ExplanationDissimilarityLoss(metric="correlation")
        exp = torch.randn(10, 10)
        loss = loss_fn(exp, exp.clone())
        assert loss.item() == pytest.approx(1.0, abs=1e-4)
    
    def test_ssim_identical(self):
        """Identical explanations should have SSIM of 1."""
        loss_fn = ExplanationDissimilarityLoss(metric="ssim")
        # SSIM needs minimum spatial dimensions for the kernel (typically 11x11 window)
        exp = torch.rand(32, 32)
        loss = loss_fn(exp, exp.clone())
        assert loss.item() == pytest.approx(1.0, abs=1e-4)
    
    def test_ssim_different(self):
        """Very different explanations should have low SSIM."""
        loss_fn = ExplanationDissimilarityLoss(metric="ssim")
        exp1 = torch.zeros(32, 32)
        exp2 = torch.ones(32, 32)
        loss = loss_fn(exp1, exp2)
        # SSIM should be low for completely different images
        assert loss.item() < 0.5
    
    def test_ssim_gradients(self):
        """SSIM should be differentiable."""
        loss_fn = ExplanationDissimilarityLoss(metric="ssim")
        exp1 = torch.rand(32, 32, requires_grad=True)
        exp2 = torch.rand(32, 32)
        loss = loss_fn(exp1, exp2)
        loss.backward()
        assert exp1.grad is not None
        assert not torch.isnan(exp1.grad).any()

    def test_topk_identical(self):
        """Identical explanations should have high top-k intersection."""
        loss_fn = ExplanationDissimilarityLoss(metric="topk", topk_percent=0.1)
        exp = torch.randn(32, 32)
        loss = loss_fn(exp, exp.clone())
        # Identical explanations have perfect overlap, loss should be high (~1)
        assert loss.item() > 0.5

    def test_topk_different(self):
        """Different random explanations should have low top-k intersection."""
        loss_fn = ExplanationDissimilarityLoss(metric="topk", topk_percent=0.1)
        torch.manual_seed(42)
        exp1 = torch.randn(32, 32)
        torch.manual_seed(123)
        exp2 = torch.randn(32, 32)
        loss = loss_fn(exp1, exp2)
        # Random explanations should have low overlap
        assert loss.item() < 0.5

    def test_topk_gradients(self):
        """Top-k should be differentiable."""
        loss_fn = ExplanationDissimilarityLoss(metric="topk", topk_percent=0.1)
        exp1 = torch.randn(32, 32, requires_grad=True)
        exp2 = torch.randn(32, 32)
        loss = loss_fn(exp1, exp2)
        loss.backward()
        assert exp1.grad is not None
        assert not torch.isnan(exp1.grad).any()

    def test_topk_custom_percent(self):
        """Top-k should work with different percentages."""
        # 50% should include more pixels
        loss_fn_50 = ExplanationDissimilarityLoss(metric="topk", topk_percent=0.5)
        # 5% should be more selective
        loss_fn_5 = ExplanationDissimilarityLoss(metric="topk", topk_percent=0.05)
        
        torch.manual_seed(42)
        exp1 = torch.randn(32, 32)
        torch.manual_seed(123)
        exp2 = torch.randn(32, 32)
        
        loss_50 = loss_fn_50(exp1, exp2)
        loss_5 = loss_fn_5(exp1, exp2)
        
        # Both should compute without error
        assert not torch.isnan(loss_50)
        assert not torch.isnan(loss_5)


class TestTopkIntersection:
    """Tests for the standalone topk_intersection function."""

    def test_identical_explanations(self):
        """Identical explanations should have IoU of 1.0."""
        exp = torch.randn(64, 64)
        iou = topk_intersection(exp, exp.clone(), k_percent=0.1)
        assert iou == pytest.approx(1.0, abs=1e-6)

    def test_different_explanations(self):
        """Different random explanations should have low IoU."""
        torch.manual_seed(42)
        exp1 = torch.randn(64, 64)
        torch.manual_seed(123)
        exp2 = torch.randn(64, 64)
        iou = topk_intersection(exp1, exp2, k_percent=0.1)
        # Random overlap should be around k_percent (10%)
        assert 0.0 <= iou <= 0.3

    def test_completely_disjoint(self):
        """Explanations with disjoint top-k should have IoU near 0."""
        # Create explanations where top values are in completely different locations
        exp1 = torch.zeros(10, 10)
        exp2 = torch.zeros(10, 10)
        # Put high values in different halves (50 pixels each)
        exp1[:5, :] = 10.0  # Top half
        exp2[5:, :] = 10.0  # Bottom half
        # With k_percent=0.5, we select 50 pixels from each
        # exp1's top 50 are all in top half, exp2's top 50 are all in bottom half
        iou = topk_intersection(exp1, exp2, k_percent=0.5)
        # IoU should be 0: intersection=0, union=100
        assert iou == pytest.approx(0.0, abs=1e-6)

    def test_various_k_percents(self):
        """Test different k_percent values."""
        exp = torch.randn(100, 100)
        
        # 1% of 10000 = 100 pixels
        iou_1 = topk_intersection(exp, exp.clone(), k_percent=0.01)
        # 50% of 10000 = 5000 pixels
        iou_50 = topk_intersection(exp, exp.clone(), k_percent=0.5)
        
        # Both should be 1.0 for identical explanations
        assert iou_1 == pytest.approx(1.0, abs=1e-6)
        assert iou_50 == pytest.approx(1.0, abs=1e-6)


class TestOutputSimilarityLoss:
    """Tests for OutputSimilarityLoss."""
    
    def test_identical_outputs(self):
        """Identical outputs should have zero loss."""
        loss_fn = OutputSimilarityLoss(metric="mse")
        out = torch.tensor([0.5])
        loss = loss_fn(out, out.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
    
    def test_different_outputs(self):
        """Different outputs should have positive loss."""
        loss_fn = OutputSimilarityLoss(metric="mse")
        out1 = torch.tensor([0.0])
        out2 = torch.tensor([1.0])
        loss = loss_fn(out1, out2)
        assert loss.item() == pytest.approx(1.0, abs=1e-6)
    
    def test_l1_loss(self):
        """Test L1 metric."""
        loss_fn = OutputSimilarityLoss(metric="l1")
        out1 = torch.tensor([0.0])
        out2 = torch.tensor([0.5])
        loss = loss_fn(out1, out2)
        assert loss.item() == pytest.approx(0.5, abs=1e-6)


class TestPerturbationNormLoss:
    """Tests for PerturbationNormLoss."""
    
    def test_zero_perturbation(self):
        """Zero perturbation should have zero norm."""
        loss_fn = PerturbationNormLoss(norm="l2")
        delta = torch.zeros(1, 3, 10, 10)
        loss = loss_fn(delta)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
    
    def test_l2_norm(self):
        """Test L2 norm computation."""
        loss_fn = PerturbationNormLoss(norm="l2")
        delta = torch.ones(1, 1, 2, 2)  # 4 elements, each 1.0
        loss = loss_fn(delta)
        # L2 norm of [1,1,1,1] is sqrt(4) = 2
        assert loss.item() == pytest.approx(2.0, abs=1e-4)
    
    def test_linf_norm(self):
        """Test L∞ norm computation."""
        loss_fn = PerturbationNormLoss(norm="linf")
        delta = torch.tensor([[[[0.1, 0.5], [-0.3, 0.2]]]])
        loss = loss_fn(delta)
        assert loss.item() == pytest.approx(0.5, abs=1e-4)
    
    def test_epsilon_threshold(self):
        """Test epsilon threshold (perturbation below epsilon is free)."""
        loss_fn = PerturbationNormLoss(norm="l2", epsilon=5.0)
        delta = torch.ones(1, 1, 2, 2)  # L2 norm is 2.0
        loss = loss_fn(delta)
        # 2.0 < 5.0, so ReLU(2.0 - 5.0) = 0
        assert loss.item() == pytest.approx(0.0, abs=1e-4)


class TestCombinedAdversarialLoss:
    """Tests for CombinedAdversarialLoss."""
    
    def test_combined_loss_components(self):
        """Test that combined loss returns all components."""
        loss_fn = CombinedAdversarialLoss(
            lambda_explanation=1.0,
            lambda_output=1.0,
            lambda_perturbation=1.0,
        )
        
        exp_orig = torch.randn(10, 10)
        exp_pert = torch.randn(10, 10)
        out_orig = torch.tensor([0.5])
        out_pert = torch.tensor([0.6])
        delta = torch.randn(1, 3, 10, 10) * 0.1
        
        total_loss, components = loss_fn(
            explanation_original=exp_orig,
            explanation_perturbed=exp_pert,
            output_original=out_orig,
            output_perturbed=out_pert,
            perturbation=delta,
        )
        
        assert isinstance(total_loss, torch.Tensor)
        assert "loss_explanation" in components
        assert "loss_output" in components
        assert "loss_perturbation" in components
        assert "total_loss" in components


class TestDifferentiableSHAP:
    """Tests for DifferentiableSHAP."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                return x.view(-1)
        
        return SimpleModel()
    
    def test_vanilla_gradient_shape(self, simple_model):
        """Test that vanilla gradient returns correct shape."""
        explainer = DifferentiableSHAP(simple_model, method="gradient")
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        saliency = explainer(x)
        assert saliency.shape == (32, 32)
    
    def test_smooth_grad_shape(self, simple_model):
        """Test that smooth_grad returns correct shape."""
        explainer = DifferentiableSHAP(simple_model, method="smooth_grad", smooth_samples=3)
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        saliency = explainer(x)
        assert saliency.shape == (32, 32)
    
    def test_integrated_grad_shape(self, simple_model):
        """Test that integrated gradients returns correct shape."""
        explainer = DifferentiableSHAP(simple_model, method="integrated_grad", ig_steps=3)
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        saliency = explainer(x)
        assert saliency.shape == (32, 32)
    
    def test_saliency_is_differentiable(self, simple_model):
        """Test that saliency computation allows gradient flow."""
        explainer = DifferentiableSHAP(simple_model, method="gradient")
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        saliency = explainer(x)
        
        # Sum saliency and backprop
        loss = saliency.sum()
        loss.backward()
        
        # x should have gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape
