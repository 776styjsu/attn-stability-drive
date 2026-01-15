"""
Tests for adversarial attacks.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from attn_stability_drive.adversarial.attacks import (
    ExplanationAdversarialAttack,
    AttackResult,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_shape=(32, 32)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def model():
    """Create a simple model."""
    return SimpleModel(input_shape=(32, 32))


@pytest.fixture
def preprocess():
    """Create preprocessing transform."""
    return Compose([
        Resize((32, 32)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


@pytest.fixture
def sample_image():
    """Create a sample PIL image."""
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestAttackResult:
    """Tests for AttackResult dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = AttackResult(
            success=True,
            iterations=50,
            original_output=0.5,
            original_explanation=np.zeros((10, 10)),
            perturbed_output=0.52,
            perturbed_explanation=np.ones((10, 10)),
            perturbation=np.zeros((3, 10, 10)),
            perturbed_image=np.zeros((10, 10, 3), dtype=np.uint8),
            output_change=0.02,
            explanation_similarity=0.5,
            perturbation_norm_l2=0.1,
            perturbation_norm_linf=0.05,
        )
        
        d = result.to_dict()
        assert d["success"] == True
        assert d["iterations"] == 50
        assert d["original_output"] == 0.5
        # Large arrays are set to None for JSON serialization
        assert d["original_explanation"] is None


class TestExplanationAdversarialAttack:
    """Tests for ExplanationAdversarialAttack."""
    
    def test_attack_initialization(self, model, preprocess):
        """Test attack can be initialized."""
        attack = ExplanationAdversarialAttack(
            model=model,
            preprocess_fn=preprocess,
            device="cpu",
            max_iterations=5,
        )
        assert attack is not None
        assert attack.epsilon == 0.1
        assert attack.max_iterations == 5
    
    def test_attack_runs(self, model, preprocess, sample_image):
        """Test that attack runs without errors."""
        attack = ExplanationAdversarialAttack(
            model=model,
            preprocess_fn=preprocess,
            device="cpu",
            max_iterations=3,  # Very few iterations for test speed
            early_stop_patience=2,
        )
        
        result = attack.attack(sample_image, verbose=False)
        
        assert isinstance(result, AttackResult)
        assert result.iterations <= 3
        assert result.original_explanation.shape == (32, 32)
        assert result.perturbed_explanation.shape == (32, 32)
        assert result.perturbation.shape == (3, 32, 32)
    
    def test_attack_result_metrics(self, model, preprocess, sample_image):
        """Test that attack result contains valid metrics."""
        attack = ExplanationAdversarialAttack(
            model=model,
            preprocess_fn=preprocess,
            device="cpu",
            max_iterations=3,
        )
        
        result = attack.attack(sample_image, verbose=False)
        
        # Check metrics are reasonable
        assert result.output_change >= 0
        assert 0 <= result.explanation_similarity <= 1 or result.explanation_similarity >= -1  # Cosine can be negative
        assert result.perturbation_norm_l2 >= 0
        assert result.perturbation_norm_linf >= 0
        assert result.perturbation_norm_linf <= attack.epsilon
    
    def test_attack_with_custom_params(self, model, preprocess, sample_image):
        """Test attack with custom parameters."""
        attack = ExplanationAdversarialAttack(
            model=model,
            preprocess_fn=preprocess,
            device="cpu",
            lambda_explanation=2.0,
            lambda_output=5.0,
            lambda_perturbation=0.1,
            epsilon=0.05,
            step_size=0.005,
            max_iterations=2,
            output_threshold=0.1,
            explanation_threshold=0.2,
        )
        
        result = attack.attack(sample_image, verbose=False)
        
        assert isinstance(result, AttackResult)
        # Perturbation should be bounded by epsilon
        assert result.perturbation_norm_linf <= 0.05 + 1e-4
    
    def test_loss_history_tracking(self, model, preprocess, sample_image):
        """Test that loss history is tracked."""
        attack = ExplanationAdversarialAttack(
            model=model,
            preprocess_fn=preprocess,
            device="cpu",
            max_iterations=5,
            early_stop_patience=10,  # Don't early stop
        )
        
        result = attack.attack(sample_image, verbose=False)
        
        assert len(result.loss_history) > 0
        assert len(result.loss_history) <= 5
        
        # Check history entries have expected keys
        entry = result.loss_history[0]
        assert "total_loss" in entry
        assert "loss_explanation" in entry
        assert "loss_output" in entry
        assert "loss_perturbation" in entry
    
    def test_save_result(self, model, preprocess, sample_image, tmp_path):
        """Test saving attack results."""
        attack = ExplanationAdversarialAttack(
            model=model,
            preprocess_fn=preprocess,
            device="cpu",
            max_iterations=2,
        )
        
        result = attack.attack(sample_image, verbose=False)
        
        output_dir = tmp_path / "attack_output"
        attack.save_result(result, output_dir, original_image=sample_image)
        
        # Check files were created
        assert (output_dir / "perturbation.npy").exists()
        assert (output_dir / "original_explanation.npy").exists()
        assert (output_dir / "perturbed_explanation.npy").exists()
        assert (output_dir / "perturbed_image.png").exists()
        assert (output_dir / "attack_result.json").exists()
    
    def test_different_explanation_methods(self, model, preprocess, sample_image):
        """Test different differentiable explanation methods."""
        for method in ["gradient", "smooth_grad", "integrated_grad"]:
            attack = ExplanationAdversarialAttack(
                model=model,
                preprocess_fn=preprocess,
                device="cpu",
                diff_explanation_method=method,
                smooth_samples=2,  # Few samples for speed
                max_iterations=2,
            )
            
            result = attack.attack(sample_image, verbose=False)
            assert isinstance(result, AttackResult)


class TestAttackWithImagePath:
    """Test attack with image path instead of PIL Image."""
    
    def test_attack_with_path(self, model, preprocess, tmp_path):
        """Test attack can load image from path."""
        # Create and save a test image
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img_path = tmp_path / "test_image.png"
        img.save(img_path)
        
        attack = ExplanationAdversarialAttack(
            model=model,
            preprocess_fn=preprocess,
            device="cpu",
            max_iterations=2,
        )
        
        result = attack.attack(str(img_path), verbose=False)
        
        assert isinstance(result, AttackResult)
        assert result.perturbed_image.shape == (32, 32, 3)
