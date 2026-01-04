"""Base model interface for autonomous driving models."""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Abstract base class for driving models.
    
    All driving models should inherit from this class and implement
    the required methods.
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
            
        Returns:
            Model output (e.g., steering angle prediction).
        """
        pass
    
    @abstractmethod
    def get_flattened_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract flattened feature representations.
        
        Used for clustering/K-means background selection in SHAP.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
            
        Returns:
            Flattened feature tensor of shape (N, D).
        """
        pass
    
    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int, int]:
        """Return expected input shape (H, W)."""
        pass
