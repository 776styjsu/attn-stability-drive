"""CUDA and reproducibility utilities."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> torch.Generator:
    """
    Set random seeds for reproducibility across all random number generators.
    
    Args:
        seed: The seed value to use.
        deterministic: If True, enables deterministic CUDA operations (may reduce performance).
        
    Returns:
        A PyTorch Generator seeded with the given seed, useful for passing to
        functions that accept a generator argument.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Enable deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # PyTorch 1.8+ deterministic flag
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        
        # Set environment variable for additional determinism
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Create a generator for use in random operations
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return generator
