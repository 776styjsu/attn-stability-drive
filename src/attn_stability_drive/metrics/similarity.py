from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
import piq


class SimilarityComputer:
    """Helper that wraps a similarity metric behind a common interface."""

    _device = None
    _lpips_model = None

    def __init__(self, metric_name: str) -> None:
        name = metric_name.lower()
        if name not in {"ssim", "fsim", "lpips"}:
            raise ValueError(f"Unsupported similarity metric: {metric_name}")
        if SimilarityComputer._device is None:
            SimilarityComputer._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if name == "lpips" and SimilarityComputer._lpips_model is None:
            model = piq.LPIPS(net_type="alex", reduction="none")
            model = model.to(SimilarityComputer._device)
            model.eval()
            SimilarityComputer._lpips_model = model

        self.name = name
        self._display_name = name.upper()

    @property
    def display_name(self) -> str:
        return self._display_name

    def compute_pair(self, path_a: Path, path_b: Path, prefer_gray: bool = True) -> float:
        arr_a = self._load(path_a, prefer_gray)
        arr_b = self._load(path_b, prefer_gray, target_size_wh=self._get_size_wh(arr_a))
        return self._compute(arr_a, arr_b)

    def _load(self, path: Path, prefer_gray: bool, target_size_wh: Optional[Tuple[int, int]] = None) -> np.ndarray:
        if self.name == "lpips":
            mode = "RGB"
        else:
            mode = "L" if prefer_gray else "RGB"
        with Image.open(path) as img:
            img = img.convert(mode)
            if target_size_wh is not None and img.size != target_size_wh:
                img = img.resize(target_size_wh, resample=Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

    @staticmethod
    def _get_size_wh(arr: np.ndarray) -> Tuple[int, int]:
        if arr.ndim == 2:
            h, w = arr.shape
        else:
            h, w = arr.shape[:2]
        return (w, h)

    def _compute(self, arr_a: np.ndarray, arr_b: np.ndarray) -> float:
        if torch is None or piq is None or SimilarityComputer._device is None:
            raise RuntimeError("PIQ metrics require both 'torch' and 'piq'.")

        tensor_a = torch.from_numpy(arr_a).float()
        tensor_b = torch.from_numpy(arr_b).float()

        if tensor_a.ndim == 2:
            tensor_a = tensor_a.unsqueeze(0).unsqueeze(0)
            tensor_b = tensor_b.unsqueeze(0).unsqueeze(0)
        else:
            tensor_a = tensor_a.permute(2, 0, 1).unsqueeze(0)
            tensor_b = tensor_b.permute(2, 0, 1).unsqueeze(0)

        tensor_a = tensor_a.to(SimilarityComputer._device)
        tensor_b = tensor_b.to(SimilarityComputer._device)

        with torch.no_grad():
            if self.name == "ssim":
                value = piq.ssim(tensor_a, tensor_b, data_range=1.0, reduction="mean")
                return float(value.item())

            if self.name == "fsim":
                chromatic = tensor_a.shape[1] == 3
                value = piq.fsim(tensor_a, tensor_b, data_range=1.0, chromatic=chromatic)
                return float(value.item())

            # LPIPS returns a distance; map to similarity in [0, 1].
            model = SimilarityComputer._lpips_model
            if model is None:
                raise RuntimeError("LPIPS model not initialised.")
            if tensor_a.shape[1] == 1:
                tensor_a = tensor_a.repeat(1, 3, 1, 1)
                tensor_b = tensor_b.repeat(1, 3, 1, 1)
            distance = model(2.0 * tensor_a - 1.0, 2.0 * tensor_b - 1.0)
            distance_val = float(distance.mean().item())
        similarity = 1.0 - distance_val
        return float(np.clip(similarity, 0.0, 1.0))
