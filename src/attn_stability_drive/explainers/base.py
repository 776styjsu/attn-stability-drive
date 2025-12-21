from abc import ABC, abstractmethod

class BaseExplainer(ABC):
    @abstractmethod
    def explain(self, image, model) -> np.ndarray:
        """Return saliency map for input image."""
        pass
    
    @abstractmethod
    def save(self, output_dir: Path) -> None:
        """Save explanation artifacts."""
        pass