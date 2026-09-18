"""
Automatic Modulation Classification (AMC) Package.
Provides Factory / Strategy pattern interface (`get_amc_classifier`).
"""

from gr_playground.dsp.amc.base import BaseAMCClassifier
from gr_playground.dsp.amc.heuristic import HeuristicAMCClassifier
from gr_playground.dsp.amc.ml import MLAMCClassifier
from gr_playground.dsp.amc.dl import DLAMCClassifier

def get_amc_classifier(mode: str = "ml", model_dir: str = None) -> BaseAMCClassifier:
    """
    Factory function for AMC Classifiers.

    Parameters:
    - mode: Strategy mode ("heuristic", "ml", "dl"). Default: "ml".
    - model_dir: Optional path to custom model artifact directory.

    Returns:
    - An instance conforming to BaseAMCClassifier.
    """
    mode = mode.lower()

    if mode == "dl":
        try:
            return DLAMCClassifier(model_path=model_dir)
        except Exception:
            mode = "ml"

    if mode == "ml":
        try:
            return MLAMCClassifier(model_dir=model_dir)
        except Exception:
            return HeuristicAMCClassifier()

    return HeuristicAMCClassifier()

__all__ = [
    "BaseAMCClassifier",
    "HeuristicAMCClassifier",
    "MLAMCClassifier",
    "DLAMCClassifier",
    "get_amc_classifier",
]

