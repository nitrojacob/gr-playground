"""
Base Abstract Interface for Automatic Modulation Classification (AMC) Strategy Pattern.
All AMC classifiers (Heuristic, ML, Deep Learning) inherit from BaseAMCClassifier
and implement a unified `classify(samples, sample_rate)` entry point.
"""

from abc import ABC, abstractmethod

class BaseAMCClassifier(ABC):
    @abstractmethod
    def classify(self, samples, sample_rate: float = 32000.0) -> dict:
        """
        Classify raw complex IQ samples (single frame or long sequence).

        Parameters:
        - samples: 1D complex64 numpy array of IQ samples.
        - sample_rate: Sample rate in Hz.

        Returns:
        - Dictionary mapping modulation class names to probability/confidence scores.
          Example: {"QPSK": 0.85, "16QAM": 0.10, ...}
        """
        pass
