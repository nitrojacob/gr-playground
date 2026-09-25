"""
Automatic Modulation Classification (AMC) Package.
Provides Factory / Strategy pattern interface (`get_amc_classifier`).
"""

MODULATION_CLASSES = [
    # Core Parent Classes (16)
    "AM", "FM", "BPSK", "GFSK", "QPSK", "8PSK",
    "16QAM", "64QAM", "256QAM", "ASK", "16APSK", "32APSK",
    "OQPSK", "OFDM", "SC-FDMA", "Noise",
    # RadioML 2018 Granular Subclasses (14)
    "OOK", "4ASK", "8ASK", "16PSK", "32PSK",
    "64APSK", "128APSK", "32QAM", "128QAM",
    "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC", "CPFSK"
]

RADIOML_24_CLASSES = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GFSK", "CPFSK"
]

RADIOML_TO_PARENT_MAP = {
    'AM-SSB-WC': 'AM', 'AM-SSB-SC': 'AM', 'AM-DSB-WC': 'AM', 'AM-DSB-SC': 'AM',
    'FM': 'FM', 'BPSK': 'BPSK', 'GFSK': 'GFSK', 'CPFSK': 'GFSK',
    'QPSK': 'QPSK', '8PSK': '8PSK', '16PSK': '8PSK', '32PSK': '8PSK',
    '16QAM': '16QAM', '32QAM': '16QAM', '64QAM': '64QAM', '128QAM': '64QAM',
    '256QAM': '256QAM', 'OOK': 'ASK', '4ASK': 'ASK', '8ASK': 'ASK',
    '16APSK': '16APSK', '32APSK': '16APSK', '64APSK': '32APSK', '128APSK': '32APSK'
}

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
    "MODULATION_CLASSES",
    "RADIOML_24_CLASSES",
    "RADIOML_TO_PARENT_MAP",
]

