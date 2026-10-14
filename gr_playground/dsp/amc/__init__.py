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

import sys
import logging

logger = logging.getLogger(__name__)

from gr_playground.dsp.amc.base import BaseAMCClassifier
from gr_playground.dsp.amc.ml import MLAMCClassifier
from gr_playground.dsp.amc.dl import DLAMCClassifier

def get_amc_classifier(mode: str = "dl", model_dir: str = None) -> BaseAMCClassifier:
    """
    Factory function for AMC Classifiers.
    Defaults to DL AMC ('dl') if weights can be loaded and required packages are found.
    Falls back to ML AMC ('ml') with an informative message if DL AMC cannot be loaded.
    Shows an error message if ML AMC fallback also fails to load.

    Parameters:
    - mode: Strategy mode ("dl", "ml", "default", "auto"). Default: "dl".
    - model_dir: Optional path to custom model artifact directory or file.

    Returns:
    - An instance conforming to BaseAMCClassifier.
    """
    mode_str = (mode or "dl").lower().strip()

    # Explicit ML mode requested
    if mode_str == "ml":
        ml_clf = MLAMCClassifier(model_dir=model_dir)
        if ml_clf.onnx_session is None:
            err_msg = (
                f"ERROR: ML AMC failed to load model weights from '{ml_clf.onnx_path}' "
                "or required packages (onnxruntime) are missing!"
            )
            logger.error(err_msg)
            sys.stderr.write(f"{err_msg}\n")
        return ml_clf

    # Default / DL mode ("dl", "default", "auto")
    # 1. Attempt to load DL AMC if required packages and weights can be found
    dl_clf = DLAMCClassifier(model_path=model_dir)
    if dl_clf.onnx_session is not None or dl_clf.torch_model is not None:
        return dl_clf

    # 2. DL AMC could not be loaded -> Fallback to ML AMC with message
    fb_msg = (
        f"Notice: DL AMC unavailable (weights not found at '{dl_clf.model_path}' or required packages missing). "
        "Falling back to ML AMC."
    )
    logger.warning(fb_msg)
    sys.stderr.write(f"{fb_msg}\n")

    # 3. Check ML AMC fallback
    ml_clf = dl_clf.fallback_classifier if dl_clf.fallback_classifier is not None else MLAMCClassifier(model_dir=model_dir)
    if ml_clf.onnx_session is None:
        err_msg = (
            "ERROR: ML AMC load also failed! Neither DL AMC nor fallback ML AMC could be initialized. "
            "Ensure model weights exist in 'gr_playground/dsp/models/' and onnxruntime is installed."
        )
        logger.error(err_msg)
        sys.stderr.write(f"{err_msg}\n")

    return ml_clf

def get_onnx_runtime_info():
    """
    Check if ONNX runtime is installed and return its availability and version.

    Returns:
        tuple (bool, str or None): (is_available, version_string)
    """
    try:
        import onnxruntime as ort
        return True, str(ort.__version__)
    except ImportError:
        return False, None

__all__ = [
    "BaseAMCClassifier",
    "MLAMCClassifier",
    "DLAMCClassifier",
    "get_amc_classifier",
    "get_onnx_runtime_info",
    "MODULATION_CLASSES",
    "RADIOML_24_CLASSES",
    "RADIOML_TO_PARENT_MAP",
]

