"""
Machine Learning AMC Classifier using Portable ONNX Runtime.
ONNX is the de-facto format for model persistence across environments.
"""

import os
import numpy as np
import logging

from gr_playground.dsp.amc.base import BaseAMCClassifier
from gr_playground.dsp.amc.common import squelch_check
from gr_playground.dsp.amc.features import extract_frame_features
from gr_playground.dsp.amc import MODULATION_CLASSES, RADIOML_TO_PARENT_MAP

logger = logging.getLogger(__name__)

class MLAMCClassifier(BaseAMCClassifier):
    """
    ML AMC Classifier leveraging ONNX model (`amc_model.onnx`).
    Reads 32 extracted physical frame features and outputs probability distribution across all MODULATION_CLASSES.
    """
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models"
            )

        self.model_dir = model_dir
        self.onnx_path = os.path.join(model_dir, "amc_model.onnx")
        self.onnx_session = None

        self._load_models()

    def _load_models(self):
        # Load ONNX Model via onnxruntime
        if os.path.exists(self.onnx_path):
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(self.onnx_path)
                logger.info(f"Loaded ONNX AMC model from {self.onnx_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load ONNX model via onnxruntime: {e}")
                self.onnx_session = None
        else:
            logger.warning(f"ONNX model artifact not found or unreadable at {self.onnx_path}")
            self.onnx_session = None

    def classify(self, samples, sample_rate: float = 32000.0) -> dict:
        """
        Classify complex IQ sample sequence into modulation class probabilities using ONNX model.
        """
        from gr_playground.dsp.amc import MODULATION_CLASSES

        num_classes = len(MODULATION_CLASSES)
        y = np.asarray(samples, dtype=np.complex64)
        if len(y) < 16:
            res = {c: 0.0 for c in MODULATION_CLASSES}
            res["Noise"] = 1.0
            return res

        # ONNX Session Fast Path
        if self.onnx_session is not None:
            try:
                feat_vec = extract_frame_features(y).reshape(1, -1).astype(np.float32)
                input_name = self.onnx_session.get_inputs()[0].name
                logits = self.onnx_session.run(None, {input_name: feat_vec})[0][0]

                # Softmax
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / (np.sum(exp_logits) + 1e-12)

                ml_raw = {MODULATION_CLASSES[i]: float(probs[i]) for i in range(min(len(probs), num_classes))}
                
                # Direct 30-class probability distribution without collapsing
                tot = sum(ml_raw.values())
                if tot > 0:
                    res_dict = {k: v / tot for k, v in ml_raw.items()}
                else:
                    res_dict = ml_raw
                return res_dict
            except Exception as e:
                logger.warning(f"ONNX inference error: {e}. Returning uniform distribution.")

        # Fallback to uniform distribution across classes
        uniform_prob = float(1.0 / max(num_classes, 1))
        return {c: uniform_prob for c in MODULATION_CLASSES}
