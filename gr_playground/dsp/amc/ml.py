"""
Machine Learning AMC Classifier using Portable ONNX Runtime.
ONNX is the de-facto format for model persistence across environments.
"""

import os
import numpy as np
import logging

from gr_playground.dsp.amc.base import BaseAMCClassifier
from gr_playground.dsp.amc.heuristic import HeuristicAMCClassifier
from gr_playground.dsp.amc.common import squelch_check
from gr_playground.dsp.amc.features import extract_frame_features
from gr_playground.dsp.amc import MODULATION_CLASSES, RADIOML_TO_PARENT_MAP

logger = logging.getLogger(__name__)

class MLAMCClassifier(BaseAMCClassifier):
    """
    ML AMC Classifier leveraging ONNX model (`amc_model.onnx`).
    Reads 26 extracted physical frame features and outputs probability distribution across all MODULATION_CLASSES.
    """
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models"
            )

        self.model_dir = model_dir
        self.onnx_path = os.path.join(model_dir, "amc_model.onnx")
        self.fallback_heuristic = None
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
                logger.warning(f"Failed to load ONNX model via onnxruntime: {e}. Falling back to HeuristicAMCClassifier.")
        
        # Fallback to HeuristicAMCClassifier if ONNX model is missing or fails to load
        logger.warning(f"ONNX model artifact not found or unreadable at {self.onnx_path}. Falling back to HeuristicAMCClassifier.")
        self.fallback_heuristic = HeuristicAMCClassifier()

    def classify(self, samples, sample_rate: float = 32000.0) -> dict:
        """
        Classify complex IQ sample sequence into modulation class probabilities using ONNX model.
        """
        from gr_playground.dsp.amc import MODULATION_CLASSES

        if self.fallback_heuristic is not None:
            return self.fallback_heuristic.classify(samples, sample_rate)

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

                ml_raw = {MODULATION_CLASSES[i]: float(probs[i]) for i in range(min(len(probs), len(MODULATION_CLASSES)))}
                
                # Aggregate granular subclass probabilities onto parent classes
                res_dict = {}
                for c, p in ml_raw.items():
                    parent = RADIOML_TO_PARENT_MAP.get(c, c)
                    res_dict[parent] = res_dict.get(parent, 0.0) + p

                for c in MODULATION_CLASSES:
                    if c not in res_dict:
                        res_dict[c] = 0.0

                tot = sum(res_dict.values())
                if tot > 0:
                    res_dict = {k: v / tot for k, v in res_dict.items()}

                # Ensemble blend with HeuristicAMCClassifier (35% ONNX ML + 65% Physical Heuristics)
                h_probs = HeuristicAMCClassifier().classify(y, sample_rate)
                blended = {}
                for c in MODULATION_CLASSES:
                    blended[c] = 0.35 * res_dict.get(c, 0.0) + 0.65 * h_probs.get(c, 0.0)

                tot_b = sum(blended.values())
                if tot_b > 0:
                    blended = {k: v / tot_b for k, v in blended.items()}
                return blended
            except Exception as e:
                logger.warning(f"ONNX inference error: {e}. Falling back to Heuristic AMC.")
                fallback = HeuristicAMCClassifier()
                return fallback.classify(samples, sample_rate)

        fallback = HeuristicAMCClassifier()
        return fallback.classify(samples, sample_rate)
