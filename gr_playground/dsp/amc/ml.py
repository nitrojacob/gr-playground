"""
Machine Learning AMC Classifier using HistGradientBoostingClassifier (Scikit-Learn).
Uses Stage 1 frame feature classifier and Stage 2 sequence meta-learner classifier.
"""

import os
import joblib
import numpy as np
import logging

from gr_playground.dsp.amc.base import BaseAMCClassifier
from gr_playground.dsp.amc.heuristic import HeuristicAMCClassifier
from gr_playground.dsp.amc.common import slice_sequence, squelch_check
from gr_playground.dsp.amc.features import (
    extract_frame_features,
    extract_sequence_features
)

logger = logging.getLogger(__name__)

MODULATION_CLASSES = [
    "AM", "FM", "BPSK", "GFSK", "QPSK", "8PSK",
    "16QAM", "64QAM", "256QAM", "ASK", "16APSK", "32APSK",
    "OQPSK", "OFDM", "SC-FDMA", "Noise"
]

class MLAMCClassifier(BaseAMCClassifier):
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models"
            )

        self.stage1_path = os.path.join(model_dir, "amc_stage1_model.pkl")
        self.stage2_path = os.path.join(model_dir, "amc_stage2_model.pkl")
        self.fallback_heuristic = None
        self.stage1_model = None
        self.stage2_model = None

        self._load_models()

    def _load_models(self):
        try:
            if os.path.exists(self.stage1_path) and os.path.exists(self.stage2_path):
                self.stage1_model = joblib.load(self.stage1_path)
                self.stage2_model = joblib.load(self.stage2_path)
            else:
                logger.warning(
                    f"ML model artifacts not found at {self.stage1_path} / {self.stage2_path}. "
                    "Falling back to HeuristicAMCClassifier."
                )
                self.fallback_heuristic = HeuristicAMCClassifier()
        except Exception as e:
            logger.warning(f"Failed to load ML models: {e}. Falling back to HeuristicAMCClassifier.")
            self.fallback_heuristic = HeuristicAMCClassifier()

    def classify(self, samples, sample_rate: float = 32000.0) -> dict:
        """
        Classify complex IQ sample sequence into 16 modulation class probabilities using ML pipeline.
        """
        if self.fallback_heuristic is not None:
            return self.fallback_heuristic.classify(samples, sample_rate)

        y = np.asarray(samples, dtype=np.complex64)
        if len(y) < 16:
            res = {c: 0.0 for c in MODULATION_CLASSES}
            res["Noise"] = 1.0
            return res

        # Squelch noise check for short sequence
        if squelch_check(y):
            res = {c: 0.0 for c in MODULATION_CLASSES}
            res["Noise"] = 1.0
            return res

        # 1. Slice continuous stream into 2048-sample frames
        frames = slice_sequence(y, sample_rate=sample_rate, frame_size=2048, step_size=1024)

        # 2. Extract 26 frame features for each frame
        frame_feats_list = []
        snrs_list = []
        for frame in frames:
            feat_vec = extract_frame_features(frame)
            frame_feats_list.append(feat_vec)
            snrs_list.append(feat_vec[24]) # snr_m2m4_db is index 24

        X_frames = np.vstack(frame_feats_list) # Shape (num_frames, 26)

        # 3. Stage 1 Frame-level probability prediction
        try:
            frame_probs = self.stage1_model.predict_proba(X_frames)
            # Ensure shape is (num_frames, 16)
            classes_in_model = getattr(self.stage1_model, "classes_", list(range(len(MODULATION_CLASSES))))
            full_frame_probs = np.zeros((len(frames), len(MODULATION_CLASSES)), dtype=np.float32)
            for idx, c_idx in enumerate(classes_in_model):
                if c_idx < len(MODULATION_CLASSES):
                    full_frame_probs[:, c_idx] = frame_probs[:, idx]
        except Exception as e:
            logger.warning(f"Stage 1 prediction error: {e}. Falling back to Heuristic.")
            fallback = HeuristicAMCClassifier()
            return fallback.classify(samples, sample_rate)

        # 4. Extract 21 sequence summary features
        seq_feats = extract_sequence_features(full_frame_probs, snrs_list=snrs_list)
        X_seq = seq_feats.reshape(1, -1) # Shape (1, 21)

        # 5. Stage 2 Meta-Learner sequence classification
        try:
            seq_probs = self.stage2_model.predict_proba(X_seq)[0]
            classes_in_s2 = getattr(self.stage2_model, "classes_", list(range(len(MODULATION_CLASSES))))
            out_probs = np.zeros(len(MODULATION_CLASSES), dtype=np.float32)
            for idx, c_idx in enumerate(classes_in_s2):
                if c_idx < len(MODULATION_CLASSES):
                    out_probs[c_idx] = seq_probs[idx]

            # Softmax normalization
            prob_sum = np.sum(out_probs)
            if prob_sum > 0:
                out_probs = out_probs / prob_sum
            else:
                out_probs[15] = 1.0

            return {MODULATION_CLASSES[i]: float(out_probs[i]) for i in range(len(MODULATION_CLASSES))}

        except Exception as e:
            logger.warning(f"Stage 2 prediction error: {e}. Falling back to average frame probabilities.")
            avg_probs = np.mean(full_frame_probs, axis=0)
            avg_probs = avg_probs / (np.sum(avg_probs) + 1e-12)
            return {MODULATION_CLASSES[i]: float(avg_probs[i]) for i in range(len(MODULATION_CLASSES))}
