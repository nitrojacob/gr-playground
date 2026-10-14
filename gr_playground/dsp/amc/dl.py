"""
Deep Learning AMC Classifier with 1D ResNet directly feeding raw IQ data.
Supports ONNX Runtime and PyTorch backends.
Falls back gracefully to MLAMCClassifier when weights are missing.
"""

import os
import sys
import logging
import numpy as np

from gr_playground.dsp.amc.base import BaseAMCClassifier
from gr_playground.dsp.amc import MODULATION_CLASSES

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    class Residual1DBlock(nn.Module):
        """
        1D Residual Convolutional Block for raw IQ time-series.
        """
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv1d(
                in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False
            )
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv1d(
                out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False
            )
            self.bn2 = nn.BatchNorm1d(out_channels)

            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels)
                )
            else:
                self.shortcut = nn.Identity()

        def forward(self, x):
            residual = self.shortcut(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.relu(out + residual)
            return out

    class ResNet1D(nn.Module):
        """
        Deep 1D Residual Neural Network for Automatic Modulation Classification
        directly operating on 2-channel [I, Q] complex baseband signals.
        """
        def __init__(self, in_channels: int = 2, num_classes: int = len(MODULATION_CLASSES)):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True)
            )
            self.stage1 = Residual1DBlock(32, 32, stride=1)
            self.stage2 = Residual1DBlock(32, 64, stride=2)
            self.stage3 = Residual1DBlock(64, 128, stride=2)
            self.stage4 = Residual1DBlock(128, 128, stride=2)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            x = self.stem(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.pool(x)
            return self.classifier(x)

except ImportError:
    Residual1DBlock = None
    ResNet1D = None


class DLAMCClassifier(BaseAMCClassifier):
    """
    Deep Learning Automatic Modulation Classifier using 1D ResNet directly feeding IQ data.
    Attempts ONNX Runtime first, falls back to PyTorch, and then to MLAMCClassifier.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            models_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models"
            )
            model_path = os.path.join(models_dir, "amc_resnet1d.onnx")
        elif os.path.isdir(model_path):
            model_path = os.path.join(model_path, "amc_resnet1d.onnx")

        self.model_path = model_path
        self.onnx_session = None
        self.input_name = None
        self.torch_model = None
        self.fallback_classifier = None

        self._load_dl_model()

    def _load_dl_model(self):
        # 1. Try loading ONNX model via onnxruntime
        if os.path.exists(self.model_path):
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(self.model_path)
                self.input_name = self.onnx_session.get_inputs()[0].name
                logger.info(f"Loaded ONNX 1D ResNet DL AMC model from {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize ONNX runtime for {self.model_path}: {e}")
                self.onnx_session = None

        # 2. Try loading PyTorch weights (.pt) if available
        pt_path = os.path.splitext(self.model_path)[0] + ".pt"
        if os.path.exists(pt_path) and ResNet1D is not None:
            try:
                model = ResNet1D(in_channels=2, num_classes=len(MODULATION_CLASSES))
                state_dict = torch.load(pt_path, map_location="cpu")
                model.load_state_dict(state_dict)
                model.eval()
                self.torch_model = model
                logger.info(f"Loaded PyTorch 1D ResNet DL AMC model from {pt_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load PyTorch model weights from {pt_path}: {e}")
                self.torch_model = None

        # 3. Fallback to MLAMCClassifier with message
        fallback_msg = (
            f"Notice: DL AMC model weights not available at '{self.model_path}' "
            "(or required packages uninitialized). Falling back to ML AMC."
        )
        logger.warning(fallback_msg)
        sys.stderr.write(f"{fallback_msg}\n")
        self._setup_fallback()

    def _setup_fallback(self):
        try:
            from gr_playground.dsp.amc.ml import MLAMCClassifier
            fallback_dir = self.model_path if os.path.isdir(self.model_path) else os.path.dirname(self.model_path)
            default_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            if fallback_dir != default_models_dir and not os.path.exists(os.path.join(fallback_dir, "amc_model.onnx")):
                self.fallback_classifier = MLAMCClassifier(model_dir=fallback_dir)
            else:
                self.fallback_classifier = MLAMCClassifier()

            if self.fallback_classifier.onnx_session is None:
                err_msg = (
                    "ERROR: ML AMC fallback failed to load! Neither DL AMC nor ML AMC models could be initialized. "
                    "Ensure model weights exist in 'gr_playground/dsp/models/' and onnxruntime is installed."
                )
                logger.error(err_msg)
                sys.stderr.write(f"{err_msg}\n")
        except Exception as e:
            err_msg = f"ERROR: Failed to initialize MLAMCClassifier fallback: {e}"
            logger.error(err_msg)
            sys.stderr.write(f"{err_msg}\n")
            self.fallback_classifier = None

    @staticmethod
    def preprocess_iq(samples: np.ndarray, target_len: int = 2048) -> np.ndarray:
        """
        Preprocess raw complex IQ samples into normalized 2-channel tensor [Batch, 2, SequenceLength].
        Channel 0: I (In-phase / Real)
        Channel 1: Q (Quadrature / Imag)
        """
        y = np.asarray(samples, dtype=np.complex64)
        if len(y) > target_len:
            start_idx = (len(y) - target_len) // 2
            y = y[start_idx : start_idx + target_len]
        elif len(y) < 64:
            y = np.pad(y, (0, 64 - len(y)), mode="wrap")

        # Energy normalization (unit average power)
        p = np.mean(np.abs(y)**2)
        if p > 1e-12:
            y = y / np.sqrt(p)

        # Convert to float32 tensor of shape (1, 2, N)
        i_comp = y.real.astype(np.float32)
        q_comp = y.imag.astype(np.float32)
        iq_tensor = np.stack([i_comp, q_comp], axis=0)[np.newaxis, :, :]
        return iq_tensor

    def classify(self, samples, sample_rate: float = 32000.0) -> dict:
        """
        Classify raw complex IQ samples into modulation class probabilities
        using the 1D ResNet directly or the configured fallback strategy.
        """
        from gr_playground.dsp.amc import RADIOML_24_CLASSES, RADIOML_TO_PARENT_MAP

        num_classes = len(MODULATION_CLASSES)
        y = np.asarray(samples, dtype=np.complex64)
        if len(y) < 16:
            res = {c: 0.0 for c in MODULATION_CLASSES}
            res["Noise"] = 1.0
            return res

        def _format_probabilities(probs):
            if len(probs) == len(RADIOML_24_CLASSES):
                raw_dict = {c: 1e-7 for c in MODULATION_CLASSES}
                for i, c_name in enumerate(RADIOML_24_CLASSES):
                    raw_dict[c_name] = float(probs[i])
                    parent = RADIOML_TO_PARENT_MAP.get(c_name)
                    if parent and parent in raw_dict:
                        raw_dict[parent] += float(probs[i])
            else:
                raw_dict = {
                    MODULATION_CLASSES[i]: float(probs[i])
                    for i in range(min(len(probs), num_classes))
                }
                for c in MODULATION_CLASSES:
                    if c not in raw_dict:
                        raw_dict[c] = 0.0

            tot = sum(raw_dict.values())
            return {k: v / tot for k, v in raw_dict.items()} if tot > 0 else raw_dict

        # 1. Fast Path: ONNX Runtime
        if self.onnx_session is not None:
            try:
                iq_tensor = self.preprocess_iq(y)
                logits = self.onnx_session.run(None, {self.input_name: iq_tensor})[0][0]
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / (np.sum(exp_logits) + 1e-12)
                return _format_probabilities(probs)
            except Exception as e:
                logger.warning(f"ONNX DL AMC inference error: {e}. Trying fallback.")

        # 2. PyTorch Forward Pass
        if self.torch_model is not None:
            try:
                iq_tensor = self.preprocess_iq(y)
                t_input = torch.from_numpy(iq_tensor)
                with torch.no_grad():
                    logits = self.torch_model(t_input).cpu().numpy()[0]
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / (np.sum(exp_logits) + 1e-12)
                return _format_probabilities(probs)
            except Exception as e:
                logger.warning(f"PyTorch DL AMC inference error: {e}. Trying fallback.")

        # 3. Fallback Classifier (MLAMCClassifier)
        if self.fallback_classifier is not None:
            return self.fallback_classifier.classify(samples, sample_rate)

        # 4. Uniform distribution fallback
        uniform_prob = float(1.0 / max(num_classes, 1))
        return {c: uniform_prob for c in MODULATION_CLASSES}
