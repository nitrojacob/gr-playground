"""
Rule-Based Heuristic AMC Classifier.
Implements decision tree rules using Higher-Order Cumulants and constellation features.
Inherits from BaseAMCClassifier.
"""

import numpy as np
from scipy import signal
from gr_playground.dsp.amc.base import BaseAMCClassifier
from gr_playground.dsp.amc.common import (
    compute_higher_order_cumulants,
    compute_constellation_features,
    squelch_check,
    slice_sequence
)

class HeuristicAMCClassifier(BaseAMCClassifier):
    """
    Multi-class decision tree using higher-order cumulants and envelope features.
    Handles single frames and long sequence streams cleanly.
    """
    def _classify_single_frame(self, samples):
        samples = np.asarray(samples, dtype=np.complex64)
        p_avg = np.mean(np.abs(samples)**2)
        if p_avg > 0:
            samples = samples / np.sqrt(p_avg)

        cumulants = compute_higher_order_cumulants(samples)
        features = compute_constellation_features(samples)
        
        c40 = cumulants["C40"]
        freq_var = features["freq_var"]
        amp_kurt = features["amp_kurtosis"]
        zero_ratio = features["zero_ratio"]

        # Spectral flatness
        nper = min(len(samples), 256)
        freqs_w, psd_w = signal.welch(samples, nperseg=nper, return_onesided=False)
        psd_valid = psd_w[psd_w > 1e-12]
        spectral_flatness = float(np.exp(np.mean(np.log(psd_valid))) / (np.mean(psd_valid) + 1e-12)) if len(psd_valid) > 0 else 1.0

        is_noise = squelch_check(samples)

        scores = {
            "AM": 0.05,
            "FM": 0.05,
            "BPSK": 0.05,
            "GFSK": 0.05,
            "QPSK": 0.05,
            "8PSK": 0.05,
            "16QAM": 0.05,
            "64QAM": 0.05,
            "256QAM": 0.05,
            "ASK": 0.05,
            "16APSK": 0.05,
            "32APSK": 0.05,
            "OQPSK": 0.05,
            "OFDM": 0.05,
            "SC-FDMA": 0.05,
            "Noise": 0.05
        }

        amp_var = features["amp_var"]
        c20 = cumulants["C20"]
        c63 = cumulants["C63"]
        if is_noise:
            scores["Noise"] += 0.85
        elif spectral_flatness < 0.80 and 2.5 <= amp_kurt <= 3.4 and c40 < 0.40 and amp_var > 0.05 and 0.20 < freq_var <= 0.35:
            # Multicarrier signals (OFDM / SC-FDMA)
            if amp_kurt > 2.85:
                scores["OFDM"] += 0.85
                scores["SC-FDMA"] += 0.30
            else:
                scores["SC-FDMA"] += 0.85
                scores["OFDM"] += 0.30
        elif freq_var < 0.08 and c20 >= 0.60 and amp_kurt >= 2.5:
            # Amplitude Modulation (AM)
            scores["AM"] += 0.85
        elif 0.05 <= amp_var < 0.15 and c40 < 0.25 and c20 < 0.30:
            # 8-PSK
            scores["8PSK"] += 0.85
            scores["QPSK"] += 0.30
        elif amp_var < 0.20 and c40 < 0.40:
            # Constant Envelope schemes: FM vs GFSK
            if freq_var > 0.35:
                scores["GFSK"] += 0.85
                scores["FM"] += 0.20
            else:
                scores["FM"] += 0.85
                scores["GFSK"] += 0.20
        elif c20 >= 0.60 and c40 >= 0.70:
            # BPSK vs ASK
            if amp_var >= 0.18:
                scores["ASK"] += 0.85
                scores["AM"] += 0.10
            else:
                scores["BPSK"] += 0.85
                scores["QPSK"] += 0.10
        elif amp_var >= 0.25 and amp_kurt < 2.2 and (zero_ratio >= 0.35 or c20 >= 0.50):
            # Amplitude Shift Keying (ASK / OOK)
            scores["ASK"] += 0.85
            scores["AM"] += 0.10
        elif c20 < 0.35 and c40 >= 0.40:
            # Digital Quadrature Modulations: QPSK vs 16-QAM vs 64-QAM
            if amp_var < 0.10:
                scores["QPSK"] += 0.85
                scores["16QAM"] += 0.30
            elif amp_var < 0.18:
                scores["16QAM"] += 0.85
                scores["64QAM"] += 0.30
            else:
                scores["64QAM"] += 0.85
                scores["16QAM"] += 0.30
        else:  # Fallback
            if freq_var > 0.10:
                scores["FM"] += 0.85
            else:
                scores["QPSK"] += 0.85

        # Normalize to sum to 1.0
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: float(v / total_score) for k, v in scores.items()}

        return scores

    def classify(self, samples, sample_rate: float = 32000.0) -> dict:
        """
        Universal Entry Point for Heuristic AMC.
        Classifies input IQ sample array using cumulant & constellation decision tree.
        """
        samples = np.asarray(samples, dtype=np.complex64)
        if len(samples) < 10:
            return {"Noise": 1.0}

        return self._classify_single_frame(samples)
