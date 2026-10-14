"""
Automatic Modulation Classification (AMC) Module
Wrapper layer delegating to gr_playground.dsp.amc strategy package.
Provides full backward compatibility for compute_higher_order_cumulants,
compute_constellation_features, and classify_modulation.
Enforces physical DSP sanity checks between occupied bandwidth and modulation schemes.
"""

import numpy as np
from gr_playground.dsp.amc.common import (
    compute_higher_order_cumulants,
    compute_constellation_features
)
from gr_playground.dsp.amc import get_amc_classifier


def apply_physical_dsp_sanity_checks(scores_dict, bandwidth_hz=None, features=None, cumulants=None):
    """
    Applies physical DSP consistency rules between estimated occupied bandwidth and modulation schemes.
    IMPORTANT: Strictly relies on physical signal properties and never uses RF frequency band assumptions.
    """
    scores = dict(scores_dict)

    if bandwidth_hz is not None and bandwidth_hz > 0:
        # High-order modulations and multicarrier schemes require wide occupied bandwidth (>25 kHz)
        if bandwidth_hz < 25000.0:
            wideband_high_order = ["64QAM", "256QAM", "16QAM", "16APSK", "32APSK", "OFDM", "SC-FDMA"]
            for mod in wideband_high_order:
                if mod in scores:
                    scores[mod] *= 0.05  # Severe penalty on physically impossible narrow-bandwidth high-order schemes

    # Envelope and Constellation Consistency Rules
    if features is not None:
        amp_var = features.get("amp_var", 0.5)
        zero_ratio = features.get("zero_ratio", 0.0)
        c20 = cumulants.get("C20", 0.0) if cumulants is not None else 0.0

        # Physical symmetry and envelope rules
        c40 = cumulants.get("C40", 0.0) if cumulants is not None else 0.0
        c42 = cumulants.get("C42", 0.0) if cumulants is not None else 0.0

        # 1. Constant-envelope physical rule (FM, GFSK, BPSK)
        if amp_var < 0.05:
            # Multi-amplitude schemes are physically impossible with near-zero amplitude variance
            multi_amp = [
                "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
                "16APSK", "32APSK", "64APSK", "128APSK",
                "4ASK", "8ASK"
            ]
            for mod in multi_amp:
                if mod in scores:
                    scores[mod] *= 0.001

            if c20 > 0.65:
                scores["BPSK"] = scores.get("BPSK", 0.01) + 2.0
            elif c20 < 0.35:
                scores["FM"] = scores.get("FM", 0.01) + 1.5
                scores["GFSK"] = scores.get("GFSK", 0.01) + 1.5

        # 2. 2-Fold Symmetry (BPSK / ASK)
        elif c20 > 0.70:
            for mod in ["16QAM", "32QAM", "64QAM", "128QAM", "256QAM", "16APSK", "32APSK", "64APSK", "128APSK"]:
                if mod in scores:
                    scores[mod] *= 0.001
            if zero_ratio < 0.20:
                scores["BPSK"] = scores.get("BPSK", 0.01) + 2.0
            else:
                scores["ASK"] = scores.get("ASK", 0.01) + 2.0
                scores["OOK"] = scores.get("OOK", 0.01) + 2.0

        # 3. 4-Fold Symmetry (QPSK vs QAM distinction via amplitude variance)
        elif c20 < 0.20 and c40 > 0.35 and c42 > 1.20:
            if amp_var < 0.12:
                for mod in ["16QAM", "32QAM", "64QAM", "128QAM", "256QAM", "16APSK", "32APSK", "64APSK", "128APSK"]:
                    if mod in scores:
                        scores[mod] *= 0.01
                scores["QPSK"] = scores.get("QPSK", 0.01) + 2.0
            else:
                for mod in ["BPSK", "QPSK", "8PSK", "16PSK", "32PSK"]:
                    if mod in scores:
                        scores[mod] *= 0.01
                scores["16QAM"] = scores.get("16QAM", 0.01) + 2.0
                scores["64QAM"] = scores.get("64QAM", 0.01) + 1.5

        # 4. ASK / OOK physical envelope rule
        if zero_ratio > 0.25 and c20 > 0.50:
            scores["ASK"] = scores.get("ASK", 0.01) + 2.0
            scores["OOK"] = scores.get("OOK", 0.01) + 2.0

        # Thermal noise squelch check
        amp_kurt = features.get("amp_kurtosis", 3.0)
        phase_var = features.get("phase_var", 0.0)
        c40 = cumulants.get("C40", 0.0) if cumulants is not None else 0.0
        if amp_kurt >= 3.0 and abs(phase_var - (np.pi**2) / 3.0) < 0.20 and c20 < 0.20 and c40 < 0.25:
            scores["Noise"] = scores.get("Noise", 0.05) + 0.90
            for k in list(scores.keys()):
                if k != "Noise":
                    scores[k] = scores.get(k, 0.05) * 0.05

    # Renormalize probabilities to sum to 1.0
    total = sum(scores.values())
    if total > 0:
        scores = {k: float(v / total) for k, v in scores.items()}
    
    return scores


def classify_modulation(samples, mode: str = "dl", bandwidth_hz: float = None):
    """
    Multi-class automatic modulation classification.
    Delegates to the requested AMC Strategy Classifier ("dl" or "ml"). Default: "dl".
    Optionally accepts bandwidth_hz for physical DSP sanity filtering.
    Returns (predictions_list, cumulants_dict, constellation_stats_dict).
    """
    samples = np.asarray(samples, dtype=np.complex64)
    cumulants = compute_higher_order_cumulants(samples)
    features = compute_constellation_features(samples)

    classifier = get_amc_classifier(mode=mode)
    scores_dict = classifier.classify(samples)

    scores_dict = apply_physical_dsp_sanity_checks(
        scores_dict,
        bandwidth_hz=bandwidth_hz,
        features=features,
        cumulants=cumulants
    )

    predictions = [(mod, score) for mod, score in scores_dict.items()]
    predictions.sort(key=lambda x: x[1], reverse=True)

    constellation_stats = {
        "phase_var": features["phase_var"],
        "amp_kurtosis": features["amp_kurtosis"],
        "estimated_symbols": 4 if predictions[0][0] == "QPSK" else (16 if "QAM" in predictions[0][0] else (0 if predictions[0][0] == "Noise" else 2))
    }

    return predictions, cumulants, constellation_stats
