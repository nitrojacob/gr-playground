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
            
            # Boost narrowband candidate scores proportionately
            if features is not None:
                amp_var = features.get("amp_var", 0.0)
                freq_var = features.get("freq_var", 0.0)
                
                if amp_var < 0.18:
                    # Constant-envelope narrowband: FM vs GFSK vs BPSK
                    if freq_var > 0.15:
                        scores["FM"] = scores.get("FM", 0.05) + 0.4
                        scores["GFSK"] = scores.get("GFSK", 0.05) + 0.3
                    else:
                        scores["BPSK"] = scores.get("BPSK", 0.05) + 0.4
                        scores["QPSK"] = scores.get("QPSK", 0.05) + 0.2
                else:
                    # Variable amplitude narrowband: AM vs ASK
                    scores["AM"] = scores.get("AM", 0.05) + 0.4
                    scores["ASK"] = scores.get("ASK", 0.05) + 0.3

    # Renormalize probabilities to sum to 1.0
    total = sum(scores.values())
    if total > 0:
        scores = {k: float(v / total) for k, v in scores.items()}
    
    return scores


def classify_modulation(samples, mode: str = "ml", bandwidth_hz: float = None):
    """
    Multi-class automatic modulation classification.
    Delegates to the requested AMC Strategy Classifier ("heuristic", "ml", or "dl"). Default: "ml".
    Optionally accepts bandwidth_hz for physical DSP sanity filtering.
    Returns (predictions_list, cumulants_dict, constellation_stats_dict).
    """
    samples = np.asarray(samples, dtype=np.complex64)
    cumulants = compute_higher_order_cumulants(samples)
    features = compute_constellation_features(samples)

    classifier = get_amc_classifier(mode=mode)
    scores_dict = classifier.classify(samples)

    if bandwidth_hz is not None:
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
