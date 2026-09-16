"""
Automatic Modulation Classification (AMC) Module
Wrapper layer delegating to gr_playground.dsp.amc strategy package.
Provides full backward compatibility for compute_higher_order_cumulants,
compute_constellation_features, and classify_modulation.
"""

import numpy as np
from gr_playground.dsp.amc.common import (
    compute_higher_order_cumulants,
    compute_constellation_features
)
from gr_playground.dsp.amc import get_amc_classifier

def classify_modulation(samples, mode: str = "heuristic"):
    """
    Multi-class automatic modulation classification.
    Delegates to the requested AMC Strategy Classifier ("heuristic", "ml", or "dl").
    Returns (predictions_list, cumulants_dict, constellation_stats_dict).
    """
    samples = np.asarray(samples, dtype=np.complex64)
    cumulants = compute_higher_order_cumulants(samples)
    features = compute_constellation_features(samples)

    classifier = get_amc_classifier(mode=mode)
    scores_dict = classifier.classify(samples)

    predictions = [(mod, score) for mod, score in scores_dict.items()]
    predictions.sort(key=lambda x: x[1], reverse=True)

    constellation_stats = {
        "phase_var": features["phase_var"],
        "amp_kurtosis": features["amp_kurtosis"],
        "estimated_symbols": 4 if predictions[0][0] == "QPSK" else (16 if "QAM" in predictions[0][0] else (0 if predictions[0][0] == "Noise" else 2))
    }

    return predictions, cumulants, constellation_stats
