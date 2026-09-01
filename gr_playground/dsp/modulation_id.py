"""
Automatic Modulation Classification (AMC) Module
Extracts Higher-Order Cumulants (C20, C21, C40, C42, C63) and constellation features.
"""

import numpy as np

def compute_higher_order_cumulants(samples):
    y = np.asarray(samples, dtype=np.complex64)
    if len(y) < 10:
        return {"C20": 0.0, "C21": 0.0, "C40": 0.0, "C42": 0.0, "C63": 0.0}

    y = y - np.mean(y)
    p_avg = np.mean(np.abs(y)**2)
    if p_avg > 0:
        y = y / np.sqrt(p_avg)

    m20 = np.mean(y**2)
    m21 = np.mean(np.abs(y)**2)
    m40 = np.mean(y**4)
    m42 = np.mean((y**3) * np.conj(y))
    m63 = np.mean((y**3) * (np.conj(y)**3))
    
    c20 = m20
    c21 = m21
    c40 = m40 - 3.0 * (m20**2)
    c42 = m42 - (np.abs(m20)**2) - 2.0 * (m21**2)
    c63 = m63 - 9.0 * c42 * m21 - 6.0 * (m21**3)
    
    return {
        "C20": float(np.abs(c20)),
        "C21": float(c21),
        "C40": float(np.abs(c40)),
        "C42": float(np.abs(c42)),
        "C63": float(np.abs(c63))
    }

def compute_constellation_features(samples):
    y = np.asarray(samples, dtype=np.complex64)
    if len(y) < 10:
        return {"amp_var": 0.0, "phase_var": 0.0, "freq_var": 0.0, "amp_kurtosis": 0.0}
    
    y = y - np.mean(y)
    mag = np.abs(y)
    phase = np.angle(y)
    
    mean_mag = np.mean(mag)
    amp_var = np.var(mag) / (mean_mag**2 + 1e-12) if mean_mag > 0 else 0.0
    phase_var = np.var(phase)
    
    inst_freq = np.diff(np.unwrap(phase))
    freq_var = np.var(inst_freq)
    
    amp_kurtosis = np.mean((mag - mean_mag)**4) / ((np.var(mag) + 1e-12)**2) if np.var(mag) > 0 else 0.0

    return {
        "amp_var": float(amp_var),
        "phase_var": float(phase_var),
        "freq_var": float(freq_var),
        "amp_kurtosis": float(amp_kurtosis)
    }

def classify_modulation(samples):
    """
    Multi-class decision tree using higher-order cumulants and envelope features.
    """
    cumulants = compute_higher_order_cumulants(samples)
    features = compute_constellation_features(samples)
    
    c40 = cumulants["C40"]
    freq_var = features["freq_var"]

    scores = {
        "BPSK": 0.05,
        "QPSK": 0.05,
        "8PSK": 0.05,
        "16QAM": 0.05,
        "64QAM": 0.05,
        "FM": 0.05,
        "AM": 0.05,
        "GFSK": 0.05
    }

    if c40 >= 1.1:
        scores["BPSK"] += 0.8
        scores["QPSK"] += 0.4
    elif 0.35 <= c40 < 1.1:
        scores["QPSK"] += 0.8
        scores["16QAM"] += 0.5
        scores["BPSK"] += 0.3
    elif 0.15 <= c40 < 0.35:
        scores["16QAM"] += 0.8
        scores["64QAM"] += 0.5
    else:  # Analog (AM or FM)
        if freq_var > 0.8:
            scores["FM"] += 0.8
            scores["GFSK"] += 0.5
        else:
            scores["AM"] += 0.8

    total = sum(scores.values())
    predictions = [(mod, score / total) for mod, score in scores.items()]
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    constellation_stats = {
        "phase_var": features["phase_var"],
        "amp_kurtosis": features["amp_kurtosis"],
        "estimated_symbols": 4 if predictions[0][0] == "QPSK" else (16 if "QAM" in predictions[0][0] else 2)
    }
    
    return predictions, cumulants, constellation_stats
