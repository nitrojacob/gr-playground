"""
Automatic Modulation Classification (AMC) Module
Extracts Higher-Order Cumulants (C20, C21, C40, C42, C63) and constellation features.
"""

import numpy as np
from scipy import signal

def compute_higher_order_cumulants(samples):
    y_in = np.asarray(samples, dtype=np.complex64)
    if len(y_in) < 10:
        return {"C20": 0.0, "C21": 0.0, "C40": 0.0, "C42": 0.0, "C63": 0.0}

    y_dc = y_in - np.mean(y_in)
    p_avg = np.mean(np.abs(y_dc)**2)
    if p_avg > 0:
        y_dc = y_dc / np.sqrt(p_avg)

    t = np.arange(len(y_dc))
    
    # 2nd-power FFT CFO derotation for BPSK / ASK
    y2 = y_dc**2
    fft2 = np.abs(np.fft.fft(y2))
    freqs2 = np.fft.fftfreq(len(y2))
    w2 = 2 * np.pi * (freqs2[np.argmax(fft2)] / 2.0)
    y_derot2 = y_dc * np.exp(-1j * w2 * t)

    # 4th-power FFT CFO derotation for QPSK / 16QAM
    y4 = y_dc**4
    fft4 = np.abs(np.fft.fft(y4))
    freqs4 = np.fft.fftfreq(len(y4))
    w4 = 2 * np.pi * (freqs4[np.argmax(fft4)] / 4.0)
    y_derot4 = y_dc * np.exp(-1j * w4 * t)

    def _calc_c(y):
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

    c_raw = _calc_c(y_dc)
    c_derot2 = _calc_c(y_derot2)
    c_derot4 = _calc_c(y_derot4)

    res = c_raw
    if c_derot2["C40"] > res["C40"]:
        res = c_derot2
    if c_derot4["C40"] > res["C40"]:
        res = c_derot4

    return res

def compute_constellation_features(samples):
    y_in = np.asarray(samples, dtype=np.complex64)
    if len(y_in) < 10:
        return {"amp_var": 0.0, "phase_var": 0.0, "freq_var": 0.0, "amp_kurtosis": 0.0, "zero_ratio": 0.0}
    
    # Focus on active signal section (exclude buffer zero padding)
    mag_all = np.abs(y_in)
    max_all = np.max(mag_all) if len(mag_all) > 0 else 1.0
    active_mask = mag_all > 0.001 * max_all
    y_raw = y_in[active_mask] if np.sum(active_mask) > 10 else y_in

    mag_raw = np.abs(y_raw)
    max_mag_raw = np.max(mag_raw) if len(mag_raw) > 0 else 1.0
    mean_mag_raw = np.mean(mag_raw)
    amp_var_raw = np.var(mag_raw) / (mean_mag_raw**2 + 1e-12) if mean_mag_raw > 0 else 0.0
    zero_ratio_raw = np.mean(mag_raw < 0.25 * max_mag_raw) if max_mag_raw > 0 else 0.0
    
    # Center y for phase/frequency variance
    y = y_raw - np.mean(y_raw)
    mag = np.abs(y)
    phase = np.angle(y)
    
    phase_var = np.var(phase)
    inst_freq = np.diff(np.unwrap(phase))
    freq_var = np.var(inst_freq)
    
    mean_mag = np.mean(mag)
    amp_kurtosis = np.mean((mag - mean_mag)**4) / ((np.var(mag) + 1e-12)**2) if np.var(mag) > 0 else 0.0

    return {
        "amp_var": float(amp_var_raw),
        "phase_var": float(phase_var),
        "freq_var": float(freq_var),
        "amp_kurtosis": float(amp_kurtosis),
        "zero_ratio": float(zero_ratio_raw)
    }

def classify_modulation(samples):
    """
    Multi-class decision tree using higher-order cumulants and envelope features.
    Includes squelch noise pre-check for unmodulated thermal noise channels.
    """
    cumulants = compute_higher_order_cumulants(samples)
    features = compute_constellation_features(samples)
    
    c40 = cumulants["C40"]
    freq_var = features["freq_var"]
    amp_kurt = features["amp_kurtosis"]
    zero_ratio = features["zero_ratio"]

    # Spectral flatness H = exp(mean(log(PSD))) / mean(PSD) (~ 0.9+ for white noise, < 0.7 for modulated signals)
    freqs_w, psd_w = signal.welch(samples, nperseg=256, return_onesided=False)
    psd_valid = psd_w[psd_w > 1e-12]
    spectral_flatness = float(np.exp(np.mean(np.log(psd_valid))) / (np.mean(psd_valid) + 1e-12)) if len(psd_valid) > 0 else 1.0

    # Squelch Check: Unmodulated Gaussian/Rayleigh thermal noise exhibits high spectral flatness (> 0.85),
    # characteristic phase variance ~ pi^2/3, and elevated amplitude kurtosis (> 3.15)
    is_noise = (spectral_flatness > 0.85 and amp_kurt >= 3.15 and abs(features["phase_var"] - (np.pi**2)/3.0) < 0.10)

    scores = {
        "BPSK": 0.05,
        "QPSK": 0.05,
        "8PSK": 0.05,
        "16QAM": 0.05,
        "64QAM": 0.05,
        "FM": 0.05,
        "AM": 0.05,
        "ASK": 0.05,
        "GFSK": 0.05,
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
    elif freq_var < 0.08 and c20 >= 0.70 and amp_kurt >= 2.8:
        # Amplitude Modulation (AM): Speech audio AM has zero frequency variance (freq_var < 0.08) and high C20
        scores["AM"] += 0.85
    elif amp_var < 0.05 and c40 < 0.20:
        # Constant Envelope schemes: FM vs GFSK
        if freq_var > 0.35:
            scores["GFSK"] += 0.85
            scores["FM"] += 0.20
        else:
            scores["FM"] += 0.85
            scores["GFSK"] += 0.20
    elif amp_var >= 0.40 and amp_kurt < 2.0:
        # Amplitude Shift Keying (ASK / OOK): high envelope variance (amp_var ~ 0.50), bimodal kurtosis < 2.0
        scores["ASK"] += 0.85
        scores["AM"] += 0.10
    elif c20 >= 0.70 and c40 >= 1.0:
        # Binary Phase Shift Keying (BPSK): C20 >= 0.70, C40 >= 1.0
        scores["BPSK"] += 0.85
        scores["QPSK"] += 0.10
    elif c20 < 0.30 and c40 >= 0.65:
        # Quadrature Phase Shift Keying (QPSK): C20 ~ 0.00, C40 ~ 0.75
        scores["QPSK"] += 0.85
        scores["16QAM"] += 0.30
    elif amp_var >= 0.05 and c40 < 0.20 and c20 < 0.30:
        # 8-PSK: Constant-ish magnitude (amp_var ~ 0.09), C40 ~ 0.05, C20 ~ 0.00
        scores["8PSK"] += 0.85
        scores["QPSK"] += 0.30
    elif 0.30 <= c40 < 0.65:
        # 16-QAM vs 64-QAM
        if c63 >= 15.0 or amp_var >= 0.185:
            scores["64QAM"] += 0.85
            scores["16QAM"] += 0.40
        else:
            scores["16QAM"] += 0.85
            scores["64QAM"] += 0.40
    else:  # Fallback
        if freq_var > 0.4:
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
        "estimated_symbols": 4 if predictions[0][0] == "QPSK" else (16 if "QAM" in predictions[0][0] else (0 if predictions[0][0] == "Noise" else 2))
    }
    
    return predictions, cumulants, constellation_stats
