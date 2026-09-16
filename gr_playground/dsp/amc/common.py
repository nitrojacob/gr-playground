"""
Shared DSP Feature Extraction & Sequence Slicing Utilities for AMC Classifiers.
Centralizes Higher-Order Cumulants, constellation statistics, spectral metrics, and sequence slicing.
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
    
    # Focus on active signal section (exclude buffer zero padding, preserving ASK off-state symbols >= 0.005)
    mag_all = np.abs(y_in)
    max_all = np.max(mag_all) if len(mag_all) > 0 else 1.0
    active_mask = mag_all > 0.001 * max_all
    y_raw = y_in[active_mask] if np.sum(active_mask) > 10 else y_in

    mag_raw = np.abs(y_raw)
    max_mag_raw = float(np.max(mag_raw)) if len(mag_raw) > 0 else 1.0
    if max_mag_raw > 0:
        y_raw = y_raw / max_mag_raw
        mag_raw = mag_raw / max_mag_raw

    mean_mag_raw = np.mean(mag_raw)
    amp_var_raw = np.var(mag_raw) / (mean_mag_raw**2 + 1e-12) if mean_mag_raw > 0 else 0.0
    zero_ratio_raw = np.mean(mag_raw < 0.25)
    amp_kurtosis_raw = np.mean((mag_raw - mean_mag_raw)**4) / ((np.var(mag_raw) + 1e-12)**2) if np.var(mag_raw) > 0 else 0.0
    
    # Center y for phase/frequency variance
    y = y_raw - np.mean(y_raw)
    phase = np.angle(y)
    
    phase_var = np.var(phase)
    inst_freq = np.diff(np.unwrap(phase))
    freq_var = np.var(inst_freq)

    return {
        "amp_var": float(amp_var_raw),
        "phase_var": float(phase_var),
        "freq_var": float(freq_var),
        "amp_kurtosis": float(amp_kurtosis_raw),
        "zero_ratio": float(zero_ratio_raw)
    }

def squelch_check(samples):
    """
    Checks if frame is unmodulated thermal noise based on spectral flatness,
    amplitude kurtosis, and phase variance characteristics.
    """
    samples = np.asarray(samples, dtype=np.complex64)
    if len(samples) < 16:
        return True

    # Spectral flatness H = exp(mean(log(PSD))) / mean(PSD)
    nper = min(len(samples), 256)
    freqs_w, psd_w = signal.welch(samples, nperseg=nper, return_onesided=False)
    psd_valid = psd_w[psd_w > 1e-12]
    if len(psd_valid) == 0:
        return True
    spectral_flatness = float(np.exp(np.mean(np.log(psd_valid))) / (np.mean(psd_valid) + 1e-12))

    features = compute_constellation_features(samples)
    amp_kurt = features["amp_kurtosis"]
    phase_var = features["phase_var"]

    return (spectral_flatness > 0.82 and amp_kurt >= 2.65 and phase_var > 3.0)

def slice_sequence(samples, sample_rate=32000.0, frame_size=2048, step_size=1024):
    """
    Slices a continuous sample sequence into consecutive frame blocks.
    Returns list of frame sample arrays.
    """
    samples = np.asarray(samples, dtype=np.complex64)
    n_samples = len(samples)

    if n_samples <= frame_size:
        return [samples]

    frames = []
    for idx in range(0, n_samples - frame_size + 1, step_size):
        frames.append(samples[idx : idx + frame_size])

    if len(frames) == 0 and n_samples > 0:
        frames.append(samples)

    return frames
