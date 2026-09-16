"""
Physical Feature Vector Extraction for Machine Learning AMC.
Extracts a 26-element frame feature vector and a 21-element sequence feature vector.
"""

import numpy as np
from scipy import signal
from gr_playground.dsp.amc.common import (
    compute_higher_order_cumulants,
    compute_constellation_features
)

FEATURE_NAMES = [
    "c20", "c21", "c40", "c42", "c63",
    "amp_var", "phase_var", "freq_var", "amp_kurtosis", "zero_ratio",
    "spectral_flatness", "spectral_kurtosis", "papr_db", "psd_asymmetry",
    "fft2_p2a", "fft4_p2a", "mean_mag", "max_mag",
    "c20_derot2", "c40_derot2", "c20_derot4", "c40_derot4",
    "inst_freq_skewness", "amp_skewness", "snr_m2m4_db", "psd_peak_ratio"
]

def extract_frame_features(samples):
    """
    Extracts a 26-element float32 feature vector from a single IQ sample frame.

    Parameters:
    - samples: 1D complex64 numpy array.

    Returns:
    - 1D numpy array of shape (26,) dtype float32.
    """
    y = np.asarray(samples, dtype=np.complex64)
    if len(y) < 10:
        return np.zeros(26, dtype=np.float32)

    # 1. DC Removal and Power Normalization
    y_dc = y - np.mean(y)
    p_avg = np.mean(np.abs(y_dc)**2)
    if p_avg > 0:
        y_norm = y_dc / np.sqrt(p_avg)
    else:
        y_norm = y_dc

    mag = np.abs(y_norm)
    mean_mag = float(np.mean(mag))
    max_mag = float(np.max(mag)) if len(mag) > 0 else 1.0

    # 2. Higher-Order Cumulants & Derotations
    cumulants = compute_higher_order_cumulants(y_norm)
    c20 = cumulants["C20"]
    c21 = cumulants["C21"]
    c40 = cumulants["C40"]
    c42 = cumulants["C42"]
    c63 = cumulants["C63"]

    # 3. Constellation & Envelope Statistics
    const_feats = compute_constellation_features(y_norm)
    amp_var = const_feats["amp_var"]
    phase_var = const_feats["phase_var"]
    freq_var = const_feats["freq_var"]
    amp_kurt = const_feats["amp_kurtosis"]
    zero_ratio = const_feats["zero_ratio"]

    # 4. Spectral Features (Welch PSD)
    nper = min(len(y_norm), 256)
    freqs_w, psd_w = signal.welch(y_norm, nperseg=nper, return_onesided=False)
    psd_valid = psd_w[psd_w > 1e-12]
    
    if len(psd_valid) > 0:
        spectral_flatness = float(np.exp(np.mean(np.log(psd_valid))) / (np.mean(psd_valid) + 1e-12))
        mean_psd = np.mean(psd_valid)
        var_psd = np.var(psd_valid)
        spectral_kurtosis = float(np.mean((psd_valid - mean_psd)**4) / ((var_psd + 1e-12)**2)) if var_psd > 0 else 0.0
    else:
        spectral_flatness = 1.0
        spectral_kurtosis = 0.0

    papr_db = float(10.0 * np.log10(max_mag**2 / (mean_mag**2 + 1e-12))) if mean_mag > 0 else 0.0

    # PSD Asymmetry (Upper vs Lower sideband power)
    mid_idx = len(psd_w) // 2
    lower_pwr = np.sum(psd_w[:mid_idx])
    upper_pwr = np.sum(psd_w[mid_idx:])
    psd_asymmetry = float(abs(upper_pwr - lower_pwr) / (upper_pwr + lower_pwr + 1e-12))

    # 5. FFT 2nd and 4th power peak-to-average ratios
    t = np.arange(len(y_norm))
    y2 = y_norm**2
    fft2 = np.abs(np.fft.fft(y2))
    freqs2 = np.fft.fftfreq(len(y2))
    w2 = 2 * np.pi * (freqs2[np.argmax(fft2)] / 2.0)
    y_derot2 = y_norm * np.exp(-1j * w2 * t)
    c20_derot2 = float(np.abs(np.mean(y_derot2**2)))
    c40_derot2 = float(np.abs(np.mean(y_derot2**4) - 3.0 * (np.mean(y_derot2**2)**2)))
    fft2_p2a = float(np.max(fft2) / (np.mean(fft2) + 1e-12))

    y4 = y_norm**4
    fft4 = np.abs(np.fft.fft(y4))
    freqs4 = np.fft.fftfreq(len(y4))
    w4 = 2 * np.pi * (freqs4[np.argmax(fft4)] / 4.0)
    y_derot4 = y_norm * np.exp(-1j * w4 * t)
    c20_derot4 = float(np.abs(np.mean(y_derot4**2)))
    c40_derot4 = float(np.abs(np.mean(y_derot4**4) - 3.0 * (np.mean(y_derot4**2)**2)))
    fft4_p2a = float(np.max(fft4) / (np.mean(fft4) + 1e-12))

    # Skewness
    phase = np.angle(y_norm)
    inst_freq = np.diff(np.unwrap(phase)) if len(phase) > 1 else np.array([0.0])
    freq_mean = np.mean(inst_freq)
    freq_std = np.std(inst_freq)
    inst_freq_skewness = float(np.mean((inst_freq - freq_mean)**3) / (freq_std**3 + 1e-12)) if freq_std > 0 else 0.0

    mag_mean = np.mean(mag)
    mag_std = np.std(mag)
    amp_skewness = float(np.mean((mag - mag_mean)**3) / (mag_std**3 + 1e-12)) if mag_std > 0 else 0.0

    # M2M4 SNR estimate
    m2 = np.mean(np.abs(y)**2)
    m4 = np.mean(np.abs(y)**4)
    denom = 2 * m2**2 - m4
    if denom > 0:
        s_pow = np.sqrt(denom)
        n_pow = m2 - s_pow
        snr_lin = s_pow / max(n_pow, 1e-6)
        snr_m2m4_db = float(10.0 * np.log10(max(snr_lin, 1e-4)))
    else:
        snr_m2m4_db = 0.0

    psd_peak_ratio = float(np.max(psd_w) / (np.median(psd_w) + 1e-12)) if len(psd_w) > 0 else 1.0

    feats = np.array([
        c20, c21, c40, c42, c63,
        amp_var, phase_var, freq_var, amp_kurt, zero_ratio,
        spectral_flatness, spectral_kurtosis, papr_db, psd_asymmetry,
        fft2_p2a, fft4_p2a, mean_mag, max_mag,
        c20_derot2, c40_derot2, c20_derot4, c40_derot4,
        inst_freq_skewness, amp_skewness, snr_m2m4_db, psd_peak_ratio
    ], dtype=np.float32)

    # Sanitize NaNs and Infs
    feats = np.nan_to_num(feats, nan=0.0, posinf=100.0, neginf=-100.0)
    return feats

def extract_sequence_features(frame_prob_matrix, snrs_list=None):
    """
    Extracts a 21-element sequence summary feature vector from frame probability matrix.

    Parameters:
    - frame_prob_matrix: 2D numpy array of shape (T, num_classes).
    - snrs_list: Optional list/array of frame SNR estimates.

    Returns:
    - 1D numpy array of shape (21,) dtype float32.
    """
    probs = np.asarray(frame_prob_matrix, dtype=np.float32)
    if probs.ndim == 1:
        probs = probs[np.newaxis, :]

    num_frames, num_classes = probs.shape

    # Truncate or pad to 16 classes
    if num_classes < 16:
        pad_width = 16 - num_classes
        probs = np.pad(probs, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
    elif num_classes > 16:
        probs = probs[:, :16]

    mean_probs = np.mean(probs, axis=0)  # Shape (16,)
    max_probs = np.max(probs, axis=0)    # Shape (16,)

    # Active duty cycle (fraction of non-noise frames)
    # Class index 15 is Noise by convention if 16 classes
    non_noise_mask = np.argmax(probs, axis=1) != 15
    active_duty_cycle = float(np.mean(non_noise_mask)) if len(non_noise_mask) > 0 else 1.0

    if snrs_list is not None and len(snrs_list) > 0:
        snrs = np.asarray(snrs_list, dtype=np.float32)
        mean_snr = float(np.mean(snrs))
        var_snr = float(np.var(snrs))
    else:
        mean_snr = 10.0
        var_snr = 0.0

    # Top class mean, top class max, active duty cycle, mean snr, var snr -> 21 elements
    top_mean = float(np.max(mean_probs))
    top_max = float(np.max(max_probs))

    seq_feats = np.concatenate([
        mean_probs,  # 16 elements
        np.array([top_mean, top_max, active_duty_cycle, mean_snr, var_snr], dtype=np.float32) # 5 elements
    ])

    return np.nan_to_num(seq_feats, nan=0.0, posinf=100.0, neginf=-100.0)
