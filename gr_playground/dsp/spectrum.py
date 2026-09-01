"""
Spectral Analysis & Feature Extraction Module
Computes PSD, M2M4 SNR estimation, occupied bandwidth, peak tone detection, DC offset, and spectral flatness.
"""

import numpy as np
from scipy import signal

def estimate_snr_m2m4(samples):
    """
    Estimate SNR using the M2M4 moment method (Signal and Noise moment ratio).
    M2 = E[|y|^2], M4 = E[|y|^4]
    """
    samples = np.asarray(samples, dtype=np.complex64)
    if len(samples) < 10:
        return 0.0
    
    m2 = np.mean(np.abs(samples)**2)
    m4 = np.mean(np.abs(samples)**4)
    
    # Avoid div zero or invalid sqrt
    denom = 2 * m2**2 - m4
    if denom <= 0:
        return 1.0  # Low SNR fallback (~0 dB)
    
    s_pow = np.sqrt(denom)
    n_pow = m2 - s_pow
    if n_pow <= 0:
        return 35.0  # High SNR cap (35 dB)
    
    snr_lin = s_pow / n_pow
    return 10.0 * np.log10(max(snr_lin, 1e-4))

def estimate_occupied_bandwidth(freqs, psd, power_fraction=0.99):
    """
    Calculate the occupied bandwidth containing power_fraction (e.g. 99%) of total signal power.
    """
    total_power = np.sum(psd)
    if total_power <= 0:
        return 0.0
    
    cum_power = np.cumsum(psd) / total_power
    low_bound = (1.0 - power_fraction) / 2.0
    high_bound = 1.0 - low_bound
    
    idx_low = np.searchsorted(cum_power, low_bound)
    idx_high = np.searchsorted(cum_power, high_bound)
    
    idx_high = min(idx_high, len(freqs) - 1)
    bw = abs(freqs[idx_high] - freqs[idx_low])
    return bw

def detect_peak_tones(freqs, psd, top_n=5, min_distance_hz=500.0):
    """
    Find dominant peak frequency tones in the PSD.
    """
    if len(psd) < 3:
        return []
    
    df = abs(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    min_dist_samples = max(1, int(min_distance_hz / df))
    
    peaks_indices, properties = signal.find_peaks(psd, distance=min_dist_samples)
    if len(peaks_indices) == 0:
        peaks_indices = np.array([np.argmax(psd)])
    
    # Sort by peak power descending
    sorted_idx = peaks_indices[np.argsort(psd[peaks_indices])[::-1]][:top_n]
    
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))
    
    results = []
    for idx in sorted_idx:
        results.append({
            "freq_hz": float(freqs[idx]),
            "power_db": float(psd_db[idx])
        })
    return results

def analyze_spectrum(samples, sample_rate=32000, nperseg=1024):
    """
    Full spectral analysis pipeline. Returns metrics dictionary and peak list.
    """
    samples = np.asarray(samples, dtype=np.complex64)
    num_samples = len(samples)
    
    # 1. Welch PSD estimation
    nfft = max(nperseg, 1024)
    freqs, psd = signal.welch(samples, fs=sample_rate, nperseg=min(len(samples), nperseg), nfft=nfft, return_onesided=False)
    
    # Shift to center 0 Hz
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    
    # 2. SNR & Bandwidth
    snr_db = estimate_snr_m2m4(samples)
    occupied_bw = estimate_occupied_bandwidth(freqs, psd, power_fraction=0.99)
    
    # 3. DC Offset Level (at center bin 0 Hz)
    center_idx = len(freqs) // 2
    dc_power = psd[center_idx]
    mean_power = np.mean(psd)
    dc_offset_db = 10.0 * np.log10(max(dc_power / (mean_power + 1e-12), 1e-6))
    
    # 4. Spectral Centroid CFO estimation
    estimated_cfo_hz = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12))

    # 5. Peaks
    peaks = detect_peak_tones(freqs, psd, top_n=5)
    
    return {
        "num_samples": num_samples,
        "sample_rate": sample_rate,
        "snr_db": snr_db,
        "occupied_bw_hz": occupied_bw,
        "dc_offset_db": dc_offset_db,
        "estimated_cfo_hz": estimated_cfo_hz,
        "peaks": peaks
    }
