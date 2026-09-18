"""
Unit tests simulating DSP analysis flaws (M2M4 SNR collapse, FIR filter bandwidth masking,
uniform phase rotation artifact, and noise squelch misclassification) and verifying fixes.
"""

import pytest
import os
import sys
import numpy as np
from scipy import signal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gr_playground.dsp.spectrum import estimate_snr_m2m4, estimate_snr_psd, analyze_spectrum, estimate_occupied_bandwidth
from gr_playground.dsp.modulation_id import compute_higher_order_cumulants, compute_constellation_features, classify_modulation

def test_m2m4_snr_collapse_on_constant_envelope_and_noise():
    """
    Test 1: Simulates constant-envelope FM and AGC-normalized noise datasets where M2M4
    collapses to 1.0 dB fallback, while PSD Peak-to-Noise Floor SNR estimator computes true SNR.
    """
    sample_rate = 240000.0
    t = np.linspace(0, 0.1, int(sample_rate * 0.1), endpoint=False)
    
    # 1. Noise dataset (where 2*M2^2 - M4 <= 0 causing M2M4 collapse to 1.0 dB fallback)
    noise = (np.random.randn(int(sample_rate * 0.1)) + 1j * np.random.randn(int(sample_rate * 0.1))).astype(np.complex64)
    snr_m2m4 = estimate_snr_m2m4(noise)

    # 2. FM signal + noise (20 dB SNR)
    fm_mod = np.exp(1j * (2 * np.pi * 5000 * t + 2.0 * np.sin(2 * np.pi * 1000 * t)))
    sig_20db = (fm_mod + 0.1 * noise[:len(t)]).astype(np.complex64)

    # PSD estimation
    nfft = 1024
    freqs, psd = signal.welch(sig_20db, fs=sample_rate, nperseg=nfft, nfft=nfft, return_onesided=False)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    snr_psd = estimate_snr_psd(freqs, psd)

    # Assert M2M4 failure mode vs robust PSD recovery
    assert snr_m2m4 <= 1.05, "Expected M2M4 to hit fallback <= 1.0 dB on noise"
    assert snr_psd > 10.0, f"Expected PSD SNR > 10 dB for 20 dB FM signal, got {snr_psd:.2f} dB"

def test_fir_filter_bandwidth_masking():
    """
    Test 2: Demonstrates how lowpass FIR filtering constrains the occupied bandwidth integral
    to the filter cutoff (~120 kHz) regardless of true underlying signal.
    """
    sample_rate = 240000.0
    
    # Generate wideband noise
    noise = (np.random.randn(24000) + 1j * np.random.randn(24000)).astype(np.complex64)
    
    # Filter with 60 kHz cutoff (120 kHz total passband)
    taps = signal.firwin(101, 60000.0 / (sample_rate / 2.0), window='hamming')
    filtered_noise = signal.lfilter(taps, 1.0, noise).astype(np.complex64)

    # PSD of filtered noise
    freqs, psd = signal.welch(filtered_noise, fs=sample_rate, nperseg=1024, return_onesided=False)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)

    bw_filtered = estimate_occupied_bandwidth(freqs, psd, power_fraction=0.99)

    # Filtering forces 99% power bandwidth to be near 118-120 kHz
    assert 110000.0 <= bw_filtered <= 122000.0, f"Expected occupied BW near 118-120 kHz, got {bw_filtered:.2f} Hz"

def test_uniform_phase_rotation_cfo_artifact():
    """
    Test 3: Demonstrates that uncompensated carrier frequency offset (CFO) creates
    uniform phase noise over [-pi, pi], yielding phase variance near theoretical pi^2/3 ~ 3.29 rad^2.
    """
    sample_rate = 240000.0
    t = np.linspace(0, 0.1, 24000, endpoint=False)
    
    # Unsynchronized tone with 500 Hz CFO
    cfo_sig = np.exp(1j * 2 * np.pi * 500 * t).astype(np.complex64)
    
    features = compute_constellation_features(cfo_sig)
    phase_var = features["phase_var"]

    # Theoretical uniform phase variance pi^2 / 3 = 3.28986 rad^2
    expected_var = (np.pi**2) / 3.0
    assert abs(phase_var - expected_var) < 0.05, f"Expected phase variance near {expected_var:.4f}, got {phase_var:.4f}"

def test_noise_squelch_precheck_in_modulation_id():
    """
    Test 4: Verifies that pure Gaussian noise or low-SNR dataset is correctly identified
    as 'Noise' / low-SNR channel instead of being misclassified as FM or GFSK.
    """
    # Pure thermal Gaussian noise
    noise_only = (np.random.randn(24000) + 1j * np.random.randn(24000)).astype(np.complex64)
    
    predictions, cumulants, constellation_stats = classify_modulation(noise_only)
    
    top_pred = predictions[0][0]
    top_prob = predictions[0][1]

    assert top_pred in ["Noise", "Low SNR Noise"], f"Expected top prediction to be Noise, got {top_pred} ({top_prob*100:.1f}%)"


def test_bandwidth_modulation_physical_sanity_check():
    """
    Test 5: Verifies that physical DSP sanity checks prevent false positive 64QAM/16QAM
    classifications on narrow-bandwidth (<25 kHz) signal slices.
    """
    # Create narrow 15 kHz slice of complex signal
    sample_rate = 240000.0
    t = np.linspace(0, 0.1, 24000, endpoint=False)
    sig = np.exp(1j * (2 * np.pi * 5000 * t + 1.5 * np.sin(2 * np.pi * 1000 * t))).astype(np.complex64)
    noise = 0.05 * (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig)))
    narrow_slice = sig + noise

    # Run classification passing physical bandwidth = 14900.0 Hz
    preds, _, _ = classify_modulation(narrow_slice, bandwidth_hz=14900.0)

    top_pred = preds[0][0]
    wideband_high_order = ["64QAM", "256QAM", "16QAM", "OFDM", "SC-FDMA"]
    assert top_pred not in wideband_high_order, f"Physical sanity check failed! Narrow 14.9 kHz signal classified as {top_pred}"

