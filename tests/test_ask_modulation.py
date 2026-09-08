"""
Unit tests for ASK (Amplitude Shift Keying) modulation classification and simulation.
Tests high SNR (~60 dB) and narrow bandwidth (~22 kHz) signals matching real-world 40.759 MHz capture.
"""

import numpy as np
import pytest
from gr_playground.dsp.modulation_id import classify_modulation
from gr_playground.simulator.channel_simulator import ChannelSimulatorFlowgraph

def test_ask_modulation_classification_high_snr():
    """
    Test synthesizing ASK signal with high SNR (60 dB) and matching bandwidth (22 kHz).
    Verifies that the AMC classifier correctly identifies ASK rather than misclassifying as BPSK.
    """
    sample_rate = 240000
    num_samples = 32768
    t = np.arange(num_samples) / sample_rate
    
    # Synthesize 2-ASK (OOK): binary data -> amplitude keying (0.05 / 1.0)
    bits = np.repeat(np.random.choice([0, 1], size=num_samples // 8), 8)
    amp = np.where(bits == 1, 1.0, 0.05).astype(np.complex64)
    carrier = np.exp(1j * 2 * np.pi * 5000.0 * t)
    noise = 0.001 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    ask_samples = (amp * carrier + noise).astype(np.complex64)

    assert len(ask_samples) == num_samples
    assert ask_samples.dtype == np.complex64
    
    predictions, cumulants, stats = classify_modulation(ask_samples)
    top_mod, top_conf = predictions[0]
    
    print(f"ASK Simulation Test - Top Prediction: {top_mod} ({top_conf*100:.1f}% confidence)")
    print(f"Extracted Cumulants: {cumulants}")
    print(f"Constellation Stats: {stats}")
    
    assert top_mod == "ASK", f"Expected ASK classification, but got {top_mod} ({top_conf*100:.1f}%)"
    assert top_conf > 0.40, f"Confidence too low: {top_conf}"

def test_bpsk_vs_ask_classification_distinction():
    """
    Test that BPSK (constant envelope, low amp_var) is classified as BPSK while ASK (varying envelope, high amp_var) is classified as ASK.
    """
    sample_rate = 240000
    num_samples = 32768
    t = np.arange(num_samples) / sample_rate
    
    # 1. Synthesize BPSK
    bits_bpsk = np.repeat(np.random.choice([-1, 1], size=num_samples // 8), 8)
    bpsk_phase = np.where(bits_bpsk > 0, 0.0, np.pi)
    bpsk_samples = np.exp(1j * (2 * np.pi * 5000.0 * t + bpsk_phase)).astype(np.complex64)
    
    bpsk_preds, _, bpsk_stats = classify_modulation(bpsk_samples)
    assert bpsk_preds[0][0] == "BPSK", f"Expected BPSK, got {bpsk_preds[0][0]}"
    
    # 2. Synthesize ASK
    bits_ask = np.repeat(np.random.choice([0, 1], size=num_samples // 8), 8)
    ask_amp = np.where(bits_ask > 0, 1.0, 0.05)
    ask_samples = (ask_amp * np.exp(1j * 2 * np.pi * 5000.0 * t)).astype(np.complex64)
    
    ask_preds, _, ask_stats = classify_modulation(ask_samples)
    assert ask_preds[0][0] == "ASK", f"Expected ASK, got {ask_preds[0][0]}"
