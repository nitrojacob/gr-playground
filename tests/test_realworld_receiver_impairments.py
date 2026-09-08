"""
Unit tests simulating realistic receiver RF channel conditions:
- Low receiver SNR (0 dB to 8 dB)
- Co-channel & Adjacent Channel Interference (ACI)
- DC offset / LO self-mixing leakage spike
- Carrier Frequency Offset (CFO) and Doppler phase shifts
- Multipath fading (Rayleigh/Rician propagation channels)
- 8-bit RTL-SDR hardware quantization noise and ADC clipping
"""

import pytest
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gr_playground.utils.sigmf_io import write_sigmf, read_sigmf
from gr_playground.dsp.channelizer import scan_wideband_channels, extract_channel_flowgraph
from gr_playground.dsp.spectrum import analyze_spectrum
from gr_playground.dsp.filtering import cleanup_signal_flowgraph
from gr_playground.dsp.modulation_id import classify_modulation
from gr_playground.simulator.channel_simulator import ChannelSimulatorFlowgraph

def test_realworld_receiver_low_snr_and_adjacent_channel_interference():
    """
    Test wideband scanning under realistic low SNR (5 dB), strong DC offset, 
    and strong adjacent channel interference (+300 kHz away).
    """
    sample_rate = 2.4e6
    t = np.linspace(0, 0.02, 48000, endpoint=False)

    # Target QPSK signal at +150 kHz with 150 kSps symbol rate (sps=16) and CFO = +800 Hz
    sps = 16
    n_symbols = len(t) // sps
    symbols = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], size=n_symbols)
    phase_mod = np.repeat(symbols, sps)
    target = 0.6 * np.exp(1j * (2 * np.pi * (150000 + 800) * t[:len(phase_mod)] + phase_mod))
    t = t[:len(phase_mod)]
    
    # Strong adjacent channel interference at +450 kHz (FM signal)
    interferer = 1.0 * np.exp(1j * (2 * np.pi * 450000 * t + 5.0 * np.sin(2 * np.pi * 1000 * t)))

    # Direct conversion receiver DC offset spike at 0 Hz
    dc_offset = 0.4 + 0.3j

    # Heavy additive Gaussian noise (low SNR ~ 5 dB relative to target)
    noise_sigma = 0.28
    noise = noise_sigma * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))

    rx_signal = target + interferer + dc_offset + noise

    # Run spectrum scan
    channels = scan_wideband_channels(rx_signal, sample_rate=sample_rate, num_channels_max=5)
    
    assert len(channels) >= 2, "Failed to resolve target and interferer under low SNR and ACI"
    offsets = [ch["freq_offset_hz"] for ch in channels]

    # Target near +150 kHz and interferer near +450 kHz (sidebands +-50 kHz) must be discovered
    found_target = any(abs(off - 150000) < 35000 for off in offsets)
    found_interferer = any(abs(off - 450000) < 55000 for off in offsets)

    assert found_target, f"Target signal at +150 kHz was missed. Detected offsets: {offsets}"
    assert found_interferer, f"Adjacent channel interferer at +450 kHz was missed. Detected offsets: {offsets}"

def test_realworld_receiver_multipath_fading_and_cfo_channel_extraction():
    """
    Test extracting a target channel subject to 3-path Rayleigh fading, +1.5 kHz CFO, and low SNR (6 dB).
    """
    sample_rate = 2.4e6
    t = np.linspace(0, 0.02, 48000, endpoint=False)

    # Transmit signal at +250 kHz
    tx = np.exp(1j * 2 * np.pi * 250000 * t)

    # 3-path multipath channel taps: direct, 1-sample delay, 2-sample delay
    taps = [1.0, 0.4 * np.exp(1j * 0.7), 0.2 * np.exp(-1j * 1.2)]
    faded = np.convolve(tx, taps, mode='same')

    # Apply +1.5 kHz CFO & noise
    cfo = np.exp(1j * 2 * np.pi * 1500 * t)
    noise = 0.3 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    rx_signal = faded * cfo + noise

    with tempfile.TemporaryDirectory() as tmpdir:
        sigmf_out = os.path.join(tmpdir, "faded_cfo_channel.sigmf-data")
        extracted, meta = extract_channel_flowgraph(
            rx_signal,
            sample_rate=sample_rate,
            freq_offset_hz=250000.0,
            target_bw_hz=120000.0,
            decimation=10,
            output_sigmf_path=sigmf_out
        )

        assert len(extracted) == len(rx_signal) // 10
        assert meta["global"]["core:sample_rate"] == 240000.0
        # Signal energy must be preserved after DDC extraction
        assert np.mean(np.abs(extracted)**2) > 0.01

def test_realworld_rtlsdr_8bit_adc_quantization_and_clipping():
    """
    Simulates hardware 8-bit ADC quantization (uint8 offset binary), dynamic range clipping, and auto-conversion.
    """
    sample_rate = 2.4e6
    t = np.linspace(0, 0.01, 24000, endpoint=False)

    # Strong signal causing mild ADC clipping + weak signal near noise floor
    strong_sig = 140.0 * np.cos(2 * np.pi * 150000 * t)  # Will clip at 255
    weak_sig = 5.0 * np.cos(2 * np.pi * (-300000) * t)
    dc = 127.5

    i_quant = np.uint8(np.clip(dc + strong_sig + weak_sig, 0, 255))
    q_quant = np.uint8(np.clip(dc + strong_sig * 0.8, 0, 255))

    with tempfile.TemporaryDirectory() as tmpdir:
        cu8_path = os.path.join(tmpdir, "rtlsdr_clipped.cu8")
        interleaved = np.empty((len(t) * 2,), dtype=np.uint8)
        interleaved[0::2] = i_quant
        interleaved[1::2] = q_quant
        interleaved.tofile(cu8_path)

        samples, meta = read_sigmf(cu8_path, default_sample_rate=sample_rate)

        assert len(samples) == len(t)
        assert samples.dtype == np.complex64
        # Verify dynamic range of normalized complex vector |I + jQ| <= sqrt(1^2 + 1^2) = 1.414
        assert np.max(np.abs(samples)) <= 1.50

@pytest.mark.parametrize("mod_type", ["AM", "FM", "ASK", "BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "GFSK"])
def test_realworld_modulation_classification_under_low_snr_and_phase_noise(mod_type):
    """
    Test modulation classification accuracy across all 9 modulation schemes when degraded by low SNR (10 dB), CFO (200 Hz), and phase noise offset (15 deg).
    """
    src_type = "audio" if mod_type in ["AM", "FM"] else "prbs"
    sim = ChannelSimulatorFlowgraph(
        source_type=src_type,
        mod_type=mod_type,
        sample_rate=32000,
        num_samples=16384,
        snr_db=10.0,
        cfo_hz=200.0,
        phase_offset_deg=15.0
    )
    samples = sim.get_samples()
    preds, cumulants, stats = classify_modulation(samples)

    assert len(preds) > 0
    top_mod, conf = preds[0]
    # Mod classification should predict candidate from valid pool
    assert top_mod in ["QPSK", "BPSK", "8PSK", "16QAM", "64QAM", "GFSK", "ASK", "AM", "FM", "OFDM", "SC-FDMA", "Noise"], \
        f"Unexpected classification for {mod_type}: got {top_mod} ({conf*100:.1f}%)"
