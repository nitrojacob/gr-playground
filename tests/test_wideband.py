"""
Unit tests for wideband spectrum scanning, channel extraction, RTL-SDR cu8 fallback auto-detection, and flowgraph builder.
"""

import pytest
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gr_playground.utils.sigmf_io import write_sigmf, read_sigmf
from gr_playground.dsp.channelizer import scan_wideband_channels, extract_channel_flowgraph
from gr_playground.dsp.flowgraph_builder import FlowgraphBuilder

def test_rtlsdr_cu8_raw_binary_auto_detection_and_sigmf_conversion():
    """
    Test auto-detection and conversion of raw RTL-SDR uint8 offset binary (.cu8) data in read_sigmf.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cu8_path = os.path.join(tmpdir, "test_capture.cu8")
        
        # Create synthetic uint8 I/Q data (127.5 is 0)
        # Tone at 100 kHz with samp_rate = 2.4 MHz
        t = np.linspace(0, 0.01, 24000, endpoint=False)
        i_synth = np.uint8(np.clip(127.5 + 100 * np.cos(2 * np.pi * 100000 * t), 0, 255))
        q_synth = np.uint8(np.clip(127.5 + 100 * np.sin(2 * np.pi * 100000 * t), 0, 255))
        
        raw_interleaved = np.empty((24000 * 2,), dtype=np.uint8)
        raw_interleaved[0::2] = i_synth
        raw_interleaved[1::2] = q_synth
        raw_interleaved.tofile(cu8_path)

        # Read back using read_sigmf fallback
        samples, meta = read_sigmf(cu8_path, default_sample_rate=2.4e6)

        assert len(samples) == 24000
        assert samples.dtype == np.complex64
        assert meta["global"]["core:sample_rate"] == 2.4e6
        # Signal power should be non-zero and mean should be near 0 DC
        assert np.abs(np.mean(samples)) < 0.1
        assert np.mean(np.abs(samples)**2) > 0.05

def test_wideband_spectrum_scanning_and_channel_offset_detection():
    """
    Test detecting multiple tones across a 2.4 MHz wideband spectrum.
    """
    sample_rate = 2.4e6
    t = np.linspace(0, 0.01, 24000, endpoint=False)
    
    # Tone 1 at +200 kHz, Tone 2 at -400 kHz
    sig1 = 0.8 * np.exp(1j * 2 * np.pi * 200000 * t)
    sig2 = 0.5 * np.exp(1j * 2 * np.pi * (-400000) * t)
    noise = 0.05 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    wideband_sig = sig1 + sig2 + noise

    channels = scan_wideband_channels(wideband_sig, sample_rate=sample_rate, num_channels_max=5)
    
    assert len(channels) >= 2
    offsets = [ch["freq_offset_hz"] for ch in channels]
    
    # Check that offsets around -400 kHz and +200 kHz were detected
    found_pos = any(abs(off - 200000) < 30000 for off in offsets)
    found_neg = any(abs(off - (-400000)) < 30000 for off in offsets)
    assert found_pos
    assert found_neg

def test_digital_downconverter_channel_extraction_to_sigmf():
    """
    Test translating a +200 kHz tone to 0 Hz baseband and decimating.
    """
    sample_rate = 2.4e6
    t = np.linspace(0, 0.01, 24000, endpoint=False)
    sig_200k = np.exp(1j * 2 * np.pi * 200000 * t)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sigmf_out = os.path.join(tmpdir, "extracted_200k.sigmf-data")
        extracted, meta = extract_channel_flowgraph(
            sig_200k,
            sample_rate=sample_rate,
            freq_offset_hz=200000.0,
            target_bw_hz=100000.0,
            decimation=10,
            output_sigmf_path=sigmf_out
        )

        assert len(extracted) == 24000 // 10
        assert meta["global"]["core:sample_rate"] == 240000.0
        assert os.path.exists(sigmf_out)

def test_wideband_signal_bandwidth_extraction_exceeds_50khz():
    """
    Testcase for Wideband Signals (e.g. WFM / 180-200 kHz bandwidth):
    Verifies that wideband signals with RF bandwidth much larger than 50 kHz are correctly
    detected by scan_wideband_channels, extracted by DDC without clipping, and analyzed
    with their true occupied bandwidth (>= 140 kHz).
    """
    from gr_playground.dsp.spectrum import analyze_spectrum

    sample_rate = 2.4e6
    num_samples = int(sample_rate * 0.05)  # 50 ms capture
    t = np.linspace(0, 0.05, num_samples, endpoint=False)

    # Wideband FM signal (75 kHz frequency deviation, 15 kHz modulating tone -> Carson's BW ~ 180 kHz)
    freq_dev = 75000.0
    audio_freq = 15000.0
    center_offset = 200000.0  # +200 kHz offset
    modulator = np.sin(2 * np.pi * audio_freq * t)
    phase = 2 * np.pi * (center_offset * t + (freq_dev / audio_freq) * (1.0 - np.cos(2 * np.pi * audio_freq * t)))
    wfm_sig = 0.8 * np.exp(1j * phase)
    noise = 0.02 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    wideband_sig = (wfm_sig + noise).astype(np.complex64)

    # 1. Scan wideband spectrum for candidate channels
    channels = scan_wideband_channels(wideband_sig, sample_rate=sample_rate, num_channels_max=5)
    assert len(channels) >= 1, "Failed to detect wideband signal channel"

    target_ch = channels[0]
    detected_bw = target_ch["bandwidth_hz"]
    print(f"Detected Channel Bandwidth: {detected_bw / 1e3:.1f} kHz")

    # 2. Extract narrowband/wideband channel with adaptive DDC
    target_bw = max(detected_bw, 180000.0)
    extracted, meta = extract_channel_flowgraph(
        wideband_sig,
        sample_rate=sample_rate,
        freq_offset_hz=target_ch["freq_offset_hz"],
        target_bw_hz=target_bw,
        decimation=None  # Adaptive decimation
    )

    out_rate = meta["global"]["core:sample_rate"]

    # 3. Analyze spectrum of extracted channel
    spec_metrics = analyze_spectrum(extracted, sample_rate=out_rate)
    extracted_bw = spec_metrics["occupied_bw_hz"]
    print(f"Extracted Signal Occupied Bandwidth: {extracted_bw / 1e3:.1f} kHz")

    # Assert that extracted occupied bandwidth is larger than 140 kHz
    assert extracted_bw >= 140000.0, f"Extracted occupied bandwidth ({extracted_bw/1e3:.1f} kHz) was clipped below 140 kHz!"

