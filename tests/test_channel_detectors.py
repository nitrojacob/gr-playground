"""
Unit Tests for Modular Channel Identification Architecture & Strategy Factory Pattern.
Tests:
1. Factory function (`get_channel_detector`) mode dispatching and fallback.
2. PPDHeuristicChannelDetector execution.
3. CFARHeuristicChannelDetector execution.
4. Interface contract compliance and state reset triggers.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gr_playground.dsp.channel_detection import (
    BaseChannelDetector,
    CFARHeuristicChannelDetector,
    PPDHeuristicChannelDetector,
    get_channel_detector,
)
from gr_playground.dsp.channelizer import scan_wideband_channels


def test_channel_detector_factory_dispatch():
    """Verify get_channel_detector returns correct instances based on mode."""
    cfar_detector = get_channel_detector("cfar_heuristic")
    assert isinstance(cfar_detector, CFARHeuristicChannelDetector)

    ppd_detector = get_channel_detector("ppd_heuristic")
    assert isinstance(ppd_detector, PPDHeuristicChannelDetector)

    # Alias check
    ppd_alias = get_channel_detector("point_peak")
    assert isinstance(ppd_alias, PPDHeuristicChannelDetector)

    # Fallback default check for unknown mode
    fallback_detector = get_channel_detector("unknown_future_mode")
    assert isinstance(fallback_detector, CFARHeuristicChannelDetector)


def test_channel_detector_interface_and_state_reset():
    """Verify BaseChannelDetector inheritance and reset_state interface contract."""
    class DummyCustomDetector(BaseChannelDetector):
        def __init__(self):
            self.reset_count = 0

        def detect_channels(self, iq_data, sample_rate, **kwargs):
            return [{"freq_offset_hz": 1000.0, "power_db": -10.0, "local_snr_db": 15.0, "bandwidth_hz": 50000.0, "channel_type": "Narrowband Signal"}]

        def reset_state(self):
            self.reset_count += 1

    custom_det = DummyCustomDetector()
    assert isinstance(custom_det, BaseChannelDetector)
    custom_det.reset_state()
    assert custom_det.reset_count == 1

    # Verify custom_det works with scan_wideband_channels
    iq = np.random.randn(4096) + 1j * np.random.randn(4096)
    res = scan_wideband_channels(iq, sample_rate=1e6, channel_detector=custom_det)
    assert len(res) == 1
    assert res[0]["freq_offset_hz"] == 1000.0


def test_cfar_and_ppd_detection_on_synthetic_signal():
    """Verify both CFAR and PPD detectors run on synthetic BPSK signal."""
    fs = 2.4e6
    num_samples = 48000
    t = np.arange(num_samples) / fs

    # Signal at +200 kHz
    bpsk_symbols = np.random.choice([1, -1], size=num_samples // 16)
    bpsk_samples = np.repeat(bpsk_symbols, 16)
    target_sig = 0.2 * bpsk_samples * np.exp(1j * 2 * np.pi * 200000.0 * t)
    noise = 0.01 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    rx_signal = (target_sig + noise).astype(np.complex64)

    # 1. Test CFAR detector via scan_wideband_channels
    channels_cfar = scan_wideband_channels(rx_signal, sample_rate=fs, detection_mode="cfar_heuristic")
    assert len(channels_cfar) >= 1
    assert any(abs(c["freq_offset_hz"] - 200000.0) < 30000.0 for c in channels_cfar)

    # 2. Test PPD detector via scan_wideband_channels
    channels_ppd = scan_wideband_channels(rx_signal, sample_rate=fs, detection_mode="ppd_heuristic")
    assert len(channels_ppd) >= 1
    assert any(abs(c["freq_offset_hz"] - 200000.0) < 30000.0 for c in channels_ppd)
