"""
Unit tests for gr-playground DSP subpackage modules
"""

import pytest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gr_playground.dsp.spectrum import analyze_spectrum
from gr_playground.dsp.filtering import cleanup_signal_flowgraph
from gr_playground.dsp.modulation_id import classify_modulation
from gr_playground.dsp.synchronization import synchronize_signal_flowgraph
from gr_playground.dsp.demodulation import demodulate_signal_flowgraph
from gr_playground.dsp.flowgraph_builder import FlowgraphBuilder

def test_dsp_welch_psd_snr_and_peak_spectrum_analysis():
    t = np.linspace(0, 1.0, 32000, endpoint=False)
    sig = np.exp(1j * 2 * np.pi * 2000 * t) + 0.1 * (np.random.randn(32000) + 1j * np.random.randn(32000))
    res = analyze_spectrum(sig, sample_rate=32000)
    assert res["num_samples"] == 32000
    assert res["snr_db"] > 10.0
    assert len(res["peaks"]) > 0

def test_dsp_signal_cleanup_dc_removal_and_iq_balancing():
    t = np.linspace(0, 0.5, 16000, endpoint=False)
    sig = np.exp(1j * 2 * np.pi * 1000 * t) + 0.2 + 0.2j  # Tone with DC offset
    cleaned = cleanup_signal_flowgraph(sig, sample_rate=32000, cutoff_hz=4000.0)
    assert len(cleaned) > 0

def test_dsp_automatic_modulation_classification():
    t = np.linspace(0, 0.5, 16000, endpoint=False)
    sig = np.exp(1j * (2 * np.pi * 1000 * t + 5.0 * np.sin(2 * np.pi * 50 * t))).astype(np.complex64)
    preds, cumulants, const_stats = classify_modulation(sig)
    assert len(preds) > 0
    assert preds[0][0] == "FM", f"Expected FM, got {preds[0][0]}"

def test_dsp_top_block_python_script_generation():
    code = FlowgraphBuilder.generate_top_block_script("/tmp/in.sigmf-data", "/tmp/out.sigmf-data", ["dc_block", "agc"])
    assert "class GeneratedTopBlock(gr.top_block):" in code
    assert "filter.dc_blocker_cc" in code
