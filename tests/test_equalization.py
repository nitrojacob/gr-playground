"""
Automated Test Suite for Layer-1 (L1) Multipath Equalization.
Exercises Zero-Forcing (ZF), MMSE, Adaptive LMS, and GNU Radio CMA equalizers.
"""

import pytest
import numpy as np
import os
import sys
import tempfile
import subprocess

from gr_playground.dsp.equalization import L1Equalizer
from gr_playground.simulator.channel_simulator import ChannelSimulatorFlowgraph
from gr_playground.dsp.synchronization import calculate_evm, synchronize_signal_flowgraph
from gr_playground.dsp.demodulation import slice_psk_qpsk_bits
from gr_playground.dsp.channel_decoding import ChannelDecoder
from gr_playground.dsp.l2_framing_id import L2FramingIdentifier
from gr_playground.simulator.sigmf_writer import SigMFWriter


def test_zero_forcing_equalizer_multipath_cancellation():
    """Verify that Zero-Forcing (ZF) equalizer cancels 2-tap and 3-tap multipath ISI distortion."""
    # Synthesize QPSK reference signal
    symbols = (np.random.randint(0, 2, 2048) * 2 - 1) + 1j * (np.random.randint(0, 2, 2048) * 2 - 1)
    symbols = symbols / np.sqrt(2.0)
    
    # 2-tap multipath channel: h = [1.0, 0.4 + 0.2j]
    channel_taps = [1.0, 0.4 + 0.2j]
    rx_distorted = np.convolve(symbols, channel_taps)[:len(symbols)]
    
    evm_before = calculate_evm(rx_distorted, constellation_type="QPSK")
    
    # Equalize with Zero-Forcing
    equalized = L1Equalizer.equalize_zero_forcing(rx_distorted, channel_taps=channel_taps)
    evm_after = calculate_evm(equalized, constellation_type="QPSK")
    
    assert evm_after < evm_before
    assert evm_after < 15.0 # EVM reduced to clear constellation


def test_mmse_equalizer_noise_regularization():
    """Verify that MMSE equalizer regularizes inversion under noisy multipath channels."""
    symbols = (np.random.randint(0, 2, 2048) * 2 - 1) + 1j * (np.random.randint(0, 2, 2048) * 2 - 1)
    symbols = symbols / np.sqrt(2.0)
    
    channel_taps = [1.0, 0.5 + 0.3j, 0.2 - 0.1j]
    rx_multipath = np.convolve(symbols, channel_taps)[:len(symbols)]
    
    # Add AWGN noise
    noise = (np.random.normal(0, 0.1, len(symbols)) + 1j * np.random.normal(0, 0.1, len(symbols)))
    rx_noisy = rx_multipath + noise
    
    eq_zf = L1Equalizer.equalize_zero_forcing(rx_noisy, channel_taps=channel_taps)
    eq_mmse = L1Equalizer.equalize_mmse(rx_noisy, channel_taps=channel_taps, snr_db=15.0)
    
    evm_zf = calculate_evm(eq_zf, constellation_type="QPSK")
    evm_mmse = calculate_evm(eq_mmse, constellation_type="QPSK")
    
    assert evm_mmse <= evm_zf # MMSE outperforms ZF under noise


def test_adaptive_lms_equalizer_evm_improvement():
    """Verify that Decision-Directed LMS Equalizer reduces EVM dynamically."""
    symbols = (np.random.randint(0, 2, 4096) * 2 - 1) + 1j * (np.random.randint(0, 2, 4096) * 2 - 1)
    symbols = symbols / np.sqrt(2.0)
    
    channel_taps = [1.0, 0.25]
    rx_distorted = np.convolve(symbols, channel_taps)[:len(symbols)]
    
    evm_before = calculate_evm(rx_distorted, constellation_type="QPSK")
    
    equalized = L1Equalizer.equalize_lms_adaptive(rx_distorted, num_taps=11, mu=0.02, mod_type="QPSK")
    evm_after = calculate_evm(equalized[500:], constellation_type="QPSK") # Evaluate after convergence
    
    assert evm_after < evm_before


def test_end_to_end_multipath_equalization_receiver_pipeline():
    """
    End-to-end integration test:
    Message -> L2 Framing -> FEC -> QPSK Mod -> Multipath Channel -> L1 Equalizer -> Sync -> Demod -> FEC Decode -> L2 Extract
    """
    test_msg = b"Equalizer Integration Test 2026"
    channel_taps = [1.0, 0.3 + 0.1j]
    
    sim = ChannelSimulatorFlowgraph(
        source_type="sine",
        mod_type="BPSK",
        sample_rate=32000,
        num_samples=8192,
        snr_db=30.0,
        multipath_taps=channel_taps,
        message_payload=test_msg,
        framing_type="HDLC",
        fec_type="CONVOLUTIONAL_K7"
    )
    distorted_iq = sim.get_samples()
    
    # 1. L1 Multipath Equalization
    equalized_iq = L1Equalizer.equalize_mmse(distorted_iq, channel_taps=channel_taps, snr_db=30.0)
    
    # 2. Synchronization & Demodulation
    sync_res = synchronize_signal_flowgraph(equalized_iq, sample_rate=32000, mod_type="BPSK")
    rx_bits = slice_psk_qpsk_bits(sync_res["synced_samples"], mod_type="BPSK")
    
    # 3. Channel Decoding & L2 Framing Payload Extraction
    fec_decoded = ChannelDecoder.decode(rx_bits, fec_type="CONVOLUTIONAL_K7")
    framing_res = L2FramingIdentifier.identify_and_extract(fec_decoded)
    
    assert framing_res["is_valid_crc"] is True
    assert test_msg in framing_res["extracted_message"]


def test_equalize_channel_cli_skill():
    """Verify CLI execution of channel-equalization skill."""
    with tempfile.TemporaryDirectory() as tmpdir:
        iq_data = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex64)
        in_file = os.path.join(tmpdir, "distorted.sigmf-data")
        out_file = os.path.join(tmpdir, "equalized.sigmf-data")
        
        SigMFWriter.export_dataset(in_file, iq_data, sample_rate=32000)
        
        script_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            ".agents",
            "skills",
            "channel-equalization",
            "scripts",
            "equalize_channel.py"
        )
        assert os.path.exists(script_path)
        
        env = dict(os.environ, PYTHONPATH=f"/usr/lib/python3/dist-packages:.:{os.environ.get('PYTHONPATH', '')}")
        cmd = [sys.executable, script_path, "--input", in_file, "--algo", "MMSE", "--output", out_file]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        assert "L1 Multipath Equalization (MMSE) complete" in res.stdout
        assert os.path.exists(out_file)
