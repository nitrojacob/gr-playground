"""
Automated Test Suite for Layer-1 (L1) Preamble Cross-Correlation & Packet Detection.
Exercises Barker codes (7, 11, 13), Zadoff-Chu CAZAC sequences, and Schmidl-Cox OFDM synchronizer.
"""

import pytest
import numpy as np
import os
import sys
import tempfile
import subprocess

from gr_playground.dsp.packet_detection import (
    L1PacketDetector,
    generate_barker_sequence,
    generate_zadoff_chu_sequence,
    generate_schmidl_cox_preamble
)
from gr_playground.simulator.channel_simulator import ChannelSimulatorFlowgraph
from gr_playground.dsp.equalization import L1Equalizer
from gr_playground.dsp.synchronization import synchronize_signal_flowgraph
from gr_playground.dsp.demodulation import slice_psk_qpsk_bits
from gr_playground.simulator.channel_coding import ChannelEncoder
from gr_playground.dsp.channel_decoding import ChannelDecoder
from gr_playground.dsp.l2_framing_id import L2FramingIdentifier
from gr_playground.simulator.sigmf_writer import SigMFWriter
from gr_playground.simulator.framing import L2FrameEncoder


def test_barker_code_preamble_cross_correlation():
    """Verify 11-bit Barker preamble matched filter detection in noisy IQ stream."""
    barker = generate_barker_sequence(11)
    samples_per_sym = 4
    ref_iq = np.repeat(barker, samples_per_sym).astype(np.complex64)
    
    # Construct stream: 500 zeros + Barker preamble + 1000 random QPSK payload
    zeros_lead = np.zeros(500, dtype=np.complex64)
    payload_iq = (np.random.randint(0, 2, 1000) * 2 - 1 + 1j * (np.random.randint(0, 2, 1000) * 2 - 1)).astype(np.complex64)
    
    stream = np.concatenate([zeros_lead, ref_iq, payload_iq])
    # Add AWGN noise
    noise = (np.random.normal(0, 0.05, len(stream)) + 1j * np.random.normal(0, 0.05, len(stream))).astype(np.complex64)
    stream_noisy = stream + noise
    
    res = L1PacketDetector.detect_barker_preamble(stream_noisy, barker_length=11, samples_per_symbol=4, threshold=0.5)
    
    assert res["num_packets_found"] >= 1
    assert abs(res["peak_indices"][0] - 500) <= 2 # Accurate timing sync within 2 samples


def test_zadoff_chu_preamble_cross_correlation():
    """Verify Zadoff-Chu CAZAC sequence preamble detection."""
    zc = generate_zadoff_chu_sequence(u=25, N=63)
    lead = np.zeros(300, dtype=np.complex64)
    tail = np.zeros(300, dtype=np.complex64)
    stream = np.concatenate([lead, zc, tail])
    
    res = L1PacketDetector.detect_zadoff_chu_preamble(stream, u=25, N=63, threshold=0.6)
    
    assert res["num_packets_found"] >= 1
    assert abs(res["peak_indices"][0] - 300) <= 1


def test_schmidl_cox_ofdm_preamble_detection_and_cfo_estimation():
    """Verify Schmidl-Cox OFDM preamble detection metric and coarse CFO estimation."""
    preamble = generate_schmidl_cox_preamble(n_fft=64)
    
    # Inject 400 Hz CFO shift
    sample_rate = 32000.0
    cfo_injected = 400.0
    t = np.arange(len(preamble)) / sample_rate
    preamble_cfo = preamble * np.exp(1j * 2.0 * np.pi * cfo_injected * t)
    
    lead = np.zeros(200, dtype=np.complex64)
    stream = np.concatenate([lead, preamble_cfo])
    
    res = L1PacketDetector.schmidl_cox_detect(stream, n_fft=64, sample_rate=sample_rate, threshold=0.5)
    
    assert res["num_packets_found"] >= 1
    assert abs(res["peak_indices"][0] - 200) <= 1
    assert abs(res["estimated_cfo_hz"] - cfo_injected) < 50.0 # Coarse CFO accurate within 50 Hz


def test_end_to_end_preamble_detection_and_receiver_pipeline():
    """
    End-to-end integration test:
    Continuous Stream (Preamble + Payload) -> L1 Packet Detector -> Equalizer -> Sync -> Demod -> FEC Decode -> L2 Extract
    """
    test_msg = b"Preamble Detector Test 2026"
    framed_bytes = L2FrameEncoder.encode(test_msg, framing_type="HDLC")
    fec_bits = ChannelEncoder.encode(framed_bytes, fec_type="CONVOLUTIONAL_K7")
    
    # Concatenate 11-bit Barker preamble bits + FEC bits + Viterbi tail flushing bits
    barker_bits = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0], dtype=np.uint8)
    tail_bits = np.zeros(16, dtype=np.uint8) # 16 zero bits to flush Viterbi decoder trellis
    all_bits = np.concatenate([barker_bits, fec_bits, tail_bits])
    
    # BPSK Modulation (1 -> +1, 0 -> -1)
    bpsk_syms = (all_bits * 2.0 - 1.0).astype(np.complex64)
    bpsk_samples = np.repeat(bpsk_syms, 4)
    
    # Prepend lead noise to simulate continuous stream with burst arrival
    lead_noise = (np.random.normal(0, 0.01, 400) + 1j * np.random.normal(0, 0.01, 400)).astype(np.complex64)
    rx_stream = np.concatenate([lead_noise, bpsk_samples])
    
    # 1. L1 Preamble Detector
    det_res = L1PacketDetector.detect_barker_preamble(rx_stream, barker_length=11, samples_per_symbol=4, threshold=0.5)
    assert det_res["num_packets_found"] >= 1
    
    pkt_start = det_res["peak_indices"][0]
    packet_slice = rx_stream[pkt_start:]
    
    # 2. Receiver Pipeline
    sync_res = synchronize_signal_flowgraph(packet_slice, sample_rate=32000, mod_type="BPSK")
    rx_bits = slice_psk_qpsk_bits(sync_res["synced_samples"], mod_type="BPSK")
    
    # Resolve BPSK phase ambiguity (if Costas loop inverted symbols)
    if not np.array_equal(rx_bits[:11], barker_bits):
        if np.array_equal(1 - rx_bits[:11], barker_bits):
            rx_bits = 1 - rx_bits
            
    payload_bits = rx_bits[11:]
    fec_decoded = ChannelDecoder.decode(payload_bits, fec_type="CONVOLUTIONAL_K7")
    framing_res = L2FramingIdentifier.identify_and_extract(fec_decoded)
    
    assert framing_res["is_valid_crc"] is True
    assert test_msg in framing_res["extracted_message"]


def test_detect_preamble_cli_skill():
    """Verify CLI execution of preamble-detection skill."""
    with tempfile.TemporaryDirectory() as tmpdir:
        barker = generate_barker_sequence(11)
        ref_iq = np.repeat(barker, 4).astype(np.complex64)
        lead = np.zeros(200, dtype=np.complex64)
        tail = np.zeros(400, dtype=np.complex64)
        stream = np.concatenate([lead, ref_iq, tail])
        
        in_file = os.path.join(tmpdir, "stream.sigmf-data")
        out_file = os.path.join(tmpdir, "extracted.sigmf-data")
        
        SigMFWriter.export_dataset(in_file, stream, sample_rate=32000)
        
        script_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            ".agents",
            "skills",
            "preamble-detection",
            "scripts",
            "detect_preamble.py"
        )
        assert os.path.exists(script_path)
        
        env = dict(os.environ, PYTHONPATH=f"/usr/lib/python3/dist-packages:.:{os.environ.get('PYTHONPATH', '')}")
        cmd = [sys.executable, script_path, "--input", in_file, "--type", "BARKER", "--length", "11", "--output", out_file]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        
        assert "Found 1 packet(s) at indices [200]" in res.stdout
        assert os.path.exists(out_file)
