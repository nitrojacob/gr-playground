"""
Automated Test Suite for Layer-2 Framing, Channel Coding (FEC), and Receiver Identification Skills.
Builds and verifies a comprehensive test matrix of L2 framing protocols and FEC channel codes.
"""

import sys
import pytest
import numpy as np
import os
import tempfile

from gr_playground.simulator.framing import L2FrameEncoder, crc16_ccitt, crc32_ieee
from gr_playground.simulator.channel_coding import ChannelEncoder
from gr_playground.dsp.channel_decoding import ChannelDecoder
from gr_playground.dsp.l2_framing_id import L2FramingIdentifier
from gr_playground.simulator.channel_simulator import ChannelSimulatorFlowgraph
from gr_playground.dsp.synchronization import synchronize_signal_flowgraph
from gr_playground.dsp.demodulation import demodulate_signal_flowgraph, slice_psk_qpsk_bits


# Comprehensive reference lists of supported L2 framing protocols and FEC channel codes
SUPPORTED_L2_FRAMING_SCHEMES = [
    "HDLC",
    "COBS",
    "AX.25",
    "CCSDS",
    "IEEE_802_15_4",
    "GENERIC_LENGTH_PREFIX"
]

SUPPORTED_FEC_CHANNEL_CODES = [
    "REPETITION",
    "HAMMING_7_4",
    "CONVOLUTIONAL_K7",
    "REED_SOLOMON"
]


def test_comprehensive_l2_framing_encoders_and_identification():
    """Verify encoding, CRC calculation, framing identification, and payload extraction for all L2 protocols."""
    original_message = b"GNU Radio L2 Framing Protocol Identification Test Payload #12345"
    
    for framing in SUPPORTED_L2_FRAMING_SCHEMES:
        encoded_frame = L2FrameEncoder.encode(original_message, framing_type=framing)
        assert len(encoded_frame) > len(original_message)
        
        # Test receiver identification skill / module
        result = L2FramingIdentifier.identify_and_extract(encoded_frame)
        assert result["is_valid_crc"] is True
        assert result["extracted_message"] == original_message
        assert result["confidence"] >= 0.7


def test_comprehensive_fec_encoders_and_decoders_with_errors():
    """Verify FEC encoding, bit error correction, and decoding across all channel codes."""
    raw_message = b"Channel Coding Test"
    raw_bits = np.unpackbits(np.frombuffer(raw_message, dtype=np.uint8))
    
    # 1. Repetition (3x) with single bit flip
    rep_bits = ChannelEncoder.encode(raw_bits, fec_type="REPETITION", rate=3)
    rep_bits_corrupted = rep_bits.copy()
    rep_bits_corrupted[1] ^= 1 # Flip one bit in first triplet
    decoded_rep = ChannelDecoder.decode(rep_bits_corrupted, fec_type="REPETITION", rate=3)
    assert np.array_equal(decoded_rep, raw_bits)
    
    # 2. Hamming(7,4) with single-bit error per 7-bit block
    ham_bits = ChannelEncoder.encode(raw_bits, fec_type="HAMMING_7_4")
    ham_bits_corrupted = ham_bits.copy()
    for b in range(len(ham_bits) // 7):
        ham_bits_corrupted[b * 7 + 2] ^= 1 # Corrupt bit index 2 in every 7-bit block
    decoded_ham = ChannelDecoder.decode(ham_bits_corrupted, fec_type="HAMMING_7_4")
    assert np.array_equal(decoded_ham, raw_bits)

    # 3. Convolutional (K=7, Rate 1/2) with Viterbi decoding under channel noise
    conv_bits = ChannelEncoder.encode(raw_bits, fec_type="CONVOLUTIONAL_K7")
    conv_bits_corrupted = conv_bits.copy()
    conv_bits_corrupted[10] ^= 1 # Corrupt isolated channel bit
    conv_bits_corrupted[25] ^= 1
    decoded_conv = ChannelDecoder.decode(conv_bits_corrupted, fec_type="CONVOLUTIONAL_K7")
    # Truncate to original length if needed
    decoded_conv = decoded_conv[:len(raw_bits)]
    assert np.array_equal(decoded_conv, raw_bits)

    # 4. Reed-Solomon RS(15,11) with symbol error correction
    rs_bytes = ChannelEncoder.encode(raw_message, fec_type="REED_SOLOMON")
    rs_corrupted = bytearray(rs_bytes)
    rs_corrupted[2] ^= 0x07 # Corrupt single byte/symbol
    decoded_rs = ChannelDecoder.decode(bytes(rs_corrupted), fec_type="REED_SOLOMON")
    assert decoded_rs[:len(raw_message)] == raw_message


@pytest.mark.parametrize("framing", ["HDLC", "COBS", "CCSDS", "GENERIC_LENGTH_PREFIX"])
@pytest.mark.parametrize("fec", ["HAMMING_7_4", "CONVOLUTIONAL_K7"])
def test_end_to_end_framed_channel_coded_transmitter_receiver_pipeline(framing, fec):
    """
    End-to-end integration test:
    Message -> L2 Framing -> FEC Channel Coding -> BPSK Modulation -> Channel -> Demod -> FEC Decode -> L2 Identify -> Extract Message
    """
    test_msg = b"Pipeline Payload 2026"
    
    sim = ChannelSimulatorFlowgraph(
        source_type="sine",
        mod_type="BPSK",
        sample_rate=32000,
        num_samples=8192,
        snr_db=35.0,
        cfo_hz=0.0,
        message_payload=test_msg,
        framing_type=framing,
        fec_type=fec
    )
    iq_samples = sim.get_samples()
    
    # Receiver Pipeline
    sync_res = synchronize_signal_flowgraph(iq_samples, sample_rate=32000, mod_type="BPSK")
    synced_iq = sync_res["synced_samples"]
    rx_bits = slice_psk_qpsk_bits(synced_iq, mod_type="BPSK")
    
    # 1. Channel Decode across block alignment shifts (for block codes like Hamming)
    framing_result = None
    shifts_to_try = range(7) if fec == "HAMMING_7_4" else [0]
    for s in shifts_to_try:
        fec_decoded_bits = ChannelDecoder.decode(rx_bits[s:], fec_type=fec)
        res = L2FramingIdentifier.identify_and_extract(fec_decoded_bits)
        if res.get("is_valid_crc"):
            framing_result = res
            break

    assert framing_result is not None, "Failed to align and identify L2 frame"
    assert framing_result["is_valid_crc"] is True
    assert framing_result["framing_type"] in [framing, "AX.25"] # HDLC and AX.25 both use 0x7E flags
    assert test_msg in framing_result["extracted_message"]


def test_cli_skills_execution():
    """Verify execution of CLI scripts for l2-framing-identification and channel-decoding skills."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_payload = b"CLI Skill Verification Message"
        framed = L2FrameEncoder.encode(test_payload, framing_type="GENERIC_LENGTH_PREFIX")
        
        in_file = os.path.join(tmpdir, "framed_input.bin")
        with open(in_file, "wb") as f:
            f.write(framed)
            
        # Run CLI skill script for l2-framing-identification
        script_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            ".agents",
            "skills",
            "l2-framing-identification",
            "scripts",
            "identify_l2_framing.py"
        )
        assert os.path.exists(script_path)
        
        import subprocess
        cmd = [sys.executable, script_path, "--input", in_file]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "GENERIC_LENGTH_PREFIX" in res.stdout
        assert "CLI Skill Verification Message" in res.stdout
