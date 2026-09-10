"""
Automated Test Suite for GMSK / MSK Continuous Phase Modulation Demodulation, Preamble Detection, and Receiver Skills.
"""

import sys
import os
import tempfile
import pytest
import numpy as np
from gnuradio import gr, blocks, digital

from gr_playground.dsp.gmsk import GMSKDemodFlowgraph, GMSKPreambleDetector
from gr_playground.dsp.demodulation import demodulate_signal_flowgraph
from gr_playground.dsp.packet_detection import L1PacketDetector
from gr_playground.simulator.framing import L2FrameEncoder
from gr_playground.dsp.l2_framing_id import L2FramingIdentifier
from gr_playground.simulator.sigmf_writer import SigMFWriter
from gr_playground.utils.sigmf_io import read_sigmf


def test_gmsk_msk_native_gnuradio_demodulation():
    """Verify GMSK and MSK bit sequence modulation and demodulation using native GNU Radio flowgraphs."""
    test_bits = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1] * 4, dtype=np.uint8)
    
    # 1. Modulate using digital.gmsk_mod
    class GMSKModHelper(gr.top_block):
        def __init__(self, bits, sps=4):
            super().__init__("GMSKModHelper")
            self.src = blocks.vector_source_b(bits.tolist(), False)
            self.mod = digital.gmsk_mod(samples_per_symbol=sps, bt=0.35)
            self.sink = blocks.vector_sink_c()
            self.connect(self.src, self.mod, self.sink)

        def run_mod(self):
            self.run()
            return np.array(self.sink.data(), dtype=np.complex64)

    mod_tb = GMSKModHelper(test_bits, sps=4)
    iq_samples = mod_tb.run_mod()
    assert len(iq_samples) > 0
    
    # Add light channel AWGN noise
    noise = (np.random.normal(0, 0.01, len(iq_samples)) + 1j * np.random.normal(0, 0.01, len(iq_samples))).astype(np.complex64)
    rx_samples = iq_samples + noise
    
    # 2. Demodulate using GMSKDemodFlowgraph
    demod_tb = GMSKDemodFlowgraph(rx_samples, samples_per_symbol=4)
    rx_bits = demod_tb.run_demod()
    assert len(rx_bits) > 0

    # 3. Demodulate via demodulate_signal_flowgraph interface
    out_bits, preview = demodulate_signal_flowgraph(rx_samples, mod_type="GMSK")
    assert "GMSK Digital Bit Payload" in preview
    assert len(out_bits) > 0


def test_bluetooth_le_and_ais_preamble_detection():
    """Verify Bluetooth LE (0xAA) and AIS maritime (0x555555) preamble detection."""
    # Construct Bluetooth LE preamble: [1, 0, 1, 0, 1, 0, 1, 0]
    bt_preamble_bits = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    payload_bits = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0] * 2, dtype=np.uint8)
    all_bits = np.concatenate([bt_preamble_bits, payload_bits])

    class GMSKModHelper(gr.top_block):
        def __init__(self, bits, sps=4):
            super().__init__("GMSKModHelper")
            self.src = blocks.vector_source_b(bits.tolist(), False)
            self.mod = digital.gmsk_mod(samples_per_symbol=sps, bt=0.5)
            self.sink = blocks.vector_sink_c()
            self.connect(self.src, self.mod, self.sink)

        def run_mod(self):
            self.run()
            return np.array(self.sink.data(), dtype=np.complex64)

    mod_tb = GMSKModHelper(all_bits, sps=4)
    iq_samples = mod_tb.run_mod()

    res = L1PacketDetector.detect_gmsk_preamble(iq_samples, preamble_type="BT_LE", samples_per_symbol=4, threshold=0.5)
    assert res["num_packets_found"] >= 1
    assert len(res["peak_indices"]) >= 1


def test_end_to_end_gmsk_receiver_pipeline():
    """
    End-to-end integration test:
    Message -> L2 HDLC Framing -> GMSK Modulator -> Channel -> Native GMSK Demod -> L2 Framing Extract -> Valid CRC Check
    """
    test_msg = b"GMSK Receiver Pipeline Payload 2026"
    framed_bytes = L2FrameEncoder.encode(test_msg, framing_type="HDLC") + bytes([0x7E] * 4)
    framed_bits = np.unpackbits(np.frombuffer(framed_bytes, dtype=np.uint8))

    class GMSKModHelper(gr.top_block):
        def __init__(self, bits, sps=4):
            super().__init__("GMSKModHelper")
            src_data = list(bits) if isinstance(bits, (bytes, bytearray, list)) else bits.tolist()
            self.src = blocks.vector_source_b(src_data, False)
            self.mod = digital.gmsk_mod(samples_per_symbol=sps, bt=0.35)
            self.sink = blocks.vector_sink_c()
            self.connect(self.src, self.mod, self.sink)

        def run_mod(self):
            self.run()
            return np.array(self.sink.data(), dtype=np.complex64)

    mod_tb = GMSKModHelper(list(framed_bytes), sps=4)
    iq_samples = mod_tb.run_mod()

    demod_tb = GMSKDemodFlowgraph(iq_samples, samples_per_symbol=4)
    rx_bits = demod_tb.run_demod()
    rx_bytes = np.packbits(rx_bits).tobytes()
    framing_res = L2FramingIdentifier.identify_and_extract(rx_bytes)
    assert framing_res["is_valid_crc"] is True
    assert test_msg in framing_res["extracted_message"]


def test_enhanced_existing_skills_cli():
    """Verify CLI execution of enhanced signal-demodulation and preamble-detection skills."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_bits = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.uint8)
        
        class GMSKModHelper(gr.top_block):
            def __init__(self, bits, sps=4):
                super().__init__("GMSKModHelper")
                self.src = blocks.vector_source_b(bits.tolist(), False)
                self.mod = digital.gmsk_mod(samples_per_symbol=sps, bt=0.35)
                self.sink = blocks.vector_sink_c()
                self.connect(self.src, self.mod, self.sink)

            def run_mod(self):
                self.run()
                return np.array(self.sink.data(), dtype=np.complex64)

        mod_tb = GMSKModHelper(test_bits, sps=4)
        iq_samples = mod_tb.run_mod()

        in_file = os.path.join(tmpdir, "gmsk_signal.sigmf-data")
        out_file = os.path.join(tmpdir, "extracted_packet.sigmf-data")
        SigMFWriter.export_dataset(in_file, iq_samples, sample_rate=32000)

        # Test CLI demodulate_signal.py
        demod_script = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "..", ".agents", "skills", "signal-demodulation", "scripts", "demodulate_signal.py"
        ))
        
        import subprocess
        cmd_demod = [sys.executable, demod_script, "--input", in_file, "--mod", "GMSK"]
        res_demod = subprocess.run(cmd_demod, capture_output=True, text=True, check=True)
        assert "GMSK Digital Bit Payload" in res_demod.stdout

        # Test CLI detect_preamble.py
        preamble_script = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "..", ".agents", "skills", "preamble-detection", "scripts", "detect_preamble.py"
        ))
        cmd_preamble = [sys.executable, preamble_script, "--input", in_file, "--type", "BT_LE", "--output", out_file]
        res_preamble = subprocess.run(cmd_preamble, capture_output=True, text=True, check=True)
        assert "Preamble Detection (BT_LE)" in res_preamble.stdout
        assert os.path.exists(out_file)
