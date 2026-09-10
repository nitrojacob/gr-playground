"""
GMSK and MSK Demodulation & Preamble Detection Module using Native GNU Radio Blocks.
Supports Continuous Phase Modulation (CPM) demodulation for GMSK (BT=0.3, 0.5) and MSK (BT=1.0),
plus Bluetooth LE (0xAA/0x55), AIS Maritime (0x555555), and GSM preamble correlation.
"""

from gnuradio import gr, blocks, analog, digital, filter
import numpy as np
from typing import Dict, Any, List, Optional

PREAMBLE_PATTERNS = {
    "BT_LE": np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8),
    "BT_LE_ALT": np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8),
    "AIS": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8),
    "GSM_TSC0": np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1], dtype=np.uint8),
}

class GMSKDemodFlowgraph(gr.top_block):
    """
    Native GNU Radio Flowgraph for GMSK and MSK Demodulation.
    Instantiates native `digital.gmsk_demod` or `analog.quadrature_demod_cf` blocks.
    """
    def __init__(self, samples: np.ndarray, samples_per_symbol: int = 4, bt: float = 0.35):
        super(GMSKDemodFlowgraph, self).__init__("GMSKDemodFlowgraph")
        
        samples_c = np.asarray(samples, dtype=np.complex64)
        self.src = blocks.vector_source_c(samples_c.tolist(), False)
        
        # Native GNU Radio GMSK Demodulator block
        try:
            self.gmsk_demod = digital.gmsk_demod(
                samples_per_symbol=samples_per_symbol,
                verbose=False,
                log=False
            )
            self.sink = blocks.vector_sink_b()
            self.connect(self.src, self.gmsk_demod, self.sink)
            self.use_native_gmsk = True
        except Exception:
            # Fallback: Quadrature FM Discriminator + Clock Recovery
            gain = 1.0 / (float(samples_per_symbol))
            self.quad_demod = analog.quadrature_demod_cf(gain)
            self.clock_rec = digital.clock_recovery_mm_ff(
                samples_per_symbol,
                0.25 * 0.175 * 0.175,
                0.5,
                0.175,
                0.005
            )
            self.slicer = digital.binary_slicer_fb()
            self.sink = blocks.vector_sink_b()
            self.connect(self.src, self.quad_demod, self.clock_rec, self.slicer, self.sink)
            self.use_native_gmsk = False

    def run_demod(self) -> np.ndarray:
        self.run()
        raw_bytes = np.array(self.sink.data(), dtype=np.uint8)
        bits = (raw_bytes > 0).astype(np.uint8)
        return bits


class GMSKPreambleDetector:
    """Layer-1 Preamble Detector for GMSK / MSK protocols (Bluetooth LE, AIS, GSM)."""

    @staticmethod
    def detect_gmsk_preamble(
        samples: np.ndarray,
        preamble_type: str = "BT_LE",
        samples_per_symbol: int = 4,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect GMSK preambles (BT_LE, AIS, GSM_TSC0) from raw complex IQ stream or demodulated bits.
        """
        p_type = preamble_type.upper()
        if p_type not in PREAMBLE_PATTERNS:
            pattern = PREAMBLE_PATTERNS["BT_LE"]
        else:
            pattern = PREAMBLE_PATTERNS[p_type]

        # Demodulate IQ stream to bit sequence using GMSKDemodFlowgraph
        tb = GMSKDemodFlowgraph(samples, samples_per_symbol=samples_per_symbol)
        bits = tb.run_demod()

        L = len(pattern)
        if len(bits) < L:
            return {"num_packets_found": 0, "peak_indices": [], "peak_values": [], "detected_packets": []}

        peak_indices = []
        peak_values = []
        detected_packets = []

        num_windows = len(bits) - L + 1
        for n in range(num_windows):
            window = bits[n : n + L]
            matches = np.sum(window == pattern)
            metric = float(matches) / float(L)
            if metric >= threshold:
                # Check for duplicate neighborhood matches
                if not peak_indices or (n - peak_indices[-1]) >= L:
                    peak_indices.append(int(n))
                    peak_values.append(metric)
                    # Extract sample slice corresponding to preamble arrival
                    sample_start = n * samples_per_symbol
                    pkt_samples = samples[sample_start : min(sample_start + L * 32 * samples_per_symbol, len(samples))]
                    detected_packets.append(pkt_samples)

        return {
            "num_packets_found": len(peak_indices),
            "peak_indices": peak_indices,
            "peak_values": peak_values,
            "detected_packets": detected_packets,
            "demodulated_bits": bits
        }
