"""
Layer-1 (L1 Physical Layer) Preamble Cross-Correlation & Packet Detection Module
Implements:
- Barker Code Matched Filter Cross-Correlator (7, 11, 13-bit)
- Zadoff-Chu (CAZAC) Sequence Correlator (LTE / 5G NR)
- Schmidl-Cox OFDM Preamble Synchronizer & Coarse CFO Estimator
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union

BARKER_CODES = {
    7: np.array([1, 1, 1, -1, -1, 1, -1], dtype=np.float32),
    11: np.array([1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1], dtype=np.float32),
    13: np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float32),
}

def generate_barker_sequence(length: int = 11) -> np.ndarray:
    """Generate 7, 11, or 13-bit Barker code sequence."""
    if length not in BARKER_CODES:
        raise ValueError(f"Unsupported Barker code length: {length}. Supported: 7, 11, 13.")
    return BARKER_CODES[length].copy()

def generate_zadoff_chu_sequence(u: int = 25, N: int = 63) -> np.ndarray:
    """
    Generate Zadoff-Chu CAZAC complex sequence.
    x_u[n] = exp(-j * pi * u * n * (n + 1) / N)
    """
    n = np.arange(N)
    zc = np.exp(-1j * np.pi * u * n * (n + 1) / float(N))
    return zc.astype(np.complex64)

def generate_schmidl_cox_preamble(n_fft: int = 64, seed: int = 42) -> np.ndarray:
    """
    Generate a Schmidl-Cox OFDM preamble symbol with repeated time-domain halves.
    Subcarriers on even indices are pseudorandom BPSK, odd indices are zero.
    """
    rng = np.random.RandomState(seed)
    half_fft = n_fft // 2
    X = np.zeros(n_fft, dtype=np.complex64)
    # Set even subcarriers to BPSK symbols
    even_symbols = (rng.randint(0, 2, size=half_fft) * 2 - 1).astype(np.complex64)
    X[0::2] = even_symbols
    
    # IFFT transformation produces two identical half-symbols in time domain
    x_time = np.fft.ifft(X) * np.sqrt(n_fft)
    return x_time.astype(np.complex64)

from gnuradio import gr, blocks, filter, digital

class L1PacketDetector:
    """Layer-1 IQ Preamble Cross-Correlator and Packet Detector using GNU Radio FIR filter flowgraphs."""

    @staticmethod
    def detect_barker_preamble(
        samples: np.ndarray,
        barker_length: int = 11,
        samples_per_symbol: int = 4,
        threshold: float = 0.5,
        packet_len_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Barker Code Matched Filter Cross-Correlator using GNU Radio filter.fir_filter_ccc flowgraph.
        """
        y = np.asarray(samples, dtype=np.complex64)
        if len(y) < barker_length * samples_per_symbol:
            return {"num_packets_found": 0, "peak_indices": [], "peak_values": [], "detected_packets": []}

        barker = generate_barker_sequence(barker_length)
        ref_signal = np.repeat(barker, samples_per_symbol).astype(np.complex64)
        L = len(ref_signal)
        ref_energy = np.sum(np.abs(ref_signal)**2)

        # Run GNU Radio FIR matched filter top_block flowgraph
        class BarkerMatchedFilterFlowgraph(gr.top_block):
            def __init__(self, samples_in, ref_sig):
                super().__init__("BarkerMatchedFilterFlowgraph")
                taps = np.conj(ref_sig[::-1]).astype(np.complex64)
                self.src = blocks.vector_source_c(samples_in.tolist(), False)
                self.fir = filter.fir_filter_ccc(1, taps.tolist())
                self.sink = blocks.vector_sink_c()
                self.connect(self.src, self.fir, self.sink)

            def run_filter(self):
                self.run()
                return np.array(self.sink.data(), dtype=np.complex64)

        tb = BarkerMatchedFilterFlowgraph(y, ref_signal)
        mf_out = tb.run_filter()

        # Compute normalized metric from FIR filter output
        num_windows = len(y) - L + 1
        corr_metric = np.zeros(num_windows, dtype=np.float32)

        for n in range(num_windows):
            window = y[n : n + L]
            win_energy = np.sum(np.abs(window)**2)
            if win_energy > 1e-9:
                mf_val = mf_out[n + L - 1] if (n + L - 1) < len(mf_out) else 0.0
                corr_metric[n] = (np.abs(mf_val)**2) / (win_energy * ref_energy)

        peak_indices = []
        peak_values = []
        
        n = 0
        min_distance = L * 2
        while n < len(corr_metric):
            if corr_metric[n] >= threshold:
                end_search = min(n + min_distance, len(corr_metric))
                local_max_rel = np.argmax(corr_metric[n:end_search])
                best_idx = n + local_max_rel
                
                peak_indices.append(int(best_idx))
                peak_values.append(float(corr_metric[best_idx]))
                n = best_idx + min_distance
            else:
                n += 1

        pkt_len = packet_len_samples if packet_len_samples is not None else L * 8
        detected_packets = [y[p_idx : min(p_idx + pkt_len, len(y))] for p_idx in peak_indices]

        return {
            "num_packets_found": len(peak_indices),
            "peak_indices": peak_indices,
            "peak_values": peak_values,
            "correlation_metric": corr_metric,
            "detected_packets": detected_packets
        }

    @staticmethod
    def detect_zadoff_chu_preamble(
        samples: np.ndarray,
        u: int = 25,
        N: int = 63,
        threshold: float = 0.5,
        packet_len_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Zadoff-Chu CAZAC sequence cross-correlator for 4G/5G PSS style frame acquisition.
        """
        y = np.asarray(samples, dtype=np.complex64)
        if len(y) < N:
            return {"num_packets_found": 0, "peak_indices": [], "peak_values": [], "detected_packets": []}

        ref_signal = generate_zadoff_chu_sequence(u=u, N=N)
        ref_energy = np.sum(np.abs(ref_signal)**2)
        num_windows = len(y) - N + 1
        corr_metric = np.zeros(num_windows, dtype=np.float32)

        for n in range(num_windows):
            window = y[n : n + N]
            win_energy = np.sum(np.abs(window)**2)
            if win_energy > 1e-9:
                cross = np.sum(window * np.conj(ref_signal))
                corr_metric[n] = (np.abs(cross)**2) / (win_energy * ref_energy)

        peak_indices = []
        peak_values = []
        n = 0
        min_dist = N * 2
        while n < len(corr_metric):
            if corr_metric[n] >= threshold:
                end_search = min(n + min_dist, len(corr_metric))
                local_max_rel = np.argmax(corr_metric[n:end_search])
                best_idx = n + local_max_rel
                
                peak_indices.append(int(best_idx))
                peak_values.append(float(corr_metric[best_idx]))
                n = best_idx + min_dist
            else:
                n += 1

        pkt_len = packet_len_samples if packet_len_samples is not None else N * 8
        detected_packets = [y[p_idx : min(p_idx + pkt_len, len(y))] for p_idx in peak_indices]

        return {
            "num_packets_found": len(peak_indices),
            "peak_indices": peak_indices,
            "peak_values": peak_values,
            "correlation_metric": corr_metric,
            "detected_packets": detected_packets
        }

    @staticmethod
    def schmidl_cox_detect(
        samples: np.ndarray,
        n_fft: int = 64,
        sample_rate: float = 32000.0,
        threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Schmidl-Cox OFDM Preamble Detection & Coarse CFO Estimator.
        Computes autocorrelation metric M(d) = |P(d)|^2 / R(d)^2 across repeated half-symbols.
        Estimates coarse CFO from phase angle: angle(P(d_peak)).
        """
        y = np.asarray(samples, dtype=np.complex64)
        H = n_fft // 2
        if len(y) < n_fft:
            return {"num_packets_found": 0, "peak_indices": [], "peak_values": [], "estimated_cfo_hz": 0.0}

        num_pos = len(y) - n_fft + 1
        m_metric = np.zeros(num_pos, dtype=np.float32)
        p_vals = np.zeros(num_pos, dtype=np.complex64)

        for d in range(num_pos):
            first_half = y[d : d + H]
            second_half = y[d + H : d + 2 * H]
            
            P_d = np.sum(np.conj(first_half) * second_half)
            R_d = np.sum(np.abs(second_half)**2)
            
            p_vals[d] = P_d
            if R_d > 1e-9:
                m_metric[d] = (np.abs(P_d)**2) / (R_d**2)

        peak_indices = []
        peak_values = []
        cfo_estimates = []

        d = 0
        min_dist = n_fft * 2
        while d < len(m_metric):
            if m_metric[d] >= threshold:
                end_search = min(d + min_dist, len(m_metric))
                best_idx = d + int(np.argmax(m_metric[d:end_search]))
                
                peak_indices.append(best_idx)
                peak_values.append(float(m_metric[best_idx]))
                
                # Estimate coarse CFO from phase of P(d) at peak
                phase_p = np.angle(p_vals[best_idx])
                # CFO = (phase / pi) * (sample_rate / n_fft)
                cfo_hz = (phase_p / np.pi) * (sample_rate / float(n_fft))
                cfo_estimates.append(float(cfo_hz))
                
                d = best_idx + min_dist
            else:
                d += 1

        primary_cfo = cfo_estimates[0] if cfo_estimates else 0.0

        return {
            "num_packets_found": len(peak_indices),
            "peak_indices": peak_indices,
            "peak_values": peak_values,
            "estimated_cfo_hz": primary_cfo,
            "metric": m_metric
        }

    @staticmethod
    def detect_gmsk_preamble(
        samples: np.ndarray,
        preamble_type: str = "BT_LE",
        samples_per_symbol: int = 4,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Detect GMSK preambles (BT_LE, AIS, GSM_TSC0) using GMSKPreambleDetector."""
        from gr_playground.dsp.gmsk import GMSKPreambleDetector
        return GMSKPreambleDetector.detect_gmsk_preamble(
            samples=samples,
            preamble_type=preamble_type,
            samples_per_symbol=samples_per_symbol,
            threshold=threshold
        )
