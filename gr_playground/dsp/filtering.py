"""
Signal Cleanup & Filtering Module using Native GNU Radio Blocks.
Applies DC offset removal, I/Q imbalance correction, lowpass/bandpass FIR filtering, and Automatic Gain Control (AGC).
"""

from gnuradio import gr, blocks, filter, analog
import numpy as np

def gram_schmidt_iq_balance(samples):
    """
    Gram-Schmidt blind I/Q imbalance estimation and compensation.
    Corrects amplitude & phase mismatch between I and Q channels.
    """
    i = np.real(samples)
    q = np.imag(samples)
    
    # 1. Remove DC component
    i = i - np.mean(i)
    q = q - np.mean(q)
    
    # 2. Estimate amplitude imbalance alpha = sqrt(E[I^2] / E[Q^2])
    p_i = np.mean(i**2)
    p_q = np.mean(q**2)
    alpha = np.sqrt(p_i / (p_q + 1e-12)) if p_q > 0 else 1.0
    q_scale = q * alpha
    
    # 3. Estimate phase imbalance sin(phi) = E[I * Q_scale] / sqrt(E[I^2] * E[Q_scale^2])
    cov_iq = np.mean(i * q_scale)
    sin_phi = cov_iq / (np.sqrt(p_i * np.mean(q_scale**2)) + 1e-12)
    sin_phi = np.clip(sin_phi, -0.99, 0.99)
    cos_phi = np.sqrt(1.0 - sin_phi**2)
    
    # 4. Orthogonalize Q_out = (Q_scale - I * sin(phi)) / cos(phi)
    q_balanced = (q_scale - i * sin_phi) / (cos_phi + 1e-12)
    
    return (i + 1j * q_balanced).astype(np.complex64)

class CleanupFlowgraph(gr.top_block):
    def __init__(self, samples, sample_rate=32000, cutoff_hz=8000.0, agc_enable=True, dc_block_enable=True):
        super(CleanupFlowgraph, self).__init__("CleanupFlowgraph")
        
        self.src = blocks.vector_source_c(samples.tolist(), False)
        last_block = self.src
        
        if dc_block_enable:
            self.dc_blocker = filter.dc_blocker_cc(32, True)
            self.connect(last_block, self.dc_blocker)
            last_block = self.dc_blocker

        if cutoff_hz and cutoff_hz < sample_rate / 2.0:
            taps = filter.firdes.low_pass(1.0, sample_rate, cutoff_hz, cutoff_hz * 0.2)
            self.lpf = filter.fir_filter_ccc(1, taps)
            self.connect(last_block, self.lpf)
            last_block = self.lpf

        if agc_enable:
            self.agc = analog.agc2_cc(1e-3, 1e-2, 1.0, 1.0)
            self.connect(last_block, self.agc)
            last_block = self.agc

        self.sink = blocks.vector_sink_c()
        self.connect(last_block, self.sink)

    def run_cleanup(self):
        self.run()
        return np.array(self.sink.data(), dtype=np.complex64)

def cleanup_signal_flowgraph(samples, sample_rate=32000, cutoff_hz=8000.0, agc_enable=True, dc_block_enable=True, iq_balance_enable=True):
    """
    Full cleanup pipeline: Gram-Schmidt I/Q balancing followed by GNU Radio flowgraph filtering.
    """
    samples = np.asarray(samples, dtype=np.complex64)
    
    if iq_balance_enable:
        samples = gram_schmidt_iq_balance(samples)
        
    tb = CleanupFlowgraph(samples, sample_rate=sample_rate, cutoff_hz=cutoff_hz, agc_enable=agc_enable, dc_block_enable=dc_block_enable)
    cleaned = tb.run_cleanup()
    return cleaned
