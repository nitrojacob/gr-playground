"""
Signal Cleanup & Filtering Module using Native GNU Radio Blocks.
Applies DC offset removal, I/Q imbalance correction, lowpass/bandpass FIR filtering, and Automatic Gain Control (AGC).
"""

from gnuradio import gr, blocks, filter, analog
import numpy as np

class IQBalanceFlowgraph(gr.top_block):
    """Native GNU Radio flowgraph for Gram-Schmidt blind I/Q imbalance compensation."""
    def __init__(self, samples):
        super(IQBalanceFlowgraph, self).__init__("IQBalanceFlowgraph")
        samples_c = np.asarray(samples, dtype=np.complex64)
        i = np.real(samples_c)
        q = np.imag(samples_c)
        
        mean_i = float(np.mean(i))
        mean_q = float(np.mean(q))
        i_dc = i - mean_i
        q_dc = q - mean_q
        
        p_i = np.mean(i_dc**2)
        p_q = np.mean(q_dc**2)
        alpha = float(np.sqrt(p_i / (p_q + 1e-12)) if p_q > 0 else 1.0)
        q_scale = q_dc * alpha
        
        cov_iq = np.mean(i_dc * q_scale)
        sin_phi = float(np.clip(cov_iq / (np.sqrt(p_i * np.mean(q_scale**2)) + 1e-12), -0.99, 0.99))
        cos_phi = float(np.sqrt(1.0 - sin_phi**2))

        # Build GNU Radio block topology
        self.src = blocks.vector_source_c(samples_c.tolist(), False)
        self.c2f = blocks.complex_to_float()
        self.sub_i = blocks.add_const_ff(-mean_i)
        self.sub_q = blocks.add_const_ff(-mean_q)
        
        self.scale_q = blocks.multiply_const_ff(alpha)
        self.cross_i = blocks.multiply_const_ff(-sin_phi)
        self.add_q = blocks.add_ff()
        self.norm_q = blocks.multiply_const_ff(1.0 / (cos_phi + 1e-12))
        
        self.f2c = blocks.float_to_complex()
        self.sink = blocks.vector_sink_c()

        # Connections: I branch -> direct to I out of f2c; Q branch -> orthog compensation
        self.connect(self.src, self.c2f)
        self.connect((self.c2f, 0), self.sub_i, (self.f2c, 0))
        self.connect((self.c2f, 1), self.sub_q, self.scale_q)
        
        self.connect(self.sub_i, self.cross_i)
        self.connect(self.scale_q, (self.add_q, 0))
        self.connect(self.cross_i, (self.add_q, 1))
        self.connect(self.add_q, self.norm_q, (self.f2c, 1))
        
        self.connect(self.f2c, self.sink)

    def run_balance(self):
        self.run()
        return np.array(self.sink.data(), dtype=np.complex64)

def gram_schmidt_iq_balance(samples):
    """
    Gram-Schmidt blind I/Q imbalance estimation and compensation using GNU Radio top_block flowgraph.
    """
    tb = IQBalanceFlowgraph(samples)
    return tb.run_balance()


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
            safe_cutoff = max(float(cutoff_hz), 100.0)
            trans_width = max(safe_cutoff * 0.2, 50.0)
            if safe_cutoff < sample_rate / 2.0:
                taps = filter.firdes.low_pass(1.0, sample_rate, safe_cutoff, trans_width)
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
