"""
Carrier Frequency, Phase, and Clock Synchronization Module using Native GNU Radio Blocks.
Implements CFO estimation via Power-of-N FFT peak, Costas Loop, Gardner Symbol Sync, and Constellation EVM calculation.
"""

from gnuradio import gr, blocks, filter, analog, digital
import numpy as np

def estimate_cfo_power_n(samples, sample_rate=32000, n=4):
    """
    Estimate Carrier Frequency Offset (CFO) by raising M-ary PSK signal to N-th power
    and finding the FFT tone peak using zero-padding.
    """
    y = np.asarray(samples, dtype=np.complex64)
    if len(y) < 16:
        return 0.0
    
    y = y - np.mean(y)
    p_avg = np.mean(np.abs(y)**2)
    if p_avg > 0:
        y = y / np.sqrt(p_avg)

    y_n = y ** n
    nfft = max(32768, 8 * len(y))
    fft_spec = np.abs(np.fft.fftshift(np.fft.fft(y_n, n=nfft)))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / sample_rate))
    
    peak_idx = np.argmax(fft_spec)
    estimated_cfo = freqs[peak_idx] / float(n)
    return float(estimated_cfo)

def calculate_evm(samples, constellation_type="QPSK"):
    """
    Calculate Error Vector Magnitude (EVM %) of synchronized constellation points.
    """
    y = np.asarray(samples, dtype=np.complex64)
    if len(y) == 0:
        return 100.0
    
    # Target reference symbols
    if constellation_type.upper() == "BPSK":
        targets = np.array([-1.0, 1.0], dtype=np.complex64)
    elif constellation_type.upper() == "8PSK":
        angles = np.arange(8) * np.pi / 4.0
        targets = np.exp(1j * angles).astype(np.complex64)
    else:  # QPSK / QAM fallback
        targets = (np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2.0)).astype(np.complex64)
    
    # Take first 10,000 samples for EVM calculation
    y = np.asarray(samples[:10000], dtype=np.complex64)
    p_sig = np.mean(np.abs(y)**2)
    if p_sig > 0:
        y = y / np.sqrt(p_sig)
    
    dists = np.abs(y[:, None] - targets[None, :])
    min_dists = np.min(dists, axis=1)
    evm_rms = np.sqrt(np.mean(min_dists**2))
    return float(evm_rms * 100.0)

class SynchronizationFlowgraph(gr.top_block):
    def __init__(self, samples, sample_rate=32000, cfo_coarse_hz=0.0, mod_type="QPSK", samples_per_symbol=4):
        super(SynchronizationFlowgraph, self).__init__("SynchronizationFlowgraph")
        
        self.src = blocks.vector_source_c(samples, False)
        last_block = self.src

        # 1. Coarse CFO Rotator Block
        if cfo_coarse_hz != 0.0:
            phase_inc = -2.0 * np.pi * cfo_coarse_hz / float(sample_rate)
            self.rotator = blocks.rotator_cc(phase_inc)
            self.connect(last_block, self.rotator)
            last_block = self.rotator

        # 2. Costas Loop (Carrier Phase & Fine Frequency Lock)
        order = 2 if mod_type.upper() == "BPSK" else (8 if mod_type.upper() == "8PSK" else 4)
        loop_bw = 2.0 * np.pi / 100.0
        self.costas = digital.costas_loop_cc(loop_bw, order, False)
        self.connect(last_block, self.costas)
        last_block = self.costas

        # 3. Symbol Timing Synchronizer (Gardner TED)
        constellation = digital.constellation_qpsk().base()
        self.symbol_sync = digital.symbol_sync_cc(
            digital.TED_GARDNER,
            samples_per_symbol,
            0.045,
            1.0,
            1.0,
            1.5,
            1,
            constellation,
            digital.IR_MMSE_8TAP,
            128
        )
        self.connect(last_block, self.symbol_sync)
        last_block = self.symbol_sync

        self.sink = blocks.vector_sink_c()
        self.connect(last_block, self.sink)

    def run_sync(self):
        self.run()
        return np.array(self.sink.data(), dtype=np.complex64)

def synchronize_signal_flowgraph(samples, sample_rate=32000, mod_type="QPSK", samples_per_symbol=4):
    """
    Full synchronization pipeline: Power-of-N CFO coarse estimation + GNU Radio Costas & Symbol Sync flowgraph.
    """
    samples = np.asarray(samples, dtype=np.complex64)
    
    # 1. Coarse CFO estimation
    n_order = 2 if mod_type.upper() == "BPSK" else 4
    cfo_coarse = estimate_cfo_power_n(samples, sample_rate=sample_rate, n=n_order)
    
    # 2. GNU Radio Flowgraph Execution
    tb = SynchronizationFlowgraph(samples, sample_rate=sample_rate, cfo_coarse_hz=cfo_coarse, mod_type=mod_type, samples_per_symbol=samples_per_symbol)
    synced_samples = tb.run_sync()
    
    # 3. EVM and residual metrics
    evm_percent = calculate_evm(synced_samples, constellation_type=mod_type)
    
    return {
        "synced_samples": synced_samples,
        "estimated_cfo_hz": cfo_coarse,
        "phase_offset_deg": 0.0,
        "symbol_rate": sample_rate / samples_per_symbol,
        "evm_percent": evm_percent,
        "snr_post_sync": 10.0 * np.log10(max(1.0 / (evm_percent / 100.0)**2, 1.0))
    }
