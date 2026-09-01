"""
Wideband Spectrum Scanning & Channel Extraction Module using Native GNU Radio Blocks.
Scans arbitrary wideband SDR captures, detects active sub-channels, and extracts narrowband channels via Frequency Translating FIR Filter (DDC).
"""

from gnuradio import gr, blocks, filter
import numpy as np
from scipy import signal
import os
from gr_playground.utils.sigmf_io import write_sigmf

def scan_wideband_channels(samples, sample_rate=2.4e6, num_channels_max=5, min_power_db=-40.0, nperseg=4096):
    """
    Scans a wideband spectrum capture for active sub-channels / signal peaks.
    Returns a list of channel dictionaries containing frequency offset (Hz), power (dB), and estimated bandwidth (Hz).
    """
    samples = np.asarray(samples, dtype=np.complex64)
    if len(samples) < 512:
        return []

    # Compute Welch PSD
    freqs, psd = signal.welch(samples, fs=sample_rate, nperseg=min(len(samples), nperseg), return_onesided=False)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))
    df = abs(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

    # Minimum distance between channels (~50 kHz)
    min_dist_samples = max(1, int(50000.0 / df))
    peaks_idx, _ = signal.find_peaks(psd_db, height=min_power_db, distance=min_dist_samples)

    if len(peaks_idx) == 0:
        peaks_idx = np.array([np.argmax(psd_db)])

    # Sort peaks by power descending
    sorted_peaks = peaks_idx[np.argsort(psd_db[peaks_idx])[::-1]][:num_channels_max]
    
    channels = []

    for idx, pk in enumerate(sorted_peaks, 1):
        peak_freq = float(freqs[pk])
        peak_pwr = float(psd_db[pk])

        # Estimate local bandwidth around peak (power drops by 10 dB)
        pk_val = psd[pk]
        threshold = pk_val * 0.1
        left = pk
        while left > 0 and psd[left] > threshold:
            left -= 1
        right = pk
        while right < len(psd) - 1 and psd[right] > threshold:
            right += 1
        
        bw = abs(freqs[right] - freqs[left])
        bw = max(bw, 10000.0)  # Minimum 10 kHz floor

        channels.append({
            "channel_id": idx,
            "freq_offset_hz": peak_freq,
            "power_db": peak_pwr,
            "bandwidth_hz": float(bw)
        })

    # Re-sort by frequency offset ascending
    channels.sort(key=lambda x: x["freq_offset_hz"])
    return channels

class ChannelizerFlowgraph(gr.top_block):
    """
    GNU Radio top_block performing Digital Downconversion (DDC):
    Translates center frequency offset to 0 Hz, applies lowpass FIR filter, and decimates.
    """
    def __init__(self, samples, sample_rate=2.4e6, freq_offset_hz=0.0, target_bw_hz=100000.0, decimation=10):
        super(ChannelizerFlowgraph, self).__init__("ChannelizerFlowgraph")

        self.src = blocks.vector_source_c(samples.tolist(), False)

        # Calculate lowpass FIR taps for frequency translating filter
        cutoff_hz = target_bw_hz / 2.0
        transition_bw = cutoff_hz * 0.2
        taps = filter.firdes.low_pass(
            1.0,               # Gain
            sample_rate,       # Input sample rate
            cutoff_hz,         # Cutoff frequency
            transition_bw      # Transition width
        )

        # Frequency Translating FIR Filter
        self.xlating_filter = filter.freq_xlating_fir_filter_ccc(
            decimation,
            taps,
            freq_offset_hz,
            sample_rate
        )

        self.sink = blocks.vector_sink_c()
        self.connect(self.src, self.xlating_filter, self.sink)

    def run_channelizer(self):
        self.run()
        return np.array(self.sink.data(), dtype=np.complex64)

def extract_channel_flowgraph(samples, sample_rate=2.4e6, freq_offset_hz=0.0, target_bw_hz=100000.0, decimation=10, center_freq=100.0e6, output_sigmf_path=None):
    """
    Extracts a narrowband channel from a wideband capture using GNU Radio DDC flowgraph.
    Returns (narrowband_samples, meta_dict). If output_sigmf_path is provided, writes to SigMF.
    """
    samples = np.asarray(samples, dtype=np.complex64)
    tb = ChannelizerFlowgraph(
        samples,
        sample_rate=sample_rate,
        freq_offset_hz=freq_offset_hz,
        target_bw_hz=target_bw_hz,
        decimation=decimation
    )
    narrowband_samples = tb.run_channelizer()

    output_rate = float(sample_rate) / float(decimation)
    output_center_freq = float(center_freq) + float(freq_offset_hz)

    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": output_rate,
            "core:description": f"Extracted narrowband channel at offset {freq_offset_hz:+.1f} Hz",
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": output_center_freq
            }
        ]
    }

    if output_sigmf_path:
        write_sigmf(
            output_sigmf_path,
            narrowband_samples,
            sample_rate=output_rate,
            center_freq=output_center_freq,
            description=f"Extracted channel from wideband capture ({sample_rate/1e6:.2f} MSPS -> {output_rate/1e3:.1f} kSPS)"
        )

    return narrowband_samples, meta
