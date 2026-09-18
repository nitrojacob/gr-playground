"""
Wideband Spectrum Scanning & Channel Extraction Module using Native GNU Radio Blocks.
Scans arbitrary wideband SDR captures, detects active sub-channels, and extracts narrowband channels via Frequency Translating FIR Filter (DDC).
"""

from gnuradio import gr, blocks, filter
import numpy as np
from scipy import signal, ndimage
import os
from gr_playground.utils.sigmf_io import write_sigmf

def scan_wideband_channels(
    samples,
    sample_rate=2.4e6,
    num_channels_max=10,
    min_snr_db=0.5,
    nperseg=32768,
    target_slice_duration_sec=0.5,
    target_channel_bw=100000.0,
    reject_spurs=True,
    detection_mode="cfar_heuristic",
    channel_detector=None,
    session_id=None,
    reset_state=False
):
    """
    Scans a wideband spectrum capture for active signal channels using modular Channel Detection strategies.
    
    Parameters:
    - detection_mode: Channel detection strategy ("cfar_heuristic", "ppd_heuristic"). Default: "cfar_heuristic".
    - channel_detector: Optional BaseChannelDetector instance override.
    
    Returns a list of channel dictionaries containing frequency offset (Hz), power (dB), local_snr_db, bandwidth (Hz), and channel_type.
    """
    samples = np.asarray(samples, dtype=np.complex64)
    n_samples = len(samples)
    if n_samples < 1024:
        return []

    actual_nfft = min(n_samples, nperseg)
    if actual_nfft < 512:
        return []

    # 1. Temporal Ensemble Averaging (~0.5s per slice)
    target_slice_len = max(actual_nfft * 2, int(sample_rate * target_slice_duration_sec))
    n_slices = max(1, int(round(n_samples / float(target_slice_len))))
    slice_len = n_samples // n_slices

    psd_accum = None
    freqs = None
    for k in range(n_slices):
        sub_samples = samples[k * slice_len : (k + 1) * slice_len]
        if len(sub_samples) < 128:
            continue
        f_sub, p_sub = signal.welch(sub_samples, fs=sample_rate, nperseg=min(len(sub_samples), actual_nfft), window='blackmanharris', return_onesided=False)
        if psd_accum is None:
            psd_accum = p_sub
            freqs = f_sub
        elif len(p_sub) == len(psd_accum):
            psd_accum += p_sub

    if psd_accum is None:
        freqs, psd_accum = signal.welch(samples, fs=sample_rate, nperseg=min(n_samples, 4096), window='blackmanharris', return_onesided=False)
        n_slices = 1

    psd_avg = psd_accum / float(n_slices)
    freqs = np.fft.fftshift(freqs)
    psd_avg = np.fft.fftshift(psd_avg)
    psd_db = 10.0 * np.log10(np.maximum(psd_avg, 1e-12))

    # 2. Delegate channel detection to Modular Strategy Engine
    from gr_playground.dsp.channel_detection import get_channel_detector, BaseChannelDetector

    if channel_detector is None:
        detector = get_channel_detector(mode=detection_mode)
    else:
        detector = channel_detector

    selected = detector.detect_channels(
        iq_data=samples,
        sample_rate=sample_rate,
        psd_db=psd_db,
        freqs=freqs,
        target_channel_bw=target_channel_bw,
        min_snr_db=min_snr_db,
        num_channels_max=num_channels_max,
        reject_spurs=reject_spurs,
        session_id=session_id,
        reset_state=reset_state
    )

    for idx, ch in enumerate(selected, 1):
        ch["channel_id"] = idx

    return selected

class ChannelizerFlowgraph(gr.top_block):
    """
    GNU Radio top_block performing Digital Downconversion (DDC):
    Translates center frequency offset to 0 Hz, applies lowpass FIR filter, and decimates.
    """
    def __init__(self, samples, sample_rate=2.4e6, freq_offset_hz=0.0, target_bw_hz=100000.0, decimation=10):
        super(ChannelizerFlowgraph, self).__init__("ChannelizerFlowgraph")

        if isinstance(samples, np.ndarray):
            self.src = blocks.vector_source_c(samples, False)
        else:
            self.src = blocks.vector_source_c(samples, False)

        # Calculate lowpass FIR taps accommodating target_bw_hz without aliasing distortion
        out_rate = float(sample_rate) / float(decimation)
        max_safe_cutoff = (out_rate / 2.0) * 0.90
        cutoff_hz = min(target_bw_hz / 2.0, max_safe_cutoff)
        cutoff_hz = max(cutoff_hz, 5000.0)
        transition_bw = max(cutoff_hz * 0.15, 2000.0)

        taps = filter.firdes.low_pass(
            1.0,               # Gain
            sample_rate,       # Input sample rate
            cutoff_hz,         # Cutoff frequency
            transition_bw      # Transition width
        )

        # Frequency Translating FIR Filter
        self.xlating_filter = filter.freq_xlating_fir_filter_ccc(
            int(decimation),
            taps,
            freq_offset_hz,
            sample_rate
        )

        self.sink = blocks.vector_sink_c()
        self.connect(self.src, self.xlating_filter, self.sink)

    def run_channelizer(self):
        self.run()
        return np.array(self.sink.data(), dtype=np.complex64)

def extract_channel_flowgraph(samples, sample_rate=2.4e6, freq_offset_hz=0.0, target_bw_hz=100000.0, decimation=None, center_freq=100.0e6, output_sigmf_path=None):
    """
    Extracts a narrowband/wideband channel from a wideband capture using GNU Radio DDC flowgraph or fast chunked FIR DDC.
    Supports adaptive decimation calculation when decimation is None or 'auto'.
    Returns (narrowband_samples, meta_dict). If output_sigmf_path is provided, writes to SigMF.
    """
    samples = np.asarray(samples, dtype=np.complex64)

    if decimation is None or str(decimation).lower() == "auto":
        desired_rate = max(target_bw_hz * 1.25, 40000.0)
        decimation = max(1, int(round(float(sample_rate) / float(desired_rate))))

    decimation = int(decimation)

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
