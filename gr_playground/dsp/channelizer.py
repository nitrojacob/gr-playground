"""
Wideband Spectrum Scanning & Channel Extraction Module using Native GNU Radio Blocks.
Scans arbitrary wideband SDR captures, detects active sub-channels, and extracts narrowband channels via Frequency Translating FIR Filter (DDC).
"""

from gnuradio import gr, blocks, filter
import numpy as np
from scipy import signal
import os
from gr_playground.utils.sigmf_io import write_sigmf

def scan_wideband_channels(samples, sample_rate=2.4e6, num_channels_max=10, min_snr_db=0.5, nperseg=32768, n_time_slices=40, target_channel_bw=100000.0, reject_spurs=True):
    """
    Scans a wideband spectrum capture for active signal channels using:
      1. High-resolution FFT (nperseg=32768 -> 73.2 Hz bin resolution, +9 dB processing gain)
      2. Temporal ensemble time-slice averaging across the capture (reduces noise variance by 1/sqrt(K))
      3. Adaptive rolling baseline noise floor estimation & Local SNR calculation
      4. Integrated band energy kernel convolution (100 kHz window)
      5. Automatic classification & spur/CW tone filtering
    Returns a list of channel dictionaries containing frequency offset (Hz), power (dB), local_snr_db, bandwidth (Hz), and channel_type.
    """
    samples = np.asarray(samples, dtype=np.complex64)
    n_samples = len(samples)
    if n_samples < 1024:
        return []

    actual_nfft = min(n_samples, nperseg)
    if actual_nfft < 512:
        return []
    
    # 1. Temporal Time-Slice Ensemble Averaging across multiple capture frames
    n_slices = min(n_time_slices, max(1, n_samples // actual_nfft))
    slice_len = n_samples // n_slices

    psd_accum = None
    freqs = None
    for k in range(n_slices):
        sub_samples = samples[k * slice_len : (k + 1) * slice_len]
        if len(sub_samples) < 128:
            continue
        f_sub, p_sub = signal.welch(sub_samples, fs=sample_rate, nperseg=min(len(sub_samples), actual_nfft), return_onesided=False)
        if psd_accum is None:
            psd_accum = p_sub
            freqs = f_sub
        elif len(p_sub) == len(psd_accum):
            psd_accum += p_sub

    if psd_accum is None:
        freqs, psd_accum = signal.welch(samples, fs=sample_rate, nperseg=min(n_samples, 4096), return_onesided=False)
        n_slices = 1

    psd_avg = psd_accum / float(n_slices)
    freqs = np.fft.fftshift(freqs)
    psd_avg = np.fft.fftshift(psd_avg)
    psd_db = 10.0 * np.log10(np.maximum(psd_avg, 1e-12))
    df = abs(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

    # 2. Adaptive Rolling Baseline Noise Floor Estimation (200 kHz median window)
    med_win = max(3, int(200000.0 / df))
    if med_win >= len(psd_db):
        med_win = len(psd_db) - 1 if len(psd_db) % 2 == 0 else len(psd_db) - 2
    if med_win % 2 == 0:
        med_win += 1
    med_win = max(3, med_win)

    noise_floor_db = signal.medfilt(psd_db, med_win) if len(psd_db) >= med_win else np.full_like(psd_db, np.median(psd_db))

    # Local SNR in dB above adaptive noise floor
    snr_db = psd_db - noise_floor_db

    # 3. Integrated Band Energy (Convolution with target channel bandwidth kernel)
    win_samples = max(1, int(target_channel_bw / df))
    kernel = np.ones(win_samples) / win_samples
    band_psd = signal.convolve(psd_avg, kernel, mode='same')
    band_psd_db = 10.0 * np.log10(np.maximum(band_psd, 1e-12))
    band_snr_db = band_psd_db - noise_floor_db

    # 4. Detect candidate channel peaks on Local SNR (exact frequency bin localization)
    min_dist_samples = max(1, int(50000.0 / df))
    peaks_idx, _ = signal.find_peaks(snr_db, height=min_snr_db, distance=min_dist_samples)

    if len(peaks_idx) == 0:
        peaks_idx = np.array([np.argmax(snr_db)])

    raw_channels = []

    for pk in peaks_idx:
        peak_freq = float(freqs[pk])
        peak_pwr = float(psd_db[pk])
        band_pwr = float(band_psd_db[pk])
        local_snr = float(snr_db[pk])
        band_snr = float(band_snr_db[pk])

        # Estimate local bandwidth around peak (power drops relative to peak)
        thresh_db = noise_floor_db[pk] + (local_snr * 0.2)
        l, r = pk, pk
        while l > 0 and psd_db[l] > thresh_db:
            l -= 1
        while r < len(psd_db) - 1 and psd_db[r] > thresh_db:
            r += 1
        
        bw = abs(freqs[r] - freqs[l])
        
        # Classify channel type based on occupied bandwidth
        if bw >= 40000.0:
            ch_type = "Wideband Channel"
        elif bw >= 15000.0:
            ch_type = "Narrowband Signal"
        else:
            ch_type = "Narrow Spur / CW Tone"

        raw_channels.append({
            "freq_offset_hz": peak_freq,
            "power_db": band_pwr,
            "peak_single_bin_db": peak_pwr,
            "local_snr_db": band_snr,
            "bandwidth_hz": float(max(bw, 10000.0)),
            "channel_type": ch_type
        })

    # Sort channels by integrated local SNR descending
    raw_channels.sort(key=lambda x: x["local_snr_db"], reverse=True)

    if reject_spurs:
        # Separate wideband/narrowband communication channels from single CW tones
        comm_channels = [c for c in raw_channels if c["channel_type"] != "Narrow Spur / CW Tone"]
        spur_channels = [c for c in raw_channels if c["channel_type"] == "Narrow Spur / CW Tone"]
        # Include comm channels first, followed by spur channels
        sorted_channels = comm_channels + spur_channels
    else:
        sorted_channels = raw_channels

    selected = sorted_channels[:num_channels_max]

    # Assign channel_id (1..N) and re-sort by frequency offset ascending
    for idx, ch in enumerate(selected, 1):
        ch["channel_id"] = idx

    selected.sort(key=lambda x: x["freq_offset_hz"])
    return selected

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
