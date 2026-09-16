"""
Wideband Spectrum Scanning & Channel Extraction Module using Native GNU Radio Blocks.
Scans arbitrary wideband SDR captures, detects active sub-channels, and extracts narrowband channels via Frequency Translating FIR Filter (DDC).
"""

from gnuradio import gr, blocks, filter
import numpy as np
from scipy import signal, ndimage
import os
from gr_playground.utils.sigmf_io import write_sigmf

def scan_wideband_channels(samples, sample_rate=2.4e6, num_channels_max=10, min_snr_db=0.5, nperseg=32768, target_slice_duration_sec=0.5, target_channel_bw=100000.0, reject_spurs=True):
    """
    Scans a wideband spectrum capture for active signal channels using:
      1. High-resolution FFT (nperseg=32768 -> 73.2 Hz bin resolution, +9 dB processing gain)
      2. Invariant temporal slice duration averaging (target_slice_duration_sec=0.5s) ensuring constant RAM bounds and burst sensitivity
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
    
    # 1. Slice-duration-based Temporal Ensemble Averaging (~0.5s per slice invariant to total capture duration)
    target_slice_len = max(actual_nfft * 2, int(sample_rate * target_slice_duration_sec))
    n_slices = max(1, int(round(n_samples / float(target_slice_len))))
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

    # 2. Adaptive Rolling Baseline Noise Floor Estimation (200 kHz median window, capped for performance)
    med_win = max(3, int(200000.0 / df))
    max_win = min(3001, max(3, len(psd_db) // 5))
    if max_win % 2 == 0:
        max_win -= 1
    med_win = min(med_win, max_win)
    if med_win % 2 == 0:
        med_win += 1
    med_win = max(3, med_win)

    noise_floor_db = signal.medfilt(psd_db, med_win) if len(psd_db) >= med_win else np.full_like(psd_db, np.median(psd_db))

    # Local SNR in dB above adaptive noise floor
    snr_db = psd_db - noise_floor_db

    # 3. Integrated Band Energy (Convolution with target channel bandwidth kernel)
    target_bw = min(target_channel_bw, float(sample_rate) * 0.8)
    win_samples = max(1, int(target_bw / df))
    kernel = np.ones(win_samples) / win_samples
    band_psd = signal.convolve(psd_avg, kernel, mode='same')
    band_psd_db = 10.0 * np.log10(np.maximum(band_psd, 1e-12))
    band_snr_db = band_psd_db - noise_floor_db

    # 4. Detect candidate channel peaks on Local SNR (exact frequency bin localization)
    min_dist_hz = max(15000.0, min(target_channel_bw * 0.5, float(sample_rate) * 0.35))
    min_dist_samples = max(1, int(min_dist_hz / df))
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

        # Estimate local occupied bandwidth around peak using dynamic smoothing window and lookahead gap tolerance
        smooth_win_bins = max(5, int(4000.0 / df))
        if smooth_win_bins % 2 == 0:
            smooth_win_bins += 1
        psd_smooth = signal.convolve(psd_db, np.ones(smooth_win_bins) / float(smooth_win_bins), mode='same')
        bw_thresh_db = noise_floor_db[pk] + 3.0
        lookahead_bins = max(3, int(8000.0 / df))

        # Search left boundary with gap tolerance for subcarrier nulls
        l = pk
        consecutive_below = 0
        first_below_l = l
        while l > 0:
            if psd_smooth[l] <= bw_thresh_db:
                if consecutive_below == 0:
                    first_below_l = l
                consecutive_below += 1
                if consecutive_below >= lookahead_bins:
                    l = min(pk, first_below_l + 1)
                    break
            else:
                consecutive_below = 0
            l -= 1
        l = max(0, l)

        # Search right boundary with gap tolerance for subcarrier nulls
        r = pk
        consecutive_below = 0
        first_below_r = r
        while r < len(psd_smooth) - 1:
            if psd_smooth[r] <= bw_thresh_db:
                if consecutive_below == 0:
                    first_below_r = r
                consecutive_below += 1
                if consecutive_below >= lookahead_bins:
                    r = max(pk, first_below_r - 1)
                    break
            else:
                consecutive_below = 0
            r += 1
        r = min(len(psd_smooth) - 1, r)

        bw = abs(freqs[r] - freqs[l])
        chan_center_freq = float(0.5 * (freqs[l] + freqs[r]))

        # Classify channel type based on occupied bandwidth
        if bw >= 40000.0:
            ch_type = "Wideband Channel"
            final_center_freq = chan_center_freq
        elif bw >= 4000.0 or sample_rate <= 100000.0:
            ch_type = "Narrowband Signal"
            final_center_freq = chan_center_freq
        else:
            ch_type = "Narrow Spur / CW Tone"
            final_center_freq = peak_freq

        raw_channels.append({
            "freq_offset_hz": final_center_freq,
            "power_db": band_pwr,
            "peak_single_bin_db": peak_pwr,
            "local_snr_db": max(local_snr, band_snr),
            "bandwidth_hz": float(max(bw, df * 2.0)),
            "channel_type": ch_type
        })

    # Deduplicate / merge adjacent channel peaks that belong to the same wideband signal
    merged_channels = []
    # Sort by power/SNR descending first so main channel center takes precedence over sidebands
    raw_channels.sort(key=lambda x: x["local_snr_db"], reverse=True)
    for ch in raw_channels:
        is_duplicate = False
        for existing in merged_channels:
            freq_diff = abs(ch["freq_offset_hz"] - existing["freq_offset_hz"])
            min_sep = max(10000.0, min(100000.0, min(ch["bandwidth_hz"], existing["bandwidth_hz"]) * 0.25))
            if freq_diff < min_sep:
                is_duplicate = True
                # Merge bandwidth
                existing["bandwidth_hz"] = max(existing["bandwidth_hz"], ch["bandwidth_hz"])
                break
        if not is_duplicate:
            merged_channels.append(ch)

    raw_channels = merged_channels

    # Sort channels by integrated local SNR descending
    raw_channels.sort(key=lambda x: x["local_snr_db"], reverse=True)

    if reject_spurs:
        # Separate wideband/narrowband communication channels from single CW tones
        comm_channels = [c for c in raw_channels if c["channel_type"] != "Narrow Spur / CW Tone"]
        spur_channels = [c for c in raw_channels if c["channel_type"] == "Narrow Spur / CW Tone"]
        sorted_channels = comm_channels + spur_channels
    else:
        sorted_channels = raw_channels

    selected = sorted_channels[:num_channels_max]

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
