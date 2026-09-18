"""
Point Peak Detection (PPD) Heuristic Channel Detector Strategy.
Encapsulates 1D scipy.signal.find_peaks searching and threshold merging.
"""

import numpy as np
from scipy import signal
from gr_playground.dsp.channel_detection.base import BaseChannelDetector


class PPDHeuristicChannelDetector(BaseChannelDetector):
    """
    Point Peak Detection (PPD) Heuristic Channel Detector.
    Searches for candidate channel peaks on combined PSD prominence and integrated band power.
    """

    def detect_channels(
        self,
        iq_data: np.ndarray,
        sample_rate: float,
        psd_db: np.ndarray = None,
        freqs: np.ndarray = None,
        target_channel_bw: float = 100000.0,
        min_snr_db: float = 3.0,
        num_channels_max: int = 10,
        reject_spurs: bool = True,
        center_freq: float = 0.0,
        session_id: str = None,
        reset_state: bool = False,
        **kwargs
    ) -> list[dict]:
        if psd_db is None or freqs is None:
            nperseg = min(32768, len(iq_data))
            freqs, psd = signal.welch(iq_data, fs=sample_rate, nperseg=nperseg, window='blackmanharris', return_onesided=False)
            freqs = np.fft.fftshift(freqs)
            psd = np.fft.fftshift(psd)
            psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))

        df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        psd_avg = 10.0 ** (psd_db / 10.0)

        med_win = max(3, int(200000.0 / df))
        max_win = min(3001, max(3, len(psd_db) // 5))
        if max_win % 2 == 0:
            max_win -= 1
        med_win = min(med_win, max_win)
        if med_win % 2 == 0:
            med_win += 1
        med_win = max(3, med_win)

        noise_floor_db = signal.medfilt(psd_db, med_win) if len(psd_db) >= med_win else np.full_like(psd_db, np.median(psd_db))
        global_noise_floor_db = float(np.median(np.sort(psd_db)[:len(psd_db)//2]))
        noise_floor_db = np.minimum(noise_floor_db, global_noise_floor_db + 4.0)

        snr_db = psd_db - noise_floor_db

        target_bw = min(target_channel_bw, float(sample_rate) * 0.8)
        win_samples = max(1, int(target_bw / df))
        kernel = np.ones(win_samples) / win_samples
        band_psd = signal.convolve(psd_avg, kernel, mode='same')
        band_psd_db = 10.0 * np.log10(np.maximum(band_psd, 1e-12))
        band_snr_db = band_psd_db - noise_floor_db

        combined_snr_db = np.maximum(snr_db, band_snr_db)

        smooth_win_bins = max(5, int(3000.0 / df))
        if smooth_win_bins % 2 == 0:
            smooth_win_bins += 1
        psd_smooth = signal.convolve(psd_db, np.ones(smooth_win_bins) / float(smooth_win_bins), mode='same')

        min_dist_hz = max(20000.0, min(50000.0, target_channel_bw * 0.35))
        min_dist_samples = max(1, int(min_dist_hz / df))
        
        peak_min_height = max(1.5, min_snr_db - 1.5)
        peaks_idx, _ = signal.find_peaks(combined_snr_db, height=peak_min_height, distance=min_dist_samples)

        if len(peaks_idx) == 0:
            peaks_idx = np.array([np.argmax(snr_db)])

        raw_channels = []

        for pk in peaks_idx:
            peak_freq = float(freqs[pk])
            peak_pwr = float(psd_db[pk])
            band_pwr = float(band_psd_db[pk])
            local_snr = float(snr_db[pk])
            band_snr = float(band_snr_db[pk])

            peak_val = psd_smooth[pk]
            noise_val = noise_floor_db[pk]
            bw_thresh_db = max(noise_val + 3.0, peak_val - 18.0)
            lookahead_bins = max(5, int(15000.0 / df))

            l = pk
            consecutive_below = 0
            first_below_l = None
            while l > 0:
                if psd_smooth[l] <= bw_thresh_db:
                    if first_below_l is None:
                        first_below_l = l
                    consecutive_below += 1
                    if consecutive_below >= lookahead_bins:
                        break
                else:
                    first_below_l = None
                    consecutive_below = 0
                l -= 1
            l = first_below_l if first_below_l is not None else max(0, l)

            r = pk
            consecutive_below = 0
            first_below_r = None
            while r < len(psd_smooth) - 1:
                if psd_smooth[r] <= bw_thresh_db:
                    if first_below_r is None:
                        first_below_r = r
                    consecutive_below += 1
                    if consecutive_below >= lookahead_bins:
                        break
                else:
                    first_below_r = None
                    consecutive_below = 0
                r += 1
            r = first_below_r if first_below_r is not None else min(len(psd_smooth) - 1, r)

            bw = abs(freqs[r] - freqs[l])

            if (peak_pwr < band_pwr - 6.0) and (local_snr < 6.0):
                continue

            win_30k = max(2, int(15000.0 / df))
            local_window_pwr = np.sum(psd_avg[max(0, pk - win_30k):min(len(psd_avg), pk + win_30k + 1)])
            bin_pwr_ratio = float(psd_avg[pk]) / float(max(1e-12, local_window_pwr))
            diff_db = peak_pwr - band_pwr
            is_single_bin_spur = (bin_pwr_ratio >= 0.50) and (diff_db >= 8.0)

            if is_single_bin_spur:
                ch_type = "Narrow Spur / CW Tone"
            elif bw >= 35000.0:
                ch_type = "Wideband Channel"
            else:
                ch_type = "Narrowband Signal"

            raw_channels.append({
                "freq_offset_hz": peak_freq + center_freq,
                "power_db": band_pwr,
                "peak_single_bin_db": peak_pwr,
                "local_snr_db": local_snr if ch_type == "Narrow Spur / CW Tone" else band_snr,
                "bandwidth_hz": float(max(bw, df * 2.0)) if ch_type != "Narrow Spur / CW Tone" else float(min(max(bw, df * 2.0), 12500.0)),
                "channel_type": ch_type,
                "confidence": float(min(1.0, max(0.1, (band_snr if ch_type != "Narrow Spur / CW Tone" else local_snr) / 20.0))),
                "bounds_hz": (float(freqs[l] + center_freq), float(freqs[r] + center_freq)),
            })

        for ch in raw_channels:
            ch["orig_freq"] = ch["freq_offset_hz"]
            ch["orig_bw"] = ch["bandwidth_hz"]

        raw_channels.sort(key=lambda x: x["local_snr_db"], reverse=True)
        merged_channels = []
        for ch in raw_channels:
            is_duplicate = False
            ch_bw = max(ch["bandwidth_hz"], ch["orig_bw"])
            ch_low = ch["freq_offset_hz"] - ch_bw * 0.5
            ch_high = ch["freq_offset_hz"] + ch_bw * 0.5

            for existing in merged_channels:
                ex_bw = max(existing["bandwidth_hz"], existing["orig_bw"])
                ex_low = existing["freq_offset_hz"] - ex_bw * 0.5
                ex_high = existing["freq_offset_hz"] + ex_bw * 0.5

                freq_diff = abs(ch["freq_offset_hz"] - existing["freq_offset_hz"])
                max_bw = max(ch_bw, ex_bw)
                edge_gap = max(0.0, max(ch_low - ex_high, ex_low - ch_high))

                if ch["channel_type"] == "Narrow Spur / CW Tone" or existing["channel_type"] == "Narrow Spur / CW Tone":
                    should_merge = (ch["channel_type"] == "Narrow Spur / CW Tone" and existing["channel_type"] == "Narrow Spur / CW Tone" and freq_diff < 10000.0)
                else:
                    should_merge = (edge_gap <= 25000.0 or freq_diff < max(120000.0, max_bw * 1.2))

                if should_merge:
                    is_duplicate = True
                    new_low = min(ch_low, ex_low)
                    new_high = max(ch_high, ex_high)
                    new_bw = new_high - new_low

                    if ch["power_db"] > existing["power_db"]:
                        existing["freq_offset_hz"] = ch["orig_freq"]
                        existing["power_db"] = ch["power_db"]

                    if edge_gap == 0.0:
                        existing["bandwidth_hz"] = max(ch_bw, ex_bw)
                    else:
                        existing["bandwidth_hz"] = new_bw

                    if existing["channel_type"] != "Narrow Spur / CW Tone":
                        if existing["bandwidth_hz"] >= 35000.0:
                            existing["channel_type"] = "Wideband Channel"
                        elif existing["bandwidth_hz"] >= 12500.0:
                            existing["channel_type"] = "Narrowband Signal"
                    break

            if not is_duplicate:
                merged_channels.append(ch)
        raw_channels = merged_channels

        raw_channels.sort(key=lambda x: x["local_snr_db"], reverse=True)

        if reject_spurs:
            comm_channels = [c for c in raw_channels if c["channel_type"] != "Narrow Spur / CW Tone" and c["local_snr_db"] >= min_snr_db]
            strong_spurs = [c for c in raw_channels if c["channel_type"] == "Narrow Spur / CW Tone" and c["local_snr_db"] >= 15.0]
            weak_spurs = [c for c in raw_channels if c["channel_type"] == "Narrow Spur / CW Tone" and c["local_snr_db"] < 15.0]
            valid_channels = sorted(comm_channels + strong_spurs, key=lambda x: x["local_snr_db"], reverse=True)
            if len(valid_channels) > 0:
                sorted_channels = valid_channels
            else:
                sorted_channels = weak_spurs
        else:
            sorted_channels = raw_channels

        return sorted_channels[:num_channels_max]
