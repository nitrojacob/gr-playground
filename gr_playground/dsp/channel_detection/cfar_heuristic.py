"""
Cell-Averaging Constant False Alarm Rate (CA-CFAR) & 99% OBW Heuristic Channel Detector.
Segments contiguous connected energy regions above local CFAR noise floor and measures
99% Occupied Bandwidth (OBW) and energy concentration.
"""

import numpy as np
from scipy import signal
from gr_playground.dsp.channel_detection.base import BaseChannelDetector


class CFARHeuristicChannelDetector(BaseChannelDetector):
    """
    Cell-Averaging / Order-Statistic CFAR & 99% Occupied Bandwidth Channel Detector.
    Default high-performance channel identification engine.
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
        psd_linear = 10.0 ** (psd_db / 10.0)
        n_bins = len(psd_db)

        # 1. Order-Statistic / Median CFAR Noise Floor (dB domain, immune to CW outlier tone masking)
        ref_win_bins = max(11, int(200000.0 / df))
        if ref_win_bins % 2 == 0:
            ref_win_bins += 1

        cfar_noise_db = signal.medfilt(psd_db, ref_win_bins) if n_bins >= ref_win_bins else np.full_like(psd_db, np.median(psd_db))
        noise_floor_db = cfar_noise_db

        snr_db = psd_db - noise_floor_db

        # Integrated band PSD for overall band prominence
        target_bw = min(target_channel_bw, float(sample_rate) * 0.8)
        win_samples = max(1, int(target_bw / df))
        kernel = np.ones(win_samples) / win_samples
        band_psd = signal.convolve(psd_linear, kernel, mode='same')
        band_psd_db = 10.0 * np.log10(np.maximum(band_psd, 1e-12))
        band_snr_db = band_psd_db - noise_floor_db

        # 2. Smooth PSD for robust region segmentation (3 kHz window, suppresses noise variance)
        smooth_win_bins = max(3, int(3000.0 / df))
        if smooth_win_bins % 2 == 0:
            smooth_win_bins += 1
        psd_smooth = signal.convolve(psd_db, np.ones(smooth_win_bins) / float(smooth_win_bins), mode='same')
        snr_smooth_db = psd_smooth - noise_floor_db
        combined_snr_db = np.maximum(snr_smooth_db, band_snr_db)

        # 2. Segment Contiguous Energy Regions Above CFAR Threshold
        cfar_thresh_db = max(2.0, min_snr_db - 0.5)
        active_mask = combined_snr_db >= cfar_thresh_db

        # Group contiguous active bins into regions
        regions = []
        in_region = False
        start_idx = 0

        for i in range(n_bins):
            if active_mask[i] and not in_region:
                in_region = True
                start_idx = i
            elif not active_mask[i] and in_region:
                in_region = False
                end_idx = i - 1
                regions.append((start_idx, end_idx))
        if in_region:
            regions.append((start_idx, n_bins - 1))

        # Bridge small gaps (< 15 kHz) between adjacent connected regions
        max_gap_bins = max(1, int(15000.0 / df))
        merged_regions = []
        for reg in regions:
            if not merged_regions:
                merged_regions.append(reg)
            else:
                last_start, last_end = merged_regions[-1]
                if reg[0] - last_end <= max_gap_bins:
                    merged_regions[-1] = (last_start, max(last_end, reg[1]))
                else:
                    merged_regions.append(reg)

        # Multi-Peak Watershed Decomposition: Split regions containing distinct local peaks separated by valleys
        final_sub_regions = []
        min_dist_bins = max(2, int(25000.0 / df))

        for l_idx, r_idx in merged_regions:
            reg_len = r_idx - l_idx + 1
            if reg_len < 3:
                final_sub_regions.append((l_idx, r_idx))
                continue

            reg_psd_smooth = psd_smooth[l_idx:r_idx + 1]
            pks_local, _ = signal.find_peaks(reg_psd_smooth, prominence=1.5, distance=min_dist_bins)

            if len(pks_local) <= 1:
                final_sub_regions.append((l_idx, r_idx))
            else:
                # Split region ONLY at deep PSD valleys (>= 12.0 dB depth or near noise floor) between adjacent local peaks
                sub_start = l_idx
                for k in range(len(pks_local) - 1):
                    p1 = pks_local[k]
                    p2 = pks_local[k + 1]
                    p1_val = reg_psd_smooth[p1]
                    p2_val = reg_psd_smooth[p2]
                    v_rel = np.argmin(reg_psd_smooth[p1:p2 + 1])
                    v_val = reg_psd_smooth[p1 + v_rel]
                    valley_global = l_idx + p1 + v_rel
                    v_depth_max = max(p1_val, p2_val) - v_val
                    v_depth_min = min(p1_val, p2_val) - v_val
                    local_noise = noise_floor_db[valley_global]

                    should_split = (v_depth_max >= 9.0) and (v_depth_min >= 4.0 or v_val <= local_noise + 6.0)

                    if should_split:
                        final_sub_regions.append((sub_start, valley_global))
                        sub_start = valley_global + 1

                final_sub_regions.append((sub_start, r_idx))

        # 3. Analyze each sub-region for 99% OBW, power, and CW spur concentration
        raw_channels = []
        win_30k_bins = max(2, int(15000.0 / df))

        for l_idx, r_idx in final_sub_regions:
            if l_idx > r_idx:
                continue
            reg_psd_lin = psd_linear[l_idx:r_idx + 1]
            reg_psd_db = psd_db[l_idx:r_idx + 1]
            reg_freqs = freqs[l_idx:r_idx + 1]

            if len(reg_psd_lin) == 0:
                continue

            pk_local = np.argmax(reg_psd_db)
            pk_global = l_idx + pk_local
            peak_freq = float(freqs[pk_global])
            peak_pwr = float(psd_db[pk_global])
            band_pwr = float(band_psd_db[pk_global])
            local_snr = float(snr_db[pk_global])
            band_snr = float(band_snr_db[pk_global])

            # 99.8% Occupied Bandwidth (OBW) within connected region (0.1% to 99.9% energy footprint)
            tot_reg_pwr = np.sum(reg_psd_lin)
            if tot_reg_pwr > 0:
                cum_pwr = np.cumsum(reg_psd_lin) / tot_reg_pwr
                idx_low = np.searchsorted(cum_pwr, 0.001)
                idx_high = np.searchsorted(cum_pwr, 0.999)
                idx_high = min(len(reg_psd_lin) - 1, max(idx_low, idx_high))
                obw_hz = float(reg_freqs[idx_high] - reg_freqs[idx_low])
                obw_hz = max(obw_hz, df * 2.0)
            else:
                obw_hz = float(reg_freqs[-1] - reg_freqs[0])

            # Tone energy ratio: mainlobe (peak ± 8 bins ~ 1.2 kHz) energy vs 30 kHz local neighborhood energy
            mainlobe_pwr = np.sum(psd_linear[max(0, pk_global - 8):min(n_bins, pk_global + 9)])
            local_window_pwr = np.sum(psd_linear[max(0, pk_global - win_30k_bins):min(n_bins, pk_global + win_30k_bins + 1)])
            tone_pwr_ratio = float(mainlobe_pwr) / float(max(1e-12, local_window_pwr))
            diff_db = peak_pwr - band_pwr

            is_single_bin_spur = (tone_pwr_ratio >= 0.50) or (local_snr >= 20.0 and tone_pwr_ratio >= 0.35)

            if is_single_bin_spur:
                ch_type = "Narrow Spur / CW Tone"
            elif obw_hz >= 35000.0:
                ch_type = "Wideband Channel"
            else:
                ch_type = "Narrowband Signal"

            raw_channels.append({
                "freq_offset_hz": peak_freq + center_freq,
                "power_db": band_pwr,
                "peak_single_bin_db": peak_pwr,
                "local_snr_db": local_snr if ch_type == "Narrow Spur / CW Tone" else band_snr,
                "bandwidth_hz": obw_hz if ch_type != "Narrow Spur / CW Tone" else float(min(obw_hz, 12500.0)),
                "channel_type": ch_type,
                "confidence": float(min(1.0, max(0.1, (band_snr if ch_type != "Narrow Spur / CW Tone" else local_snr) / 20.0))),
                "bounds_hz": (float(reg_freqs[0] + center_freq), float(reg_freqs[-1] + center_freq)),
            })

        # Filter candidate sub-regions that lie within window leakage skirts of strong CW spurs
        strong_cw_spurs = [
            c for c in raw_channels
            if c["peak_single_bin_db"] >= 15.0 or c["local_snr_db"] >= 15.0
        ]

        filtered_raw = []
        for ch in raw_channels:
            if ch["channel_type"] != "Narrow Spur / CW Tone" and ch["bandwidth_hz"] < 35000.0:
                is_skirt_artifact = any(
                    abs(ch["freq_offset_hz"] - spur["freq_offset_hz"]) < 120000.0
                    and ch["peak_single_bin_db"] < spur["peak_single_bin_db"] - 15.0
                    for spur in strong_cw_spurs
                )
                if is_skirt_artifact:
                    continue
            filtered_raw.append(ch)
        raw_channels = filtered_raw

        # 4. Deduplicate & Merge adjacent sub-regions belonging to the same wideband signal
        for ch in raw_channels:
            ch["orig_freq"] = ch["freq_offset_hz"]
            ch["orig_bw"] = ch["bandwidth_hz"]

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
                    should_merge = (ch["channel_type"] == "Narrow Spur / CW Tone" and existing["channel_type"] == "Narrow Spur / CW Tone" and freq_diff < 60000.0)
                else:
                    should_merge = (edge_gap <= 20000.0 and freq_diff < max(80000.0, max_bw * 1.0))

                if should_merge:
                    is_duplicate = True
                    new_low = min(ch_low, ex_low)
                    new_high = max(ch_high, ex_high)
                    new_bw = new_high - new_low

                    existing["local_snr_db"] = max(existing["local_snr_db"], ch["local_snr_db"])

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

        # 4. Filter and select top channels prioritized by integrated energy prominence & local SNR
        def channel_priority(c):
            if c["channel_type"] == "Narrow Spur / CW Tone":
                return c["local_snr_db"]
            bw_factor_db = 10.0 * np.log10(max(1.0, float(c["bandwidth_hz"]) / 1000.0))
            return c["local_snr_db"] + bw_factor_db

        raw_channels.sort(key=channel_priority, reverse=True)

        if reject_spurs:
            comm_channels = [c for c in raw_channels if c["channel_type"] != "Narrow Spur / CW Tone" and c["local_snr_db"] >= min_snr_db]
            strong_spurs = [c for c in raw_channels if c["channel_type"] == "Narrow Spur / CW Tone" and c["local_snr_db"] >= max(6.0, min_snr_db)]
            weak_spurs = [c for c in raw_channels if c["channel_type"] == "Narrow Spur / CW Tone" and c["local_snr_db"] < max(6.0, min_snr_db)]
            valid_channels = sorted(comm_channels + strong_spurs, key=channel_priority, reverse=True)
            if len(valid_channels) > 0:
                sorted_channels = valid_channels
            else:
                sorted_channels = weak_spurs
        else:
            sorted_channels = raw_channels

        return sorted_channels[:num_channels_max]
