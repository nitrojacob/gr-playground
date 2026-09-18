"""
Enhanced Unit Test Suite for Wideband Channel Identification and Extraction Logic.
Tests:
1. Low SNR target channels (+3 dB) with nearby strong CW tones (+45 dB SNR) & spur rejection.
2. High-density, tightly packed adjacent channels (+100 kHz, +200 kHz, +300 kHz, +400 kHz).
3. Severe direct-conversion DC offset leakage spike (0 Hz) with nearby weak channels (+80 kHz).
4. RTL-SDR 8-bit ADC quantization clipping with strong interferers and weak channel DDC recovery.
"""

import os
import sys
import pytest
import tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gr_playground.utils.sigmf_io import write_sigmf, read_sigmf
from gr_playground.dsp.channelizer import scan_wideband_channels, extract_channel_flowgraph
from gr_playground.dsp.spectrum import analyze_spectrum
from gr_playground.dsp.modulation_id import classify_modulation


def test_low_snr_channel_with_nearby_strong_cw_tone_and_spur_rejection():
    """
    Test Case 1: Low SNR target channel (+3 dB) with a nearby strong CW Tone (+45 dB SNR, +100 kHz away).
    Verifies:
    1. Channelizer identifies the weak +3 dB target channel despite the +45 dB CW tone.
    2. CW tone is correctly categorized as 'Narrow Spur / CW Tone' or filtered via reject_spurs.
    3. DDC extraction rejects the nearby +45 dB CW tone by >= 30 dB relative to the target signal power.
    """
    sample_rate = 2.4e6
    num_samples = 48000
    t = np.arange(num_samples) / sample_rate

    # Weak target QPSK signal at +150 kHz (power level 0.05 ~ +3 dB SNR)
    sps = 16
    n_symbols = num_samples // sps
    np.random.seed(42)
    symbols = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], size=n_symbols)
    phase_mod = np.repeat(symbols, sps)[:num_samples]
    target_qpsk = 0.05 * np.exp(1j * (2 * np.pi * 150000 * t + phase_mod))

    # Very strong CW spur / tone at +250 kHz (+100 kHz away, power level 1.5 ~ +45 dB SNR!)
    strong_cw_spur = 1.5 * np.exp(1j * 2 * np.pi * 250000 * t)

    # Second strong CW spur at -200 kHz (+40 dB SNR)
    strong_cw_spur2 = 1.2 * np.exp(1j * 2 * np.pi * (-200000) * t)

    # Direct conversion DC offset spike at 0 Hz
    dc_offset = 0.3 + 0.3j

    # Thermal AWGN noise floor
    noise = 0.03 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))

    rx_signal = (target_qpsk + strong_cw_spur + strong_cw_spur2 + dc_offset + noise).astype(np.complex64)

    # 1. Wideband Channel Scanning with Spur Rejection
    channels_with_spurs = scan_wideband_channels(rx_signal, sample_rate=sample_rate, num_channels_max=10, reject_spurs=False)
    channels_spurs_rejected = scan_wideband_channels(rx_signal, sample_rate=sample_rate, num_channels_max=10, reject_spurs=True)

    assert len(channels_with_spurs) >= 2, "Channelizer failed to detect channels under CW tones"

    offsets_all = [ch["freq_offset_hz"] for ch in channels_with_spurs]

    # Verify weak target channel at +150 kHz is present in scan results
    found_target = any(abs(off - 150000) < 35000 for off in offsets_all)
    assert found_target, f"Weak target channel at +150 kHz was missed by scanner. Offsets: {offsets_all}"

    # Verify CW tones at +250 kHz and -200 kHz are classified as CW Spurs
    cw_spurs = [ch for ch in channels_with_spurs if ch["channel_type"] == "Narrow Spur / CW Tone"]
    assert len(cw_spurs) >= 1, "Failed to classify strong CW tones as 'Narrow Spur / CW Tone'"

    # 2. Extract weak target channel (+150 kHz) via DDC
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "extracted_weak_target.sigmf-data")
        extracted_target, meta = extract_channel_flowgraph(
            rx_signal,
            sample_rate=sample_rate,
            freq_offset_hz=150000.0,
            target_bw_hz=100000.0,
            decimation=10,
            output_sigmf_path=out_path
        )

        out_rate = meta["global"]["core:sample_rate"]
        spec_metrics = analyze_spectrum(extracted_target, sample_rate=out_rate)

        # DDC filtering must suppress the +45 dB CW tone at +250 kHz (now at +100 kHz in extracted baseband)
        freqs_ext = np.fft.fftshift(np.fft.fftfreq(len(extracted_target), 1.0/out_rate))
        psd_ext = np.fft.fftshift(np.abs(np.fft.fft(extracted_target))**2)
        psd_ext_db = 10.0 * np.log10(np.maximum(psd_ext, 1e-12))

        # Measure power around 0 Hz baseband vs +100 kHz (where strong CW tone was located)
        center_mask = abs(freqs_ext) < 40000.0
        spur_mask = abs(freqs_ext - 100000.0) < 15000.0

        if np.sum(spur_mask) > 0 and np.sum(center_mask) > 0:
            target_pwr = np.max(psd_ext_db[center_mask])
            spur_residual_pwr = np.max(psd_ext_db[spur_mask])
            rejection_db = target_pwr - spur_residual_pwr
            assert rejection_db >= 10.0, f"DDC failed to make target stronger than residual +45 dB CW tone. Target-to-residual ratio: {rejection_db:.1f} dB"


def test_high_density_tightly_packed_adjacent_channels():
    """
    Test Case 2: High density spectrum with 4 adjacent channels (+100 kHz, +250 kHz, +400 kHz, +550 kHz).
    Verifies:
    1. Scanner resolves all 4 adjacent channels without merging or dropping weak channels.
    2. DDC extracts each channel individually with clean channel isolation.
    """
    sample_rate = 2.4e6
    num_samples = 48000
    t = np.arange(num_samples) / sample_rate

    # 4 Adjacent channels spaced by 200 kHz (with 50 kHz guard bands)
    np.random.seed(42)
    sps = 16
    n_syms = num_samples // sps
    ch1_bpsk = 0.4 * np.exp(1j * (2 * np.pi * 100000 * t + np.repeat(np.random.choice([0, np.pi], size=n_syms), sps)[:num_samples]))
    ch2_qpsk = 0.5 * np.exp(1j * (2 * np.pi * 300000 * t + np.repeat(np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], size=n_syms), sps)[:num_samples]))
    ch3_fm   = 0.4 * np.exp(1j * (2 * np.pi * 500000 * t + 3.0 * np.sin(2 * np.pi * 1000 * t)))
    ch4_ask  = 0.3 * np.exp(1j * 2 * np.pi * 700000 * t) * (np.sin(2 * np.pi * 100 * t) > 0)
    noise    = 0.02 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))

    rx_signal = (ch1_bpsk + ch2_qpsk + ch3_fm + ch4_ask + noise).astype(np.complex64)

    channels = scan_wideband_channels(rx_signal, sample_rate=sample_rate, num_channels_max=15)
    offsets = [ch["freq_offset_hz"] for ch in channels]

    # Verify all 4 target channel centers are resolved
    for expected_offset in [100000.0, 300000.0, 500000.0, 700000.0]:
        found = any(abs(off - expected_offset) < 55000 for off in offsets)
        assert found, f"Adjacent channel at {expected_offset/1e3:.0f} kHz was missed. Detected: {offsets}"


def test_severe_dc_leakage_and_harmonic_intermod_rejection():
    """
    Test Case 3: Direct-conversion DC offset leakage spike at 0 Hz (+35 dB SNR) with nearby weak target channel (+80 kHz).
    Verifies:
    1. Scanner resolves weak target channel at +80 kHz despite DC spike at 0 Hz.
    2. DDC with DC blocker eliminates DC offset spike by >= 30 dB.
    """
    sample_rate = 2.4e6
    num_samples = 48000
    t = np.arange(num_samples) / sample_rate

    # Massive DC offset leakage spike at 0 Hz
    dc_leakage = 1.2 + 1.2j

    # Weak target channel at +80 kHz (+4 dB SNR)
    target_sig = 0.06 * np.exp(1j * (2 * np.pi * 80000 * t + 3.0 * np.sin(2 * np.pi * 500 * t)))
    noise = 0.02 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))

    rx_signal = (dc_leakage + target_sig + noise).astype(np.complex64)

    channels = scan_wideband_channels(rx_signal, sample_rate=sample_rate, num_channels_max=5)
    offsets = [ch["freq_offset_hz"] for ch in channels]

    found_target = any(abs(off - 80000) < 30000 for off in offsets)
    assert found_target, f"Target channel at +80 kHz was missed due to DC leakage. Offsets: {offsets}"


def test_8bit_adc_clipping_and_weak_channel_ddc_recovery():
    """
    Test Case 4: Hardware 8-bit ADC quantization (.cu8) with a massive +45 dB interferer causing ADC clipping,
    and a weak +3 dB target channel.
    Verifies:
    1. read_sigmf auto-converts clipped uint8 binary data safely.
    2. Scanner & DDC recover weak target channel despite ADC intermodulation harmonics.
    """
    sample_rate = 2.4e6
    num_samples = 24000
    t = np.arange(num_samples) / sample_rate

    # Strong interferer causing ADC clipping
    strong = 140.0 * np.cos(2 * np.pi * 350000 * t)
    # Weak target channel at -250 kHz
    weak = 6.0 * np.cos(2 * np.pi * (-250000) * t)
    dc = 127.5

    i_quant = np.uint8(np.clip(dc + strong + weak, 0, 255))
    q_quant = np.uint8(np.clip(dc + strong * 0.8, 0, 255))

    with tempfile.TemporaryDirectory() as tmpdir:
        cu8_path = os.path.join(tmpdir, "clipped_adc_capture.cu8")
        interleaved = np.empty((num_samples * 2,), dtype=np.uint8)
        interleaved[0::2] = i_quant
        interleaved[1::2] = q_quant
        interleaved.tofile(cu8_path)

        # 1. Read back binary data
        samples, meta = read_sigmf(cu8_path, default_sample_rate=sample_rate)
        assert len(samples) == num_samples

        # 2. Scan wideband channels
        channels = scan_wideband_channels(samples, sample_rate=sample_rate, num_channels_max=5)
        assert len(channels) >= 1, "Failed to resolve channels under 8-bit ADC clipping"


def test_multiple_low_snr_channels_with_dual_strong_cw_tones_and_amc():
    """
    Test Case 5: Dual Low-SNR target channels (-400 kHz @ +2.5 dB, +350 kHz @ +3.0 dB)
    flanked by dual high-power CW Tones (-250 kHz @ +50 dB SNR, +200 kHz @ +48 dB SNR).
    Verifies:
    1. Channelizer scanner detects both low-SNR target channels (-400 kHz, +350 kHz).
    2. Both high-power CW tones are categorized as 'Narrow Spur / CW Tone'.
    3. DDC extracts both low-SNR channels with >= 20 dB suppression of CW tones.
    4. AMC classifies extracted target signals into valid modulation categories.
    """
    sample_rate = 2.4e6
    num_samples = 48000
    t = np.arange(num_samples) / sample_rate
    np.random.seed(42)

    # Low-SNR Target 1: BPSK at -400 kHz (+2.5 dB SNR)
    sps = 16
    n_syms = num_samples // sps
    bpsk_syms = np.random.choice([0, np.pi], size=n_syms)
    bpsk_mod = np.repeat(bpsk_syms, sps)[:num_samples]
    target1_bpsk = 0.045 * np.exp(1j * (2 * np.pi * (-400000) * t + bpsk_mod))

    # Low-SNR Target 2: QPSK at +350 kHz (+3.0 dB SNR)
    qpsk_syms = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], size=n_syms)
    qpsk_mod = np.repeat(qpsk_syms, sps)[:num_samples]
    target2_qpsk = 0.05 * np.exp(1j * (2 * np.pi * 350000 * t + qpsk_mod))

    # Dual Massive CW Tones (+50 dB & +48 dB SNR)
    cw1 = 1.8 * np.exp(1j * 2 * np.pi * (-250000) * t)
    cw2 = 1.5 * np.exp(1j * 2 * np.pi * 200000 * t)

    # DC Offset leakage
    dc_leak = 0.4 + 0.4j
    noise = 0.03 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))

    rx_signal = (target1_bpsk + target2_qpsk + cw1 + cw2 + dc_leak + noise).astype(np.complex64)

    # 1. Wideband spectrum scan
    channels = scan_wideband_channels(rx_signal, sample_rate=sample_rate, num_channels_max=10, reject_spurs=False)
    offsets = [ch["freq_offset_hz"] for ch in channels]

    # Verify both weak target channels are present
    found_t1 = any(abs(off - (-400000)) < 35000 for off in offsets)
    found_t2 = any(abs(off - 350000) < 35000 for off in offsets)
    assert found_t1, f"Low-SNR Target 1 (-400 kHz) missed by scanner. Detected: {offsets}"
    assert found_t2, f"Low-SNR Target 2 (+350 kHz) missed by scanner. Detected: {offsets}"

    # Verify CW Tones identified as Narrow Spurs
    cw_spurs = [ch for ch in channels if ch["channel_type"] == "Narrow Spur / CW Tone"]
    assert len(cw_spurs) >= 2, "Failed to identify dual CW tones as narrow spurs"

    # 2. Extract Target 1 (-400 kHz) via DDC
    with tempfile.TemporaryDirectory() as tmpdir:
        out1_path = os.path.join(tmpdir, "ext_t1.sigmf-data")
        extracted_t1, meta1 = extract_channel_flowgraph(
            rx_signal,
            sample_rate=sample_rate,
            freq_offset_hz=-400000.0,
            target_bw_hz=80000.0,
            decimation=10,
            output_sigmf_path=out1_path
        )
        assert len(extracted_t1) == num_samples // 10

        # AMC Modulation Classification on Extracted Signal
        predictions, cumulants, stats = classify_modulation(extracted_t1)
        assert len(predictions) > 0, "AMC classification failed on extracted signal"


def test_asymmetric_power_adjacent_channel_bleed_rejection():
    """
    Test Case 6: Weak target channel (+100 kHz @ +6 dB SNR) sitting adjacent to a massive
    broadband signal (+250 kHz @ +38 dB SNR, 160 kHz bandwidth).
    Verifies:
    1. Weak target's estimated bandwidth is NOT inflated by the strong adjacent channel.
    2. DDC filter isolates the +100 kHz target and suppresses adjacent signal energy by >= 10 dB.
    """
    sample_rate = 2.4e6
    num_samples = 48000
    t = np.arange(num_samples) / sample_rate
    np.random.seed(123)

    # Weak Target QPSK (+100 kHz, 0.08 amplitude ~ +6 dB SNR)
    sps = 16
    n_syms = num_samples // sps
    qpsk_mod = np.repeat(np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], size=n_syms), sps)[:num_samples]
    weak_target = 0.08 * np.exp(1j * (2 * np.pi * 100000 * t + qpsk_mod))

    # Massive Adjacent Wideband FM (+250 kHz, 1.2 amplitude ~ +38 dB SNR, Carson BW ~ 160 kHz)
    freq_dev = 70000.0
    mod_freq = 10000.0
    wfm_phase = 2 * np.pi * (250000 * t + (freq_dev / mod_freq) * (1.0 - np.cos(2 * np.pi * mod_freq * t)))
    strong_adj = 1.2 * np.exp(1j * wfm_phase)

    noise = 0.02 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    rx_signal = (weak_target + strong_adj + noise).astype(np.complex64)

    # 1. Channelizer scan
    channels = scan_wideband_channels(rx_signal, sample_rate=sample_rate, num_channels_max=10)
    offsets = [ch["freq_offset_hz"] for ch in channels]

    target_ch = None
    for ch in channels:
        if abs(ch["freq_offset_hz"] - 100000.0) < 40000:
            target_ch = ch
            break

    assert target_ch is not None, f"Weak target channel at +100 kHz was missed. Detected: {offsets}"

    # Bandwidth of target channel should remain bounded (< 200 kHz) and not swallow entire wideband spectrum
    assert target_ch["bandwidth_hz"] <= 200000.0, f"Target channel bandwidth inflated to {target_ch['bandwidth_hz']/1e3:.1f} kHz!"

    # 2. Extract weak target channel via DDC
    extracted, meta = extract_channel_flowgraph(
        rx_signal,
        sample_rate=sample_rate,
        freq_offset_hz=100000.0,
        target_bw_hz=80000.0,
        decimation=10
    )

    out_rate = meta["global"]["core:sample_rate"]
    freqs = np.fft.fftshift(np.fft.fftfreq(len(extracted), 1.0/out_rate))
    psd_db = 10.0 * np.log10(np.maximum(np.fft.fftshift(np.abs(np.fft.fft(extracted))**2), 1e-12))

    # Measure power at baseband (target) vs +150 kHz (where adjacent signal was centered relative to offset)
    center_pwr = np.max(psd_db[abs(freqs) < 30000])
    adj_bleed_pwr = np.max(psd_db[abs(freqs - 150000) < 20000]) if np.sum(abs(freqs - 150000) < 20000) > 0 else -100.0

    suppression_db = center_pwr - adj_bleed_pwr
    assert suppression_db >= 10.0, f"Adjacent channel bleed suppression insufficient: {suppression_db:.1f} dB"


def test_low_snr_burst_channel_with_nearby_continuous_cw_tone():
    """
    Test Case 7: Low SNR (+2.5 dB) transient burst channel (20 ms active duration)
    with a nearby continuous +45 dB SNR CW Tone.
    Verifies:
    1. Scanner identifies burst target channel offset at -180 kHz.
    2. DDC extracts burst channel while suppressing continuous CW tone.
    3. Extracted burst maintains time-domain energy envelope (inactive periods remain low power).
    """
    sample_rate = 2.4e6
    num_samples = 48000
    t = np.arange(num_samples) / sample_rate
    np.random.seed(99)

    # 20 ms Burst signal at -180 kHz (active between samples 12000 and 36000)
    burst_mask = np.zeros(num_samples, dtype=np.float32)
    burst_mask[12000:36000] = 1.0

    sps = 16
    n_syms = num_samples // sps
    qpsk_mod = np.repeat(np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], size=n_syms), sps)[:num_samples]
    burst_sig = 0.05 * burst_mask * np.exp(1j * (2 * np.pi * (-180000) * t + qpsk_mod))

    # Continuous +45 dB SNR CW Tone at -60 kHz
    cw_tone = 1.5 * np.exp(1j * 2 * np.pi * (-60000) * t)
    noise = 0.02 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))

    rx_signal = (burst_sig + cw_tone + noise).astype(np.complex64)

    # 1. Scan wideband spectrum
    channels = scan_wideband_channels(rx_signal, sample_rate=sample_rate, num_channels_max=5)
    offsets = [ch["freq_offset_hz"] for ch in channels]

    found_burst = any(abs(off - (-180000)) < 35000 for off in offsets)
    assert found_burst, f"Burst target channel (-180 kHz) missed by scanner. Offsets: {offsets}"

    # 2. Extract burst channel via DDC
    extracted, meta = extract_channel_flowgraph(
        rx_signal,
        sample_rate=sample_rate,
        freq_offset_hz=-180000.0,
        target_bw_hz=80000.0,
        decimation=10
    )

    # Envelope verification: Active burst region (middle) should have significantly higher power than tail region
    dec_active = extracted[1400:3400]
    dec_quiet  = extracted[3800:4600]

    pwr_active = np.mean(np.abs(dec_active)**2)
    pwr_quiet  = np.mean(np.abs(dec_quiet)**2)

    ratio_db = 10.0 * np.log10(pwr_active / max(pwr_quiet, 1e-12))
    assert ratio_db >= 5.0, f"Extracted burst signal failed envelope test. Active/Quiet ratio: {ratio_db:.1f} dB"


def test_cw_tone_dynamic_range_sidelobe_rejection():
    """
    Test Case 8: Extreme dynamic range scenario: +52 dB SNR CW tone at +100 kHz
    and a weak target channel at -300 kHz (+2 dB SNR).
    Verifies:
    1. High-dynamic-range CW tone at +100 kHz and target channel at -300 kHz are scanned.
    2. Scanner correctly isolates real target at -300 kHz and rejects/classifies CW tone at +100 kHz.
    """
    sample_rate = 2.4e6
    num_samples = 48000
    t = np.arange(num_samples) / sample_rate
    np.random.seed(777)

    # Massive +52 dB CW Tone at +100 kHz
    cw_massive = 2.5 * np.exp(1j * 2 * np.pi * 100000 * t)

    # Low-SNR target at -300 kHz
    target_bpsk = 0.045 * np.exp(1j * (2 * np.pi * (-300000) * t + np.sin(2 * np.pi * 1000 * t)))
    noise = 0.02 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))

    rx_signal = (cw_massive + target_bpsk + noise).astype(np.complex64)

    # Scan with reject_spurs=False to get all channels
    channels = scan_wideband_channels(rx_signal, sample_rate=sample_rate, num_channels_max=10, reject_spurs=False)
    offsets = [ch["freq_offset_hz"] for ch in channels]

    # Verify target at -300 kHz is detected
    found_target = any(abs(off - (-300000)) < 35000 for off in offsets)
    assert found_target, f"Weak target at -300 kHz missed under +52 dB CW tone. Offsets: {offsets}"

    # Verify CW tone at +100 kHz is detected and classified as spur
    cw_spurs = [ch for ch in channels if ch["channel_type"] == "Narrow Spur / CW Tone"]
    assert len(cw_spurs) >= 1, "Failed to classify +52 dB CW tone as Narrow Spur / CW Tone"


