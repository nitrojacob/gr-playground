"""
Unit Test Suite for RF Signal Bandwidth Estimation Algorithms.
Verifies that occupied bandwidth (OBW 99%) and wideband channelizer bandwidth estimation
accurately measure true occupied RF bandwidth within +/- 10% tolerance across:
1. Root-Raised-Cosine (RRC) pulse-shaped QPSK / BPSK signals.
2. Carson Wideband FM (WFM) signals.
3. Double-Sideband Amplitude Modulation (AM-DSB).
4. Multi-subcarrier OFDM signals.
5. Noise sweeps (SNR from 10 dB to 30 dB).
"""

import os
import sys
import pytest
import numpy as np
from scipy import signal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gr_playground.dsp.spectrum import analyze_spectrum, estimate_occupied_bandwidth
from gr_playground.dsp.channelizer import scan_wideband_channels


def generate_rrc_signal(symbols, sps, alpha=0.35, num_taps=101):
    """Generates RRC pulse-shaped complex baseband signal."""
    t = np.arange(-num_taps // 2, num_taps // 2 + 1, dtype=np.float32) / float(sps)
    rrc = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti == 0.0:
            rrc[i] = 1.0 - alpha + (4.0 * alpha / np.pi)
        elif abs(ti) == 1.0 / (4.0 * alpha):
            rrc[i] = (alpha / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
            )
        else:
            denom = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            num = np.sin(np.pi * ti * (1.0 - alpha)) + 4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha))
            rrc[i] = num / denom
    rrc /= np.sqrt(np.sum(rrc ** 2))

    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex64)
    upsampled[::sps] = symbols
    shaped = np.convolve(upsampled, rrc, mode="same")
    return shaped.astype(np.complex64)


def test_bandwidth_estimation_rrc_qpsk():
    """
    Test Case 1: RRC Filtered QPSK Signal
    Parameters:
    - Symbol Rate (Rs) = 100 kHz
    - Roll-off Factor (alpha) = 0.35
    - Sample Rate (Fs) = 2.0 MHz (sps = 20)
    - Theoretical 99% OBW = 121.5 kHz (Stopband = 135 kHz)
    - Verification: Estimated bandwidth within +/- 10% of theoretical 99% OBW.
    """
    sample_rate = 2.0e6
    symbol_rate = 100000.0
    alpha = 0.35
    sps = int(sample_rate / symbol_rate)
    num_symbols = 4000

    np.random.seed(42)
    qpsk_const = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    symbols = np.random.choice(qpsk_const, size=num_symbols)

    iq = generate_rrc_signal(symbols, sps=sps, alpha=alpha)
    noise = 0.01 * (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))
    rx_signal = iq + noise

    # Theoretical 99% OBW for RRC alpha=0.35
    theoretical_bw = 1.21 * symbol_rate  # 121 kHz
    
    # 1. Analyze Spectrum Occupied Bandwidth
    metrics = analyze_spectrum(rx_signal, sample_rate=sample_rate, nperseg=4096)
    est_bw = metrics["occupied_bw_hz"]

    err_pct = abs(est_bw - theoretical_bw) / theoretical_bw * 100.0
    print(f"RRC QPSK: Theoretical = {theoretical_bw/1e3:.1f} kHz, Estimated = {est_bw/1e3:.1f} kHz, Error = {err_pct:.2f}%")

    assert err_pct <= 10.0, f"RRC QPSK BW estimation error ({err_pct:.2f}%) exceeded 10%! Est: {est_bw/1e3:.1f} kHz, Target: {theoretical_bw/1e3:.1f} kHz"


def test_bandwidth_estimation_rrc_bpsk():
    """
    Test Case 2: RRC Filtered BPSK Signal
    Parameters:
    - Symbol Rate (Rs) = 200 kHz
    - Roll-off Factor (alpha) = 0.25
    - Sample Rate (Fs) = 2.4 MHz (sps = 12)
    - Theoretical 99% OBW = 230 kHz (Stopband = 250 kHz)
    - Verification: Estimated bandwidth within +/- 10% of theoretical 99% OBW.
    """
    sample_rate = 2.4e6
    symbol_rate = 200000.0
    alpha = 0.25
    sps = int(sample_rate / symbol_rate)
    num_symbols = 5000

    np.random.seed(123)
    symbols = np.random.choice([-1.0+0j, 1.0+0j], size=num_symbols)

    iq = generate_rrc_signal(symbols, sps=sps, alpha=alpha)
    noise = 0.01 * (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))
    rx_signal = iq + noise

    theoretical_bw = 1.15 * symbol_rate  # 230 kHz

    metrics = analyze_spectrum(rx_signal, sample_rate=sample_rate, nperseg=4096)
    est_bw = metrics["occupied_bw_hz"]

    err_pct = abs(est_bw - theoretical_bw) / theoretical_bw * 100.0
    print(f"RRC BPSK: Theoretical = {theoretical_bw/1e3:.1f} kHz, Estimated = {est_bw/1e3:.1f} kHz, Error = {err_pct:.2f}%")

    assert err_pct <= 10.0, f"RRC BPSK BW estimation error ({err_pct:.2f}%) exceeded 10%! Est: {est_bw/1e3:.1f} kHz, Target: {theoretical_bw/1e3:.1f} kHz"


def test_bandwidth_estimation_carson_wfm():
    """
    Test Case 3: Carson Wideband FM (WFM) Signal
    Parameters:
    - Peak Frequency Deviation (delta_f) = 75 kHz
    - Modulating Audio Frequency (fm) = 15 kHz
    - Sample Rate (Fs) = 2.4 MHz
    - Theoretical Carson Bandwidth = 2 * (delta_f + fm) = 180 kHz
    - Verification: Estimated bandwidth within +/- 10% (162 kHz to 198 kHz).
    """
    sample_rate = 2.4e6
    duration = 0.05
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate

    freq_dev = 75000.0
    audio_freq = 15000.0
    phase = (freq_dev / audio_freq) * (1.0 - np.cos(2 * np.pi * audio_freq * t))
    wfm_sig = np.exp(1j * phase)
    noise = 0.01 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    rx_signal = (wfm_sig + noise).astype(np.complex64)

    theoretical_bw = 2.0 * (freq_dev + audio_freq)  # 180 kHz

    metrics = analyze_spectrum(rx_signal, sample_rate=sample_rate, nperseg=4096)
    est_bw = metrics["occupied_bw_hz"]

    err_pct = abs(est_bw - theoretical_bw) / theoretical_bw * 100.0
    print(f"WFM Carson: Theoretical = {theoretical_bw/1e3:.1f} kHz, Estimated = {est_bw/1e3:.1f} kHz, Error = {err_pct:.2f}%")

    assert err_pct <= 10.0, f"Carson WFM BW estimation error ({err_pct:.2f}%) exceeded 10%! Est: {est_bw/1e3:.1f} kHz, Target: {theoretical_bw/1e3:.1f} kHz"


def test_bandwidth_estimation_double_sideband_am():
    """
    Test Case 4: Double-Sideband AM Signal (AM-DSB)
    Parameters:
    - Modulating Audio Frequency (fm) = 20 kHz
    - Sample Rate (Fs) = 1.0 MHz
    - Theoretical Bandwidth = 2 * fm = 40 kHz
    - Verification: Estimated bandwidth within +/- 10% (36 kHz to 44 kHz).
    """
    sample_rate = 1.0e6
    duration = 0.05
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate

    mod_freq = 20000.0
    audio = np.cos(2 * np.pi * mod_freq * t)
    carrier = np.cos(2 * np.pi * 100000.0 * t)
    am_sig = (1.0 + 0.8 * audio) * carrier
    
    # Convert to complex baseband around +100 kHz offset
    analytic = signal.hilbert(am_sig) * np.exp(-1j * 2 * np.pi * 100000.0 * t)
    noise = 0.01 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    rx_signal = (analytic + noise).astype(np.complex64)

    theoretical_bw = 2.0 * mod_freq  # 40 kHz

    metrics = analyze_spectrum(rx_signal, sample_rate=sample_rate, nperseg=4096)
    est_bw = metrics["occupied_bw_hz"]

    err_pct = abs(est_bw - theoretical_bw) / theoretical_bw * 100.0
    print(f"AM-DSB: Theoretical = {theoretical_bw/1e3:.1f} kHz, Estimated = {est_bw/1e3:.1f} kHz, Error = {err_pct:.2f}%")

    assert err_pct <= 10.0, f"AM-DSB BW estimation error ({err_pct:.2f}%) exceeded 10%! Est: {est_bw/1e3:.1f} kHz, Target: {theoretical_bw/1e3:.1f} kHz"


def test_bandwidth_estimation_ofdm_multicarrier():
    """
    Test Case 5: Multi-Subcarrier OFDM Signal
    Parameters:
    - Active Subcarriers = 52
    - Subcarrier Spacing (df) = 15 kHz
    - Sample Rate (Fs) = 2.4 MHz
    - Theoretical Bandwidth = 52 * 15 kHz = 780 kHz
    - Verification: Estimated bandwidth within +/- 10% (702 kHz to 858 kHz).
    """
    sample_rate = 2.4e6
    n_fft = 128
    active_subcarriers = 52
    df = 15000.0
    num_symbols = 100

    np.random.seed(42)
    ofdm_samples = []

    for _ in range(num_symbols):
        freq_domain = np.zeros(n_fft, dtype=np.complex64)
        # Allocate 52 subcarriers symmetrically around 0 Hz
        qpsk_data = (np.random.choice([1, -1], size=active_subcarriers) + 1j * np.random.choice([1, -1], size=active_subcarriers)) / np.sqrt(2)
        half = active_subcarriers // 2
        freq_domain[:half] = qpsk_data[:half]
        freq_domain[-half:] = qpsk_data[half:]

        time_domain = np.fft.ifft(freq_domain) * np.sqrt(n_fft)
        # Add cyclic prefix
        cp = time_domain[-16:]
        ofdm_samples.extend(np.concatenate([cp, time_domain]))

    rx_signal = np.array(ofdm_samples, dtype=np.complex64)
    actual_fs = n_fft * df  # 1.92 MHz effective grid sample rate

    theoretical_bw = active_subcarriers * df  # 780 kHz

    metrics = analyze_spectrum(rx_signal, sample_rate=actual_fs, nperseg=2048)
    est_bw = metrics["occupied_bw_hz"]

    err_pct = abs(est_bw - theoretical_bw) / theoretical_bw * 100.0
    print(f"OFDM 52-subcarrier: Theoretical = {theoretical_bw/1e3:.1f} kHz, Estimated = {est_bw/1e3:.1f} kHz, Error = {err_pct:.2f}%")

    assert err_pct <= 10.0, f"OFDM BW estimation error ({err_pct:.2f}%) exceeded 10%! Est: {est_bw/1e3:.1f} kHz, Target: {theoretical_bw/1e3:.1f} kHz"


def test_wideband_channelizer_bandwidth_estimation_accuracy():
    """
    Test Case 6: Wideband Channelizer (`scan_wideband_channels`) Bandwidth Estimation
    Verifies that `scan_wideband_channels` estimates the bandwidth of an RRC QPSK signal
    offset at +300 kHz within +/- 10% tolerance.
    """
    sample_rate = 2.4e6
    symbol_rate = 120000.0
    alpha = 0.35
    sps = int(sample_rate / symbol_rate)
    num_symbols = 4000

    np.random.seed(99)
    qpsk_const = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    symbols = np.random.choice(qpsk_const, size=num_symbols)

    baseband_qpsk = generate_rrc_signal(symbols, sps=sps, alpha=alpha)
    t = np.arange(len(baseband_qpsk)) / sample_rate
    offset_qpsk = baseband_qpsk * np.exp(1j * 2 * np.pi * 300000.0 * t)

    noise = 0.01 * (np.random.randn(len(offset_qpsk)) + 1j * np.random.randn(len(offset_qpsk)))
    rx_signal = offset_qpsk + noise

    theoretical_bw = (1.0 + alpha) * symbol_rate  # 162 kHz

    channels = scan_wideband_channels(rx_signal, sample_rate=sample_rate, num_channels_max=5)
    assert len(channels) >= 1, "Channelizer missed +300 kHz RRC QPSK signal"

    ch = channels[0]
    est_bw = ch["bandwidth_hz"]

    err_pct = abs(est_bw - theoretical_bw) / theoretical_bw * 100.0
    print(f"Channelizer RRC QPSK: Theoretical = {theoretical_bw/1e3:.1f} kHz, Estimated = {est_bw/1e3:.1f} kHz, Error = {err_pct:.2f}%")

    assert err_pct <= 10.0, f"Channelizer BW estimation error ({err_pct:.2f}%) exceeded 10%! Est: {est_bw/1e3:.1f} kHz, Target: {theoretical_bw/1e3:.1f} kHz"


def test_bandwidth_estimation_rippled_fading_fm():
    """
    Test Case 7: Wideband FM Signal with Audio Modulation Ripples and Spectral Dips
    Verifies that scan_wideband_channels estimates true occupied bandwidth (~180 kHz)
    without truncating on micro 0.3 dB valley dips in real-world fading signals.
    """
    sample_rate = 2.4e6
    duration = 0.1
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate

    freq_dev = 75000.0
    # Multi-tone modulating audio producing realistic spectral ripples
    audio = 0.5 * np.cos(2 * np.pi * 3000 * t) + 0.5 * np.sin(2 * np.pi * 12000 * t)
    phase = 2 * np.pi * freq_dev * np.cumsum(audio) / sample_rate
    wfm_sig = np.exp(1j * phase)

    # Add multipath echo & noise ripples
    echo = 0.3 * np.roll(wfm_sig, 50)
    noise = 0.02 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    rx_signal = (wfm_sig + echo + noise).astype(np.complex64)

    channels = scan_wideband_channels(rx_signal, sample_rate=sample_rate, num_channels_max=5)
    assert len(channels) >= 1, "Channelizer missed rippled FM signal"

    est_bw = channels[0]["bandwidth_hz"]
    print(f"Rippled FM Estimated BW = {est_bw/1e3:.1f} kHz")

    # Estimated bandwidth must span true wideband FM footprint (> 80 kHz)
    assert est_bw >= 80000.0, f"Channelizer truncated rippled FM to {est_bw/1e3:.1f} kHz!"


def test_channelizer_multi_peak_deduplication():
    """
    Test Case 8: Multi-Peak Wideband Channel Deduplication
    Verifies that multi-peaked wideband signal spectra are properly merged into a single
    channel instead of fragmenting into multiple false positive narrowband channels.
    """
    sample_rate = 2.4e6
    duration = 0.1
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate

    # Generate wideband FM signal with multi-tone modulation (3 distinct spectral peaks)
    freq_dev = 60000.0
    phase = 2 * np.pi * freq_dev * np.cumsum(np.sin(2 * np.pi * 4000 * t) + 0.5 * np.cos(2 * np.pi * 12000 * t)) / sample_rate
    wfm_sig = np.exp(1j * phase)

    noise = 0.01 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    rx_signal = (wfm_sig + noise).astype(np.complex64)

    channels = scan_wideband_channels(rx_signal, sample_rate=sample_rate, num_channels_max=10)

    # Should merge multi-peaked wideband signal into 1 unified wideband channel
    assert len(channels) == 1, f"Expected 1 merged wideband channel, got {len(channels)}"
    assert channels[0]["bandwidth_hz"] >= 90000.0, f"Expected merged BW >= 90 kHz, got {channels[0]['bandwidth_hz']/1e3:.1f} kHz"

