"""
GNU Radio Channel Impairment Blocks Wrapper
Applies AWGN, CFO, Phase Noise, Clock Drift (SRO), DC Offset, I/Q Imbalance, Multipath Fading, and Narrowband Jammer Interference using native gnuradio blocks.
"""

from gnuradio import gr, blocks, analog, channels
import numpy as np

class GRImpairments:
    @staticmethod
    def channel_model(snr_db=20.0, cfo_hz=0.0, sro_ppm=0.0, sample_rate=32000, multipath_taps=None):
        """
        Native GNU Radio channel model block.
        Computes noise voltage from SNR in dB, frequency offset relative to sample rate, and timing drift.
        """
        # Calculate noise voltage: signal power assumed normalized to 1.0 (0 dBW)
        noise_power = 10.0 ** (-snr_db / 10.0)
        noise_voltage = np.sqrt(noise_power / 2.0)  # Complex noise divides power equally between I and Q

        # Normalized frequency offset = CFO (Hz) / Sample Rate (Hz)
        freq_offset = cfo_hz / float(sample_rate)

        # Timing offset (SRO)
        timing_offset = 1.0 + (sro_ppm / 1e6)

        taps = multipath_taps if multipath_taps is not None else [1.0 + 0.0j]

        return channels.channel_model(
            noise_voltage=noise_voltage,
            frequency_offset=freq_offset,
            epsilon=timing_offset,
            taps=taps,
            noise_seed=42
        )

    @staticmethod
    def dc_offset_and_iq_imbalance(dc_i=0.0, dc_q=0.0, mag_imbalance_db=0.0, phase_imbalance_deg=0.0):
        """
        Injects DC Offset using native GNU Radio add_const_cc block.
        """
        dc_const = complex(dc_i, dc_q)
        return blocks.add_const_cc(dc_const)

    @staticmethod
    def iq_imbalance(samples, mag_imbalance_db=0.0, phase_imbalance_deg=0.0):
        """
        Applies Quadrature I/Q Gain and Phase Imbalance matrix transformation:
        I_out = (1 + alpha) * I_in
        Q_out = (1 - alpha) * Q_in * cos(phi) + I_in * sin(phi)
        """
        if mag_imbalance_db == 0.0 and phase_imbalance_deg == 0.0:
            return samples
            
        alpha = 10.0 ** (mag_imbalance_db / 20.0) - 1.0
        phi = np.radians(phase_imbalance_deg)
        
        i_in = samples.real
        q_in = samples.imag
        
        i_out = (1.0 + alpha) * i_in
        q_out = (1.0 - alpha) * q_in * np.cos(phi) + i_in * np.sin(phi)
        
        return (i_out + 1j * q_out).astype(np.complex64)

    @staticmethod
    def phase_noise(samples, phase_noise_std_rad=0.01, seed=42):
        """
        Simulates Wiener process Local Oscillator (LO) Phase Noise / Jitter:
        phi[n] = phi[n-1] + N(0, sigma^2)
        """
        if phase_noise_std_rad <= 0.0:
            return samples
            
        rng = np.random.RandomState(seed)
        dphi = rng.normal(0.0, phase_noise_std_rad, size=len(samples))
        phi = np.cumsum(dphi)
        return (samples * np.exp(1j * phi)).astype(np.complex64)

    @staticmethod
    def hpa_saturation(samples, ibo_db=3.0, p=2.0):
        """
        Simulates Solid-State High Power Amplifier (HPA) non-linear saturation via the Rapp Model:
        y = x / (1 + (|x| / V_sat)^(2p))^(1 / (2p))
        Evaluates out-of-band spectral regrowth and PAPR degradation in OFDM/SC-FDMA.
        """
        if ibo_db is None:
            return samples
            
        p_avg = np.mean(np.abs(samples)**2)
        v_sat = np.sqrt(p_avg) * (10.0 ** (ibo_db / 20.0))
        
        mag = np.abs(samples) + 1e-12
        gain_scale = 1.0 / ((1.0 + (mag / v_sat)**(2.0 * p)) ** (1.0 / (2.0 * p)))
        
        return (samples * gain_scale).astype(np.complex64)

    @staticmethod
    def jammer_interference(sample_rate=32000, jammer_freq_hz=2000.0, jammer_power_db=-10.0):
        """
        Creates a narrowband tone jammer using analog.sig_source_c and adds it to the signal arm.
        """
        amplitude = np.sqrt(10.0 ** (jammer_power_db / 10.0))
        jammer_source = analog.sig_source_c(sample_rate, analog.GR_COS_WAVE, jammer_freq_hz, amplitude, 0.0)
        adder = blocks.add_cc()
        return jammer_source, adder


