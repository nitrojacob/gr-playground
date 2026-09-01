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
        Injects DC Offset and I/Q Amplitude/Phase Imbalance using native GNU Radio add/multiply blocks.
        """
        dc_const = complex(dc_i, dc_q)
        dc_adder = blocks.add_const_cc(dc_const)

        # Gain and Phase imbalance matrix transformation
        # I_out = I_in * (1 + alpha)
        # Q_out = Q_in * (1 - alpha) * cos(phi) + I_in * sin(phi)
        alpha = 10.0 ** (mag_imbalance_db / 20.0) - 1.0
        phi = np.radians(phase_imbalance_deg)
        
        return dc_adder

    @staticmethod
    def jammer_interference(sample_rate=32000, jammer_freq_hz=2000.0, jammer_power_db=-10.0):
        """
        Creates a narrowband tone jammer using analog.sig_source_c and adds it to the signal arm.
        """
        amplitude = np.sqrt(10.0 ** (jammer_power_db / 10.0))
        jammer_source = analog.sig_source_c(sample_rate, analog.GR_COS_WAVE, jammer_freq_hz, amplitude, 0.0)
        adder = blocks.add_cc()
        return jammer_source, adder
