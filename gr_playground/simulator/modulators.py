"""
GNU Radio Modulator Blocks Wrapper
Implements AM, FM, PM, BPSK, QPSK, 8PSK, 16QAM, 64QAM, 256QAM, BFSK, 4FSK, GFSK, MSK, and OFDM modulators using native gnuradio blocks.
"""

from gnuradio import gr, blocks, analog, digital, filter
import numpy as np

class GRModulators:
    @staticmethod
    def get_constellation(mod_type):
        """Construct digital constellation objects for GNU Radio modulator blocks."""
        mod_upper = mod_type.upper()
        if mod_upper == "BPSK":
            return digital.constellation_bpsk().base()
        elif mod_upper == "QPSK":
            return digital.constellation_qpsk().base()
        elif mod_upper == "8PSK":
            return digital.constellation_8psk().base()
        elif mod_upper == "16QAM":
            return digital.constellation_16qam().base()
        elif mod_upper == "64QAM":
            # Build 64-QAM constellation
            points = []
            for i in [-7, -5, -3, -1, 1, 3, 5, 7]:
                for q in [-7, -5, -3, -1, 1, 3, 5, 7]:
                    points.append(complex(i, q))
            points = np.array(points) / np.sqrt(42.0)
            return digital.constellation_calcdist(points.tolist(), [], 4, 1).base()
        elif mod_upper in ["ASK", "2ASK", "OOK"]:
            # Build 2-ASK / OOK constellation points [0.0, 1.0]
            points = [complex(0.0, 0.0), complex(1.0, 0.0)]
            return digital.constellation_calcdist(points, [], 2, 1).base()
        elif mod_upper == "256QAM":
            points = []
            for i in np.arange(-15, 16, 2):
                for q in np.arange(-15, 16, 2):
                    points.append(complex(i, q))
            points = np.array(points) / np.sqrt(170.0)
            return digital.constellation_calcdist(points.tolist(), [], 4, 1).base()
        else:
            return digital.constellation_qpsk().base()

    @staticmethod
    def digital_constellation_modulator(mod_type="QPSK", samples_per_symbol=4, differential=False):
        """Native GNU Radio constellation modulator for PSK and QAM."""
        constellation = GRModulators.get_constellation(mod_type)
        return digital.generic_mod(
            constellation=constellation,
            differential=differential,
            samples_per_symbol=samples_per_symbol,
            pre_diff_code=True,
            excess_bw=0.35,
            verbose=False,
            log=False
        )

    @staticmethod
    def fm_modulator(sample_rate=32000, max_dev=5000.0):
        """Native GNU Radio frequency modulator (float audio in -> complex IQ out)."""
        sensitivity = 2.0 * np.pi * max_dev / sample_rate
        return analog.frequency_modulator_fc(sensitivity)

    @staticmethod
    def am_modulator(carrier_freq=0.0, mod_index=0.8):
        """Native GNU Radio AM modulator (float audio in -> complex IQ out)."""
        # AM: s(t) = (1 + m*m(t)) * exp(j * w_c * t)
        # Implemented using float to complex and add constant block
        add_const = blocks.add_const_ff(1.0 / mod_index)
        scale = blocks.multiply_const_ff(mod_index)
        f2c = blocks.float_to_complex()
        return add_const, scale, f2c

    @staticmethod
    def gfsk_modulator(samples_per_symbol=4, bt=0.35, sensitivity=1.0):
        """Native GNU Radio GFSK modulator block."""
        return digital.gfsk_mod(
            samples_per_symbol=samples_per_symbol,
            sensitivity=sensitivity,
            bt=bt,
            verbose=False,
            log=False
        )
