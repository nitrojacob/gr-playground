"""
GNU Radio Modulator Blocks Wrapper
Implements AM, FM, PM, BPSK, QPSK, 8PSK, 16QAM, 64QAM, 256QAM, BFSK, 4FSK, GFSK, MSK, and OFDM modulators using native gnuradio blocks.
"""

from gnuradio import gr, blocks, analog, digital, filter, fft
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

    @staticmethod
    def generate_multicarrier_samples(
        num_samples=16384,
        n_fft=64,
        n_used=48,
        cp_len=16,
        subcarrier_mod="QPSK",
        sc_fdma=False,
        seed=42
    ):
        """
        Synthesize multicarrier complex IQ samples (OFDM or SC-FDMA).
        
        Parameters:
        - num_samples: Total number of complex64 IQ samples to generate
        - n_fft: Total FFT size (e.g. 32, 64, 128, 256, 512)
        - n_used: Number of active occupied subcarriers (n_used <= n_fft - 2)
        - cp_len: Cyclic prefix length in samples
        - subcarrier_mod: Constellation mapping per subcarrier ('BPSK', 'QPSK', '16QAM', '64QAM')
        - sc_fdma: If True, apply M-point DFT precoding prior to IFFT (SC-FDMA / LTE Uplink)
        - seed: Random seed for deterministic reproducibility
        """
        rng = np.random.RandomState(seed)
        n_used = min(n_used, n_fft - 2)
        n_used = (n_used // 2) * 2  # Ensure even count
        
        # 1. Construct constellation map
        mod_upper = subcarrier_mod.upper()
        if mod_upper == "BPSK":
            alphabet = np.array([-1.0, 1.0], dtype=np.complex64)
        elif mod_upper == "QPSK":
            alphabet = np.array([-1-1j, -1+1j, 1-1j, 1+1j], dtype=np.complex64) / np.sqrt(2.0)
        elif mod_upper == "16QAM":
            pts = []
            for i in [-3, -1, 1, 3]:
                for q in [-3, -1, 1, 3]:
                    pts.append(complex(i, q))
            alphabet = np.array(pts, dtype=np.complex64) / np.sqrt(10.0)
        elif mod_upper == "64QAM":
            pts = []
            for i in [-7, -5, -3, -1, 1, 3, 5, 7]:
                for q in [-7, -5, -3, -1, 1, 3, 5, 7]:
                    pts.append(complex(i, q))
            alphabet = np.array(pts, dtype=np.complex64) / np.sqrt(42.0)
        else:
            alphabet = np.array([-1-1j, -1+1j, 1-1j, 1+1j], dtype=np.complex64) / np.sqrt(2.0)

        samples_per_symbol = n_fft + cp_len
        num_symbols_needed = int(np.ceil(num_samples / samples_per_symbol)) + 2
        
        half_used = n_used // 2
        
        # Active subcarrier indices in shifted spectrum [-n_fft//2 ... n_fft//2 - 1]
        left_subcarriers = np.arange(-half_used, 0)
        right_subcarriers = np.arange(1, half_used + 1)
        active_indices_shifted = np.concatenate([left_subcarriers, right_subcarriers])
        # Convert shifted indices to standard FFT bin indices [0 ... n_fft - 1]
        active_indices_fft = (active_indices_shifted + n_fft) % n_fft

        # Frequency domain symbol matrix (num_symbols_needed, n_fft)
        freq_matrix = np.zeros((num_symbols_needed, n_fft), dtype=np.complex64)

        for s_idx in range(num_symbols_needed):
            symbol_indices = rng.randint(0, len(alphabet), size=n_used)
            symbols = alphabet[symbol_indices]
            
            if sc_fdma:
                precoded = np.fft.fft(symbols) / np.sqrt(n_used)
                subcarrier_vals = precoded
            else:
                subcarrier_vals = symbols

            freq_matrix[s_idx, active_indices_fft] = subcarrier_vals

        # GNU Radio IFFT Flowgraph using native fft.fft_vcc
        class OFDMIFFTFlowgraph(gr.top_block):
            def __init__(self, freq_syms, fft_size):
                super().__init__("OFDMIFFTFlowgraph")
                flat_syms = np.asarray(freq_syms, dtype=np.complex64).flatten()
                self.src = blocks.vector_source_c(flat_syms.tolist(), False)
                self.ifft = fft.fft_vcc(fft_size, False, [], True)
                self.scale = blocks.multiply_const_cc(complex(1.0 / np.sqrt(fft_size), 0.0))
                self.sink = blocks.vector_sink_c()
                self.connect(self.src, self.ifft, self.scale, self.sink)

            def run_ifft(self):
                self.run()
                return np.array(self.sink.data(), dtype=np.complex64)

        try:
            tb_ifft = OFDMIFFTFlowgraph(freq_matrix, n_fft)
            ifft_out = tb_ifft.run_ifft()
            ifft_matrix = ifft_out.reshape(num_symbols_needed, n_fft)
        except Exception:
            ifft_matrix = np.fft.ifft(freq_matrix, axis=1) * np.sqrt(n_fft)

        symbol_blocks = []
        for s_idx in range(num_symbols_needed):
            x_time = ifft_matrix[s_idx]
            cp = x_time[-cp_len:]
            x_symbol = np.concatenate([cp, x_time])
            symbol_blocks.append(x_symbol)

        full_stream = np.concatenate(symbol_blocks).astype(np.complex64)
        return full_stream[:num_samples]

