"""
Layer-1 (L1 Physical Layer) Multipath Channel Equalization Module
Combats frequency-selective multipath fading and Inter-Symbol Interference (ISI) using:
- Zero-Forcing (ZF) Frequency Equalizer
- Minimum Mean Square Error (MMSE) Regularized Equalizer
- Decision-Directed Adaptive LMS Equalizer
- Native GNU Radio Constant Modulus Algorithm (CMA) Equalizer
"""

from gnuradio import gr, blocks, digital
import numpy as np
from typing import Optional, Union, Tuple

class L1Equalizer:
    """Layer-1 Multipath Channel Equalizer for complex IQ signals."""

    @staticmethod
    def equalize_zero_forcing(
        samples: np.ndarray,
        channel_taps: Optional[Union[list, np.ndarray]] = None,
        n_fft: Optional[int] = None
    ) -> np.ndarray:
        """
        Zero-Forcing (ZF) Equalizer.
        Inverts the estimated channel frequency response H(f).
        W(f) = 1 / H(f)
        """
        y = np.asarray(samples, dtype=np.complex64)
        if len(y) == 0:
            return y

        if channel_taps is None:
            # Simple 2-tap auto-estimate fallback if no taps provided
            channel_taps = [1.0, 0.2]

        h = np.asarray(channel_taps, dtype=np.complex64)
        n = n_fft if n_fft is not None else max(1024, len(y))

        # Channel frequency response
        H = np.fft.fft(h, n=n)
        
        # Zero-forcing inverse filter with regularization eps to prevent division by zero
        eps = 1e-4 * np.max(np.abs(H))
        W = 1.0 / (H + eps)

        # Equalize in frequency domain
        Y = np.fft.fft(y, n=n)
        X_hat = Y * W
        x_equalized = np.fft.ifft(X_hat)[:len(y)]

        return x_equalized.astype(np.complex64)

    @staticmethod
    def equalize_mmse(
        samples: np.ndarray,
        channel_taps: Optional[Union[list, np.ndarray]] = None,
        snr_db: float = 20.0,
        n_fft: Optional[int] = None
    ) -> np.ndarray:
        """
        Minimum Mean Square Error (MMSE) Equalizer.
        Regularizes channel inversion to prevent noise amplification at spectral nulls.
        W(f) = H*(f) / (|H(f)|^2 + 1/SNR)
        """
        y = np.asarray(samples, dtype=np.complex64)
        if len(y) == 0:
            return y

        if channel_taps is None:
            channel_taps = [1.0, 0.2]

        h = np.asarray(channel_taps, dtype=np.complex64)
        n = n_fft if n_fft is not None else max(1024, len(y))

        H = np.fft.fft(h, n=n)
        
        # Noise-to-signal ratio factor
        gamma = 10.0 ** (-snr_db / 10.0)
        
        # MMSE equalizer frequency response
        W = np.conj(H) / (np.abs(H)**2 + gamma)

        Y = np.fft.fft(y, n=n)
        X_hat = Y * W
        x_equalized = np.fft.ifft(X_hat)[:len(y)]

        return x_equalized.astype(np.complex64)

    @staticmethod
    def equalize_lms_adaptive(
        samples: np.ndarray,
        num_taps: int = 11,
        mu: float = 0.01,
        mod_type: str = "QPSK"
    ) -> np.ndarray:
        """
        Decision-Directed Adaptive Least Mean Squares (LMS) Equalizer.
        Dynamically adjusts complex FIR filter taps: w[n+1] = w[n] + mu * e[n] * x*[n]
        """
        y = np.asarray(samples, dtype=np.complex64)
        if len(y) < num_taps:
            return y

        # Center tap initialized to 1.0, others 0.0
        w = np.zeros(num_taps, dtype=np.complex64)
        w[num_taps // 2] = 1.0 + 0.0j

        output = np.zeros_like(y)
        buffer = np.zeros(num_taps, dtype=np.complex64)

        mod_upper = mod_type.upper()

        def quantize_symbol(val: complex) -> complex:
            """Hard decision slicer for decision-directed error calculation."""
            if mod_upper == "BPSK":
                return complex(1.0 if val.real > 0 else -1.0, 0.0)
            elif mod_upper in ["ASK", "2ASK", "OOK"]:
                return complex(1.0 if np.abs(val) > 0.5 else 0.0, 0.0)
            elif mod_upper == "8PSK":
                ang = np.angle(val) % (2 * np.pi)
                idx = int(np.round(ang / (np.pi / 4.0))) % 8
                return np.exp(1j * idx * np.pi / 4.0)
            else: # QPSK / QAM fallback
                re = 1.0 / np.sqrt(2.0) if val.real > 0 else -1.0 / np.sqrt(2.0)
                im = 1.0 / np.sqrt(2.0) if val.imag > 0 else -1.0 / np.sqrt(2.0)
                return complex(re, im)

        for i in range(len(y)):
            buffer = np.roll(buffer, 1)
            buffer[0] = y[i]

            # Filter output
            y_hat = np.dot(w, buffer)
            output[i] = y_hat

            # Decision directed error
            d_hat = quantize_symbol(y_hat)
            e = d_hat - y_hat

            # LMS tap update
            w += mu * e * np.conj(buffer)

        return output.astype(np.complex64)

    class CMAEqualizerFlowgraph(gr.top_block):
        def __init__(self, samples: np.ndarray, num_taps: int = 15, modulus: float = 1.0):
            super().__init__("CMAEqualizerFlowgraph")
            self.src = blocks.vector_source_c(samples.tolist(), False)
            self.cma = digital.cma_equalizer_cc(num_taps, modulus, 0.01, 1)
            self.sink = blocks.vector_sink_c()
            self.connect(self.src, self.cma, self.sink)

        def run_equalizer(self) -> np.ndarray:
            self.run()
            return np.array(self.sink.data(), dtype=np.complex64)

    @classmethod
    def equalize_gnuradio_cma(
        cls,
        samples: np.ndarray,
        num_taps: int = 15,
        modulus: float = 1.0
    ) -> np.ndarray:
        """Native GNU Radio Constant Modulus Algorithm (CMA) Adaptive Equalizer wrapper."""
        y = np.asarray(samples, dtype=np.complex64)
        tb = cls.CMAEqualizerFlowgraph(y, num_taps=num_taps, modulus=modulus)
        return tb.run_equalizer()
