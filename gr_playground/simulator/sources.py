"""
GNU Radio Source Blocks Wrapper
Provides Sine, Square, Audio File Loop, PRBS Bit Stream, and Noise sources using native gnuradio blocks.
"""

import os
from gnuradio import gr, blocks, analog, digital
from scipy.io import wavfile
import numpy as np

class GRSources:
    @staticmethod
    def sine_source(sample_rate=32000, tone_freq=1000.0, amplitude=1.0):
        """Native GNU Radio complex sine tone source."""
        return analog.sig_source_c(sample_rate, analog.GR_COS_WAVE, tone_freq, amplitude, 0.0)

    @staticmethod
    def square_source(sample_rate=32000, tone_freq=500.0, amplitude=1.0):
        """Native GNU Radio float square wave source."""
        return analog.sig_source_f(sample_rate, analog.GR_SQUARE_WAVE, tone_freq, amplitude, 0.0)

    @staticmethod
    def audio_loop_source(wav_path, target_sample_rate=32000):
        """
        Native GNU Radio WAV file loop source. Reads WAV file into float vector source with repeat=True.
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"WAV audio file not found: {wav_path}")
        
        sr, data = wavfile.read(wav_path)
        if data.ndim > 1:
            data = data[:, 0]  # Mono
        
        # Normalize float samples between -1.0 and 1.0
        if data.dtype == np.int16:
            data_float = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data_float = data.astype(np.float32) / 2147483648.0
        else:
            data_float = data.astype(np.float32)

        return blocks.vector_source_f(data_float.tolist(), repeat=True)

    @staticmethod
    def prbs_source(degree=7):
        """Native GNU Radio PRBS byte source for digital data payloads."""
        np.random.seed(42)
        random_bytes = np.random.randint(0, 256, 4096, dtype=np.uint8).tolist()
        return blocks.vector_source_b(random_bytes, repeat=True)

    @staticmethod
    def noise_source(noise_type=analog.GR_GAUSSIAN, amplitude=1.0, seed=42):
        """Native GNU Radio fast noise source."""
        return analog.fastnoise_source_c(noise_type, amplitude, seed)
