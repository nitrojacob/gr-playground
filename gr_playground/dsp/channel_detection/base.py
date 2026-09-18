"""
Base Abstract Interface for Wideband Channel Detection Strategy Pattern.
All channel detectors (CFAR, Point Peak Detection, ML, Deep Learning) inherit
from BaseChannelDetector and implement a unified `detect_channels(...)` entry point.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseChannelDetector(ABC):
    @abstractmethod
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
        """
        Detect active sub-channels in wideband IQ signal / spectrum.

        Parameters:
        - iq_data: 1D complex64 numpy array of time-domain IQ samples (full frame or streaming segment).
        - sample_rate: Total sample rate in Hz.
        - psd_db: Optional precomputed 1D float64 array of Welch PSD in dB.
        - freqs: Optional precomputed 1D float64 array of frequency offsets in Hz.
        - target_channel_bw: Expected nominal target channel bandwidth in Hz (default 100 kHz).
        - min_snr_db: Prominence threshold above local noise floor in dB (default 3.0 dB).
        - num_channels_max: Maximum number of channels to return (default 10).
        - reject_spurs: Whether to filter out narrow single-bin CW spurs & LO leakage.
        - center_freq: RF center frequency in Hz (default 0.0 Hz).
        - session_id: Optional RF stream / acquisition session ID for stateful models.
        - reset_state: Explicit flag to flush internal STFT overlap buffers or tracking state.

        Returns:
        - List of dictionary descriptors for each detected channel:
          [
            {
              "freq_offset_hz": float,         # Center frequency offset relative to center_freq
              "power_db": float,               # Integrated band power across occupied bandwidth in dB
              "peak_single_bin_db": float,     # Peak single-bin PSD value in dB
              "local_snr_db": float,           # Estimated local SNR above noise floor in dB
              "bandwidth_hz": float,           # Occupied bandwidth (99% OBW) in Hz
              "channel_type": str,             # "Wideband Channel", "Narrowband Signal", or "Narrow Spur / CW Tone"
              "confidence": float,             # Detection confidence score (0.0 to 1.0)
              "bounds_hz": (float, float),     # (lower_freq_hz, upper_freq_hz) boundary tuple
            },
            ...
          ]
        """
        pass

    def reset_state(self) -> None:
        """
        Reset internal temporal memory, STFT overlap buffers, and tracking state.
        Default implementation is a no-op for stateless detectors.
        """
        pass
