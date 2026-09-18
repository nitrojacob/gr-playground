"""
Modular Wideband Channel Identification Package (`gr_playground.dsp.channel_detection`).
Provides Strategy & Factory pattern interface (`get_channel_detector`).
"""

from gr_playground.dsp.channel_detection.base import BaseChannelDetector
from gr_playground.dsp.channel_detection.cfar_heuristic import CFARHeuristicChannelDetector
from gr_playground.dsp.channel_detection.ppd_heuristic import PPDHeuristicChannelDetector

def get_channel_detector(mode: str = "cfar_heuristic", **kwargs) -> BaseChannelDetector:
    """
    Factory function for Wideband Channel Detectors.

    Parameters:
    - mode: Strategy mode ("cfar_heuristic", "ppd_heuristic"). Default: "cfar_heuristic".

    Returns:
    - An instance conforming to BaseChannelDetector.
    """
    mode_str = str(mode).lower().strip()

    if mode_str in ("ppd_heuristic", "ppd", "point_peak"):
        return PPDHeuristicChannelDetector()

    # Default engine: CFARHeuristicChannelDetector
    return CFARHeuristicChannelDetector()


__all__ = [
    "BaseChannelDetector",
    "CFARHeuristicChannelDetector",
    "PPDHeuristicChannelDetector",
    "get_channel_detector",
]
