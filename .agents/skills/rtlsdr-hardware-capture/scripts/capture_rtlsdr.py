#!/usr/bin/env python3
"""
CLI Tool: Generic Headless RTL-SDR RF Live Capture with Tunable Gain, Frequency, and DDC Channelization.
Location: gr-playground/.agents/skills/rtlsdr-hardware-capture/scripts/capture_rtlsdr.py
Exports natively to SigMF (.sigmf-data + .sigmf-meta) using native gr-osmosdr driver.
"""

import sys
import os
import time
import json
import argparse

# Resolve paths to load gr_playground package
SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SKILL_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gnuradio import gr, blocks, filter, analog
import osmosdr
from gnuradio.filter import firdes
from gnuradio.fft import window
from gr_playground.utils.summary import format_spectrum_summary

def parse_frequency(val_str):
    """Parse frequency string supporting Hz, kHz (k/kHz), MHz (M/MHz), GHz (G/GHz), and exponential notation."""
    if isinstance(val_str, (int, float)):
        return float(val_str)
    s = str(val_str).strip().upper()
    if s.endswith("GHZ") or s.endswith("G"):
        clean = s.replace("GHZ", "").replace("G", "").strip()
        return float(clean) * 1e9
    if s.endswith("MHZ") or s.endswith("M"):
        clean = s.replace("MHZ", "").replace("M", "").strip()
        return float(clean) * 1e6
    if s.endswith("KHZ") or s.endswith("K"):
        clean = s.replace("KHZ", "").replace("K", "").strip()
        return float(clean) * 1e3
    return float(s)

class HeadlessRtlSdrCaptureFlowgraph(gr.top_block):
    """
    Headless GNU Radio top block for live RTL-SDR RF capture.
    Constructs a pipeline:
      osmosdr.source (RTL-SDR) -> filter.freq_xlating_fir_filter_ccc (DDC) -> analog.agc2_cc -> blocks.file_sink
    No Qt GUI dependencies.
    """
    def __init__(
        self,
        center_freq=92.0e6,
        samp_rate=2400000,
        tuner_gain=20.0,
        if_gain=20.0,
        gain_mode="manual",
        offset_hz=200000.0,
        cutoff_hz=60000.0,
        decimation=10,
        pure_wideband=False,
        out_path="/tmp/rtlsdr_capture.sigmf-data"
    ):
        super().__init__("Headless RTL-SDR Live Capture", catch_exceptions=True)

        self.samp_rate = samp_rate
        self.center_freq = center_freq
        self.offset_hz = offset_hz
        self.decimation = decimation
        self.out_path = out_path

        # 1. RTL-SDR Hardware Source (gr-osmosdr)
        try:
            self.rtlsdr_source = osmosdr.source(args="rtl=0")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize RTL-SDR hardware via gr-osmosdr: {str(e)}\n"
                "Ensure an RTL-SDR USB dongle is connected and kernel DVB module is not locking the device."
            ) from e

        self.rtlsdr_source.set_sample_rate(samp_rate)
        self.rtlsdr_source.set_center_freq(center_freq, 0)
        self.rtlsdr_source.set_freq_corr(0, 0)

        # Gain Configuration
        use_auto_agc = (gain_mode.lower() in ["auto", "agc", "true", "1"])
        self.rtlsdr_source.set_gain_mode(use_auto_agc, 0)
        if not use_auto_agc:
            try:
                self.rtlsdr_source.set_gain(float(tuner_gain), 0)
            except Exception:
                pass
            try:
                self.rtlsdr_source.set_if_gain(float(if_gain), 0)
            except Exception:
                pass

        # 2. Digital Downconverter (DDC) or Direct Wideband Bypass
        self.pure_wideband = pure_wideband
        if pure_wideband:
            self.file_sink = blocks.file_sink(gr.sizeof_gr_complex, out_path, False)
            self.file_sink.set_unbuffered(False)
            self.connect((self.rtlsdr_source, 0), (self.file_sink, 0))
        else:
            lpf_taps = firdes.low_pass(1.0, samp_rate, cutoff_hz, cutoff_hz * 0.2, window.WIN_HAMMING, 6.76)
            self.freq_xlating_filter = filter.freq_xlating_fir_filter_ccc(
                int(decimation), lpf_taps, offset_hz, samp_rate
            )
            self.agc = analog.agc2_cc(1e-3, 1e-2, 1.0, 1.0, 65536)
            self.file_sink = blocks.file_sink(gr.sizeof_gr_complex, out_path, False)
            self.file_sink.set_unbuffered(False)

            self.connect((self.rtlsdr_source, 0), (self.freq_xlating_filter, 0))
            self.connect((self.freq_xlating_filter, 0), (self.agc, 0))
            self.connect((self.agc, 0), (self.file_sink, 0))

def execute_rtlsdr_capture(
    center_freq=92.0e6,
    samp_rate=2400000,
    duration_sec=30.0,
    tuner_gain=20.0,
    if_gain=20.0,
    gain_mode="manual",
    offset_hz=200000.0,
    cutoff_hz=60000.0,
    decimation=10,
    pure_wideband=False,
    output_path="/tmp/rtlsdr_capture.sigmf-data",
    description="Live RTL-SDR RF Capture"
):
    """
    Executes live RTL-SDR capture and exports SigMF data and metadata.
    Returns (data_path, meta_path, sample_count, output_rate).
    """
    base, _ = os.path.splitext(output_path)
    data_path = base + ".sigmf-data"
    meta_path = base + ".sigmf-meta"

    os.makedirs(os.path.dirname(os.path.abspath(data_path)), exist_ok=True)

    if os.path.exists(data_path):
        try:
            os.remove(data_path)
        except OSError:
            pass
    if os.path.exists(meta_path):
        try:
            os.remove(meta_path)
        except OSError:
            pass

    tb = HeadlessRtlSdrCaptureFlowgraph(
        center_freq=center_freq,
        samp_rate=samp_rate,
        tuner_gain=tuner_gain,
        if_gain=if_gain,
        gain_mode=gain_mode,
        offset_hz=offset_hz,
        cutoff_hz=cutoff_hz,
        decimation=decimation,
        pure_wideband=pure_wideband,
        out_path=data_path
    )

    output_rate = samp_rate if pure_wideband else (samp_rate // decimation)
    target_rf_freq = center_freq if pure_wideband else (center_freq + offset_hz)

    print(f"📡 Starting {duration_sec:.1f}s RTL-SDR {'PURE WIDEBAND' if pure_wideband else 'DDC'} capture at {center_freq/1e6:.3f} MHz...")
    print(f"   Parameters: SampRate={samp_rate/1e6:.2f}MSps, Gain={tuner_gain}dB ({gain_mode})" + (f", DDC Offset={offset_hz/1e3:.1f}kHz" if not pure_wideband else ""))

    tb.start()
    t_start = time.time()
    time.sleep(duration_sec)
    tb.stop()
    tb.wait()
    t_elapsed = time.time() - t_start

    file_bytes = os.path.getsize(data_path) if os.path.exists(data_path) else 0
    sample_count = file_bytes // 8

    # Write SigMF v1.0 Metadata
    metadata = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": float(output_rate),
            "core:version": "1.0.0",
            "core:description": f"{description} ({center_freq/1e6:.3f} MHz center, {tuner_gain} dB gain, {'Pure Wideband' if pure_wideband else 'DDC Channelized'})",
            "core:author": "gr-playground rtlsdr-hardware-capture skill",
            "hw:frontend": "RTL-SDR USB Dongle",
            "hw:tuner_gain_db": float(tuner_gain),
            "hw:if_gain_db": float(if_gain),
            "hw:gain_mode": gain_mode
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": float(target_rf_freq),
                "core:datetime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        ],
        "annotations": []
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Capture finished in {t_elapsed:.2f}s!")
    print(f"   Data File: {data_path} ({file_bytes:,} bytes, {sample_count:,} complex samples)")
    print(f"   Meta File: {meta_path} (Sample Rate: {output_rate/1e6:.2f} MSps)")

    return data_path, meta_path, sample_count, output_rate

def main():
    parser = argparse.ArgumentParser(description="Generic Headless RTL-SDR Live Capture Tool.")
    parser.add_argument("--freq", "--center_freq", type=str, default="92.0e6", help="RF Center Frequency in Hz, MHz (e.g. 92M, 100.2e6, 1420M)")
    parser.add_argument("--samp_rate", "--sample_rate", type=float, default=2400000.0, help="SDR wideband sampling rate (Hz)")
    parser.add_argument("--duration", "--time", type=float, default=30.0, help="Capture duration in seconds")
    parser.add_argument("--gain", "--tuner_gain", type=float, default=20.0, help="Tuner / RF gain in dB")
    parser.add_argument("--if_gain", type=float, default=20.0, help="IF / Intermediate Frequency gain in dB")
    parser.add_argument("--gain_mode", type=str, default="manual", choices=["manual", "auto", "agc"], help="Gain control mode (manual or auto/agc)")
    parser.add_argument("--offset", "--channel_offset", type=float, default=200000.0, help="DDC frequency offset in Hz")
    parser.add_argument("--cutoff", type=float, default=60000.0, help="DDC lowpass filter cutoff in Hz")
    parser.add_argument("--decimation", type=int, default=10, help="DDC decimation factor")
    parser.add_argument("--wideband", "--pure_wideband", action="store_true", help="Perform pure wideband capture (no DDC during capture)")
    parser.add_argument("--output", type=str, default="/tmp/rtlsdr_capture.sigmf-data", help="Output SigMF data filepath (.sigmf-data)")
    parser.add_argument("--description", type=str, default="Live RTL-SDR RF Capture", help="Dataset description string")

    args = parser.parse_args()

    center_freq_hz = parse_frequency(args.freq)

    execute_rtlsdr_capture(
        center_freq=center_freq_hz,
        samp_rate=args.samp_rate,
        duration_sec=args.duration,
        tuner_gain=args.gain,
        if_gain=args.if_gain,
        gain_mode=args.gain_mode,
        offset_hz=args.offset,
        cutoff_hz=args.cutoff,
        decimation=args.decimation,
        pure_wideband=args.wideband,
        output_path=args.output,
        description=args.description
    )

if __name__ == "__main__":
    main()
