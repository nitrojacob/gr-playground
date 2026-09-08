#!/usr/bin/env python3
"""
Top-Level RF Signal Analysis Orchestrator Script.
Location: gr-playground/.agents/skills/analyze-rf-signal/scripts/analyze_rf_pipeline.py

Orchestrates the complete end-to-end RF signal analysis pipeline across prepackaged skills:
1. Spectrum Scanning & Channel Discovery (scan_wideband_channels)
2. Target Channel Extraction & DDC (extract_channel_flowgraph)
3. Signal Analysis & Spectrum Inspection (analyze_spectrum)
4. Signal Cleanup & IQ Balancing (cleanup_signal_flowgraph)
5. Automatic Modulation Classification (classify_modulation)
6. Carrier & Symbol Synchronization (synchronize_signal_flowgraph)
7. Payload Demodulation & Byte Decoding (demodulate_ask_ook / demodulate_signal_flowgraph)
8. Receiver Top Block Flowgraph Generation (FlowgraphBuilder)
"""

import sys
import os
import argparse
import json
import numpy as np

# Add project root and skill script directories to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEMOD_SCRIPT_DIR = os.path.join(PROJECT_ROOT, ".agents", "skills", "signal-demodulation", "scripts")
if DEMOD_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, DEMOD_SCRIPT_DIR)

from gr_playground.utils.sigmf_io import read_sigmf, write_sigmf
from gr_playground.dsp.channelizer import scan_wideband_channels, extract_channel_flowgraph
from gr_playground.dsp.spectrum import analyze_spectrum
from gr_playground.dsp.filtering import cleanup_signal_flowgraph
from gr_playground.dsp.modulation_id import classify_modulation
from gr_playground.dsp.synchronization import synchronize_signal_flowgraph
from gr_playground.dsp.demodulation import demodulate_signal_flowgraph
from gr_playground.dsp.flowgraph_builder import FlowgraphBuilder
from demodulate_ask import demodulate_ask_ook

def run_rf_analysis_pipeline(input_path, scan_only=False, channel_index=1, manual_offset=None, decimation=10, output_dir="/tmp", auto_all=False):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input RF file not found: {input_path}")

    samples, meta = read_sigmf(input_path)
    sample_rate = meta.get("global", {}).get("core:sample_rate", 2.4e6)
    center_freq = 0.0
    if meta.get("captures"):
        center_freq = meta["captures"][0].get("core:frequency", 0.0)

    print(f"\n=========================================================================")
    print(f"🛰️  Top-Level RF Signal Analysis Orchestrator")
    print(f"=========================================================================")
    print(f"Input Dataset        : {input_path}")
    print(f"Center Frequency     : {center_freq/1e6:.3f} MHz")
    print(f"Wideband Sample Rate : {sample_rate/1e6:.2f} MSps ({len(samples):,} samples)")

    # 1. Scan wideband spectrum for candidate channels
    detected_channels = scan_wideband_channels(samples, sample_rate=sample_rate, num_channels_max=10)

    print("\n### Localized Wideband Signal Peaks")
    print("| Rank | Frequency Offset | Absolute RF Freq | Local SNR | Integrated Band Power | Est Bandwidth | Channel Type |")
    print("|------|------------------|------------------|-----------|-----------------------|---------------|--------------|")
    for ch in detected_channels:
        abs_rf = center_freq + ch["freq_offset_hz"]
        ch_type = ch.get("channel_type", "Wideband Channel")
        local_snr = ch.get("local_snr_db", 0.0)
        print(f"| #{ch['channel_id']} | {ch['freq_offset_hz']/1e3:+8.1f} kHz | {abs_rf/1e6:10.4f} MHz | +{local_snr:4.1f} dB | {ch['power_db']:+6.1f} dB | {ch['bandwidth_hz']/1e3:6.1f} kHz | {ch_type} |")

    if scan_only:
        print("\n📋 Wideband spectral scan complete. Select a channel index (--channel_index N) or manual offset (--manual_offset OFFSET_HZ) to execute full DSP pipeline.")
        return {"detected_channels": detected_channels}

    # Determine channels to process
    channels_to_process = []
    if manual_offset is not None:
        channels_to_process.append({"channel_id": 1, "freq_offset_hz": manual_offset, "bandwidth_hz": 100000.0})
    elif auto_all:
        channels_to_process = detected_channels
    else:
        selected_ch = detected_channels[min(channel_index - 1, len(detected_channels) - 1)]
        channels_to_process.append(selected_ch)

    pipeline_reports = []

    for target_ch in channels_to_process:
        offset_hz = target_ch["freq_offset_hz"]
        abs_rf = center_freq + offset_hz
        print(f"\n-------------------------------------------------------------------------")
        print(f"🎯 Processing Channel #{target_ch.get('channel_id', 1)} at {offset_hz/1e3:+.1f} kHz (RF: {abs_rf/1e6:.4f} MHz)")
        print(f"-------------------------------------------------------------------------")

        # Step 2: DDC Channel Extraction
        extracted_sigmf = os.path.join(output_dir, f"extracted_ch_{int(abs_rf/1e3)}k.sigmf-data")
        if len(samples) > 1000000:
            from scipy import signal
            target_bw = target_ch.get("bandwidth_hz", 100000.0)
            taps = signal.firwin(101, (target_bw / 2.0) / (sample_rate / 2.0))
            chunk_size = 2000000
            extracted_chunks = []
            for c_i in range(0, len(samples), chunk_size):
                chunk = samples[c_i:c_i+chunk_size]
                t_chunk = (np.arange(c_i, c_i + len(chunk), dtype=np.float64)) / sample_rate
                phase = (-2.0 * np.pi * offset_hz * t_chunk).astype(np.float32)
                rot = np.cos(phase) + 1j * np.sin(phase)
                trans = chunk * rot
                filt = signal.lfilter(taps, 1.0, trans)
                extracted_chunks.append(filt[::decimation].astype(np.complex64))
            extracted_samples = np.concatenate(extracted_chunks)
            write_sigmf(extracted_sigmf, extracted_samples, sample_rate=sample_rate/decimation, center_freq=abs_rf)
            ext_meta = {}
        else:
            extracted_samples, ext_meta = extract_channel_flowgraph(
                samples,
                sample_rate=sample_rate,
                freq_offset_hz=offset_hz,
                target_bw_hz=target_ch.get("bandwidth_hz", 100000.0),
                decimation=decimation,
                center_freq=center_freq,
                output_sigmf_path=extracted_sigmf
            )
        ch_rate = sample_rate / decimation

        # Step 3: Spectral Analysis
        spec_metrics = analyze_spectrum(extracted_samples, sample_rate=ch_rate)
        print(f"📊 [1/5 Signal Analysis] SNR: {spec_metrics['snr_m2m4_db']:.1f} dB (PSD Peak: {spec_metrics['snr_psd_db']:.1f} dB), Occupied BW: {spec_metrics['occupied_bw_hz']/1e3:.1f} kHz")

        # Step 4: Signal Cleanup
        cleaned_samples = cleanup_signal_flowgraph(extracted_samples, sample_rate=ch_rate, cutoff_hz=spec_metrics['occupied_bw_hz']/2.0)
        cleaned_sigmf = os.path.join(output_dir, f"cleaned_ch_{int(abs_rf/1e3)}k.sigmf-data")
        write_sigmf(cleaned_sigmf, cleaned_samples, sample_rate=ch_rate, center_freq=abs_rf)
        print(f"🧼 [2/5 Signal Cleanup] Applied Gram-Schmidt IQ balancing, DC blocker, and bandpass filter.")

        # Slice active frame segment for AMC, sync, and demodulation (max 500k samples ~ 2 sec)
        proc_samples = cleaned_samples[:500000]

        # Step 5: Automatic Modulation Recognition
        preds, cumulants, const_stats = classify_modulation(proc_samples)
        top_mod, top_conf = preds[0] if preds else ("Unknown", 0.0)
        print(f"🔍 [3/5 Modulation ID] Predicted Modulation: {top_mod} ({top_conf*100:.1f}% confidence)")

        # Step 6: Synchronization & Demodulation
        synced_sigmf = os.path.join(output_dir, f"synced_ch_{int(abs_rf/1e3)}k.sigmf-data")
        payload_result = {}

        if top_mod == "ASK":
            print(f"⏱️ [4/5 Synchronization] ASK envelope keying detected. Performing pulse timing recovery...")
            ask_report = demodulate_ask_ook(cleaned_sigmf, output_json=os.path.join(output_dir, "ask_demod_report.json"))
            print(f"🎙️ [5/5 Demodulation] Extracted {ask_report['total_frames_detected']} PWM/OOK packet frames.")
            payload_result = {"ask_report": ask_report}
        elif top_mod in ["AM", "FM"]:
            print(f"⏱️ [4/5 Synchronization] Analog {top_mod} signal. Proceeding to demodulation...")
            audio_out = os.path.join(output_dir, f"demodulated_audio_{top_mod.lower()}.wav")
            audio_samples = demodulate_signal_flowgraph(proc_samples, sample_rate=ch_rate, mod_type=top_mod, output_wav_path=audio_out)
            print(f"🎙️ [5/5 Demodulation] Demodulated analog {top_mod} audio to: {audio_out}")
            payload_result = {"audio_path": audio_out, "audio_samples_count": len(audio_samples)}
        else:
            sync_res = synchronize_signal_flowgraph(proc_samples, sample_rate=ch_rate, mod_type=top_mod)
            synced_samples = sync_res["synced_samples"]
            evm_pct = sync_res["evm_percent"]
            write_sigmf(synced_sigmf, synced_samples, sample_rate=ch_rate, center_freq=abs_rf)
            demod_bits, preview_str = demodulate_signal_flowgraph(synced_samples, sample_rate=ch_rate, mod_type=top_mod)
            print(f"🎙️ [5/5 Demodulation] Synchronized (EVM: {evm_pct:.1f}%) and demodulated {top_mod} bitstream: {preview_str}")
            payload_result = {"evm_pct": evm_pct, "demod_bits_count": len(demod_bits), "preview": preview_str}

        # Step 7: GNU Radio Flowgraph Top Block Generator
        script_path = os.path.join(output_dir, f"receiver_top_block_{top_mod.lower()}.py")
        ops_list = ["freq_xlating_filter", "dc_block", "agc"]
        if top_mod in ["BPSK", "QPSK", "16QAM"]:
            ops_list.extend(["costas_loop", "symbol_sync"])
        script_code = FlowgraphBuilder.generate_top_block_script(input_path, os.path.join(output_dir, f"output_ch_{int(abs_rf/1e3)}k.sigmf-data"), ops_list, sample_rate=sample_rate, center_freq_offset=offset_hz)
        with open(script_path, "w") as f:
            f.write(script_code)
        print(f"🛠️ [Flowgraph Generator] Generated standalone GNU Radio top_block script: {script_path}")

        ch_report = {
            "channel_id": target_ch.get("channel_id", 1),
            "freq_offset_hz": offset_hz,
            "abs_rf_freq_hz": abs_rf,
            "snr_db": spec_metrics["snr_m2m4_db"],
            "occupied_bw_hz": spec_metrics["occupied_bw_hz"],
            "predicted_modulation": top_mod,
            "confidence": top_conf,
            "extracted_sigmf": extracted_sigmf,
            "cleaned_sigmf": cleaned_sigmf,
            "generated_top_block_script": script_path,
            "payload_result": payload_result
        }
        pipeline_reports.append(ch_report)

    summary_report = {
        "input_dataset": input_path,
        "sample_rate": sample_rate,
        "center_freq": center_freq,
        "processed_channels_count": len(pipeline_reports),
        "channel_reports": pipeline_reports
    }

    report_path = os.path.join(output_dir, "rf_analysis_summary.json")
    with open(report_path, "w") as f:
        json.dump(summary_report, f, indent=2)

    print(f"\n=========================================================================")
    print(f"✅ End-to-End RF Signal Analysis Pipeline Complete!")
    print(f"✅ Full Report Saved to: {report_path}")
    print(f"=========================================================================\n")

    return summary_report

def main():
    parser = argparse.ArgumentParser(description="Top-Level RF Signal Analysis Orchestrator Script")
    parser.add_argument("--input", type=str, required=True, help="Input SigMF file path (.sigmf-data or .sigmf-meta)")
    parser.add_argument("--scan_only", action="store_true", help="Only scan wideband spectrum and list candidate channels")
    parser.add_argument("--channel_index", type=int, default=1, help="Target candidate channel index to process (1 = strongest active peak)")
    parser.add_argument("--manual_offset", type=float, default=None, help="Explicit frequency offset in Hz to process")
    parser.add_argument("--decimation", type=int, default=10, help="DDC decimation factor (default 10 -> 240 kSps)")
    parser.add_argument("--output_dir", type=str, default="/tmp", help="Output directory for generated datasets and scripts")
    parser.add_argument("--auto_all", action="store_true", help="Process all localized candidate channels automatically in batch")

    args = parser.parse_args()

    run_rf_analysis_pipeline(
        input_path=args.input,
        scan_only=args.scan_only,
        channel_index=args.channel_index,
        manual_offset=args.manual_offset,
        decimation=args.decimation,
        output_dir=args.output_dir,
        auto_all=args.auto_all
    )

if __name__ == "__main__":
    main()
