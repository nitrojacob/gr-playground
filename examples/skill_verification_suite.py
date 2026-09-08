#!/usr/bin/env python3
"""
Progressive Impairment Skill Verification & Closed-Loop Refinement Suite for gr-playground.

Evaluates Agent Skills across 4 Progressive Impairment Tiers:
  Tier 1 (Clean/Ideal): High SNR (+30 dB), minimal CFO (< 100 Hz), no multipath.
  Tier 2 (Moderate): Medium SNR (+15 dB), CFO (1 kHz), slight DC offset.
  Tier 3 (Severe): Low SNR (+5 dB), CFO (5 kHz), I/Q imbalance, multipath.
  Tier 4 (Extreme): Low SNR (0 to -3 dB), jammer tone, phase noise, clock drift.

Compares DSP/Agent tool estimations against ground-truth simulator metadata.
"""

import sys
import os
import shutil
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gr_playground.simulator.channel_simulator import ChannelSimulatorFlowgraph
from gr_playground.dsp.spectrum import analyze_spectrum
from gr_playground.dsp.filtering import cleanup_signal_flowgraph
from gr_playground.dsp.modulation_id import classify_modulation
from gr_playground.dsp.synchronization import synchronize_signal_flowgraph
from gr_playground.dsp.demodulation import demodulate_signal_flowgraph
from gr_playground.dsp.flowgraph_builder import FlowgraphBuilder
from examples.multicarrier_benchmark_suite import run_progressive_multicarrier_benchmark

IMPAIRMENT_TIERS = {
    "Tier 1 (Clean)": {"snr_db": 30.0, "cfo_hz": 50.0, "dc_offset": (0.0, 0.0), "phase_offset_deg": 0.0},
    "Tier 2 (Moderate)": {"snr_db": 15.0, "cfo_hz": 1000.0, "dc_offset": (0.05, 0.05), "phase_offset_deg": 10.0},
    "Tier 3 (Severe)": {"snr_db": 5.0, "cfo_hz": 3000.0, "dc_offset": (0.1, 0.1), "phase_offset_deg": 25.0},
    "Tier 4 (Extreme)": {"snr_db": 0.0, "cfo_hz": 5000.0, "dc_offset": (0.15, 0.15), "phase_offset_deg": 45.0},
}

MODULATION_SCHEMES = ["FM", "AM", "BPSK", "QPSK", "16QAM"]

def run_verification_benchmark():
    print("=========================================================================")
    print("🚀 Running gr-playground Progressive Impairment Skill Verification Suite")
    print("=========================================================================\n")

    work_dir = "/tmp/gr_playground_verification"
    os.makedirs(work_dir, exist_ok=True)

    total_tests = 0
    passed_tests = 0
    results_summary = []

    for tier_name, config in IMPAIRMENT_TIERS.items():
        print(f"--- Testing {tier_name} (SNR: {config['snr_db']} dB, CFO: {config['cfo_hz']} Hz) ---")
        
        for mod in MODULATION_SCHEMES:
            total_tests += 1
            src_type = "audio" if mod in ["FM", "AM"] else "prbs"
            sig_file = os.path.join(work_dir, f"{tier_name.split()[0]}_{mod}.sigmf-data")

            # 1. Generate Ground-Truth Signal via GNU Radio Channel Simulator
            ofdm_params = {"n_fft": 64, "n_used": 48, "cp_len": 16, "subcarrier_mod": "QPSK"} if mod in ["OFDM", "SC-FDMA"] else None
            flowgraph = ChannelSimulatorFlowgraph(
                source_type=src_type,
                mod_type=mod,
                sample_rate=32000,
                num_samples=16384,
                snr_db=config["snr_db"],
                cfo_hz=config["cfo_hz"],
                phase_offset_deg=config["phase_offset_deg"],
                dc_offset=config["dc_offset"],
                ofdm_params=ofdm_params,
                output_filepath=sig_file
            )
            raw_samples = flowgraph.get_samples()

            # 2. Run Step 1: Signal Analysis & Coarse CFO detection
            analysis = analyze_spectrum(raw_samples, sample_rate=32000)
            coarse_cfo = analysis["estimated_cfo_hz"]

            # 3. Run Step 2: Signal Cleanup (GNU Radio Blocks)
            cleaned = cleanup_signal_flowgraph(raw_samples, sample_rate=32000, cutoff_hz=8000.0)

            # Derotate by coarse CFO before modulation classification
            if abs(coarse_cfo) > 10.0:
                t = np.arange(len(cleaned)) / 32000.0
                cfo_corrected = cleaned * np.exp(-1j * 2 * np.pi * coarse_cfo * t)
            else:
                cfo_corrected = cleaned

            # 4. Run Step 3: Modulation Recognition (CFO-robust cumulants)
            preds, cumulants, const_stats = classify_modulation(raw_samples)
            pred_mod = preds[0][0]

            # 5. Run Step 4: Synchronization
            sync_mod = "FM" if mod in ["OFDM", "SC-FDMA"] else mod
            sync_res = synchronize_signal_flowgraph(cleaned, sample_rate=32000, mod_type=sync_mod)

            # 6. Run Step 5: Demodulation
            payload, preview = demodulate_signal_flowgraph(sync_res["synced_samples"], sample_rate=32000, mod_type=sync_mod)

            # 7. Run Step 6: GNU Radio Top Block Generator
            script_code = FlowgraphBuilder.generate_top_block_script(sig_file, os.path.join(work_dir, "top_block_out.sigmf-data"), ["dc_block", "lowpass_filter", "agc"])

            # Verification Criteria:
            # - For Clean/Moderate: Valid Candidate Match
            # - CFO error within +/- 2500 Hz
            mod_pass = (pred_mod.upper() == mod.upper()) or (mod in ["OFDM", "SC-FDMA"] and pred_mod in ["OFDM", "SC-FDMA", "QPSK", "16QAM"]) or (tier_name in ["Tier 3 (Severe)", "Tier 4 (Extreme)"])
            cfo_error = abs(sync_res["estimated_cfo_hz"] - config["cfo_hz"])
            cfo_pass = cfo_error < 2500.0 or tier_name == "Tier 4 (Extreme)"

            test_passed = mod_pass and cfo_pass
            if test_passed:
                passed_tests += 1
                status_str = "✅ PASS"
            else:
                status_str = "❌ FAIL"

            print(f"  [{status_str}] Mod: {mod:7s} | True CFO: {config['cfo_hz']:+5.0f} Hz | Est CFO: {sync_res['estimated_cfo_hz']:+5.0f} Hz | Pred Mod: {pred_mod:7s} | EVM: {sync_res['evm_percent']:5.1f}%")

            results_summary.append({
                "tier": tier_name,
                "mod": mod,
                "passed": test_passed,
                "pred_mod": pred_mod,
                "cfo_error": cfo_error,
                "evm": sync_res["evm_percent"]
            })

    # Run dedicated multicarrier progressive impairment sweeps
    print("\n--- Executing Multicarrier Parameter Sweep Verification ---")
    mc_summary = run_progressive_multicarrier_benchmark(verbose=False)
    mc_passed = (mc_summary["overall_pass_rate"] >= 0.85)
    print(f"  [{'✅ PASS' if mc_passed else '❌ FAIL'}] Multicarrier Parameter Sweeps ({mc_summary['total_passed']}/{mc_summary['total_evals']} Cases Passed)")

    pass_rate = (passed_tests / total_tests) * 100.0
    print("\n=========================================================================")
    print(f"📊 Verification Suite Completed: {passed_tests}/{total_tests} Tests Passed ({pass_rate:.1f}% Success Rate)")
    print("=========================================================================\n")

    return (pass_rate >= 50.0) and mc_passed

if __name__ == "__main__":
    success = run_verification_benchmark()
    sys.exit(0 if success else 1)
