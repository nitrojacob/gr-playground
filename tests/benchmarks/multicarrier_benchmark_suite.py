"""
Multicarrier (OFDM & SC-FDMA) Progressive Impairment Benchmark Suite & Sweeper
Progressively evaluates sub-parameter sweeps (subcarrier spacing, subcarrier bandwidth/count, cyclic prefix ratio, SC-FDMA vs OFDM PAPR under HPA Saturation)
across 6 progressive channel impairment levels (AWGN, CFO, SRO, DC offset, I/Q imbalance, LO phase noise, HPA saturation, multipath fading).
"""

import os
import sys
import json
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gr_playground.simulator.channel_simulator import ChannelSimulatorFlowgraph
from gr_playground.dsp.spectrum import analyze_spectrum
from gr_playground.utils.sigmf_io import read_sigmf

PROGRESSIVE_IMPAIRMENT_LEVELS = [
    {
        "level": 0,
        "name": "Level 0: Ideal Channel",
        "snr_db": 30.0,
        "cfo_hz": 0.0,
        "sro_ppm": 0.0,
        "dc_offset": (0.0, 0.0),
        "mag_imbalance_db": 0.0,
        "phase_imbalance_deg": 0.0,
        "phase_noise_std": 0.0,
        "hpa_ibo_db": None,
        "multipath_taps": None
    },
    {
        "level": 1,
        "name": "Level 1: Low AWGN, CFO & Phase Noise",
        "snr_db": 22.0,
        "cfo_hz": 50.0,
        "sro_ppm": 0.0,
        "dc_offset": (0.0, 0.0),
        "mag_imbalance_db": 0.0,
        "phase_imbalance_deg": 0.0,
        "phase_noise_std": 0.005,
        "hpa_ibo_db": None,
        "multipath_taps": None
    },
    {
        "level": 2,
        "name": "Level 2: Moderate Noise, Clock Drift & LO Jitter",
        "snr_db": 16.0,
        "cfo_hz": 250.0,
        "sro_ppm": 10.0,
        "dc_offset": (0.0, 0.0),
        "mag_imbalance_db": 0.5,
        "phase_imbalance_deg": 1.0,
        "phase_noise_std": 0.01,
        "hpa_ibo_db": None,
        "multipath_taps": None
    },
    {
        "level": 3,
        "name": "Level 3: I/Q Imbalance, DC Offset & CFO",
        "snr_db": 14.0,
        "cfo_hz": 1000.0,
        "sro_ppm": 20.0,
        "dc_offset": (0.10, -0.05),
        "mag_imbalance_db": 1.0,
        "phase_imbalance_deg": 2.0,
        "phase_noise_std": 0.015,
        "hpa_ibo_db": 5.0,
        "multipath_taps": None
    },
    {
        "level": 4,
        "name": "Level 4: Multipath Fading & HPA Saturation",
        "snr_db": 12.0,
        "cfo_hz": 1500.0,
        "sro_ppm": 25.0,
        "dc_offset": (0.10, -0.05),
        "mag_imbalance_db": 1.5,
        "phase_imbalance_deg": 3.0,
        "phase_noise_std": 0.02,
        "hpa_ibo_db": 3.5,
        "multipath_taps": [1.0 + 0.0j, 0.5 * np.exp(1j * 0.3), 0.25 * np.exp(1j * 0.7)]
    },
    {
        "level": 5,
        "name": "Level 5: Severe Stress Test & Heavy HPA Clipping",
        "snr_db": 8.0,
        "cfo_hz": 2500.0,
        "sro_ppm": 50.0,
        "dc_offset": (0.20, -0.10),
        "mag_imbalance_db": 2.0,
        "phase_imbalance_deg": 5.0,
        "phase_noise_std": 0.03,
        "hpa_ibo_db": 2.0,
        "multipath_taps": [1.0 + 0.0j, 0.7 * np.exp(1j * 0.5), 0.4 * np.exp(1j * 1.2)]
    }
]

def compute_papr_db(samples):
    """Calculate Peak-to-Average Power Ratio (PAPR) in dB."""
    p_instantaneous = np.abs(samples)**2
    p_peak = np.max(p_instantaneous)
    p_avg = np.mean(p_instantaneous) + 1e-12
    return float(10.0 * np.log10(p_peak / p_avg))

def run_multicarrier_benchmark_suite(impairment_profile=None, verbose=True):
    """
    Run multicarrier sub-parameter sweeps under a specified channel impairment profile.
    """
    if impairment_profile is None:
        impairment_profile = PROGRESSIVE_IMPAIRMENT_LEVELS[1]

    snr_db = impairment_profile.get("snr_db", 22.0)
    cfo_hz = impairment_profile.get("cfo_hz", 50.0)
    sro_ppm = impairment_profile.get("sro_ppm", 0.0)
    dc_offset = impairment_profile.get("dc_offset", (0.0, 0.0))
    mag_imbalance_db = impairment_profile.get("mag_imbalance_db", 0.0)
    phase_imbalance_deg = impairment_profile.get("phase_imbalance_deg", 0.0)
    phase_noise_std = impairment_profile.get("phase_noise_std", 0.0)
    hpa_ibo_db = impairment_profile.get("hpa_ibo_db", None)
    multipath_taps = impairment_profile.get("multipath_taps", None)

    sample_rate = 2400000  # 2.4 MSPS
    num_samples = 16384
    output_dir = "/tmp/multicarrier_benchmark"
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "impairment_profile": impairment_profile,
        "sweeps": {},
        "total_cases": 0,
        "passed_cases": 0,
        "pass_rate": 0.0,
        "summary": ""
    }

    test_count = 0
    passed_count = 0

    # -------------------------------------------------------------------------
    # SWEEP 1: Subcarrier Spacing / Separation (Delta f = fs / N_fft)
    # -------------------------------------------------------------------------
    sweep1_results = []
    n_fft_list = [64, 128, 256, 512]
    for n_fft in n_fft_list:
        test_count += 1
        delta_f_hz = sample_rate / n_fft
        n_used = int(0.75 * n_fft)
        cp_len = n_fft // 4
        
        out_file = os.path.join(output_dir, f"ofdm_spacing_fft{n_fft}.sigmf-data")
        fg = ChannelSimulatorFlowgraph(
            source_type="sine",
            mod_type="OFDM",
            sample_rate=sample_rate,
            num_samples=num_samples,
            snr_db=snr_db,
            cfo_hz=cfo_hz,
            sro_ppm=sro_ppm,
            dc_offset=dc_offset,
            mag_imbalance_db=mag_imbalance_db,
            phase_imbalance_deg=phase_imbalance_deg,
            phase_noise_std=phase_noise_std,
            hpa_ibo_db=hpa_ibo_db,
            multipath_taps=multipath_taps,
            ofdm_params={"n_fft": n_fft, "n_used": n_used, "cp_len": cp_len, "subcarrier_mod": "QPSK"},
            output_filepath=out_file
        )
        samples = fg.get_samples()
        
        read_samps, meta = read_sigmf(out_file)
        sigmf_ok = (len(read_samps) == num_samples and meta["global"]["core:sample_rate"] == sample_rate)
        
        papr_db = compute_papr_db(samples)
        passed = sigmf_ok and (papr_db > 1.2)
        if passed:
            passed_count += 1

        sweep1_results.append({
            "n_fft": n_fft,
            "subcarrier_spacing_hz": delta_f_hz,
            "n_used": n_used,
            "cp_len": cp_len,
            "papr_db": round(papr_db, 2),
            "sigmf_valid": sigmf_ok,
            "passed": passed
        })

    results["sweeps"]["subcarrier_spacing_sweep"] = sweep1_results

    # -------------------------------------------------------------------------
    # SWEEP 2: Subcarrier Bandwidth / Occupied Subcarriers (N_used)
    # -------------------------------------------------------------------------
    sweep2_results = []
    n_used_list = [12, 24, 36, 48]
    n_fft = 64
    subcarrier_spacing = sample_rate / n_fft  # 37.5 kHz
    
    for n_used in n_used_list:
        test_count += 1
        expected_bw_hz = n_used * subcarrier_spacing
        cp_len = 16

        out_file = os.path.join(output_dir, f"ofdm_bw_used{n_used}.sigmf-data")
        fg = ChannelSimulatorFlowgraph(
            source_type="sine",
            mod_type="OFDM",
            sample_rate=sample_rate,
            num_samples=num_samples,
            snr_db=snr_db,
            cfo_hz=cfo_hz,
            sro_ppm=sro_ppm,
            dc_offset=dc_offset,
            mag_imbalance_db=mag_imbalance_db,
            phase_imbalance_deg=phase_imbalance_deg,
            phase_noise_std=phase_noise_std,
            hpa_ibo_db=hpa_ibo_db,
            multipath_taps=multipath_taps,
            ofdm_params={"n_fft": n_fft, "n_used": n_used, "cp_len": cp_len, "subcarrier_mod": "QPSK"},
            output_filepath=out_file
        )
        samples = fg.get_samples()
        
        spec = analyze_spectrum(samples, sample_rate=sample_rate)
        meas_bw_hz = spec["occupied_bw_hz"]
        
        bw_scaling_ok = (meas_bw_hz > 0.30 * expected_bw_hz)
        passed = bw_scaling_ok
        if passed:
            passed_count += 1

        sweep2_results.append({
            "n_used": n_used,
            "expected_bw_hz": expected_bw_hz,
            "measured_bw_hz": meas_bw_hz,
            "bw_scaling_ok": bw_scaling_ok,
            "passed": passed
        })

    results["sweeps"]["subcarrier_bandwidth_sweep"] = sweep2_results

    # -------------------------------------------------------------------------
    # SWEEP 3: Cyclic Prefix Length / Ratio (CP / N_fft)
    # -------------------------------------------------------------------------
    sweep3_results = []
    cp_list = [4, 8, 16, 32]
    n_fft = 64
    n_used = 48

    for cp_len in cp_list:
        test_count += 1
        cp_ratio = cp_len / n_fft
        symbol_len = n_fft + cp_len

        out_file = os.path.join(output_dir, f"ofdm_cp_{cp_len}.sigmf-data")
        fg = ChannelSimulatorFlowgraph(
            source_type="sine",
            mod_type="OFDM",
            sample_rate=sample_rate,
            num_samples=num_samples,
            snr_db=snr_db,
            cfo_hz=cfo_hz,
            sro_ppm=sro_ppm,
            dc_offset=dc_offset,
            mag_imbalance_db=mag_imbalance_db,
            phase_imbalance_deg=phase_imbalance_deg,
            phase_noise_std=phase_noise_std,
            hpa_ibo_db=hpa_ibo_db,
            multipath_taps=multipath_taps,
            ofdm_params={"n_fft": n_fft, "n_used": n_used, "cp_len": cp_len, "subcarrier_mod": "QPSK"},
            output_filepath=out_file
        )
        samples = fg.get_samples()
        
        papr_db = compute_papr_db(samples)
        passed = (len(samples) == num_samples) and (papr_db > 1.2)
        if passed:
            passed_count += 1

        sweep3_results.append({
            "cp_len": cp_len,
            "cp_ratio": round(cp_ratio, 4),
            "symbol_length_samples": symbol_len,
            "papr_db": round(papr_db, 2),
            "passed": passed
        })

    results["sweeps"]["cyclic_prefix_sweep"] = sweep3_results

    # -------------------------------------------------------------------------
    # SWEEP 4: SC-FDMA vs OFDM PAPR Comparison under HPA Saturation
    # -------------------------------------------------------------------------
    sweep4_results = []
    constellations = ["QPSK", "16QAM"]
    n_fft = 64
    n_used = 48
    cp_len = 16

    for sub_mod in constellations:
        test_count += 1
        
        # Standard OFDM
        fg_ofdm = ChannelSimulatorFlowgraph(
            source_type="sine",
            mod_type="OFDM",
            sample_rate=sample_rate,
            num_samples=num_samples,
            snr_db=snr_db,
            cfo_hz=cfo_hz,
            sro_ppm=sro_ppm,
            dc_offset=dc_offset,
            mag_imbalance_db=mag_imbalance_db,
            phase_imbalance_deg=phase_imbalance_deg,
            phase_noise_std=phase_noise_std,
            hpa_ibo_db=hpa_ibo_db,
            multipath_taps=multipath_taps,
            ofdm_params={"n_fft": n_fft, "n_used": n_used, "cp_len": cp_len, "subcarrier_mod": sub_mod}
        )
        s_ofdm = fg_ofdm.get_samples()
        papr_ofdm = compute_papr_db(s_ofdm)

        # SC-FDMA (DFT-precoded OFDM)
        fg_scfdma = ChannelSimulatorFlowgraph(
            source_type="sine",
            mod_type="SC-FDMA",
            sample_rate=sample_rate,
            num_samples=num_samples,
            snr_db=snr_db,
            cfo_hz=cfo_hz,
            sro_ppm=sro_ppm,
            dc_offset=dc_offset,
            mag_imbalance_db=mag_imbalance_db,
            phase_imbalance_deg=phase_imbalance_deg,
            phase_noise_std=phase_noise_std,
            hpa_ibo_db=hpa_ibo_db,
            multipath_taps=multipath_taps,
            ofdm_params={"n_fft": n_fft, "n_used": n_used, "cp_len": cp_len, "subcarrier_mod": sub_mod}
        )
        s_scfdma = fg_scfdma.get_samples()
        papr_scfdma = compute_papr_db(s_scfdma)

        papr_reduction_db = papr_ofdm - papr_scfdma
        papr_check = (papr_scfdma <= papr_ofdm + 1.2)
        passed = papr_check
        if passed:
            passed_count += 1

        sweep4_results.append({
            "subcarrier_mod": sub_mod,
            "papr_ofdm_db": round(papr_ofdm, 2),
            "papr_scfdma_db": round(papr_scfdma, 2),
            "papr_reduction_db": round(papr_reduction_db, 2),
            "passed": passed
        })

    results["sweeps"]["scfdma_vs_ofdm_papr_sweep"] = sweep4_results

    # -------------------------------------------------------------------------
    # Summary Calculations
    # -------------------------------------------------------------------------
    results["total_cases"] = test_count
    results["passed_cases"] = passed_count
    results["pass_rate"] = round(passed_count / test_count, 4) if test_count > 0 else 0.0
    profile_name = impairment_profile.get("name", f"SNR={snr_db}dB")
    results["summary"] = f"[{profile_name}] Passed {passed_count}/{test_count} cases ({results['pass_rate']*100:.1f}%)"

    if verbose:
        print(f"\n=========================================================================")
        print(f"       MULTICARRIER BENCHMARK: {profile_name} ({results['pass_rate']*100:.1f}% PASS RATE)")
        print("=========================================================================")
        print(f"Profile Specs: SNR={snr_db}dB | CFO={cfo_hz}Hz | PhaseNoise={phase_noise_std}rad | HPA_IBO={'None' if hpa_ibo_db is None else f'{hpa_ibo_db}dB'}")
        print("-------------------------------------------------------------------------")
        print("[Sweep 1: Subcarrier Spacing Sweep (Delta f)]")
        for res in sweep1_results:
            print(f"  N_fft={res['n_fft']:3d} | Delta_f={res['subcarrier_spacing_hz']/1e3:6.2f} kHz | PAPR={res['papr_db']:5.2f} dB | Passed={res['passed']}")
        print("-------------------------------------------------------------------------")
        print("[Sweep 2: Subcarrier Bandwidth Sweep (N_used)]")
        for res in sweep2_results:
            print(f"  N_used={res['n_used']:2d} | Exp BW={res['expected_bw_hz']/1e3:6.1f} kHz | Meas BW={res['measured_bw_hz']/1e3:6.1f} kHz | Passed={res['passed']}")
        print("-------------------------------------------------------------------------")
        print("[Sweep 3: Cyclic Prefix Sweep (CP / N_fft)]")
        for res in sweep3_results:
            print(f"  CP={res['cp_len']:2d} | Ratio={res['cp_ratio']:6.3f} | PAPR={res['papr_db']:5.2f} dB | Passed={res['passed']}")
        print("-------------------------------------------------------------------------")
        print("[Sweep 4: SC-FDMA vs OFDM PAPR Comparison]")
        for res in sweep4_results:
            print(f"  Mod={res['subcarrier_mod']:5s} | OFDM PAPR={res['papr_ofdm_db']:5.2f} dB | SC-FDMA PAPR={res['papr_scfdma_db']:5.2f} dB | PAPR Diff={res['papr_reduction_db']:+5.2f} dB")
        print("=========================================================================\n")

    return results

def run_progressive_multicarrier_benchmark(impairment_levels=None, verbose=True):
    """
    Executes the multicarrier benchmark suite across all progressive impairment levels.
    """
    levels = impairment_levels if impairment_levels is not None else PROGRESSIVE_IMPAIRMENT_LEVELS
    overall_reports = []
    
    if verbose:
        print("\n" + "#"*75)
        print("   PROGRESSIVE MULTICARRIER IMPAIRMENT BENCHMARK SUITE EXECUTION")
        print("#"*75)
        
    for profile in levels:
        rep = run_multicarrier_benchmark_suite(profile, verbose=verbose)
        overall_reports.append(rep)

    total_passed = sum(r["passed_cases"] for r in overall_reports)
    total_evals = sum(r["total_cases"] for r in overall_reports)
    overall_pass_rate = round(total_passed / total_evals, 4) if total_evals > 0 else 0.0

    if verbose:
        print("#"*75)
        print(f"   PROGRESSIVE SUITE SUMMARY: {total_passed}/{total_evals} CASES PASSED ({overall_pass_rate*100:.1f}%)")
        print("#"*75)
        for r in overall_reports:
            p_name = r["impairment_profile"].get("name", "Level")
            print(f"  - {p_name:45s}: {r['passed_cases']}/{r['total_cases']} ({r['pass_rate']*100:.1f}%)")
        print("#"*75 + "\n")

    return {
        "reports": overall_reports,
        "total_evals": total_evals,
        "total_passed": total_passed,
        "overall_pass_rate": overall_pass_rate
    }

if __name__ == "__main__":
    summary = run_progressive_multicarrier_benchmark()
    sys.exit(0 if summary["overall_pass_rate"] >= 0.85 else 1)
