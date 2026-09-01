"""
Summary & Report Formatting Utilities for LLM Agent Consumption.
Formated reports present structured metrics, peak tables, and EVM stats without dumping raw signal samples.
"""

import json

def format_spectrum_summary(num_samples, sample_rate, estimated_snr_db, occupied_bw_hz, dc_offset_db, peaks):
    """
    Format spectral analysis into a readable markdown report.
    """
    lines = [
        "## Signal Analysis Summary Report",
        "",
        f"- **Sample Count**: {num_samples:,}",
        f"- **Sample Rate**: {sample_rate / 1e3:.2f} kHz",
        f"- **Estimated SNR (M2M4)**: {estimated_snr_db:.2f} dB",
        f"- **Occupied Bandwidth (99% Power)**: {occupied_bw_hz / 1e3:.2f} kHz",
        f"- **DC Offset Level**: {dc_offset_db:.2f} dB",
        "",
        "### Spectral Dominant Peaks",
        "| Peak Index | Frequency (Hz) | Power (dB) | Relative Freq (kHz) |",
        "|------------|----------------|------------|---------------------|"
    ]
    for idx, peak in enumerate(peaks[:5], 1):
        lines.append(f"| {idx} | {peak['freq_hz']:.1f} | {peak['power_db']:.2f} | {peak['freq_hz']/1e3:.3f} |")
    
    return "\n".join(lines)

def format_cleanup_summary(original_snr, cleaned_snr, dc_removed_db, iq_imbalance_corr_db):
    """
    Format signal cleanup operations into a markdown report.
    """
    snr_gain = cleaned_snr - original_snr
    lines = [
        "## Signal Cleanup Report",
        "",
        f"- **Original Estimated SNR**: {original_snr:.2f} dB",
        f"- **Cleaned Signal SNR**: {cleaned_snr:.2f} dB",
        f"- **SNR Gain**: {snr_gain:+.2f} dB",
        f"- **DC Bias Removed**: {dc_dc_removed_db:.2f} dB" if 'dc_dc_removed_db' in locals() else f"- **DC Offset Suppressed**: {dc_removed_db:.2f} dB",
        f"- **I/Q Imbalance Correction**: {iq_imbalance_corr_db:.2f} dB",
        "- **Status**: Signal filtered and stabilized for modulation identification."
    ]
    return "\n".join(lines)

def format_modulation_id_summary(predictions, cumulants, constellation_stats):
    """
    Format modulation classification into a markdown report.
    """
    top_pred = predictions[0] if predictions else ("Unknown", 0.0)
    lines = [
        "## Modulation Recognition Report",
        "",
        f"### **Predicted Modulation: `{top_pred[0]}`** (Confidence: {top_pred[1]*100:.1f}%)",
        "",
        "#### Top Predictions",
        "| Modulation Scheme | Probability / Confidence |",
        "|-------------------|-------------------------|"
    ]
    for mod, conf in predictions[:5]:
        lines.append(f"| {mod} | {conf*100:.1f}% |")

    lines.extend([
        "",
        "#### Extracted Higher-Order Cumulants",
        f"- **C20 (Direct Energy)**: {cumulants.get('C20', 0):.4f}",
        f"- **C21 (Average Power)**: {cumulants.get('C21', 0):.4f}",
        f"- **C40 (Phase Symmetry)**: {cumulants.get('C40', 0):.4f}",
        f"- **C42 (Kurtosis Metric)**: {cumulants.get('C42', 0):.4f}",
        "",
        "#### Constellation Properties",
        f"- **Phase Variance**: {constellation_stats.get('phase_var', 0):.4f} rad²",
        f"- **Amplitude Kurtosis**: {constellation_stats.get('amp_kurtosis', 0):.4f}",
        f"- **Estimated Symbol States**: {constellation_stats.get('estimated_symbols', 'N/A')}"
    ])
    return "\n".join(lines)

def format_synchronization_summary(estimated_cfo_hz, phase_offset_deg, symbol_rate, evm_percent, snr_post_sync):
    """
    Format carrier & clock synchronization results into markdown report.
    """
    lines = [
        "## Signal Synchronization Report",
        "",
        f"- **Carrier Frequency Offset (CFO)**: {estimated_cfo_hz:+.2f} Hz",
        f"- **Phase Offset**: {phase_offset_deg:.2f}°",
        f"- **Estimated Symbol Rate**: {symbol_rate:.1f} Baud",
        f"- **Constellation EVM (Error Vector Magnitude)**: {evm_percent:.2f}%",
        f"- **Post-Sync SNR**: {snr_post_sync:.2f} dB",
        "- **Status**: Carrier locked & symbol clock aligned."
    ]
    return "\n".join(lines)

def format_demodulation_summary(mod_type, payload_type, payload_length, preview_text=""):
    """
    Format demodulation payload summary into markdown report.
    """
    lines = [
        "## Demodulation Output Report",
        "",
        f"- **Modulation Type**: `{mod_type}`",
        f"- **Extracted Payload Type**: `{payload_type}`",
        f"- **Payload Size**: {payload_length} units",
    ]
    if preview_text:
        lines.extend([
            "",
            "#### Extracted Content Preview",
            "```",
            preview_text[:500],
            "```"
        ])
    return "\n".join(lines)
