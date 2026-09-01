---
name: signal-cleanup
description: Clean up complex IQ signals using native GNU Radio blocks (filter.dc_blocker_cc, filter.fir_filter_ccc, analog.agc2_cc) and Gram-Schmidt I/Q imbalance compensation.
---

# Signal Cleanup Skill 🧼

Use this skill when a signal has DC bias offset, amplitude/phase I/Q imbalance, out-of-band noise, or unnormalized gain.

## Instructions

1. **Execute Signal Cleanup Tool**:
   ```bash
   PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/signal-cleanup/scripts/cleanup_signal.py --input <input_sigmf_file> --output <output_cleaned_file> --cutoff <bandwidth_cutoff_hz>
   ```

2. **DSP Pipeline Applied (Native GNU Radio Blocks)**:
   - Gram-Schmidt Orthogonalization: Corrects I/Q amplitude imbalance ($\alpha$) and phase skew ($\phi$).
   - `filter.dc_blocker_cc`: Suppresses DC bias spikes.
   - `filter.fir_filter_ccc`: Band-limits signal to occupied bandwidth, rejecting out-of-band noise.
   - `analog.agc2_cc`: Stabilizes average signal energy to unit power ($1.0$).

3. **Verify SNR Gain**:
   Check the output report to confirm SNR gain and stabilization before passing to modulation recognition.
