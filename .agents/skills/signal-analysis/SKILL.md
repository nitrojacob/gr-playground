---
name: signal-analysis
description: Inspect signal spectrum, detect peak tones, calculate SNR via M2M4 and Welch PSD, estimate occupied bandwidth, and measure DC offset level without running end-to-end demodulation.
---

# Signal Analysis Skill 📡

Use this micro-worker skill when the user asks for targeted spectral metrics and spectrum inspection of a SigMF signal (e.g., *"measure SNR"*, *"calculate occupied bandwidth"*, *"inspect Welch PSD noise floor"*, or *"list spectral peak tones"*). Do NOT use this skill for end-to-end multi-stage pipeline analysis (use `analyze-rf-signal` instead).

## Critical Constraint
**DO NOT ATTEMPT TO VIEW OR READ RAW COMPLEX IQ DATA SAMPLES.** Always process signals using the `analyze_signal.py` script provided in this skill.

## Instructions

1. **Execute Signal Analysis Tool**:
   Run the skill script via command line:
   ```bash
   PYTHONPATH=/usr/lib/python3/dist-packages:./ .agents/skills/signal-analysis/scripts/analyze_signal.py --input <path_to_sigmf_file>
   ```

2. **Interpret Report Metrics**:
   - **Estimated SNR (M2M4)**: Evaluate if signal is high quality (> 20 dB), moderate (10–20 dB), or degraded (< 10 dB).
   - **Occupied Bandwidth**: Identifies the 99% power bandwidth footprint.
   - **DC Offset Level**: If > 3 dB above mean noise, signal requires DC offset suppression during cleanup.
   - **Dominant Peaks Table**: Lists tone frequencies and relative powers.

3. **Next Steps Recommendation**:
   - If DC offset or noise is high $\rightarrow$ invoke `signal-cleanup`.
   - If clean $\rightarrow$ proceed directly to `modulation-recognition`.
