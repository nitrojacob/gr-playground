---
name: signal-analysis
description: Inspect unknown signal spectrum, detect peak tones, calculate SNR via M2M4, estimate occupied bandwidth, and measure DC offset without looking at raw IQ samples.
---

# Signal Analysis Skill 📡

Use this skill when you receive a raw or unknown complex64 signal (SigMF file) and need to understand its spectral properties, noise floor, center frequency offset, occupied bandwidth, and peak tone frequencies.

## Critical Constraint
**DO NOT ATTEMPT TO VIEW OR READ RAW COMPLEX IQ DATA SAMPLES.** Always process signals using the `analyze_signal.py` script provided in this skill.

## Instructions

1. **Execute Signal Analysis Tool**:
   Run the skill script via command line:
   ```bash
   PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/signal-analysis/scripts/analyze_signal.py --input <path_to_sigmf_file>
   ```

2. **Interpret Report Metrics**:
   - **Estimated SNR (M2M4)**: Evaluate if signal is high quality (> 20 dB), moderate (10–20 dB), or degraded (< 10 dB).
   - **Occupied Bandwidth**: Identifies the 99% power bandwidth footprint.
   - **DC Offset Level**: If > 3 dB above mean noise, signal requires DC offset suppression during cleanup.
   - **Dominant Peaks Table**: Lists tone frequencies and relative powers.

3. **Next Steps Recommendation**:
   - If DC offset or noise is high $\rightarrow$ invoke `signal-cleanup`.
   - If clean $\rightarrow$ proceed directly to `modulation-recognition`.
