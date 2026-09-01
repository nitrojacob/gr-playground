---
name: rtlsdr-hardware-capture
description: Capture live RF signals from RTL-SDR hardware dongles with tunable center frequency, sample rate, duration, tuner/RF/IF gain, offset frequency, and DDC channelization, exporting directly to SigMF (.sigmf-meta and .sigmf-data).
---

# RTL-SDR Hardware Capture Skill 📡

Use this skill when the user requests to capture live RF spectrum signals using an RTL-SDR hardware USB dongle.

## Triggering Context
Activate this skill whenever the user asks to:
- "capture from RTL-SDR"
- "capture around <frequency> for <duration>"
- "record live RF signal at <freq>"
- "capture <frequency> MHz with gain <gain>"

## Instructions

1. **Execute Headless RTL-SDR Capture Script**:
   Run the generic capture tool with requested tuning parameters:
   ```bash
   PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/rtlsdr-hardware-capture/scripts/capture_rtlsdr.py \
     --freq <center_freq_hz_or_mhz> \
     --duration <seconds> \
     --gain <tuner_rf_gain_db> \
     --if_gain <if_gain_db> \
     --gain_mode <manual|auto> \
     --samp_rate <wideband_sample_rate> \
     --offset <ddc_offset_hz> \
     --output <output_sigmf_path>
   ```

2. **Tunable Parameters**:
   - `--freq` / `--center_freq`: Center RF frequency in Hz, MHz, or GHz (e.g. `92.0e6`, `92M`, `100.2MHz`, `433.92M`, `1420M`).
   - `--duration` / `--time`: Capture duration in seconds (default: `30`).
   - `--gain` / `--tuner_gain`: Tuner / RF gain in dB (default: `20.0`).
   - `--if_gain`: Intermediate Frequency (IF) gain in dB (default: `20.0`).
   - `--gain_mode`: Gain mode (`manual` or `auto`/`agc`).
   - `--samp_rate`: Wideband SDR hardware sampling rate in Hz (default: `2400000` / 2.4 MSps).
   - `--offset`: DDC frequency offset in Hz (default: `200000` / 200 kHz).
   - `--cutoff`: DDC lowpass filter cutoff frequency in Hz (default: `60000` / 60 kHz).
   - `--decimation`: DDC decimation factor (default: `10`).
   - `--output`: Target path for `.sigmf-data` file (default: `/tmp/rtlsdr_capture.sigmf-data`).

3. **Downstream DSP Workflow**:
   The capture tool generates paired SigMF files (`.sigmf-data` + `.sigmf-meta`). Continue the analysis pipeline:
   `rtlsdr-hardware-capture` $\rightarrow$ `signal-analysis` $\rightarrow$ `signal-cleanup` $\rightarrow$ `modulation-recognition` $\rightarrow$ `signal-demodulation`.
