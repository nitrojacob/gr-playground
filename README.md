# gr-playground

`gr-playground` is for AI LLM agents to learn and develop skills in signal processing. It includes a GNU Radio simulation playground, DSP library, and benchmark testbed. The thought behind this is simple: You give your agent a testcase with transimitter and channel impairment simulation, and a task to demodulate/decode and get back original signal; The agent, will be able to try various algos and come up with one that works. Have more testcases for the same schemes at harder and harder channel impariment models, so that once your coding agent passes all the testcases, it has built the agent skills necessary for demodulating the signal.

Alternately the skills that come prepackaged with this repo in .agents/ can be used as is with your coding agent, and you can prompt it like "capture wideband signal from rtl-sdr centered at 433MHz", "Analyse the captured rf band" etc. Your coding agent will scan the wideband capture for potential information channels, list and ask you to select channel to extract, extract it and identify modulation and demodulates the signal for you. You can also prompt it to "build a flowgraph for the receiver" to create a gnuradio-companion flow graph for realtime and interactive analysis.

The playground consists of three core pillars:
1. **The Signal & Channel Simulator** (`gr_playground.simulator`): Synthesizes impaired complex IQ signals with AWGN, CFO, SRO, DC offset, multipath fading, phase noise, and jammer interference.
2. **The Modular DSP Libraries** (`gr_playground.dsp` & `gr_playground.utils`): Provides wideband spectrum scanning, Digital Downconversion (DDC), automatic modulation classification (AMC) via higher-order cumulants, signal cleanup, synchronization, and demodulation.
3. **The Automated Testcases & Benchmark Suite** (`tests/` & `examples/`): Evaluates DSP algorithms and agent skills against realistic real-world receiver non-idealities across 22 automated test cases.

## Documentation & Specifications

- **[Product Requirements Document (PRD)](file:///mnt/wksp/kaggle_5dag/experiments/gr-playground/specs/PRD.md)**: Product goals, functional/non-functional specifications, and pillar requirements.
- **[Architecture & System Design](file:///mnt/wksp/kaggle_5dag/experiments/gr-playground/specs/ARCHITECTURE.md)**: Three-pillar system architecture diagrams, mathematical formulations, GRC schema rules, and test strategies.

---

## Key Principles

1. **No Raw Signals to LLM**: Time-domain samples and complex IQ arrays are processed entirely via DSP tools. The agent interacts strictly with concise summary metrics (PSD peaks, SNR, bandwidth, cumulants, EVM, decoded bit/audio text).
2. **Native GNU Radio Reuse**: All signal generation, impairment modeling, filtering, synchronization, and demodulation leverage native GNU Radio blocks.
3. **Native SigMF Inter-Tool Standard**: All intermediate data, wideband captures, and channel extractions exchange data using standard **SigMF** format (`.sigmf-data` + `.sigmf-meta`).
4. **Executable Flowgraph Generation**: The agent's final goal for any new or unknown signal is to generate an executable GNU Radio Python flowgraph (`gr.top_block`) or `.grc` file.
5. **Skills Housing Tools**: All executable tool scripts are housed directly inside their respective skill directories under `.agents/skills/<skill_name>/scripts/`.

---

## Directory Structure & Agent Skills

```
gr-playground/
├── gr_playground/             # Core Python package & GNU Radio wrappers
│   ├── simulator/             # Source, modulation, impairment & SigMF flowgraphs
│   ├── dsp/                   # GNU Radio DSP modules (spectrum, filtering, sync, demod, channelizer, flowgraph builder)
│   │   ├── channelizer.py     # Wideband spectrum scanning & DDC channel extraction
│   │   ├── spectrum.py        # Welch PSD, M2M4 SNR, occupied BW, peak tones
│   │   ├── filtering.py       # Gram-Schmidt I/Q balance, DC blocker, AGC
│   │   ├── modulation_id.py   # Higher-order cumulants (C20-C63) & AMC tree
│   │   ├── synchronization.py # Power-of-N CFO, Costas loop, Gardner symbol sync
│   │   ├── demodulation.py    # AM/FM quadrature demod & digital bit slicing
│   │   └── flowgraph_builder.py # Executable Python top_block script generator
│   └── utils/                 # SigMF I/O with auto-detection & LLM summary report formatters
├── .agents/skills/            # Agent Skill definitions & Executable Tool Scripts
│   ├── generate-test-signal/
│   │   ├── SKILL.md
│   │   └── scripts/generate_test_signal.py
│   ├── signal-analysis/
│   │   ├── SKILL.md
│   │   └── scripts/analyze_signal.py
│   ├── signal-cleanup/
│   │   ├── SKILL.md
│   │   └── scripts/cleanup_signal.py
│   ├── modulation-recognition/
│   │   ├── SKILL.md
│   │   └── scripts/classify_modulation.py
│   ├── signal-synchronization/
│   │   ├── SKILL.md
│   │   └── scripts/synchronize_signal.py
│   ├── signal-demodulation/
│   │   ├── SKILL.md
│   │   └── scripts/demodulate_signal.py
│   ├── build-gnuradio-flowgraph/
│   │   ├── SKILL.md
│   │   └── scripts/build_gnuradio_flowgraph.py
│   └── rtlsdr-hardware-capture/
│       ├── SKILL.md
│       └── scripts/
│           ├── capture_rtlsdr.py
│           └── localize_and_extract.py
├── examples/                  # Audio generators & skill verification benchmark suite
│   ├── audio/
│   └── skill_verification_suite.py
├── grc/                       # Unified GNU Radio Companion (.grc) frontend flowgraph
│   └── rtlsdr_wideband_frontend.grc
└── tests/                     # Automated pytest suite (DSP, simulator, wideband)
    ├── test_ask_modulation.py
    ├── test_dsp.py
    ├── test_dsp_flaws_and_fixes.py
    ├── test_realworld_receiver_impairments.py
    ├── test_simulator.py
    ├── test_skill_verification.py
    └── test_wideband.py
```

---

## How-To: Analyzing Arbitrary Bands from RTL-SDR & SDR Hardware

This section details how to capture, scan, extract, and analyze arbitrary RF bands recorded from SDR hardware (e.g. RTL-SDR, HackRF, USRP, LimeSDR) using natural language agent prompts.

### Workflow Overview

```
[SDR Capture / Raw cu8] ──> [RTL-SDR Capture Skill / read_sigmf] ──> [Wideband SigMF]
                                                                            │
                                                                  (Scan Channels)
                                                                            │
                                                                            ▼
[Demodulated Audio/Data] <── [AMC / Sync] <── [Narrowband SigMF] <── (DDC Channelizer)
```

---

### Step 1: Live Capture or Ingest RF Data into Native SigMF

- **Live RTL-SDR Hardware Capture** (via `rtlsdr-hardware-capture` skill):
  Prompt your agent:
  > *"Capture 5 seconds of live wideband RF spectrum at 433.92 MHz from RTL-SDR dongle and save to `/tmp/wideband_raw_capture.sigmf-data`."*

- **File Ingestion with Auto Format Detection**:
  Prompt your agent:
  > *"Ingest the raw `capture.cu8` file recorded at 100 MHz with sample rate 2.4 MSPS into standard SigMF format."*

- **GUI Ingestion via GRC Frontend**:
  Open [rtlsdr_wideband_frontend.grc](file:///mnt/wksp/kaggle_5dag/experiments/gr-playground/grc/rtlsdr_wideband_frontend.grc) in GNU Radio Companion (`gnuradio-companion`) for live visual capture and interactive playback.

---

### Step 2: Scan Wideband Spectrum & Discover Channels

Prompt your agent to scan the wideband spectrum capture for active sub-channels:

> *"Scan wideband spectrum `/tmp/wideband_raw_capture.sigmf-data` to discover candidate sub-channels, their frequency offsets, power levels, and estimated bandwidths."*

*Expected Agent Output:*
```
Found 3 candidate sub-channels:
- Channel 1: Offset +150.0 kHz | Power -12.4 dBFS | BW 100.0 kHz
- Channel 2: Offset -320.0 kHz | Power -18.1 dBFS | BW 50.0 kHz
- Channel 3: Offset +450.0 kHz | Power -22.0 dBFS | BW 200.0 kHz
```

---

### Step 3: Extract Narrowband Channel via Digital Downconverter (DDC)

Prompt your agent to extract and downconvert a specific sub-channel:

> *"Extract sub-channel 1 at +150 kHz offset from `/tmp/wideband_raw_capture.sigmf-data` with decimation 10 and save to `/tmp/channel_150k.sigmf-data`."*

---

### Step 4: Run Narrowband Channel Analysis & Demodulation Pipeline

Once the target sub-channel is extracted into a narrowband SigMF file, execute the full analysis and demodulation pipeline using agent prompts:

1. **Analyze Channel Spectrum & SNR** (`signal-analysis` skill):
   > *"Analyze the signal spectrum, SNR, and occupied bandwidth of `/tmp/channel_150k.sigmf-data`."*

2. **Perform Signal Cleanup** (`signal-cleanup` skill):
   > *"Remove DC offset and perform I/Q imbalance correction on `/tmp/channel_150k.sigmf-data` and save to `/tmp/cleaned_channel.sigmf-data`."*

3. **Classify Modulation Scheme** (`modulation-recognition` skill):
   > *"Identify the modulation scheme of `/tmp/cleaned_channel.sigmf-data` using higher-order cumulants."*

4. **Synchronize Carrier & Symbol Clock** (`signal-synchronization` skill):
   > *"Synchronize carrier frequency offset and symbol clock timing for FM modulation on `/tmp/cleaned_channel.sigmf-data` and save to `/tmp/synced_channel.sigmf-data`."*

5. **Demodulate Payload** (`signal-demodulation` skill):
   > *"Demodulate FM payload from `/tmp/synced_channel.sigmf-data` and write audio output to `/tmp/demodulated_audio.wav`."*

---

### Step 5: Generate Executable Top Block Python Script

Prompt your agent to generate a standalone GNU Radio Python flowgraph (`gr.top_block`) for the complete receiver chain (`build-gnuradio-flowgraph` skill):

> *"Generate an executable standalone GNU Radio top block script for the receiver pipeline processing `/tmp/channel_150k.sigmf-data` and save to `/tmp/receiver_top_block.py`."*

---

## Agent Prompts & Prepackaged Skills

The table below lists the prepackaged skills in `.agents/skills/` alongside representative natural language prompts that invoke them:

| Skill Directory | Skill Name | Example Agent Prompt |
| :--- | :--- | :--- |
| `.agents/skills/analyze-rf-signal` | `analyze-rf-signal` | *"Analyse this RF signal `/tmp/wideband_raw_capture.sigmf-data` and run the full end-to-end DSP analysis pipeline."* |
| `.agents/skills/generate-test-signal` | `generate-test-signal` | *"Generate a 15 dB SNR FM test signal with 2.5 kHz CFO from audio source and save to `/tmp/test_signal.sigmf-data`."* |
| `.agents/skills/rtlsdr-hardware-capture` | `rtlsdr-hardware-capture` | *"Capture 5s of live wideband RF spectrum at 433.92 MHz from RTL-SDR, list channels, and extract channel 1 to `/tmp/target.sigmf-data`."* |
| `.agents/skills/signal-analysis` | `signal-analysis` | *"Inspect spectrum, measure SNR via M2M4, and estimate bandwidth for `/tmp/target.sigmf-data`."* |
| `.agents/skills/signal-cleanup` | `signal-cleanup` | *"Apply DC blocker, Gram-Schmidt I/Q balancing, and lowpass filtering to `/tmp/target.sigmf-data`."* |
| `.agents/skills/modulation-recognition` | `modulation-recognition` | *"Classify modulation scheme of `/tmp/target.sigmf-data` using higher-order cumulants."* |
| `.agents/skills/signal-synchronization` | `signal-synchronization` | *"Recover CFO and synchronize symbol clock timing for QPSK signal `/tmp/target.sigmf-data`."* |
| `.agents/skills/signal-demodulation` | `signal-demodulation` | *"Demodulate BPSK payload from `/tmp/synced.sigmf-data` into decoded bits or audio file."* |
| `.agents/skills/build-gnuradio-flowgraph` | `build-gnuradio-flowgraph` | *"Generate executable GNU Radio Python flowgraph receiver script for `/tmp/target.sigmf-data`."* |

### Running Test Verification Suite

```bash
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 -m pytest tests/
```

## Open Problems
* Decoding is not yet part of the skills, but would be a great addition, for the agent to have knowledge about all the popular encoding/decoding schemes for digital communication.
  * In fact, you can add more blocks of the ideal receiver(channel decode, decryption, error correction, source decoding etc..) in the testsuite objective loop and build LLM agent skill for them.
* In real systems at very high channel impairment, there will errors in received message. Rather than the testsuite being structured for a hard equality with input (which may be theoretically impossible), a BER threshold for pass may be more appropriate to learn more advanced strategies.
