# gr-playground

`gr-playground` is for AI LLM agents to learn and develop skills in signal processing. It includes a GNU Radio simulation playground, DSP library, and benchmark testbed. The thought behind this is simple: You give your agent a testcase with transimitter and channel impairment simulation, and a task to demodulate/decode and get back original signal; The agent, will be able to try various algos and come up with one that works. Have more testcases for the same schemes at harder and harder channel impariment models, so that once your coding agent passes all the testcases, it has built the agent skills necessary for demodulating the signal.

Alternately the skills that come prepackaged with this repo in .agents/ can be used as is with your coding agent, and you can prompt it like "capture wideband signal from rtl-sdr centered at 433MHz", "Analyse the captured rf band" etc. Your coding agent will scan the wideband capture for potential information channels, list and ask you to select channel to extract, extract it and identify modulation and demodulates the signal for you. You can also prompt it to "build a flowgraph for the receiver" to create a gnuradio-companion flow graph for realtime and interactive analysis.

The playground consists of three core pillars:
1. **The Signal & Channel Simulator** (`gr_playground.simulator`): Synthesizes impaired complex IQ signals with AWGN, CFO, SRO, DC offset, multipath fading, phase noise, and jammer interference.
2. **The Modular DSP Libraries** (`gr_playground.dsp` & `gr_playground.utils`): Provides wideband spectrum scanning, Digital Downconversion (DDC), automatic modulation classification (AMC) via higher-order cumulants, signal cleanup, synchronization, and demodulation.
3. **The Automated Testcases & Benchmark Suite** (`tests/`): Evaluates DSP algorithms and agent skills against realistic real-world receiver non-idealities across 33 automated test cases.

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
├── grc/                       # Unified GNU Radio Companion (.grc) frontend flowgraph
│   └── rtlsdr_wideband_frontend.grc
└── tests/                     # Automated pytest suite & benchmark testbed
    ├── assets/audio/          # Audio WAV reference generators & sample files
    ├── benchmarks/            # Benchmark suites (multicarrier & skill verification)
    │   ├── multicarrier_benchmark_suite.py
    │   └── skill_verification_suite.py
    ├── conftest.py            # Pytest layer ordering & base-layer fail-fast hooks
    ├── test_simulator.py      # Layer 1: Foundation simulator & SigMF I/O
    ├── test_dsp.py            # Layer 2: Core DSP & analytics
    ├── test_wideband.py       # Layer 3: Wideband scanning & DDC
    ├── test_ask_modulation.py # Layer 4: Modulation ID edge cases
    ├── test_dsp_flaws_and_fixes.py # Layer 5: DSP regression fixes
    ├── test_multicarrier.py   # Layer 6: Multicarrier parameter sweeps
    ├── test_realworld_receiver_impairments.py # Layer 7: Receiver channel suite
    ├── test_skill_verification.py # Layer 8: Impairment benchmark
    └── test_analyze_rf_signal_skill.py # Layer 9: Top-level skill pipeline
```

---

## How-To: Analyzing Arbitrary Bands from RTL-SDR & SDR Hardware

This section details how to analyze arbitrary RF bands recorded from SDR hardware (e.g., RTL-SDR, HackRF, USRP, LimeSDR) using natural language agent prompts.

### 🌟 Primary End-to-End Skill: `analyze-rf-signal`

The **`analyze-rf-signal`** skill is the main orchestrator for gr-playground. Giving a single prompt to your agent triggers the entire DSP pipeline automatically:

> *"Analyse this RF signal `/tmp/wideband_raw_capture.sigmf-data` and run the full end-to-end DSP analysis pipeline."*

#### Automated Pipeline Steps (Executed Under the Hood):
```
[Wideband SigMF Capture]
          │
          ├──> 1. Scan Wideband Spectrum & Discover Candidate Channels
          ├──> 2. Extract Narrowband Target Channel (DDC Channelizer)
          ├──> 3. Spectral Analysis (Welch PSD, SNR, Occupied BW)
          ├──> 4. Signal Cleanup (DC Blocker, Gram-Schmidt I/Q Balancing)
          ├──> 5. Modulation Recognition (Higher-Order Cumulants AMC)
          ├──> 6. CFO Recovery & Symbol Timing Synchronization
          ├──> 7. Demodulate Payload (WAV Audio / Decoded Bits)
          └──> 8. Generate Executable Top Block Python Script (`receiver_top_block.py`)
```

---

### Step-by-Step Execution Using Sub-Skills (Lower-Level Components)

If you require isolated, step-by-step control over individual DSP steps, you can invoke the individual **sub-skills** that power `analyze-rf-signal` under the hood:

#### 1. Hardware Capture & Ingestion
- **Live RTL-SDR Hardware Capture** (`rtlsdr-hardware-capture` skill):
  > *"Capture 5 seconds of live wideband RF spectrum at 433.92 MHz from RTL-SDR dongle and save to `/tmp/wideband_raw_capture.sigmf-data`."*
- **File Ingestion with Auto Format Detection**:
  > *"Ingest the raw `capture.cu8` file recorded at 100 MHz with sample rate 2.4 MSPS into standard SigMF format."*
- **GUI Ingestion via GRC Frontend**:
  Open [rtlsdr_wideband_frontend.grc](file:///mnt/wksp/kaggle_5dag/experiments/gr-playground/grc/rtlsdr_wideband_frontend.grc) in GNU Radio Companion for visual capture and interactive playback.

#### 2. Individual Sub-Skill Steps
- **Spectrum Analysis** (`signal-analysis` sub-skill):
  > *"Measure SNR, Welch PSD noise floor, and occupied bandwidth of `/tmp/channel_150k.sigmf-data`."*
- **Signal Cleanup** (`signal-cleanup` sub-skill):
  > *"Remove DC offset and perform I/Q imbalance correction on `/tmp/channel_150k.sigmf-data`."*
- **Modulation Classification** (`modulation-recognition` sub-skill):
  > *"Identify the modulation scheme of `/tmp/cleaned_channel.sigmf-data` using higher-order cumulants."*
- **Synchronization** (`signal-synchronization` sub-skill):
  > *"Synchronize carrier frequency offset and symbol clock timing on `/tmp/cleaned_channel.sigmf-data`."*
- **Demodulation** (`signal-demodulation` sub-skill):
  > *"Demodulate FM payload from `/tmp/synced_channel.sigmf-data` to `/tmp/demodulated_audio.wav`."*
- **Flowgraph Script Generator** (`build-gnuradio-flowgraph` sub-skill):
  > *"Generate an executable standalone GNU Radio top block script for `/tmp/channel_150k.sigmf-data`."*

---

## Agent Prompts & Prepackaged Skills

The table below lists all prepackaged skills in `.agents/skills/`. **`analyze-rf-signal`** is the primary end-to-end entrypoint, while sub-skills handle specific modular DSP tasks:

### 🎯 Primary End-to-End Orchestration Skill
| Skill Directory | Skill Name | Primary Agent Prompt | Description |
| :--- | :--- | :--- | :--- |
| `.agents/skills/analyze-rf-signal` | `analyze-rf-signal` | *"Analyse this RF signal `/tmp/wideband_raw_capture.sigmf-data` and run the full end-to-end DSP analysis pipeline."* | **Primary Entrypoint**: Automatically scans spectrum, prompts for channel selection, and runs end-to-end DDC, cleanup, AMC, sync, demod, and top_block script generation. |

### 🛠️ Hardware Capture & Data Generation Skills
| Skill Directory | Skill Name | Example Agent Prompt | Description |
| :--- | :--- | :--- | :--- |
| `.agents/skills/rtlsdr-hardware-capture` | `rtlsdr-hardware-capture` | *"Capture 5s of live wideband RF spectrum at 433.92 MHz from RTL-SDR, list channels, and extract channel 1 to `/tmp/target.sigmf-data`."* | Captures live RF from RTL-SDR hardware dongles directly into SigMF. |
| `.agents/skills/generate-test-signal` | `generate-test-signal` | *"Generate a 15 dB SNR FM test signal with 2.5 kHz CFO from audio source and save to `/tmp/test_signal.sigmf-data`."* | Synthesizes impaired test signals (AM, FM, PSK, QAM) with SigMF headers. |

### 🧩 Sub-Skills & Component DSP Modules (Triggered by `analyze-rf-signal`)
| Skill Directory | Skill Name | Example Agent Prompt | Component Role |
| :--- | :--- | :--- | :--- |
| `.agents/skills/signal-analysis` | `signal-analysis` | *"Measure SNR, Welch PSD noise floor, and occupied bandwidth of `/tmp/target.sigmf-data` without running full demodulation."* | Sub-skill: Inspects PSD, SNR, occupied BW, and DC offset. |
| `.agents/skills/signal-cleanup` | `signal-cleanup` | *"Apply DC blocker, Gram-Schmidt I/Q balancing, and lowpass filtering to `/tmp/target.sigmf-data`."* | Sub-skill: Cleans up IQ imbalance and DC bias. |
| `.agents/skills/modulation-recognition` | `modulation-recognition` | *"Classify modulation scheme of `/tmp/target.sigmf-data` using higher-order cumulants."* | Sub-skill: Classifies modulation scheme via $C_{20}\dots C_{63}$ cumulants. |
| `.agents/skills/signal-synchronization` | `signal-synchronization` | *"Recover CFO and synchronize symbol clock timing for QPSK signal `/tmp/target.sigmf-data`."* | Sub-skill: Recovers CFO and Gardner symbol clock locks. |
| `.agents/skills/signal-demodulation` | `signal-demodulation` | *"Demodulate BPSK payload from `/tmp/synced.sigmf-data` into decoded bits or audio file."* | Sub-skill: Demodulates WAV audio or sliced bitstreams. |
| `.agents/skills/build-gnuradio-flowgraph` | `build-gnuradio-flowgraph` | *"Generate executable GNU Radio Python flowgraph receiver script for `/tmp/target.sigmf-data`."* | Sub-skill: Generates standalone Python `top_block` scripts & validates GRC schemas. |

### Running Test Verification Suite

Execute all tests in proper architectural layer order:

```bash
./run_all_tests.sh
# Or with verbose output:
./run_all_tests.sh -v
```

## Open Problems
* Decoding is not yet part of the skills, but would be a great addition, for the agent to have knowledge about all the popular encoding/decoding schemes for digital communication.
  * In fact, you can add more blocks of the ideal receiver(channel decode, decryption, error correction, source decoding etc..) in the testsuite objective loop and build LLM agent skill for them.
* In real systems at very high channel impairment, there will errors in received message. Rather than the testsuite being structured for a hard equality with input (which may be theoretically impossible), a BER threshold for pass may be more appropriate to learn more advanced strategies.
