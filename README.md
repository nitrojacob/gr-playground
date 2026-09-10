# gr-playground

`gr-playground` is for AI LLM agents to learn and develop skills in signal processing. It includes a GNU Radio simulation playground, DSP library, and benchmark testbed. The thought behind this is simple: You give your agent a testcase with transmitter and channel impairment simulation, and a task to demodulate/decode and get back original signal; The agent will be able to try various algos and come up with one that works. Have more testcases for the same schemes at harder and harder channel impairment models, so that once your coding agent passes all the testcases, it has built the agent skills necessary for demodulating and decoding the signal.

Alternately, the skills that come prepackaged with this repo in `.agents/skills/` can be used as is with your coding agent, and you can prompt it like "capture wideband signal from rtl-sdr centered at 433MHz", "Analyse the captured rf band" etc. Your coding agent will scan the wideband capture for potential information channels, list and ask you to select a channel to extract, extract it, identify modulation, demodulate, detect L1 preambles, and decode L2 message payloads for you. You can also prompt it to "build a flowgraph for the receiver" to create a gnuradio-companion flow graph for realtime and interactive analysis.

The playground consists of three core pillars:
1. **The Signal & Channel Simulator** (`gr_playground.simulator`): Synthesizes impaired complex IQ signals with AWGN, CFO, SRO, DC offset, multipath fading, phase noise, jammer interference, Layer-2 framing (HDLC, COBS, AX.25, CCSDS, IEEE 802.15.4), and FEC channel coding (Viterbi K=7, Hamming, Repetition, Reed-Solomon).
2. **The Modular DSP Libraries** (`gr_playground.dsp` & `gr_playground.utils`): Provides wideband spectrum scanning, Digital Downconversion (DDC), automatic modulation classification (AMC), signal cleanup, synchronization, Layer-1 equalization (ZF, MMSE, LMS, CMA), Layer-1 preamble detection (Barker, Zadoff-Chu, Schmidl-Cox, Bluetooth LE, AIS, GSM), native GMSK/MSK demodulation, Layer-2 framing identification, and FEC channel decoding.
3. **The Automated Testcases & Benchmark Suite** (`tests/`): Evaluates DSP algorithms and agent skills against realistic real-world receiver non-idealities across 58 automated test cases.

## Documentation & Specifications

- **[Product Requirements Document (PRD)](specs/PRD.md)**: Product goals, functional/non-functional specifications, and pillar requirements.
- **[Architecture & System Design](specs/ARCHITECTURE.md)**: System architecture overview, key architectural principles, complete directory breakdown, mathematical formulations, GRC schema rules, and test strategies.

---

## How-To: Setting Up & Analyzing RF Spectrum with AI Agents

This section details how to configure **Antigravity** and **Claude Code** to use the prepackaged agent skills in `.agents/skills/`, and how to prompt them for wideband spectrum scanning, signal demodulation, and flowgraph generation.

---

### 1. Setting Up the Project for AI Agents (Google Antigravity)

1. **Clone the project**: `git clone github.com/nitrojacob/gr-playground`. This creates the `gr-playground` folder required in next step.
2. **Open Workspace**: Create a new project with `gr-playground` folder in Antigravity.
3. **Automatic Skill & Rule Discovery**: Antigravity automatically discovers and loads all prepackaged agent skills in `.agents/skills/` (e.g., `analyze-rf-signal`, `rtlsdr-hardware-capture`, `build-gnuradio-flowgraph`) and workspace guidelines from `.agents/AGENTS.md`. No manual registration or setup is required.
4. **Execution**: Prompt Antigravity in natural language chat as shown below

---

### 2. Primary End-to-End Skills

> *"Capture 5 seconds of live wideband RF spectrum at 433.92 MHz from RTL-SDR dongle and save to `/tmp/wideband_raw_capture.sigmf-data`."*

> *"Analyse this RF signal `/tmp/wideband_raw_capture.sigmf-data` and run the full end-to-end DSP analysis pipeline."*

---

### Step-by-Step Execution Using Sub-Skills (Lower-Level Components)

If you require isolated, step-by-step control over individual DSP steps, you can invoke the individual **sub-skills** that power `analyze-rf-signal` under the hood:

#### 1. Hardware Capture & Ingestion
- **Live RTL-SDR Hardware Capture** (`rtlsdr-hardware-capture` skill):
  > *"Capture 5 seconds of live wideband RF spectrum at 433.92 MHz from RTL-SDR dongle and save to `/tmp/wideband_raw_capture.sigmf-data`."*
- **File Ingestion with Auto Format Detection**:
  > *"Ingest the raw `capture.cu8` file recorded at 100 MHz with sample rate 2.4 MSPS into standard SigMF format."*
- **GUI Ingestion via GRC Frontend**:
  Open [rtlsdr_wideband_frontend.grc](grc/rtlsdr_wideband_frontend.grc) in GNU Radio Companion for visual capture and interactive playback.

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
  > *"Demodulate GMSK payload from `/tmp/synced_channel.sigmf-data` to `/tmp/demodulated_bits.bin`."*
- **Equalization** (`channel-equalization` sub-skill):
  > *"Perform MMSE channel equalization on multipath signal `/tmp/channel_150k.sigmf-data`."*
- **Preamble Detection** (`preamble-detection` sub-skill):
  > *"Detect Bluetooth LE preamble 0xAA in `/tmp/channel_150k.sigmf-data` and extract burst packets."*
- **Channel Decoding** (`channel-decoding` sub-skill):
  > *"Perform Viterbi K=7 convolutional decoding on `/tmp/demodulated_bits.bin`."*
- **Framing Identification** (`l2_framing_identification` sub-skill):
  > *"Identify L2 framing scheme and extract message payload from `/tmp/demodulated_bits.bin`."*
- **Flowgraph Script Generator** (`build-gnuradio-flowgraph` sub-skill):
  > *"Generate an executable standalone GNU Radio top block script for `/tmp/channel_150k.sigmf-data`."*

---

## Directory Structure, Key Principles & Architecture Specifications

For the complete **Directory Structure**, **Key Architectural Principles**, and **Module Specifications**, see:
👉 **[specs/ARCHITECTURE.md](specs/ARCHITECTURE.md)**

---

## Running Test Verification Suite

Execute all tests across all 13 architectural layers:

```bash
./run_all_tests.sh
# Or with verbose output:
./run_all_tests.sh -v
```

## Open Problems
* In real systems at very high channel impairment, there will be errors in received message. Rather than the testsuite being structured for a hard equality with input (which may be theoretically impossible), a BER threshold for pass may be more appropriate to learn more advanced strategies.
* A hardcoded decision tree (`modulation_id.py`) is used for classification of modulation schemes. Explore possibility of the LLM agent training a small MLP for classification (AI training AI).
