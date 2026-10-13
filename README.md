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
- **[Wideband Channel Identification Technical Note](specs/channel_identification.md)**: Specifications for modular channel detection strategies (`cfar_heuristic`, `ppd_heuristic`, ML/DL interface contract, zero-copy mechanics, and CA-CFAR migration rationale).
- **[Automatic Modulation Classification Specification](specs/modulation_classifier.md)**: Specifications for 26-feature vector extraction, synthetic/real dataset collection pipeline, HistGradientBoosting training, and temporal sequence meta-stacking.

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
**[specs/ARCHITECTURE.md](specs/ARCHITECTURE.md)**

---

## Running Test Verification Suite

Execute all tests across all 13 architectural layers:

```bash
./run_all_tests.sh
# Or with verbose output:
./run_all_tests.sh -v
```

---

## Benchmarks

### Zero-Shot Modulation Recognition on RadioML 2018.01A

The Automatic Modulation Classification engine (`MLAMCClassifier` backed by `amc_model.onnx`) is trained strictly on **100% synthetic training data** synthesized by `gr_playground.simulator`. It is evaluated against the real-world **RadioML 2018.01A** dataset (`GOLD_XYZ_OSHF.hdf5`), which contains over 2.55 million over-the-air frame slices across 24 modulation formats from -20 dB to +30 dB SNR.

- **Kaggle Benchmark Notebook**: [RadioML 2018 MLAMC Benchmark](https://www.kaggle.com/code/jacobmgec/radioml2018-mlamc-benchmark)
- **Zero-Shot Domain Transfer**: The model is never exposed to real-world over-the-air signals during training. The synthetic dataset synthesizes realistic physical channel impairments:
  - Random carrier phase rotation (uniform [-pi, pi])
  - Bidirectional carrier frequency offset (+/- delta_f up to +/- 0.05 f_s)
  - Continuous sub-sample fractional timing delays (via FFT phase ramp)
  - Additive White Gaussian Noise (AWGN, -20 dB to +30 dB SNR)
  - Oscillator phase noise and I/Q amplitude/phase imbalance
- **Feature Representation**: 40-dimensional physical feature vector combining higher-order cumulants (C20, C21, C40, C41, C42, C60, C63), spectral and envelope moments, and dual-order Costas loop tracking features (2nd-order and 4th-order Costas loop phase error variances and lock indicators).
- **Performance Highlights**:
  - **Individual 24-Class Exact Accuracy**: Reaches ~27% to 30% top-1 exact accuracy on raw, unsynchronized 1024-sample frames at moderate-to-high SNR (random chance across 24 classes is 4.17%).
  - **Family-Level Classification Accuracy**: Achieves ~50% family accuracy across all 24 classes, and 80% to 90% accuracy on dominant families (QPSK ~90%, PSK ~80%, AM ~80%-90%, FM/GFSK ~80%-90%).
  - **Inference Latency**: Highly efficient CPU execution via ONNX Runtime (~3.7 to 7.7 ms per 1024-sample frame), enabling real-time streaming classification on edge SDR hardware.

---

## Open Problems
* In real systems at very high channel impairment, there will be errors in received message. Rather than the testsuite being structured for a hard equality with input (which may be theoretically impossible), a BER threshold for pass may be more appropriate to learn more advanced strategies.
* Testcases currently are passing if the family is identified correctly. See whether the agent is able to put in further optimization effort if strict modulation matching is enforced.
* Feature-based ML classification (`MLAMCClassifier` / `amc_model.onnx`) is now integrated. Explore extending this to end-to-end 1D/2D neural network or transformer models directly on raw IQ sequences.
* Currently I am in the loop for architectural decisions like (heuristic AMC -> ML AMC). Is it possible to have an architect agent that identifies the implementation agent has hit a wall and rearchitects the problem? (Multiple Agents.md in the same project directory each agent with its own goals/intents)