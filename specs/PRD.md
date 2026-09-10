# Product Requirements Document (PRD): `gr-playground`

## 1. Product Overview & Vision

`gr-playground` is an open-source **GNU Radio Simulation Playground, DSP Library, and Testbed**. It is designed to provide autonomous LLM coding agents and SDR developers with an isolated, reproducible sandbox to:
1. **Simulate RF Channels & Impairments**: Synthesize complex IQ signals with precise physical layer impairments (AWGN, CFO, SRO, DC offset, multipath fading, phase noise, jammer interference), Layer-2 framing (HDLC, COBS, AX.25, CCSDS, IEEE 802.15.4), and FEC channel encoding.
2. **Execute Modular DSP Operations**: Perform wideband spectrum scanning, Digital Downconversion (DDC), spectral analysis, signal cleanup, automatic modulation classification (AMC), carrier/clock synchronization, Layer-1 channel equalization (ZF, MMSE, LMS, CMA), Layer-1 preamble detection (Barker, Zadoff-Chu, Schmidl-Cox, Bluetooth LE, AIS, GSM), native GMSK/MSK demodulation, Layer-2 framing identification, and FEC channel decoding.
3. **Run Verification Testcases & Benchmarks**: Evaluate DSP algorithms and agent skills against realistic real-world receiver non-idealities across a 58-test automated verification suite.
4. **Deploy Prepackaged Agent Skills**: Provide 12 autonomous skills in `.agents/skills/` for Antigravity and Claude Code to enable end-to-end spectrum discovery, demodulation, and flowgraph generation.

The **Playground** is built entirely on native GNU Radio blocks (`gnuradio.analog`, `gnuradio.digital`, `gnuradio.channels`, `gnuradio.filter`, `gnuradio.blocks`) and uses **SigMF** (`.sigmf-data` + `.sigmf-meta`) as the standard data inter-tool standard.

---

## 2. Core Pillars of the Playground

```
+---------------------------------------------------------------------------------------------------+
|                                           gr-playground                                           |
+---------------------------------------------------------------------------------------------------+
        │                                         │                                         │
        ▼                                         ▼                                         ▼
+-----------------------+             +-----------------------+             +-----------------------+
|  Pillar I: Simulator  |             | Pillar II: DSP & HW   |             | Pillar III: Testcases |
| (gr_playground.sim)   |             | (gr_playground.dsp/hw)|             | (tests/ & .agents/)   |
+-----------------------+             +-----------------------+             +-----------------------+
| • Signal Sources      |             | • Spectrum Scanning   |             | • 58 Automated Tests  |
| • Modulators (AM/FM/  |             | • DDC & Cleanup       |             | • 12 Agent Skills     |
|   PSK/QAM/GFSK/GMSK/  |             | • AMC (Higher Cum.)   |             | • Real-World Impair-  |
|   MSK/OFDM/SC-FDMA)   |             | • Sync & Equalization |             |   ments Benchmark     |
| • L2 Framing & FEC    |             | • Preamble Correlation|             | • Progressive Impair- |
| • Impairment Chain    |             | • Native Demodulation |             |   ment Verification   |
| • SigMF Export        |             | • L2 Frame & FEC Dec  |             | • GRC Schema & Block  |
|                       |             | • RTL-SDR HW Capture  |             |   Presence Validator  |
+-----------------------+             +-----------------------+             +-----------------------+
```

---

## 3. Target Personas & Use Cases

1. **AI Agent Developers**: Require a deterministic GNU Radio environment where agents can generate test datasets, analyze unknown spectrums, classify modulations, detect preambles, decode payloads, and generate executable Python `top_block` scripts or `.grc` flowgraphs.
2. **SDR & DSP Engineers**: Need modular Python DSP libraries for fast signal analysis, DDC channelization, preamble detection, equalization, and modulation recognition without manually building GRC flowgraphs.
3. **Algorithmic Researchers**: Need a reproducible simulation testbed to bench test receiver algorithms against progressive physical layer impairments and FEC decoding bounds.

---

## 4. Pillar Specifications & Requirements

### 4.1. Pillar I: The Channel & Signal Simulator (`gr_playground.simulator`)

The simulator provides automated signal generation, framing, FEC encoding, and physical layer impairment modeling.

- **PRD-SIM-1 (Sources)**: Support continuous complex/float signal sources: `sine` (single-tone), `square` (harmonic-rich), `audio` (WAV speech/music loop), `prbs` (pseudo-random bit sequences), and `noise` (Gaussian noise).
- **PRD-SIM-2 (Modulation Schemes)**: Modulate input sources into analog and digital schemes:
  - Digital Single-Carrier Constellations: `BPSK`, `QPSK`, `8PSK`, `16QAM`, `64QAM`, `256QAM`.
  - Continuous Phase / Frequency: `FM`, `AM`, `GFSK`, `BFSK`, `GMSK`, `MSK`.
  - Multicarrier Schemes: `OFDM` (Orthogonal Frequency Division Multiplexing) and `SC-FDMA` (Single Carrier Frequency Division Multiple Access with $M$-point DFT precoding).
- **PRD-SIM-3 (L2 Framing & FEC Synthesis)**: Encapsulate payload bits into L2 frames (`HDLC`, `COBS`, `AX.25`, `CCSDS`, `IEEE 802.15.4`) with CRCs and encode using FEC (`Viterbi K=7 R=1/2`, `Hamming(7,4)`, `Repetition(3)`, `Reed-Solomon RS(15,11)`).
- **PRD-SIM-4 (Impairment Pipeline)**: Apply chainable physical channel impairments via native GNU Radio blocks:
  - Additive White Gaussian Noise (AWGN) specified in `snr_db` ($0\text{ to }30\text{ dB}$).
  - Carrier Frequency Offset (`cfo_hz`) & Doppler phase rotation.
  - Sampling Rate Offset (`sro_ppm`).
  - Direct Conversion DC Offset (`dc_offset` $I/Q$ tuple).
  - Quadrature I/Q Amplitude & Phase Imbalance (`mag_imbalance_db`, `phase_imbalance_deg`).
  - Frequency Selective Multipath Fading (`multipath_taps` array for Rayleigh/Rician channels).
- **PRD-SIM-5 (SigMF Dataset Export)**: Automatically export generated impaired signals to SigMF v1.0 specification files (`.sigmf-data` + `.sigmf-meta`).

### 4.2. Pillar II: Modular DSP & Hardware Ingestion (`gr_playground.dsp` & `gr_playground.hardware`)

The DSP library provides decoupled, reusable modules for inspecting, filtering, synchronizing, and decoding signals.

- **PRD-DSP-1 (Wideband Spectrum Scanning & DDC Channelizer)**:
  - Scan wideband spectrum captures ($2.4+\text{ MSPS}$) using Temporal Ensemble Welch PSD (`nperseg=32768`, $+9\text{ dB}$ processing gain).
  - Estimate adaptive rolling noise floor ($200\text{ kHz}$ median window) and local SNR prominence ($SNR_{\text{local}} = PSD_{\text{dB}} - NoiseFloor_{\text{dB}}$) to discover active sub-channels while rejecting $0\text{ Hz}$ LO leakage spikes.
  - Isolate sub-channels using high-performance unified chunked FIR DDC (`extract_channel_flowgraph` / `extract_channel`), lowpass filter, decimate, AGC normalize, and export to SigMF.
- **PRD-DSP-2 (Spectrum Analysis & SNR Metrics)**:
  - Compute Welch Power Spectral Density (PSD), peak tone frequencies, occupied bandwidth (99% power), DC offset level, and overall SNR via M2M4 ratio.
- **PRD-DSP-3 (Signal Cleanup & Filtering)**:
  - Remove DC bias via `filter.dc_blocker_cc`, apply Gram-Schmidt I/Q imbalance compensation, and perform lowpass FIR filtering (`filter.fir_filter_ccc`).
- **PRD-DSP-4 (Automatic Modulation Classification)**:
  - Calculate Higher-Order Cumulants ($C_{20}, C_{21}, C_{40}, C_{42}$) and constellation statistics to classify modulations (`FM`, `AM`, `BPSK`, `QPSK`, `8PSK`, `16QAM`, `64QAM`, `256QAM`, `GFSK`, `BFSK`, `GMSK`, `MSK`, `OFDM`, `SC-FDMA`).
  - Squelch low SNR noise ($SNR < 1.5\text{ dB}$) to prevent false positive classifications.
- **PRD-DSP-5 (Carrier & Clock Synchronization)**:
  - Perform Carrier Frequency Offset (CFO) recovery, Costas Loop phase locking (`digital.costas_loop_cc`), and Gardner symbol clock timing synchronization (`digital.symbol_sync_cc`).
- **PRD-DSP-6 (Demodulation)**:
  - Demodulate analog AM/FM into 16-bit PCM WAV audio files (`.wav`).
  - Slice digital modulations (BPSK, QPSK, GFSK, BFSK, GMSK, MSK) into bit/byte symbol strings using native GNU Radio blocks (`digital.gmsk_demod`, `digital.quadrature_demod_cf`).
- **PRD-DSP-7 (Layer-1 Preamble Detection & Frame Sync)**:
  - Cross-correlate complex IQ samples against known preamble sequences (Barker 7/11/13, Zadoff-Chu CAZAC, Schmidl-Cox OFDM, Bluetooth LE, AIS, GSM 26-bit midamble) to locate burst packet boundaries and extract synchronized frame slices.
- **PRD-DSP-8 (Layer-1 Channel Equalization)**:
  - Eliminate inter-symbol interference (ISI) and multipath fading using Zero-Forcing (ZF), Minimum Mean Square Error (MMSE), Adaptive LMS/DFE, and Constant Modulus Algorithm (CMA) equalizers.
- **PRD-DSP-9 (Layer-2 Framing Identification & Extraction)**:
  - Parse L2 framing overhead (`HDLC`, `COBS`, `AX.25`, `CCSDS`, `IEEE 802.15.4`, `Generic Length-Prefix`), perform bit un-stuffing, verify CRC-16/32 checksums, and extract raw message payloads.
- **PRD-DSP-10 (Layer-2 FEC Channel Decoding)**:
  - Perform error correction decoding using Viterbi K=7 R=1/2 convolutional decoder, Hamming(7,4), Repetition majority voting, and Reed-Solomon RS(15,11) decoders.
- **PRD-DSP-11 (SigMF I/O & Auto Format Detection)**:
  - Read standard SigMF files and provide fallback auto-conversion for raw 8-bit unsigned offset binary (`.cu8`) hardware captures `(x - 127.5)/127.5`.
- **PRD-DSP-12 (Flowgraph Builder & In-Skill GRC Validation)**:
  - Generate executable standalone GNU Radio Python top_block scripts (`gr.top_block`).
  - Include in-skill GRC YAML schema and block presence validation (`FlowgraphBuilder.validate_grc_flowgraph` / `build_gnuradio_flowgraph.py --validate_grc`) to programmatically verify that $0$ dummy or missing blocks exist upon creating or editing flowgraphs.
- **PRD-HW-1 (Live RTL-SDR Hardware Capture)**:
  - Capture live RF spectrum from RTL-SDR USB dongles (`0bda:2838`) with tunable center frequency, sample rate, duration, gain, and automatic kernel DVB tuner module unloading (`dvb_usb_rtl28xxu`). Export directly to SigMF format.

### 4.3. Pillar III: Automated Test Suite & Prepackaged Agent Skills

- **PRD-TEST-1 (58-Test Verification Suite)**:
  - Maintain 100% test pass rate across 58 automated unit and end-to-end integration test cases in `tests/` covering all 13 core DSP, simulator, hardware, and skill modules.
- **PRD-TEST-2 (Real-World Receiver Impairments Suite)**:
  - Test wideband scanning under $5\text{ dB}$ SNR, strong DC offset, and Adjacent Channel Interference (ACI).
  - Test DDC channel extraction under 3-path Rayleigh multipath fading and $+1.5\text{ kHz}$ CFO.
  - Test 8-bit RTL-SDR hardware ADC quantization noise and full-scale dynamic range clipping (`.cu8`).
- **PRD-TEST-3 (Progressive Impairment Benchmark Suite)**:
  - Run end-to-end benchmark suite across progressive SNR drops ($30\text{ dB} \to 0\text{ dB}$), CFO sweeps, and multipath channels to verify agent skill pass rates.
- **PRD-TEST-4 (Multicarrier Benchmark & Sub-Parameter Sweep Suite)**:
  - Perform sub-parameter sweeps over subcarrier spacing ($\Delta f$), subcarrier bandwidth ($N_{\text{used}}$), cyclic prefix ratios ($CP/N_{\text{fft}}$), and SC-FDMA DFT precoding PAPR reduction under configurable impairment profiles.
- **PRD-SKILL-1 (Prepackaged Agent Skills Suite)**:
  - Ship 12 modular skills in `.agents/skills/`: `analyze-rf-signal`, `build-gnuradio-flowgraph`, `channel-decoding`, `channel-equalization`, `generate-test-signal`, `l2-framing-identification`, `modulation-recognition`, `preamble-detection`, `rtlsdr-hardware-capture`, `signal-analysis`, `signal-cleanup`, `signal-demodulation`, and `signal-synchronization`.

---

## 5. Non-Functional Requirements

- **Performance**: Spectrum scanning of 72M samples (30s at 2.4 MSPS) must complete in under 5 seconds.
- **SigMF Compliance**: All datasets produced or consumed by the playground must comply with SigMF v1.0 specifications.
- **Zero Dummy Blocks**: In-skill GRC flowgraph validator must verify `b.is_dummy_block == False` for all blocks in generated flowgraphs.
- **Automated Test Pass Rate**: 100% pass rate across all 58 automated test cases in `tests/`.
- **Cross-Platform AI Support**: Full compatibility with both Antigravity and Claude Code agent platforms.
