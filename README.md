# gr-playground 📡🤖

A GNU Radio simulation playground and DSP skill suite designed for AI LLM agents.

`gr-playground` allows LLM agents to perform signal analysis, modulation recognition, channel impairment compensation, carrier/clock synchronization, and signal cleanup by **composing and executing native GNU Radio flowgraphs** (`gnuradio.analog`, `gnuradio.digital`, `gnuradio.channels`, `gnuradio.filter`, `gnuradio.blocks`).

---

## Key Principles

1. **No Raw Signals to LLM**: Time-domain samples and complex IQ arrays are processed entirely via DSP tools. The agent interacts strictly with concise summary metrics (PSD peaks, SNR, bandwidth, cumulants, EVM, decoded bit/audio text).
2. **Native GNU Radio Reuse**: All signal generation, impairment modeling, filtering, synchronization, and demodulation leverage native GNU Radio blocks.
3. **Executable Flowgraph Generation**: The agent's final goal for any new or unknown signal is to generate an executable GNU Radio Python flowgraph (`gr.top_block`) or `.grc` file.
4. **Skills Housing Tools**: All executable tool scripts are housed directly inside their respective skill directories under `.agents/skills/<skill_name>/scripts/`.

---

## Directory Structure & Agent Skills

```
gr-playground/
├── gr_playground/             # Core Python package & GNU Radio wrappers
│   ├── simulator/             # Source, modulation, impairment & SigMF flowgraphs
│   ├── dsp/                   # GNU Radio DSP modules (spectrum, filtering, sync, demod, flowgraph builder)
│   └── utils/                 # SigMF I/O & LLM summary report formatters
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
│   └── build-gnuradio-flowgraph/
│       ├── SKILL.md
│       └── scripts/build_gnuradio_flowgraph.py
├── examples/                  # Audio generators & skill verification benchmark suite
│   ├── audio/
│   └── skill_verification_suite.py
├── grc/                       # GNU Radio Companion (.grc) flowgraphs
└── tests/                     # Automated pytest suite
```

---

## Running Skill Tool Scripts

```bash
# 1. Generate test signal with SigMF metadata
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/generate-test-signal/scripts/generate_test_signal.py --source audio --mod FM --snr 15 --cfo 2500 --output /tmp/test_signal/signal.sigmf-data

# 2. Analyze signal spectrum and SNR
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/signal-analysis/scripts/analyze_signal.py --input /tmp/test_signal/signal.sigmf-data

# 3. Perform DC removal, I/Q balancing, filtering
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/signal-cleanup/scripts/cleanup_signal.py --input /tmp/test_signal/signal.sigmf-data --output /tmp/cleaned.sigmf-data

# 4. Classify modulation scheme
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/modulation-recognition/scripts/classify_modulation.py --input /tmp/cleaned.sigmf-data

# 5. Synchronize carrier & clock
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/signal-synchronization/scripts/synchronize_signal.py --input /tmp/cleaned.sigmf-data --mod QPSK --output /tmp/synced.sigmf-data

# 6. Demodulate payload
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/signal-demodulation/scripts/demodulate_signal.py --input /tmp/synced.sigmf-data --mod FM --audio_out /tmp/demod_audio.wav

# 7. Generate GNU Radio Top Block script
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/build-gnuradio-flowgraph/scripts/build_gnuradio_flowgraph.py --input /tmp/test_signal/signal.sigmf-data --output_script /tmp/receiver_top_block.py
```

### Running the Progressive Impairment Verification Suite
```bash
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 examples/skill_verification_suite.py
```
