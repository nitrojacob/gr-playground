---
name: build-gnuradio-flowgraph
description: Compose and generate executable standalone GNU Radio Python top_block scripts (.py) or GRC block diagram files (.grc) using native GNU Radio DSP blocks.
---

# Build GNU Radio Flowgraph Skill 🛠️

Use this skill when you need to construct a complete, standalone GNU Radio Python flowgraph (`gr.top_block`) to process, clean, synchronize, and demodulate a new or custom signal.

## Instructions

1. **Execute Flowgraph Generator Tool**:
   ```bash
   PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/build-gnuradio-flowgraph/scripts/build_gnuradio_flowgraph.py --input <input_sigmf_file> --output_script <output_script.py> --ops "dc_block,lowpass_filter,agc,costas_loop,symbol_sync"
   ```

2. **Generated Python Top Block Architecture**:
   The output script imports `from gnuradio import gr, blocks, filter, analog, digital` and wires native blocks into an executable pipeline:
   - `blocks.file_source` $\rightarrow$ `filter.dc_blocker_cc` $\rightarrow$ `filter.fir_filter_ccc` $\rightarrow$ `analog.agc2_cc` $\rightarrow$ `digital.costas_loop_cc` $\rightarrow$ `digital.symbol_sync_cc` $\rightarrow$ `blocks.file_sink`.

3. **Execute Generated Script**:
   Run the generated GNU Radio top block script directly:
   ```bash
   PYTHONPATH=/usr/lib/python3/dist-packages python3 <output_script.py> <input_sigmf> <output_iq>
   ```
