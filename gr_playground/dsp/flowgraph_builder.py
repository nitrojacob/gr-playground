"""
Flowgraph Builder for GNU Radio Python Scripts and GRC Diagrams.
Allows agents to generate executable Python top_block files and GNU Radio Companion (.grc) flowgraphs.
"""

import os

class FlowgraphBuilder:
    @staticmethod
    def generate_top_block_script(input_sigmf_path, output_sigmf_path, operations, sample_rate=32000, center_freq_offset=0.0):
        """
        Generates a standalone Python script instantiating a GNU Radio gr.top_block
        that executes the requested DSP operations (DC block, filtering, Costas loop, DDC channelizer, etc.).
        """
        script_code = f"""#!/usr/bin/env python3
# Automatically generated GNU Radio Top Block script by gr-playground Agent

import sys
import numpy as np
from gnuradio import gr, blocks, filter, analog, digital

class GeneratedTopBlock(gr.top_block):
    def __init__(self, input_file, output_file, sample_rate={sample_rate}):
        super(GeneratedTopBlock, self).__init__("GeneratedTopBlock")

        # 1. Source: Read SigMF/Complex64 File
        self.src = blocks.file_source(gr.sizeof_gr_complex, input_file, False)
        
        last_block = self.src

"""
        # Append operations
        if "freq_xlating_filter" in operations:
            script_code += f"""        # Frequency Translating FIR Filter (Digital Downconverter)
        cutoff = {sample_rate} * 0.1
        taps = filter.firdes.low_pass(1.0, {sample_rate}, cutoff, cutoff * 0.2)
        self.xlating = filter.freq_xlating_fir_filter_ccc(1, taps, {center_freq_offset}, {sample_rate})
        self.connect(last_block, self.xlating)
        last_block = self.xlating
"""
        if "dc_block" in operations:
            script_code += """        # DC Blocker Block
        self.dc_blocker = filter.dc_blocker_cc(32, True)
        self.connect(last_block, self.dc_blocker)
        last_block = self.dc_blocker
"""
        if "lowpass_filter" in operations:
            script_code += f"""        # FIR Lowpass Filter Block
        taps = filter.firdes.low_pass(1.0, {sample_rate}, {sample_rate*0.25}, {sample_rate*0.05})
        self.lpf = filter.fir_filter_ccc(1, taps)
        self.connect(last_block, self.lpf)
        last_block = self.lpf
"""
        if "agc" in operations:
            script_code += """        # Automatic Gain Control (AGC2) Block
        self.agc = analog.agc2_cc(1e-3, 1e-2, 1.0, 1.0)
        self.connect(last_block, self.agc)
        last_block = self.agc
"""
        if "costas_loop" in operations:
            script_code += """        # Costas Loop Carrier Sync Block
        self.costas = digital.costas_loop_cc(np.pi / 50.0, 4, False)
        self.connect(last_block, self.costas)
        last_block = self.costas
"""
        if "symbol_sync" in operations:
            script_code += """        # Symbol Timing Sync Block
        self.sym_sync = digital.symbol_sync_cc(
            digital.TED_GARDNER, 4, 0.045, 1.0, 1.0, 1.5, 1, digital.constellation_qpsk().base(), digital.IR_MMSE_8TAP, 128
        )
        self.connect(last_block, self.sym_sync)
        last_block = self.sym_sync
"""

        script_code += f"""
        # Sink: File Sink
        self.sink = blocks.file_sink(gr.sizeof_gr_complex, output_file, False)
        self.connect(last_block, self.sink)

def main():
    if len(sys.argv) > 2:
        inp, out = sys.argv[1], sys.argv[2]
    else:
        inp, out = "{input_sigmf_path}", "{output_sigmf_path}"
    
    tb = GeneratedTopBlock(inp, out)
    tb.run()
    print(f"GNU Radio Flowgraph Execution Complete. Output saved to: {{out}}")

if __name__ == "__main__":
    main()
"""
        return script_code

    @staticmethod
    def save_script(filepath, script_content):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(script_content)
        os.chmod(filepath, 0o755)
        return filepath
