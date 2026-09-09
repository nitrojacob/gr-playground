"""
Demodulation Module using Native GNU Radio Blocks.
Demodulates AM, FM, BPSK, QPSK, 8PSK, and QAM signals into audio WAV output or decoded bit/text payloads.
"""

from gnuradio import gr, blocks, analog, digital, filter
import numpy as np
from scipy.io import wavfile
import os

class FMDemodFlowgraph(gr.top_block):
    def __init__(self, samples, sample_rate=32000, max_dev=5000.0):
        super(FMDemodFlowgraph, self).__init__("FMDemodFlowgraph")
        self.src = blocks.vector_source_c(samples.tolist(), False)
        # Quadrature FM Demod Block
        gain = float(sample_rate) / (2.0 * np.pi * max_dev)
        self.fm_demod = analog.quadrature_demod_cf(gain)
        self.sink = blocks.vector_sink_f()
        self.connect(self.src, self.fm_demod, self.sink)

    def run_demod(self):
        self.run()
        return np.array(self.sink.data(), dtype=np.float32)

class AMDemodFlowgraph(gr.top_block):
    def __init__(self, samples, sample_rate=32000):
        super(AMDemodFlowgraph, self).__init__("AMDemodFlowgraph")
        self.src = blocks.vector_source_c(samples.tolist(), False)
        # Complex to magnitude for envelope detection
        self.c2m = blocks.complex_to_mag()
        # DC removal for audio
        self.dc_block = filter.dc_blocker_ff(32, True)
        self.sink = blocks.vector_sink_f()
        self.connect(self.src, self.c2m, self.dc_block, self.sink)

    def run_demod(self):
        self.run()
        return np.array(self.sink.data(), dtype=np.float32)

def slice_psk_qpsk_bits(samples, mod_type="QPSK"):
    """
    Slices synchronized IQ constellation points into binary bit sequence.
    """
    y = np.asarray(samples, dtype=np.complex64)
    bits = []
    mod_upper = mod_type.upper()
    
    if mod_upper == "BPSK":
        for val in y:
            bits.append(1 if np.real(val) > 0 else 0)
    elif mod_upper in ["ASK", "2ASK", "OOK"]:
        mag = np.abs(y)
        thresh = (np.max(mag) + np.min(mag)) / 2.0
        for m in mag:
            bits.append(1 if m > thresh else 0)
    elif mod_upper in ["8PSK"]:
        for val in y:
            phase = np.angle(val) % (2 * np.pi)
            sym = int(np.floor(phase / (np.pi / 4.0))) & 7
            bits.extend([(sym >> 2) & 1, (sym >> 1) & 1, sym & 1])
    elif mod_upper in ["16QAM"]:
        for val in y:
            i_bit0 = 1 if np.real(val) > 0 else 0
            i_bit1 = 1 if np.abs(np.real(val)) < 0.5 else 0
            q_bit0 = 1 if np.imag(val) > 0 else 0
            q_bit1 = 1 if np.abs(np.imag(val)) < 0.5 else 0
            bits.extend([i_bit0, i_bit1, q_bit0, q_bit1])
    else:  # QPSK / Default
        for val in y:
            bits.append(1 if np.real(val) > 0 else 0)
            bits.append(1 if np.imag(val) > 0 else 0)
            
    return np.array(bits, dtype=np.uint8)

def demodulate_signal_flowgraph(samples, sample_rate=32000, mod_type="QPSK", output_wav_path=None):
    """
    Full demodulation pipeline. Returns (payload_data, preview_string).
    """
    samples = np.asarray(samples, dtype=np.complex64)
    mod_upper = mod_type.upper()
    
    if mod_upper == "FM":
        tb = FMDemodFlowgraph(samples, sample_rate=sample_rate)
        audio = tb.run_demod()
        
        if output_wav_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
            norm_audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.9
            wavfile.write(output_wav_path, int(sample_rate), (norm_audio * 32767).astype(np.int16))
            preview = f"FM Audio Demodulated successfully ({len(audio)} samples -> {output_wav_path})"
        else:
            preview = f"FM Audio Demodulated ({len(audio)} samples)"
        return audio, preview

    elif mod_upper in ["AM", "AM-DSB", "AM-SSB"]:
        tb = AMDemodFlowgraph(samples, sample_rate=sample_rate)
        audio = tb.run_demod()
        
        if output_wav_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
            norm_audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.9
            wavfile.write(output_wav_path, int(sample_rate), (norm_audio * 32767).astype(np.int16))
            preview = f"AM Audio Demodulated successfully ({len(audio)} samples -> {output_wav_path})"
        else:
            preview = f"AM Audio Demodulated ({len(audio)} samples)"
        return audio, preview

    else: # Digital PSK / QAM / FSK
        bits = slice_psk_qpsk_bits(samples, mod_type=mod_upper)
        # Convert first 64 bits to hex string preview
        bit_str = "".join(str(b) for b in bits[:64])
        preview = f"Digital Bit Payload (Total {len(bits)} bits). Preview first 64 bits: [{bit_str}]"
        return bits, preview
