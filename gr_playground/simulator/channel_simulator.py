"""
Unified GNU Radio Channel Simulator Flowgraph (gr.top_block)
Connects source selection, modulation scheme, channel impairments, and output sinks using native gnuradio blocks.
"""

import os
from gnuradio import gr, blocks, analog, digital, channels, filter
import numpy as np

from gr_playground.simulator.sources import GRSources
from gr_playground.simulator.modulators import GRModulators
from gr_playground.simulator.impairments import GRImpairments
from gr_playground.simulator.sigmf_writer import SigMFWriter

class ChannelSimulatorFlowgraph(gr.top_block):
    def __init__(self, 
                 source_type="sine", 
                 mod_type="QPSK", 
                 sample_rate=32000, 
                 num_samples=16384,
                 tone_freq=1000.0,
                 wav_path=None,
                 snr_db=20.0, 
                 cfo_hz=0.0, 
                 phase_offset_deg=0.0,
                 sro_ppm=0.0,
                 dc_offset=(0.0, 0.0),
                 mag_imbalance_db=0.0,
                 phase_imbalance_deg=0.0,
                 multipath_taps=None,
                 jammer_power_db=None,
                 output_filepath=None):
        
        super(ChannelSimulatorFlowgraph, self).__init__("ChannelSimulatorFlowgraph")

        self.source_type = source_type.lower()
        self.mod_type = mod_type.upper()
        self.sample_rate = int(sample_rate)
        self.num_samples = int(num_samples)
        self.snr_db = float(snr_db)
        self.cfo_hz = float(cfo_hz)
        self.phase_offset_deg = float(phase_offset_deg)
        self.sro_ppm = float(sro_ppm)

        # 1. Instantiate Signal Source
        if self.source_type == "sine":
            self.src_block = GRSources.sine_source(self.sample_rate, tone_freq)
            is_complex_src = True
        elif self.source_type == "square":
            self.src_block = GRSources.square_source(self.sample_rate, tone_freq)
            is_complex_src = False
        elif self.source_type == "audio":
            if not wav_path or not os.path.exists(wav_path):
                # Fallback to sample audio path
                default_audio = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "audio", "speech.wav")
                wav_path = default_audio if os.path.exists(default_audio) else wav_path
            self.src_block = GRSources.audio_loop_source(wav_path, self.sample_rate)
            is_complex_src = False
        elif self.source_type == "prbs":
            self.src_block = GRSources.prbs_source()
            is_complex_src = False
        elif self.source_type == "noise":
            self.src_block = GRSources.noise_source(amplitude=0.5)
            is_complex_src = True
        else:
            self.src_block = GRSources.sine_source(self.sample_rate, tone_freq)
            is_complex_src = True

        # 2. Instantiate Modulator Block
        if self.mod_type in ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "256QAM"]:
            if not is_complex_src:
                # Convert float/byte source to byte/float for constellation modulator
                if self.source_type == "prbs":
                    mod_in = self.src_block
                else:
                    # Float audio/square source -> convert to byte via thresholding
                    f2b = blocks.float_to_uchar()
                    self.connect(self.src_block, f2b)
                    mod_in = f2b
            else:
                # Complex sine/noise -> convert magnitude to byte
                c2m = blocks.complex_to_mag()
                f2b = blocks.float_to_uchar()
                self.connect(self.src_block, c2m, f2b)
                mod_in = f2b

            self.mod_block = GRModulators.digital_constellation_modulator(self.mod_type)
            self.connect(mod_in, self.mod_block)
            tx_out = self.mod_block

        elif self.mod_type == "FM":
            if is_complex_src:
                c2f = blocks.complex_to_real()
                self.connect(self.src_block, c2f)
                audio_in = c2f
            else:
                audio_in = self.src_block
            self.mod_block = GRModulators.fm_modulator(self.sample_rate, max_dev=5000.0)
            self.connect(audio_in, self.mod_block)
            tx_out = self.mod_block

        elif self.mod_type in ["AM", "AM-DSB", "AM-SSB"]:
            if is_complex_src:
                c2f = blocks.complex_to_real()
                self.connect(self.src_block, c2f)
                audio_in = c2f
            else:
                audio_in = self.src_block
            add_const, scale, f2c = GRModulators.am_modulator()
            self.connect(audio_in, scale, add_const, f2c)
            tx_out = f2c

        elif self.mod_type in ["GFSK", "BFSK"]:
            if not is_complex_src:
                if self.source_type == "prbs":
                    b_in = self.src_block
                else:
                    f2b = blocks.float_to_uchar()
                    self.connect(self.src_block, f2b)
                    b_in = f2b
            else:
                c2m = blocks.complex_to_mag()
                f2b = blocks.float_to_uchar()
                self.connect(self.src_block, c2m, f2b)
                b_in = f2b
            self.mod_block = GRModulators.gfsk_modulator()
            self.connect(b_in, self.mod_block)
            tx_out = self.mod_block

        else: # RAW / PASS-THROUGH
            if not is_complex_src:
                f2c = blocks.float_to_complex()
                self.connect(self.src_block, f2c)
                tx_out = f2c
            else:
                tx_out = self.src_block

        # 3. Impairment Chain
        self.chan_block = GRImpairments.channel_model(
            snr_db=self.snr_db,
            cfo_hz=self.cfo_hz,
            sro_ppm=self.sro_ppm,
            sample_rate=self.sample_rate,
            multipath_taps=multipath_taps
        )
        self.connect(tx_out, self.chan_block)
        impaired_out = self.chan_block

        # DC Offset & Phase Rotator
        if dc_offset != (0.0, 0.0):
            dc_add = blocks.add_const_cc(complex(dc_offset[0], dc_offset[1]))
            self.connect(impaired_out, dc_add)
            impaired_out = dc_add

        if phase_offset_deg != 0.0:
            rot = blocks.rotator_cc(np.radians(phase_offset_deg))
            self.connect(impaired_out, rot)
            impaired_out = rot

        # 4. Vector Sink to collect complex64 output
        self.head_block = blocks.head(gr.sizeof_gr_complex, self.num_samples)
        self.sink_block = blocks.vector_sink_c()
        self.connect(impaired_out, self.head_block, self.sink_block)

        self.output_filepath = output_filepath

    def get_samples(self):
        """Run the GNU Radio flowgraph and return numpy complex64 array."""
        self.run()
        data = np.array(self.sink_block.data(), dtype=np.complex64)
        
        if self.output_filepath:
            SigMFWriter.export_dataset(
                self.output_filepath,
                data,
                sample_rate=self.sample_rate,
                center_freq=0.0,
                source_type=self.source_type,
                mod_type=self.mod_type,
                snr_db=self.snr_db,
                cfo_hz=self.cfo_hz,
                phase_offset_deg=self.phase_offset_deg,
                sro_ppm=self.sro_ppm,
                iq_imbalance_db=0.0
            )
        
        return data
