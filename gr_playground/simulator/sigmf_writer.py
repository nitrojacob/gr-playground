"""
SigMF Header Exporter for GNU Radio Channel Simulator
"""

from gr_playground.utils.sigmf_io import write_sigmf

class SigMFWriter:
    @staticmethod
    def export_dataset(filepath, samples, sample_rate, center_freq, source_type, mod_type, snr_db, cfo_hz, phase_offset_deg, sro_ppm, iq_imbalance_db):
        """
        Exports complex64 dataset and writes SigMF metadata JSON header.
        """
        annotations = [
            {
                "core:sample_start": 0,
                "core:sample_count": len(samples),
                "core:label": f"{mod_type}_{source_type}",
                "gr_playground:source_type": source_type,
                "gr_playground:modulation_type": mod_type,
                "gr_playground:snr_db": float(snr_db),
                "gr_playground:cfo_hz": float(cfo_hz),
                "gr_playground:phase_offset_deg": float(phase_offset_deg),
                "gr_playground:sro_ppm": float(sro_ppm),
                "gr_playground:iq_imbalance_db": float(iq_imbalance_db),
            }
        ]
        description = f"gr-playground simulation: {mod_type} with {source_type} source (SNR={snr_db}dB, CFO={cfo_hz}Hz)"
        return write_sigmf(filepath, samples, sample_rate=sample_rate, center_freq=center_freq, description=description, annotations=annotations)
