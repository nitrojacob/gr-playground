"""
Dataset Generation Script for Automatic Modulation Classification (AMC).
Synthesizes IQ frames across 16 modulation schemes under variable sample rates,
durations, SNRs, CFO, phase noise, and channel impairments.

Discards raw IQ immediately and persists compact feature matrices:
- `data/amc_stage1_features.npz` (Stage 1 frame features: [N_frames, 26])
- `data/amc_stage2_features.npz` (Stage 2 sequence features: [N_seqs, 21])
"""

import os
import argparse
import numpy as np
from scipy import signal

from gr_playground.dsp.amc.features import (
    extract_frame_features,
    extract_sequence_features
)
from gr_playground.dsp.amc.common import squelch_check

MODULATION_CLASSES = [
    "AM", "FM", "BPSK", "GFSK", "QPSK", "8PSK",
    "16QAM", "64QAM", "256QAM", "ASK", "16APSK", "32APSK",
    "OQPSK", "OFDM", "SC-FDMA", "Noise"
]

CLASS_MAP = {name: idx for idx, name in enumerate(MODULATION_CLASSES)}

def generate_raw_iq_frame(mod_type: str, num_samples: int = 2048, sps: int = 4,
                          snr_db: float = 20.0, cfo_rel: float = 0.0,
                          phase_noise_std: float = 0.0, mag_imbal_db: float = 0.0,
                          phase_imbal_deg: float = 0.0) -> np.ndarray:
    """
    Synthesize a single raw complex64 IQ frame (length num_samples) for a target modulation scheme.
    """
    num_symbols = int(np.ceil(num_samples / sps)) + 10

    if mod_type == "Noise":
        iq = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2.0)
        return iq.astype(np.complex64)

    elif mod_type in ["AM", "AM-DSB-WC", "AM-DSB-SC", "AM-SSB-WC", "AM-SSB-SC", "FM"]:
        t = np.arange(num_samples)
        if np.random.rand() > 0.5:
            f_m = np.random.uniform(0.001, 0.015)
            audio = np.sin(2 * np.pi * f_m * t)
        else:
            audio = 0.6 * np.sin(2 * np.pi * 0.005 * t) + 0.4 * np.sin(2 * np.pi * 0.012 * t)

        if mod_type in ["AM", "AM-DSB-WC"]:
            m_depth = np.random.uniform(0.5, 0.95)
            audio_hilbert = np.imag(signal.hilbert(audio))
            iq = (1.0 + m_depth * audio) + 1j * (m_depth * audio_hilbert)
        elif mod_type == "AM-DSB-SC":
            iq = audio + 1j * 0.0
        elif mod_type == "AM-SSB-WC":
            m_depth = np.random.uniform(0.5, 0.95)
            audio_hilbert = np.imag(signal.hilbert(audio))
            iq = (1.0 + m_depth * audio) + 1j * (m_depth * audio_hilbert)
        elif mod_type == "AM-SSB-SC":
            audio_hilbert = np.imag(signal.hilbert(audio))
            iq = audio + 1j * audio_hilbert
        else: # FM
            freq_dev = np.random.uniform(0.005, 0.25)
            phase = 2 * np.pi * freq_dev * np.cumsum(audio)
            iq = np.exp(1j * phase)

    elif mod_type in ["ASK", "OOK", "4ASK", "8ASK"]:
        if mod_type in ["ASK", "OOK"]:
            symbols = np.random.choice([0.0, 1.0], size=num_symbols)
        elif mod_type == "4ASK":
            symbols = np.random.choice([-3.0, -1.0, 1.0, 3.0], size=num_symbols)
        else: # 8ASK
            symbols = np.random.choice([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0], size=num_symbols)
        iq = np.repeat(symbols, sps)[:num_samples] + 1j * 0.0

    elif mod_type == "BPSK":
        bits = np.random.choice([0, 1], size=num_symbols)
        symbols = 2.0 * bits - 1.0
        iq = np.repeat(symbols, sps)[:num_samples] + 1j * 0.0

    elif mod_type == "QPSK":
        bits_i = np.random.choice([0, 1], size=num_symbols)
        bits_q = np.random.choice([0, 1], size=num_symbols)
        sym_i = (2.0 * bits_i - 1.0) / np.sqrt(2.0)
        sym_q = (2.0 * bits_q - 1.0) / np.sqrt(2.0)
        iq = np.repeat(sym_i + 1j * sym_q, sps)[:num_samples]

    elif mod_type == "OQPSK":
        bits_i = np.random.choice([0, 1], size=num_symbols)
        bits_q = np.random.choice([0, 1], size=num_symbols)
        sym_i = (2.0 * bits_i - 1.0) / np.sqrt(2.0)
        sym_q = (2.0 * bits_q - 1.0) / np.sqrt(2.0)
        iq_i = np.repeat(sym_i, sps)[:num_samples]
        iq_q = np.repeat(sym_q, sps)[:num_samples]
        offset = max(1, sps // 2)
        iq_q = np.roll(iq_q, offset)
        iq = iq_i + 1j * iq_q

    elif mod_type in ["8PSK", "16PSK", "32PSK"]:
        m_order = 8 if mod_type == "8PSK" else (16 if mod_type == "16PSK" else 32)
        phases = (2 * np.pi / float(m_order)) * np.random.randint(0, m_order, size=num_symbols)
        symbols = np.exp(1j * phases)
        iq = np.repeat(symbols, sps)[:num_samples]

    elif mod_type in ["16QAM", "32QAM", "64QAM", "128QAM", "256QAM"]:
        if mod_type == "16QAM":
            grid = np.array([-3, -1, 1, 3])
            constellation = (grid[:, None] + 1j * grid[None, :]).ravel()
        elif mod_type == "32QAM":
            grid = np.array([-5, -3, -1, 1, 3, 5])
            c_square = (grid[:, None] + 1j * grid[None, :]).ravel()
            # Remove 4 outer corners
            constellation = np.array([p for p in c_square if abs(p.real) + abs(p.imag) <= 8])
        elif mod_type == "64QAM":
            grid = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            constellation = (grid[:, None] + 1j * grid[None, :]).ravel()
        elif mod_type == "128QAM":
            grid = np.array([-11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11])
            c_square = (grid[:, None] + 1j * grid[None, :]).ravel()
            constellation = np.array([p for p in c_square if abs(p.real) + abs(p.imag) <= 16])
        else: # 256QAM
            grid = np.linspace(-15, 15, 16)
            constellation = (grid[:, None] + 1j * grid[None, :]).ravel()

        constellation = constellation / np.sqrt(np.mean(np.abs(constellation)**2))
        syms = np.random.choice(constellation, size=num_symbols)
        iq = np.repeat(syms, sps)[:num_samples]

    elif mod_type in ["16APSK", "32APSK", "64APSK", "128APSK"]:
        if mod_type == "16APSK":
            r1, r2 = 1.0, 2.6
            p1 = np.exp(1j * (2 * np.pi / 4 * np.arange(4) + np.pi/4)) * r1
            p2 = np.exp(1j * (2 * np.pi / 12 * np.arange(12))) * r2
            constellation = np.concatenate([p1, p2])
        elif mod_type == "32APSK":
            r1, r2, r3 = 1.0, 2.54, 4.33
            p1 = np.exp(1j * (2 * np.pi / 4 * np.arange(4) + np.pi/4)) * r1
            p2 = np.exp(1j * (2 * np.pi / 12 * np.arange(12))) * r2
            p3 = np.exp(1j * (2 * np.pi / 16 * np.arange(16) + np.pi/16)) * r3
            constellation = np.concatenate([p1, p2, p3])
        elif mod_type == "64APSK":
            r1, r2, r3, r4 = 1.0, 2.2, 3.6, 5.0
            p1 = np.exp(1j * (2 * np.pi / 4 * np.arange(4) + np.pi/4)) * r1
            p2 = np.exp(1j * (2 * np.pi / 12 * np.arange(12))) * r2
            p3 = np.exp(1j * (2 * np.pi / 20 * np.arange(20))) * r3
            p4 = np.exp(1j * (2 * np.pi / 28 * np.arange(28))) * r4
            constellation = np.concatenate([p1, p2, p3, p4])
        else: # 128APSK
            r1, r2, r3, r4 = 1.0, 2.0, 3.4, 4.8
            p1 = np.exp(1j * (2 * np.pi / 8 * np.arange(8))) * r1
            p2 = np.exp(1j * (2 * np.pi / 16 * np.arange(16))) * r2
            p3 = np.exp(1j * (2 * np.pi / 40 * np.arange(40))) * r3
            p4 = np.exp(1j * (2 * np.pi / 64 * np.arange(64))) * r4
            constellation = np.concatenate([p1, p2, p3, p4])

        constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
        syms = np.random.choice(constellation, size=num_symbols)
        iq = np.repeat(syms, sps)[:num_samples]

    elif mod_type in ["GFSK", "CPFSK"]:
        bits = np.random.choice([0, 1], size=num_symbols)
        nrz = 2.0 * bits - 1.0
        upsampled = np.repeat(nrz, sps)
        if mod_type == "GFSK":
            t_g = np.linspace(-2, 2, 4 * sps)
            h_g = np.exp(-t_g**2 / (2 * 0.5**2))
            h_g /= np.sum(h_g)
            filtered = np.convolve(upsampled, h_g, mode='same')[:num_samples]
        else: # CPFSK
            filtered = upsampled[:num_samples]
        h_mod = 0.5 if mod_type == "GFSK" else 0.75
        phase = np.pi * h_mod * np.cumsum(filtered) / sps
        iq = np.exp(1j * phase)

    elif mod_type in ["OFDM", "SC-FDMA"]:
        n_fft = 64
        n_used = 48
        cp_len = 16
        n_syms = int(np.ceil(num_samples / (n_fft + cp_len)))
        ofdm_syms = []
        for _ in range(n_syms):
            subcarriers = np.zeros(n_fft, dtype=np.complex64)
            data_syms = (np.random.choice([-1, 1], size=n_used) + 1j * np.random.choice([-1, 1], size=n_used)) / np.sqrt(2)
            if mod_type == "SC-FDMA":
                data_syms = np.fft.fft(data_syms) / np.sqrt(n_used)
            subcarriers[1:n_used+1] = data_syms
            time_domain = np.fft.ifft(subcarriers) * np.sqrt(n_fft)
            cp = time_domain[-cp_len:]
            sym_with_cp = np.concatenate([cp, time_domain])
            ofdm_syms.append(sym_with_cp)
        iq = np.concatenate(ofdm_syms)[:num_samples]

    else:
        iq = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2.0)

    # Apply Impairments
    # 1. Power normalization
    p_sig = np.mean(np.abs(iq)**2)
    if p_sig > 0:
        iq = iq / np.sqrt(p_sig)

    # 2. CFO
    if cfo_rel != 0.0:
        t = np.arange(num_samples)
        iq = iq * np.exp(1j * 2 * np.pi * cfo_rel * t)

    # 3. Phase Noise
    if phase_noise_std > 0.0:
        pn = np.random.normal(0.0, phase_noise_std, size=num_samples)
        iq = iq * np.exp(1j * pn)

    # 4. IQ Amplitude / Phase Imbalance
    if mag_imbal_db != 0.0 or phase_imbal_deg != 0.0:
        gain = 10.0**(mag_imbal_db / 20.0)
        phase_rad = np.radians(phase_imbal_deg)
        i_chan = iq.real * gain * np.cos(phase_rad / 2.0) - iq.imag * gain * np.sin(phase_rad / 2.0)
        q_chan = -iq.real * (1.0 / gain) * np.sin(phase_rad / 2.0) + iq.imag * (1.0 / gain) * np.cos(phase_rad / 2.0)
        iq = i_chan + 1j * q_chan

    # 5. AWGN Noise Addition
    if snr_db < 100.0:
        snr_lin = 10.0**(snr_db / 10.0)
        noise_pwr = 1.0 / max(snr_lin, 1e-4)
        noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * np.sqrt(noise_pwr / 2.0)
        iq = iq + noise

    return iq.astype(np.complex64)

def build_dataset(num_stage1_frames: int = 80000, num_stage2_seqs: int = 8000,
                  output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating Stage 1 Dataset ({num_stage1_frames} frame instances)...")

    num_classes = len(MODULATION_CLASSES)
    frames_per_class = num_stage1_frames // num_classes

    X1_list = []
    y1_list = []

    for cls_name in MODULATION_CLASSES:
        cls_idx = CLASS_MAP[cls_name]
        print(f"  Synthesizing {frames_per_class} frames for class {cls_name} (id {cls_idx})...")
        for _ in range(frames_per_class):
            # Sweep parameters
            snr_db = float(np.random.uniform(-10.0, 30.0))
            sps = int(np.random.choice([2, 4, 8, 16]))
            cfo_rel = float(np.random.uniform(-0.08, 0.08))
            phase_noise_std = float(np.random.uniform(0.0, 0.025))
            mag_imbal_db = float(np.random.uniform(0.0, 1.2))
            phase_imbal_deg = float(np.random.uniform(0.0, 8.0))

            frame = generate_raw_iq_frame(
                cls_name, num_samples=2048, sps=sps, snr_db=snr_db,
                cfo_rel=cfo_rel, phase_noise_std=phase_noise_std,
                mag_imbal_db=mag_imbal_db, phase_imbal_deg=phase_imbal_deg
            )

            feats = extract_frame_features(frame)
            X1_list.append(feats)
            y1_list.append(cls_idx)

    X1 = np.vstack(X1_list).astype(np.float32)
    y1 = np.array(y1_list, dtype=np.int64)

    s1_path = os.path.join(output_dir, "amc_stage1_features.npz")
    np.savez_compressed(s1_path, X=X1, y=y1)
    print(f"Stage 1 features saved to {s1_path} (Shape X: {X1.shape}, y: {y1.shape})")

    # Generate Stage 2 Sequence Dataset
    print(f"\nGenerating Stage 2 Sequence Dataset ({num_stage2_seqs} sequence instances)...")
    seqs_per_class = num_stage2_seqs // num_classes

    # Train a fast temporary Stage 1 classifier to produce realistic frame predictions for Stage 2
    from sklearn.ensemble import HistGradientBoostingClassifier
    print("Fitting temporary Stage 1 model to extract sequence frame probability distributions...")
    temp_stage1 = HistGradientBoostingClassifier(max_iter=50, random_state=42)
    temp_stage1.fit(X1, y1)

    X2_list = []
    y2_list = []

    for cls_name in MODULATION_CLASSES:
        cls_idx = CLASS_MAP[cls_name]
        print(f"  Synthesizing {seqs_per_class} sequence summaries for class {cls_name}...")
        for _ in range(seqs_per_class):
            num_frames = int(np.random.randint(4, 20))
            snr_db = float(np.random.uniform(-10.0, 30.0))

            frame_feats_seq = []
            snrs_seq = []
            for _ in range(num_frames):
                sps = int(np.random.choice([2, 4, 8, 16]))
                cfo_rel = float(np.random.uniform(-0.08, 0.08))
                phase_noise_std = float(np.random.uniform(0.0, 0.025))

                f = generate_raw_iq_frame(
                    cls_name, num_samples=2048, sps=sps, snr_db=snr_db,
                    cfo_rel=cfo_rel, phase_noise_std=phase_noise_std
                )
                ff = extract_frame_features(f)
                frame_feats_seq.append(ff)
                snrs_seq.append(ff[24])

            X_frames = np.vstack(frame_feats_seq)
            frame_probs = temp_stage1.predict_proba(X_frames)

            # Map predicted classes to 16 classes array
            classes_in_temp = getattr(temp_stage1, "classes_", list(range(16)))
            full_frame_probs = np.zeros((num_frames, 16), dtype=np.float32)
            for idx, c_idx in enumerate(classes_in_temp):
                if c_idx < 16:
                    full_frame_probs[:, c_idx] = frame_probs[:, idx]

            seq_feats = extract_sequence_features(full_frame_probs, snrs_list=snrs_seq)
            X2_list.append(seq_feats)
            y2_list.append(cls_idx)

    X2 = np.vstack(X2_list).astype(np.float32)
    y2 = np.array(y2_list, dtype=np.int64)

    s2_path = os.path.join(output_dir, "amc_stage2_features.npz")
    np.savez_compressed(s2_path, X=X2, y=y2)
    print(f"Stage 2 features saved to {s2_path} (Shape X: {X2.shape}, y: {y2.shape})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AMC Feature Datasets")
    parser.add_argument("--stage1_frames", type=int, default=80000, help="Total Stage 1 frame instances")
    parser.add_argument("--stage2_seqs", type=int, default=8000, help="Total Stage 2 sequence instances")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for feature files")
    args = parser.parse_args()

    build_dataset(
        num_stage1_frames=args.stage1_frames,
        num_stage2_seqs=args.stage2_seqs,
        output_dir=args.output_dir
    )
