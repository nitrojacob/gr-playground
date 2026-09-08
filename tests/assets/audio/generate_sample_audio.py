#!/usr/bin/env python3
"""
Generate synthetic audio reference files for gr-playground audio source testing.
Creates speech, chirp, tone sweep, and music WAV files under examples/audio/
"""

import os
import numpy as np
from scipy.io import wavfile

AUDIO_DIR = os.path.dirname(os.path.abspath(__file__))

def create_synthetic_speech(sample_rate=32000, duration=3.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Formant frequencies for synthetic vowels /a/, /i/, /u/
    f0 = 130 + 15 * np.sin(2 * np.pi * 3 * t)  # Pitch modulation
    # Synthesis via harmonics and envelope modulation
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 2.5 * t)) * (np.sin(2 * np.pi * 0.5 * t) > -0.2)
    signal = (np.sin(2 * np.pi * f0 * t) + 
              0.5 * np.sin(2 * np.pi * f0 * 2 * t) + 
              0.3 * np.sin(2 * np.pi * f0 * 3 * t) + 
              0.2 * np.sin(2 * np.pi * 800 * t) + 
              0.15 * np.sin(2 * np.pi * 2400 * t))
    audio = signal * envelope
    audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.9
    return sample_rate, (audio * 32767).astype(np.int16)

def create_chirp(sample_rate=32000, duration=3.0, f_start=200, f_end=4000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Linear chirp phase: 2*pi * (f_start * t + 0.5 * (f_end - f_start)/duration * t^2)
    k = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    audio = np.sin(phase) * 0.8
    return sample_rate, (audio * 32767).astype(np.int16)

def create_tone_sweep(sample_rate=32000, duration=3.0, f_min=100, f_max=6000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Logarithmic frequency sweep
    log_freq = f_min * (f_max / f_min) ** (t / duration)
    phase = 2 * np.pi * f_min * duration / np.log(f_max / f_min) * ((f_max / f_min) ** (t / duration) - 1)
    audio = np.sin(phase) * 0.85
    return sample_rate, (audio * 32767).astype(np.int16)

def create_music(sample_rate=32000, duration=4.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Arpeggiated chord (C major: C4=261.63, E4=329.63, G4=392.00, C5=523.25)
    notes = [261.63, 329.63, 392.00, 523.25]
    audio = np.zeros_like(t)
    num_notes = len(notes)
    note_duration = duration / num_notes
    for idx, f in enumerate(notes):
        mask = (t >= idx * note_duration) & (t < (idx + 1) * note_duration)
        sub_t = t[mask] - idx * note_duration
        note_env = np.exp(-3 * sub_t / note_duration)
        audio[mask] = (np.sin(2 * np.pi * f * sub_t) + 0.4 * np.sin(2 * np.pi * f * 2 * sub_t)) * note_env
    audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.85
    return sample_rate, (audio * 32767).astype(np.int16)

def generate_all():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    files = {
        "speech.wav": create_synthetic_speech(),
        "chirp.wav": create_chirp(),
        "tone_sweep.wav": create_tone_sweep(),
        "music.wav": create_music(),
    }
    for filename, (sr, data) in files.items():
        path = os.path.join(AUDIO_DIR, filename)
        wavfile.write(path, sr, data)
        print(f"Generated {filename} ({sr} Hz, {len(data)} samples) -> {path}")

if __name__ == "__main__":
    generate_all()
