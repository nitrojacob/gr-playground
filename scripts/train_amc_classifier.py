"""
Training Script for Automatic Modulation Classification (AMC) Model.
Trains PyTorch AMCFeatureNet across all 28 MODULATION_CLASSES (including all RadioML 2018.01A schemes)
and exports the trained model directly to portable ONNX format (`gr_playground/dsp/models/amc_model.onnx`).

ONNX is the de-facto model format for persistence across runtime environments (including Kaggle).
"""

import os
import sys
import argparse

# Ensure GNU Radio system path and workspace root are in sys.path
gnuradio_path = "/usr/lib/python3/dist-packages"
if os.path.exists(gnuradio_path) and gnuradio_path not in sys.path:
    sys.path.append(gnuradio_path)

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gr_playground.dsp.amc import MODULATION_CLASSES, RADIOML_TO_PARENT_MAP
from gr_playground.dsp.amc.features import extract_frame_features
from gr_playground.simulator.channel_simulator import ChannelSimulatorFlowgraph
from scripts.generate_amc_dataset import generate_raw_iq_frame

class AMCFeatureNet(nn.Module):
    """
    PyTorch Feature Classifier mapping 26 physical DSP features -> 28 modulation logits.
    Exportable directly to portable ONNX format via torch.onnx.export.
    """
    def __init__(self, input_dim=26, num_classes=len(MODULATION_CLASSES)):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(self.input_bn(x))


def _generate_single_sample(args):
    mod_name, target_idx, snr, num_samples, cfo_rel, phase_noise, mag_imbal, phase_imbal, use_simulator = args
    if use_simulator:
        try:
            cfo_hz = cfo_rel * 32000.0
            phase_deg = phase_noise * 180.0
            sim = ChannelSimulatorFlowgraph(
                mod_type=mod_name,
                num_samples=num_samples,
                snr_db=snr,
                cfo_hz=cfo_hz,
                phase_offset_deg=phase_deg
            )
            samples = sim.get_samples()
        except Exception:
            samples = generate_raw_iq_frame(
                mod_type=mod_name,
                num_samples=num_samples,
                snr_db=snr,
                cfo_rel=cfo_rel,
                phase_noise_std=phase_noise,
                mag_imbal_db=mag_imbal,
                phase_imbal_deg=phase_imbal
            )
    else:
        samples = generate_raw_iq_frame(
            mod_type=mod_name,
            num_samples=num_samples,
            snr_db=snr,
            cfo_rel=cfo_rel,
            phase_noise_std=phase_noise,
            mag_imbal_db=mag_imbal,
            phase_imbal_deg=phase_imbal
        )
    feat = extract_frame_features(samples)
    return feat, target_idx


def generate_training_data(samples_per_class=300):
    from concurrent.futures import ProcessPoolExecutor
    print(f"⚡ Generating synthetic DSP feature dataset across {len(MODULATION_CLASSES)} modulation classes (in-memory parallel)...")
    
    class_to_idx = {c: i for i, c in enumerate(MODULATION_CLASSES)}
    snrs = [-2.0, 0.0, 3.0, 6.0, 10.0, 15.0, 24.0, 30.0]

    tasks = []
    for mod_name in MODULATION_CLASSES:
        parent_mod = RADIOML_TO_PARENT_MAP.get(mod_name, mod_name)
        target_idx = class_to_idx[parent_mod]

        for _ in range(samples_per_class):
            snr = float(np.random.choice(snrs))
            num_samples = int(np.random.choice([2048, 4096, 8192]))
            cfo_rel = float(np.random.uniform(-0.02, 0.02))
            phase_noise = float(np.random.uniform(0.0, 0.03))
            mag_imbal = float(np.random.uniform(0.0, 1.0))
            phase_imbal = float(np.random.uniform(0.0, 10.0))
            use_simulator = (np.random.rand() < 0.5)

            tasks.append((mod_name, target_idx, snr, num_samples, cfo_rel, phase_noise, mag_imbal, phase_imbal, use_simulator))

    X_list = []
    y_list = []
    
    # Run parallel feature extraction in RAM memory
    max_workers = min(os.cpu_count() or 4, 16)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_generate_single_sample, tasks, chunksize=100)
        for feat, target_idx in results:
            X_list.append(feat)
            y_list.append(target_idx)

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Standardize feature scaling to prevent infs
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    return X, y


def train_and_export_onnx(output_model_dir: str = "gr_playground/dsp/models", samples_per_class: int = 1000, epochs: int = 40):
    os.makedirs(output_model_dir, exist_ok=True)
    onnx_path = os.path.join(output_model_dir, "amc_model.onnx")

    X, y = generate_training_data(samples_per_class=samples_per_class)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Train PyTorch Model for ONNX Export
    device = torch.device("cpu")
    model = AMCFeatureNet(input_dim=26, num_classes=len(MODULATION_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"🏋️ Training PyTorch Feature Model ({epochs} epochs)...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for b_x, b_y in loader:
            optimizer.zero_grad()
            out = model(b_x)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_tensor), dim=1)
        acc = (preds == y_tensor).float().mean().item()
        print(f"✅ PyTorch Model Accuracy: {acc * 100:.2f}%")

    # Export ONNX Model Artifact directly
    dummy_input = torch.randn(1, 26, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"🎉 Successfully exported ONNX model artifact to: {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AMC PyTorch Model & Export ONNX Binary")
    parser.add_argument("--output_model_dir", type=str, default="gr_playground/dsp/models", help="Directory to save amc_model.onnx")
    parser.add_argument("--samples_per_class", type=int, default=1000, help="Number of synthetic frames per class")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    args = parser.parse_args()

    train_and_export_onnx(output_model_dir=args.output_model_dir, samples_per_class=args.samples_per_class, epochs=args.epochs)
