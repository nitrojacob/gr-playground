# AMC Classifier Upgrade: Training Data Collection, Feature Engineering & ML Plan

## 1. Executive Summary & Goals

This document specifies the end-to-end plan of action to upgrade the Automatic Modulation Classification (AMC) subsystem in `gr_playground` from a hand-tuned decision tree to a high-performance, feature-based Machine Learning classifier (XGBoost / LightGBM).

### Objectives
- **Replace Fragile Thresholds**: Eliminate hardcoded cutoff boundaries in `classify_modulation()` with a trained classifier.
- **Maintain Real-Time Speed**: Keep inference latency under $1.0\text{ ms}$ per 2048-sample block on a single CPU thread without requiring GPU acceleration.
- **High Classification Accuracy**:
  - $> 90\%$ overall accuracy at $\text{SNR} \ge 5\text{ dB}$.
  - $> 95\%$ overall accuracy at $\text{SNR} \ge 10\text{ dB}$.
- **Supported Modulation Classes (13 Total)**:
  - **Analog**: AM, FM
  - **Digital Constant Envelope**: BPSK, GFSK
  - **Digital Phase/Quadrature**: QPSK, 8PSK, 16QAM, 64QAM, 256QAM
  - **Digital Amplitude Shift Keying**: ASK / OOK
  - **Multicarrier / Broadband**: OFDM, SC-FDMA
  - **Unmodulated Channel**: Thermal Noise

---

## 2. Dataset Synthesis & Collection Pipeline

To train a robust classifier, training data will be generated using synthetic channel simulation supplemented by real-world RTL-SDR captures.

```
+------------------------------------+
|  GR-Playground Channel Simulator   |
| (gr_playground/simulator/...)      |
+-----------------+------------------+
                  |
                  v
+------------------------------------+
| Synthetic Signal Generation Sweep  |
| - 13 Modulation Classes            |
| - SNR Sweep: -10 dB to +30 dB      |
| - Impairments: CFO, SRO, IQ, Fading|
+-----------------+------------------+
                  |
                  +-----------------------------------+
                  |                                   |
                  v                                   v
+------------------------------------+   +------------------------------------+
| Synthetic IQ Frame Dataset         |   | Real RTL-SDR SigMF Annotations     |
| (250,000 Frames)                   |   | (25,000 Frames)                    |
+-----------------+------------------+   +----------------+-------------------+
                  |                                   |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Combined Feature Matrix           |
                  | (X: [275,000 x 26], Y: [275,000]) |
                  +-----------------------------------+
```

### 2.1 Synthetic Dataset Generation

Synthetic samples will be produced using `ChannelSimulatorFlowgraph` (`gr_playground/simulator/channel_simulator.py`).

* **Total Samples**: 275,000 frame instances ($20,000$ synthetic frames per class + $25,000$ real hardware frames).
* **Frame Size**: $N = 2048$ complex64 IQ samples per frame.
* **Parameter Sweep Matrix**:

| Parameter | Sweep Range / Distribution | Description |
| :--- | :--- | :--- |
| **SNR** | $-10\text{ dB}$ to $+30\text{ dB}$ ($1\text{ dB}$ step size) | Additive White Gaussian Noise (AWGN) level |
| **Carrier Frequency Offset (CFO)** | Uniform $[-0.10 \cdot f_s, +0.10 \cdot f_s]$ | Residual frequency drift relative to sample rate |
| **Symbol Rate / SPS** | $SPS \in \{2, 4, 8, 16\}$ | Samples per symbol for digital schemes |
| **IQ Amplitude Imbalance** | Uniform $[0.0\text{ dB}, 1.5\text{ dB}]$ | Receiver front-end magnitude mismatch |
| **IQ Phase Imbalance** | Uniform $[0^\circ, 10^\circ]$ | Quadrature mixer phase skew |
| **Phase Noise** | Uniform $[0.000, 0.030]\text{ rad}$ standard dev | Oscillator phase jitter |
| **Multipath Fading** | Rayleigh / Rician 2-tap & 4-tap models | L1 wireless channel multipath profiles |
| **HPA Non-linearity** | Input Back-off (IBO) $3\text{ dB}$ to $12\text{ dB}$ | High-power amplifier saturation for QAM |

### 2.2 Real-World Data Capture Integration

To prevent synthetic-to-real domain shift:
1. **Source Capture**: Collect 30-second SigMF captures from live hardware across:
   - $88 - 108\text{ MHz}$ (FM Broadcast)
   - $433.92\text{ MHz}$ ISM band (ASK/GFSK weather sensors/remotes)
   - $1090\text{ MHz}$ ADS-B (PPM/ASK)
   - NOAA APT satellite band ($137.5\text{ MHz}$, FM/AM)
   - Thermal noise baseline (antenna disconnected / 50-ohm termination)
2. **Annotation & Segmenting**: Slice continuous streams into 2048-sample frames labeled with ground truth modulation.
3. **Data Mix**: Integrate 10% real annotated frames into the training set and 20% into the final validation test set.

### 2.3 Real-World Signal Labeling Protocol

Real hardware RF captures are labeled through a three-stage semi-automated protocol:

1. **SigMF Metadata Ground Truth**:
   - Captures are recorded with native `.sigmf-meta` headers. Standard frequency allocations (e.g. 88-108 MHz WFM, 1090 MHz ADS-B PPM/OOK, 433.92 MHz ISM key fobs/sensors) provide known ground-truth labels written into the SigMF `annotations` array:
     ```json
     "annotations": [
       {
         "core:sample_start": 0,
         "core:sample_count": 2048000,
         "core:freq_lower_edge": 433870000,
         "core:freq_upper_edge": 433970000,
         "core:label": "GFSK",
         "core:comment": "433.92MHz ISM Weather Sensor"
       }
     ]
     ```
2. **Spectrogram Energy Slicing**:
   - Long continuous captures are sliced into candidate burst segments using energy detection (Welch PSD power thresholding above noise floor).
   - Inactive frames during silent periods are automatically flagged as `Noise`.
   - Active signal frames inherit the modulation label from the corresponding SigMF annotation block.
3. **Visual & IQ Demodulation Verification**:
   - High-SNR captures are visually verified using FFT waterfalls and constellation diagrams prior to dataset ingestion.

### 2.4 Streaming Generation & Feature Matrix Persistence Architecture

To balance **storage efficiency** with **experimentation flexibility** (allowing fast retraining across different ML algorithms without repeating CPU-heavy DSP feature extraction):

```
Generator (In-Memory IQ Block) ──► Extract 26 Features ──► Append to Matrix ──► Save Feature File (.npz)
  (2048 Samples ~ 16 KB)            (DSP Pipeline)         (104 Bytes/Frame)        (~31.1 MB Total on Disk)
                                                                 │
                                                                 └──► DISCARD Raw IQ Block (0 GB Raw Storage)
```

#### Protocol:
1. **Discard Raw IQ Samples Immediately**: Raw IQ frames ($2048$ samples per block) are synthesized transiently in memory, immediately processed by `extract_feature_vector()`, and garbage collected ($0.0\text{ GB}$ raw sample disk storage).
2. **Persist Compact Feature Matrices**: The resulting extracted feature arrays are saved to disk as compressed `.npz` or `.parquet` files (`amc_stage1_features.npz` and `amc_stage2_features.npz`).
3. **Re-usability**: Any ML algorithm (e.g. `HistGradientBoostingClassifier`, `RandomForest`, `XGBoost`, SVM, MLP Neural Networks) can be trained instantly from the persistent `~31.1 MB` feature files without re-running DSP calculations.

#### Memory & Storage Footprint:

| Dataset Component | Target Unit | Instance Count | Matrix Shape | Persistent Feature File (Disk) | Raw IQ Sample Caching |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1 (Frame-Level)** | 2048-sample IQ frame | **286,000 frames** (260k synth + 26k real) | `[286,000 , 26]` | **~29.7 MB** (`amc_stage1_features.npz`) | **0.0 GB (Discarded)** |
| **Stage 2 (Meta-Learner)** | Multi-frame sequence | **19,500 sequences** (1,500 / class) | `[19,500 , 18]` | **~1.4 MB** (`amc_stage2_features.npz`) | **0.0 GB (Discarded)** |
| **Total Feature Storage** | Extracted Feature Files | **305,500 total** | — | **~31.1 MB Total Disk Space** | **0.0 GB Storage** |

#### Deployable Model Artifact Payload:
* `amc_frame_classifier.pkl` / `.json`: **~1.5 MB**
* `amc_meta_classifier.pkl` / `.json`: **~0.3 MB**
* **Total Stored Model Artifact Payload**: **~1.8 MB** (embedded into repository assets).

---

## 3. Feature Extraction Pipeline

Instead of training complex deep neural networks on raw IQ samples, the model will consume an **extended 26-element physical feature vector** calculated per frame.

```
Complex IQ Frame (2048 samples)
    ├── 1. Higher-Order Cumulants (C20, C21, C40, C42, C63 + derotations)
    ├── 2. Envelope & Constellation Statistics (amp_var, phase_var, freq_var, amp_kurtosis, zero_ratio)
    ├── 3. Spectral & Frequency Features (spectral_flatness, spectral_kurtosis, PAPR, PSD_asymmetry)
    └── 4. Cyclic / FFT Peak Features (fft2_p2a, fft4_p2a, max_fft_peak)
                               │
                               v
                     26-Dimensional Feature Vector
```

### 3.1 Feature Definitions

1. **Higher-Order Cumulants (HOC)**:
   - $C_{20}, C_{21}$: Second-order cumulants (raw, 2nd-power derotated, 4th-power derotated).
   - $C_{40}, C_{42}$: Fourth-order cumulants measuring constellation geometry & rotational symmetry.
   - $C_{63}$: Sixth-order cumulant for high-density QAM separation.
2. **Envelope & Constellation Features**:
   - `amp_var`: Normalized amplitude variance $\text{Var}(|y|) / \mu_{|y|}^2$.
   - `phase_var`: Phase variance $\text{Var}(\angle y)$.
   - `freq_var`: Instantaneous frequency variance $\text{Var}(\Delta \text{unwrap}(\angle y))$.
   - `amp_kurtosis`: Amplitude kurtosis $\mathbb{E}[(|y|-\mu)^4] / \sigma^4$.
   - `zero_ratio`: Ratio of samples with magnitude $< 0.25 \cdot \max(|y|)$ (detects ASK/OOK bimodal distributions).
3. **Spectral Features**:
   - `spectral_flatness`: Wiener entropy of Welch PSD ($\exp(\mathbb{E}[\ln(PSD)]) / \mathbb{E}[PSD]$).
   - `spectral_kurtosis`: Fourth moment of power spectral density.
   - `papr_db`: Peak-to-Average Power Ratio in dB.
   - `psd_asymmetry`: Difference between upper and lower sideband power (distinguishes SSB/AM from FM).
4. **Cyclic / FFT Peak Ratios**:
   - `fft2_p2a`: Peak-to-average ratio of $\text{FFT}(y^2)$ spectrum (detects BPSK/ASK carrier lines).
   - `fft4_p2a`: Peak-to-average ratio of $\text{FFT}(y^4)$ spectrum (detects QPSK/QAM carrier lines).

---

## 4. Model Selection & Training Methodology

### 4.1 Algorithm Selection
* **Primary Algorithm**: **`sklearn.ensemble.HistGradientBoostingClassifier`** (Pre-installed via `scikit-learn 1.4.1`)
  * Loss: `log_loss` (multi-class cross-entropy)
  * Number of Classes: 13
  * Parallelization: OpenMP / multi-threading built-in
  * Advantages: Requires zero external package installations, offers sub-millisecond inference per block, and matches LightGBM/XGBoost classification performance.
* **Secondary / Optional Algorithm**: `xgboost.XGBClassifier` (if explicitly installed)

### 4.2 Cross-Validation & Data Split
* **Train / Val / Test Split**:
  * 70% Training (~192,500 samples)
  * 15% Validation (~41,250 samples)
  * 15% Test (~41,250 samples)
* **Stratification**: Stratified by both **Modulation Class** and **SNR bin** to ensure uniform noise representation.

### 4.3 Hyperparameter Optimization
Hyperparameter tuning search space:

| Parameter | Search Space | Target Range |
| :--- | :--- | :--- |
| `max_depth` | $4$ to $10$ | $6 - 8$ |
| `learning_rate` | $0.01$ to $0.20$ | $0.05 - 0.10$ |
| `max_iter` | $100$ to $500$ | $250$ (with early stopping) |
| `l2_regularization` | $0.0$ to $10.0$ | Regularization parameter |
| `min_samples_leaf` | $10$ to $100$ | Overfitting control |

---

## 5. Model Serialization & Runtime Deployment

To integrate the trained classifier seamlessly into `gr_playground`:

1. **Model Persistence**:
   - Save trained model binary to `gr_playground/dsp/models/amc_xgboost.json`.
   - Include feature normalization parameters (mean/scale vectors) in `amc_xgboost_scaler.json`.
2. **Runtime Inference Integration**:
   - Update `classify_modulation()` in `gr_playground/dsp/modulation_id.py`:
     ```python
     # Extract 26-element feature vector
     features = extract_feature_vector(samples)
     # Run XGBoost inference
     probs = amc_model.predict_proba(features)[0]
     # Return dictionary mapping class name -> probability score
     return {class_names[i]: float(probs[i]) for i in range(len(class_names))}
     ```
3. **Fallback Protection**:
   - If `scikit-learn` model file fails to load, fall back gracefully to the legacy rule-based heuristic classifier.

### 5.2 Multi-Frame Temporal Aggregation (For 30s+ Captures)

When processing continuous long-duration captures (e.g. 30 seconds $\approx 60\times 10^6$ samples), the stream is divided into $M$ sliding frames ($N = 2048$ samples each). Predictions are combined into a single signal-level verdict using a two-stage aggregation pipeline:

1. **Squelch / Silence Filtering**:
   - Frames where `spectral_flatness > 0.85` or local power falls below the channel squelch threshold are categorized as `Noise` and excluded from modulation voting.

2. **SNR-Weighted Soft Probability Averaging (Soft Voting)**:
   - For all active frames $i \in \{1 \dots M_{active}\}$, extract the class probability vector $\mathbf{p}_i$.
   - Compute the weighted ensemble probability vector:
     $$\mathbf{P}_{\text{overall}} = \frac{\sum_{i=1}^{M_{active}} w_i \cdot \mathbf{p}_i}{\sum_{i=1}^{M_{active}} w_i}$$
     where $w_i = \max(0, \text{SNR}_i - \text{SNR}_{\text{floor}})$ weights high-SNR frames more heavily.
   - The overall modulation is declared as $C_{\text{final}} = \arg\max_k P_{\text{overall}, k}$.

3. **Intermittent / Burst Spectrum Logging**:
   - For bursty transmissions (e.g. push-to-talk FM, ISM 433 MHz sensor bursts), the scanner logs both the overall primary modulation AND a temporal duty-cycle breakdown (e.g., `GFSK [Active 12% of 30s capture]`).

### 5.3 Meta-Learner ML Stacking Model (Single Combined Classifier)

To replace heuristic averaging rules with a single data-driven ML decision, a **Meta-Learner Stacking Model** (`sklearn.ensemble.HistGradientBoostingClassifier`) aggregates sequence-level summary statistics across all frames $t = 1 \dots T$:

```
Frame 1..T Predictions ──► Summary Feature Matrix ──► Meta-ML Classifier ──► Single Verdict
                           - Mean Probabilities      (HistGradientBoosting)   (e.g., "16QAM")
                           - Max Probabilities
                           - Probability Variances
                           - Active Duty Cycle %
                           - Mean/Var SNR
```

1. **Meta-Feature Matrix Construction**:
   For a capture of $T$ frames, compute an 18-element sequence feature vector:
   * `mean_prob_[1..13]`: Mean probability vector across active frames ($\bar{\mathbf{p}}$).
   * `max_prob_[1..13]`: Maximum probability vector ($\max \mathbf{p}_t$) to detect transient burst modulations.
   * `prob_var_[1..13]`: Variance of probabilities across time (distinguishes continuous vs. bursty modulations).
   * `active_duty_cycle`: Fraction of active non-noise frames ($\frac{M_{active}}{T}$).
   * `mean_snr`, `var_snr`: Average and variance of frame-level SNR estimates.

2. **Single Combined Verdict**:
   - The Meta-Classifier maps the 18-element sequence feature vector to a single final modulation class.
   - Preserves fast sub-millisecond CPU inference while learning non-linear temporal interactions automatically.

---

## 6. Verification & Evaluation Plan

### 6.1 Automated Test Suite
1. **SNR vs Accuracy Curve**:
   - Plot classification accuracy across SNR range ($-10\text{ dB}$ to $+30\text{ dB}$) in 2 dB steps.
   - Verify $> 90\%$ accuracy at $\ge 5\text{ dB}$ SNR.
2. **Confusion Matrix Analysis**:
   - Validate low cross-talk between similar constellations (e.g. QPSK vs 16QAM, FM vs GFSK).
3. **Latency Benchmarking**:
   - Measure 1,000 inference calls using `pytest-benchmark`. Target total per-block execution time $< 1.0\text{ ms}$.

### 6.2 Implementation Checklist

- [ ] Create synthetic dataset generation script (`scripts/generate_amc_dataset.py`).
- [ ] Implement feature vector extractor (`gr_playground/dsp/amc_features.py`).
- [ ] Train XGBoost model and tune hyperparameters (`scripts/train_amc_classifier.py`).
- [ ] Export model artifacts to `gr_playground/dsp/models/`.
- [ ] Update `classify_modulation()` in `gr_playground/dsp/modulation_id.py`.
- [ ] Add regression and accuracy unit tests in `tests/test_amc_classifier.py`.
