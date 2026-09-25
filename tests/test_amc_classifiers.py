"""
Unit Test Suite for Modular Automatic Modulation Classification (AMC) Subsystem.
Tests HeuristicAMCClassifier, MLAMCClassifier, DLAMCClassifier, Feature Extractors, and Factory.
"""

import pytest
import numpy as np

from gr_playground.dsp.amc import (
    BaseAMCClassifier,
    HeuristicAMCClassifier,
    MLAMCClassifier,
    DLAMCClassifier,
    get_amc_classifier,
    MODULATION_CLASSES,
    RADIOML_24_CLASSES,
    RADIOML_TO_PARENT_MAP
)
from gr_playground.dsp.amc.features import (
    extract_frame_features,
    extract_sequence_features,
    FEATURE_NAMES
)

@pytest.fixture
def sample_iq():
    """Generates synthetic QPSK IQ samples."""
    np.random.seed(42)
    syms = (np.random.choice([-1, 1], size=1024) + 1j * np.random.choice([-1, 1], size=1024)) / np.sqrt(2)
    sps = 4
    iq = np.repeat(syms, sps)
    noise = (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq))) * 0.1
    return (iq + noise).astype(np.complex64)

def test_feature_extractors(sample_iq):
    """Test 26-element frame feature extractor and sequence feature extractor."""
    frame_feats = extract_frame_features(sample_iq[:2048])
    assert isinstance(frame_feats, np.ndarray)
    assert frame_feats.shape == (26,)
    assert frame_feats.dtype == np.float32
    assert not np.isnan(frame_feats).any()
    assert not np.isinf(frame_feats).any()
    assert len(FEATURE_NAMES) == 26

    # Test sequence feature extractor
    fake_probs = np.random.uniform(size=(10, len(MODULATION_CLASSES))).astype(np.float32)
    seq_feats = extract_sequence_features(fake_probs, snrs_list=[15.0]*10)
    assert isinstance(seq_feats, np.ndarray)
    assert seq_feats.shape == (21,)
    assert seq_feats.dtype == np.float32
    assert not np.isnan(seq_feats).any()
    assert not np.isinf(seq_feats).any()

def test_heuristic_classifier(sample_iq):
    """Test HeuristicAMCClassifier interface conformance and output dictionary."""
    clf = HeuristicAMCClassifier()
    res = clf.classify(sample_iq, sample_rate=32000.0)

    assert isinstance(res, dict)
    assert "QPSK" in res
    assert "16QAM" in res
    assert "FM" in res
    assert len(res) == len(MODULATION_CLASSES)
    assert pytest.approx(sum(res.values()), abs=1e-2) == 1.0

def test_ml_classifier_fallback(sample_iq):
    """Test MLAMCClassifier fallback behavior when model files are not present."""
    clf = MLAMCClassifier(model_dir="/tmp/non_existent_model_dir")
    assert clf.fallback_heuristic is not None

    res = clf.classify(sample_iq, sample_rate=32000.0)
    assert isinstance(res, dict)
    assert len(res) == len(MODULATION_CLASSES)
    assert pytest.approx(sum(res.values()), abs=1e-2) == 1.0

def test_dl_classifier_fallback(sample_iq):
    """Test DLAMCClassifier fallback behavior when DL model weights are missing."""
    clf = DLAMCClassifier(model_path="/tmp/non_existent_dl.onnx")
    assert clf.fallback_classifier is not None

    res = clf.classify(sample_iq, sample_rate=32000.0)
    assert isinstance(res, dict)
    assert pytest.approx(sum(res.values()), abs=1e-2) == 1.0

def test_factory_get_amc_classifier():
    """Test factory strategy selector function."""
    h_clf = get_amc_classifier("heuristic")
    assert isinstance(h_clf, BaseAMCClassifier)
    assert isinstance(h_clf, HeuristicAMCClassifier)

    m_clf = get_amc_classifier("ml")
    assert isinstance(m_clf, BaseAMCClassifier)

    d_clf = get_amc_classifier("dl")
    assert isinstance(d_clf, BaseAMCClassifier)

def test_ml_classifier_with_trained_models(sample_iq):
    """Test MLAMCClassifier loading trained model binaries/ONNX and running inference."""
    clf = MLAMCClassifier()
    assert clf.fallback_heuristic is None or clf.onnx_session is not None or clf.stage1_model is not None

    res = clf.classify(sample_iq, sample_rate=32000.0)
    assert isinstance(res, dict)
    assert len(res) == len(MODULATION_CLASSES)
    assert pytest.approx(sum(res.values()), abs=1e-2) == 1.0

    top_class = max(res.items(), key=lambda x: x[1])[0]
    assert top_class in MODULATION_CLASSES

def test_classify_modulation_with_ml_mode(sample_iq):
    """Test top-level classify_modulation function with mode='ml'."""
    from gr_playground.dsp.modulation_id import classify_modulation
    preds, cumulants, const_stats = classify_modulation(sample_iq, mode="ml")
    
    assert len(preds) == len(MODULATION_CLASSES)
    assert isinstance(cumulants, dict)
    assert isinstance(const_stats, dict)
    assert pytest.approx(sum(score for _, score in preds), abs=1e-2) == 1.0

def test_ml_classifier_across_core_modulations_and_impairments():
    """
    Sweeps MLAMCClassifier across core modulation schemes under various impairment levels.
    """
    from scripts.generate_amc_dataset import generate_raw_iq_frame

    clf = MLAMCClassifier()

    test_classes = ["AM", "FM", "BPSK", "GFSK", "QPSK", "8PSK", "ASK", "OFDM", "SC-FDMA", "Noise"]
    snr_levels = [10.0, 20.0]

    for mod in test_classes:
        for snr in snr_levels:
            iq = generate_raw_iq_frame(
                mod, num_samples=8192, sps=4, snr_db=snr,
                cfo_rel=0.02, phase_noise_std=0.01, mag_imbal_db=0.5
            )
            res = clf.classify(iq, sample_rate=32000.0)

            assert isinstance(res, dict)
            assert len(res) == len(MODULATION_CLASSES)
            assert pytest.approx(sum(res.values()), abs=1e-2) == 1.0
            assert not any(np.isnan(v) for v in res.values())

@pytest.mark.next_level
def test_next_level_radioml_modulations():
    """
    Next-Level PyTest Suite for fine-grained RadioML 24 modulation schemes.
    Evaluates MLAMCClassifier on fine-grained APSK, PSK, QAM, and Analog sub-classes.
    """
    clf = MLAMCClassifier()

    for r_mod in RADIOML_24_CLASSES:
        parent_mod = RADIOML_TO_PARENT_MAP.get(r_mod, r_mod)
        # Verify MLAMCClassifier returns scores for all RadioML 24 classes
        dummy_iq = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex64)
        res = clf.classify(dummy_iq)

        assert isinstance(res, dict)
        assert r_mod in res
        assert parent_mod in res
        assert pytest.approx(sum(res.values()), abs=1e-2) == 1.0


