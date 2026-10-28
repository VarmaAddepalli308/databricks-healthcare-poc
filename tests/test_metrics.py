import numpy as np
from poc_lib.metrics import psi, categorical_psi, precision_recall_at_k, risk_tier, age_band, expected_calibration_error


def test_psi_identical_is_zero():
    x = np.random.RandomState(0).normal(size=5000)
    assert psi(x, x) < 1e-6


def test_psi_detects_shift():
    r = np.random.RandomState(0)
    assert psi(r.normal(0, 1, 5000), r.normal(1.5, 1, 5000)) > 0.25


def test_categorical_psi():
    assert categorical_psi(["a"] * 90 + ["b"] * 10, ["a"] * 50 + ["b"] * 50) > 0.25


def test_precision_at_k_perfect_ranking():
    y = np.array([1] * 10 + [0] * 90)
    s = np.array([1.0] * 10 + [0.0] * 90)
    out = precision_recall_at_k(y, s, 0.10)
    assert out["precision"] == 1.0 and out["recall"] == 1.0 and out["lift"] == 10.0


def test_risk_tier_and_age_band():
    assert risk_tier(0.5, 0.4, 0.2) == "HIGH" and risk_tier(0.3, 0.4, 0.2) == "MEDIUM" and risk_tier(0.1, 0.4, 0.2) == "LOW"
    assert age_band(45) == "<50" and age_band(70) == "65-79" and age_band(88) == "80+"


def test_ece_calibrated_data_is_small():
    r = np.random.RandomState(1)
    p = r.uniform(0, 1, 20000)
    y = (r.uniform(0, 1, 20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.03
