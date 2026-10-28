"""Pure numpy/pandas helpers (no Spark) so they are trivially unit-testable."""
import numpy as np


def psi(expected, actual, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), eps, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def categorical_psi(expected, actual, eps: float = 1e-6) -> float:
    import pandas as pd
    e = pd.Series(expected).value_counts(normalize=True)
    a = pd.Series(actual).value_counts(normalize=True)
    cats = e.index.union(a.index)
    e = e.reindex(cats).fillna(0).clip(lower=eps)
    a = a.reindex(cats).fillna(0).clip(lower=eps)
    return float(((a - e) * np.log(a / e)).sum())


def precision_recall_at_k(y_true, score, k_frac: float = 0.10) -> dict:
    """Care-management teams have fixed capacity: how good are the top k% of patients?"""
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    k = max(1, int(len(score) * k_frac))
    idx = np.argsort(-score)[:k]
    tp = float(y_true[idx].sum())
    prevalence = float(y_true.mean())
    precision = tp / k
    recall = tp / max(float(y_true.sum()), 1.0)
    lift = precision / prevalence if prevalence > 0 else float("nan")
    return {"precision": precision, "recall": recall, "lift": lift, "k": k}


def expected_calibration_error(y_true, prob, bins: int = 10) -> float:
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob > lo) & (prob <= hi)
        if m.sum():
            ece += m.mean() * abs(y_true[m].mean() - prob[m].mean())
    return float(ece)


def risk_tier(score: float, high_thr: float, med_thr: float) -> str:
    if score >= high_thr:
        return "HIGH"
    if score >= med_thr:
        return "MEDIUM"
    return "LOW"


def age_band(age) -> str:
    age = float(age)
    if age < 50:
        return "<50"
    if age < 65:
        return "50-64"
    if age < 80:
        return "65-79"
    return "80+"
