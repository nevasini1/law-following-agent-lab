"""Small, dependency-light statistics used across experiments."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as st


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion with Wilson score interval (well-behaved at 0/n and n/n, unlike the Wald interval)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def bootstrap_ci(x, fn=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    boots = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)]
    return float(fn(x)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def mcnemar(a, b) -> float:
    """Exact McNemar p-value for paired binary outcomes a, b (same items under two conditions)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(st.binomtest(n01, n01 + n10, 0.5).pvalue)


def cohen_kappa(y1, y2) -> float:
    y1, y2 = np.asarray(y1), np.asarray(y2)
    po = float((y1 == y2).mean())
    labels = np.union1d(y1, y2)
    pe = float(sum((y1 == l).mean() * (y2 == l).mean() for l in labels))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def ece(probs, labels, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(probs[bins == b].mean() - labels[bins == b].mean()) * (bins == b).mean()
                     for b in range(n_bins) if (bins == b).any()))
