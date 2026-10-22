"""Dawid & Skene (1979) EM for aggregating noisy judges without gold labels.

Each judge j has a confusion matrix theta_j[k, l] = P(judge says l | truth k). EM alternates between the
posterior over true labels and the judges' confusion matrices. Its by-product -- each judge's *estimated*
sensitivity and specificity, learned without gold -- is what makes it useful for validating judges.
"""
from __future__ import annotations

import numpy as np


def dawid_skene(labels: np.ndarray, n_classes: int = 2, iters: int = 100, smoothing: float = 0.01, tol=1e-6):
    """labels: (items, judges) int array, -1 for missing. Returns (posteriors, confusions, prior)."""
    L = np.asarray(labels)
    n, J = L.shape
    # initialize posteriors with (soft) majority vote
    T = np.zeros((n, n_classes))
    for k in range(n_classes):
        T[:, k] = (L == k).sum(1)
    T = (T + smoothing) / (T + smoothing).sum(1, keepdims=True)
    for _ in range(iters):
        prior = T.mean(0)
        conf = np.full((J, n_classes, n_classes), smoothing)
        for j in range(J):
            for l in range(n_classes):
                conf[j, :, l] += T[L[:, j] == l].sum(0)
        conf /= conf.sum(2, keepdims=True)
        logT = np.tile(np.log(prior), (n, 1))
        for j in range(J):
            seen = L[:, j] >= 0
            logT[seen] += np.log(conf[j][:, L[seen, j]].T)
        T_new = np.exp(logT - logT.max(1, keepdims=True))
        T_new /= T_new.sum(1, keepdims=True)
        if np.abs(T_new - T).max() < tol:
            T = T_new
            break
        T = T_new
    return T, conf, prior
