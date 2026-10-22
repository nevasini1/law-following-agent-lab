"""Prediction-powered inference for a mean (e.g. an agent's violation rate) from an LLM judge.

Angelopoulos et al., "Prediction-powered inference", Science 2023; power tuning (PPI++) from
Angelopoulos, Duchi & Zrnic 2023. With n gold labels Y and judge predictions f on those n items and on N
unlabeled items:

    theta_PP   = mean_N(f) - mean_n(f - Y)                       (unbiased whatever the judge's errors)
    theta_PP++ = lam * mean_N(f) + mean_n(Y - lam * f),  lam* = Cov(Y, f) / ((1 + n/N) Var(f))

The judge only buys variance reduction; validity comes from the n gold labels. A naive judge-only estimate
inherits the judge's bias, and its confidence interval shrinks around the wrong number as N grows.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def classical(y_lab, alpha=0.05):
    y = np.asarray(y_lab, float)
    se = y.std(ddof=1) / np.sqrt(len(y))
    z = norm.ppf(1 - alpha / 2)
    return y.mean(), y.mean() - z * se, y.mean() + z * se


def naive(f_all, alpha=0.05):
    f = np.asarray(f_all, float)
    se = f.std(ddof=1) / np.sqrt(len(f))
    z = norm.ppf(1 - alpha / 2)
    return f.mean(), f.mean() - z * se, f.mean() + z * se


def ppi_mean(y_lab, f_lab, f_unl, alpha=0.05, tuned=True):
    y, fl, fu = (np.asarray(a, float) for a in (y_lab, f_lab, f_unl))
    n, N = len(y), len(fu)
    if tuned:
        f_all = np.concatenate([fl, fu])
        var_f = f_all.var(ddof=1)
        lam = float(np.cov(y, fl)[0, 1] / ((1 + n / N) * var_f)) if var_f > 0 else 0.0
        lam = min(max(lam, 0.0), 1.0)
    else:
        lam = 1.0
    est = lam * fu.mean() + (y - lam * fl).mean()
    se = np.sqrt(lam ** 2 * fu.var(ddof=1) / N + (y - lam * fl).var(ddof=1) / n)
    z = norm.ppf(1 - alpha / 2)
    return est, est - z * se, est + z * se


def coverage_study(y, f, n_labeled, reps=2000, alpha=0.05, seed=0):
    """Treat a fully labeled pool as the population; repeatedly reveal n gold labels and compare estimators."""
    rng = np.random.default_rng(seed)
    y, f = np.asarray(y, float), np.asarray(f, float)
    truth = y.mean()
    out = {k: {"cover": [], "width": [], "err": []} for k in ("classical", "naive", "ppi", "ppi++")}
    for _ in range(reps):
        idx = rng.permutation(len(y))
        lab, unl = idx[:n_labeled], idx[n_labeled:]
        res = {"classical": classical(y[lab], alpha), "naive": naive(f, alpha),
               "ppi": ppi_mean(y[lab], f[lab], f[unl], alpha, tuned=False),
               "ppi++": ppi_mean(y[lab], f[lab], f[unl], alpha, tuned=True)}
        for k, (est, lo, hi) in res.items():
            out[k]["cover"].append(lo <= truth <= hi)
            out[k]["width"].append(hi - lo)
            out[k]["err"].append(est - truth)
    return {k: {"coverage": float(np.mean(v["cover"])), "width": float(np.mean(v["width"])),
                "bias": float(np.mean(v["err"])), "rmse": float(np.sqrt(np.mean(np.square(v["err"]))))}
            for k, v in out.items()} | {"truth": float(truth), "n_labeled": n_labeled, "N": len(y)}
