import numpy as np

from lawful.judge.dawid_skene import dawid_skene
from lawful.judge.ppi import coverage_study


def test_ppi_is_valid_and_naive_is_not_under_judge_bias():
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.3
    # a judge that misses 30% of violations and false-alarms 5% of the time: biased low
    f = np.where(y, rng.random(5000) < 0.7, rng.random(5000) < 0.05).astype(float)
    res = coverage_study(y, f, n_labeled=200, reps=400)
    assert res["ppi"]["coverage"] > 0.92 and res["ppi++"]["coverage"] > 0.92
    assert res["naive"]["coverage"] < 0.05
    assert res["ppi++"]["width"] < res["classical"]["width"]


def test_dawid_skene_recovers_judge_quality():
    rng = np.random.default_rng(1)
    z = (rng.random(3000) < 0.4).astype(int)
    acc = [0.9, 0.8, 0.65, 0.55]
    L = np.stack([np.where(rng.random(3000) < a, z, 1 - z) for a in acc], 1)
    T, conf, prior = dawid_skene(L)
    est_acc = [(conf[j, 0, 0] * prior[0] + conf[j, 1, 1] * prior[1]) for j in range(4)]
    assert np.allclose(est_acc, acc, atol=0.04)
    assert ((T[:, 1] > 0.5) == z).mean() > (np.round(L.mean(1)) == z).mean()
