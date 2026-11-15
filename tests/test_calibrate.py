"""Checks for the calibration code on synthetic crops (docs/evaluation.md, "Validity checks").

Run: python -m pytest
"""
import numpy as np
import pytest

from wiser.calibrate import flag_two_cutoffs, pixel_pooled_cutoff, two_cutoffs, wiser_cutoff
from wiser.metrics import crop_weights, misses

KINDS = ["per_crop", "pooled", "capped"]
CAP = 8  # the synthetic crops are 8 x 8, so a smaller cap than the real 32


def synthetic_crops(n, size=8, seed=0):
    """Fake crops that behave roughly like NDWS ones.

    - Fire sizes are heavy-tailed: mostly small fires, a few big ones, about 6 % without fire.
    - Fire pixels score higher on average.
    - Each crop has its own random offset, so pixels inside a crop are correlated.
    """
    rng = np.random.default_rng(seed)
    fire = np.zeros((n, size, size), dtype=bool)
    for i in range(n):
        k = 0 if rng.random() < 0.06 else min(int(rng.zipf(1.8)), size * size)
        fire[i].flat[rng.choice(size * size, k, replace=False)] = True
    crop_offset = rng.normal(size=(n, 1, 1))
    scores = crop_offset + rng.normal(size=fire.shape) + 1.5 * fire
    return scores.astype(np.float32), fire


def brute_force_cutoff(scores, positives, alpha, kind):
    """Try every possible cutoff and keep the largest one that satisfies the WISER condition."""
    P, _ = misses(np.ones_like(positives), positives)
    w = crop_weights(P, kind, CAP, n_px=positives[0].size)
    budget = alpha * w.sum() - (1 - alpha)
    if budget < 0:
        return -np.inf
    best = -np.inf
    for t in np.concatenate([np.unique(scores[positives]).astype(np.float64), [np.inf]]):
        _, FN = misses(scores >= t, positives)
        rho = np.divide(FN, P, out=np.zeros(len(P)), where=P > 0)
        if (w * rho).sum() <= budget * (1 - 1e-12):
            best = max(best, t)
    return best


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_exact_solver_matches_brute_force(kind, alpha):
    for seed in range(5):
        scores, fire = synthetic_crops(60, seed=seed)
        assert wiser_cutoff(scores, fire, alpha, kind, cap=CAP) == brute_force_cutoff(scores, fire, alpha, kind)


def test_not_enough_data_flags_everything():
    scores, fire = synthetic_crops(10)  # per-crop at alpha = 0.05 needs at least 19 crops with fire
    assert wiser_cutoff(scores, fire, 0.05, "per_crop") == -np.inf


@pytest.mark.parametrize("kind", KINDS)
def test_promise_holds_on_average(kind):
    """Validity check 1: over random calibration/test splits, the promise holds on average.

    The promise is E[w * (rho - alpha)] <= 0 for a new crop, so we average that quantity
    over the test crops of each split and over many splits.
    """
    alpha, n_cal, reps = 0.1, 300, 2000
    scores, fire = synthetic_crops(600, seed=1)
    rng = np.random.default_rng(2)
    values = []
    for _ in range(reps):
        order = rng.permutation(len(scores))
        cal, test = order[:n_cal], order[n_cal:]
        t = wiser_cutoff(scores[cal], fire[cal], alpha, kind, cap=CAP)
        P, FN = misses(scores[test] >= t, fire[test])
        w = crop_weights(P, kind, CAP, n_px=fire[0].size)
        rho = np.divide(FN, P, out=np.zeros(len(P)), where=P > 0)
        values.append(np.mean(w * (rho - alpha)))
    values = np.array(values)
    assert np.isfinite(values).all()
    assert values.mean() <= 3 * values.std(ddof=1) / np.sqrt(reps)


def test_pixel_pooled_rule_matches_its_definition():
    scores, fire = synthetic_crops(50)
    alpha, m = 0.1, int(fire.sum())
    t = pixel_pooled_cutoff(scores, fire, alpha)
    assert (fire & (scores < t)).sum() == int(np.ceil(alpha * (m + 1))) - 1


def test_two_cutoffs_use_the_right_cutoff_in_each_part():
    scores, fire = synthetic_crops(300, seed=3)
    populated = np.zeros_like(fire)
    populated[:, :, :3] = True  # pretend the left three columns of every crop are populated
    t_pop, t_rest = two_cutoffs(scores, fire, populated, alpha=0.1, alpha_pop=0.05, cap=CAP)
    flagged = flag_two_cutoffs(scores, populated, t_pop, t_rest)
    assert (flagged[populated] == (scores[populated] >= t_pop)).all()
    assert (flagged[~populated] == (scores[~populated] >= t_rest)).all()
