"""Choose the cutoff.

- The WISER rule for the per-crop, pooled, capped and populated miss rates,
  solved exactly.
- The comparison rules: a fixed 0.5 cutoff and the pixel-pooled rule of Dayan 2026.

Works only on saved scores, in float64 on the CPU. A pixel is flagged when its score >= the cutoff.
"""
import numpy as np

from wiser.metrics import CAP, crop_weights

FIXED_CUTOFF = 0.0  # scores are logits, and a logit of 0 is a probability of 0.5


def wiser_cutoff(scores, positives, alpha, kind="capped", cap=CAP):
    """The WISER rule on a set of calibration crops.

    Returns the largest cutoff t with
        sum_i w_i * (rho_i(t) - alpha) + (1 - alpha) <= 0,
    where w_i is crop i's weight and rho_i(t) its share of fire missed. The extra (1 - alpha)
    is room for one worst-case crop. Returns -inf (flag everything) if there is not enough data.
    """
    n = len(scores)
    P = positives.reshape(n, -1).sum(axis=1)
    w = crop_weights(P, kind, cap, n_px=positives[0].size)
    cost = np.divide(w, P, out=np.zeros(n), where=P > 0)  # cost of missing one fire pixel of crop i
    # scores[positives] lists fire pixels crop by crop, so repeating each crop's cost P_i times lines up.
    return exact_cutoff(scores[positives], np.repeat(cost, P), w.sum(), alpha)


def exact_cutoff(fire_scores, fire_costs, n_eff, alpha):
    """Largest t such that the fire pixels with score < t cost at most alpha * n_eff - (1 - alpha)."""
    budget = alpha * n_eff - (1.0 - alpha)
    if budget < 0:
        return -np.inf  # not enough calibration data: flag everything (useless, but still valid)
    order = np.argsort(fire_scores, kind="stable")
    s = np.asarray(fire_scores, dtype=np.float64)[order]
    cum = np.cumsum(np.asarray(fire_costs, dtype=np.float64)[order])
    k = int(np.searchsorted(cum, budget * (1 - 1e-12), side="right"))  # we can afford to miss the k lowest
    return np.inf if k == len(s) else float(s[k])


def pixel_pooled_cutoff(scores, positives, alpha):
    """The rule of Dayan 2026: a quantile of all calibration fire pixels pooled together.

    It misses ceil(alpha * (m + 1)) - 1 of the m fire pixels. It treats pixels as independent,
    so it carries no real guarantee. Kept only as a comparison.
    """
    s = np.sort(np.asarray(scores[positives], dtype=np.float64))
    k = min(int(np.ceil(alpha * (len(s) + 1))), len(s))
    return float(s[k - 1])


def two_cutoffs(scores, fire, populated, alpha, alpha_pop, cap=CAP):
    """Exposure-aware version.

    Populated pixels get their own cutoff for a per-crop miss rate of at most alpha_pop.
    All other pixels use the capped rule at alpha.
    """
    t_pop = wiser_cutoff(scores, fire & populated, alpha_pop, kind="per_crop")
    t_rest = wiser_cutoff(scores, fire & ~populated, alpha, kind="capped", cap=cap)
    return t_pop, t_rest


def flag_two_cutoffs(scores, populated, t_pop, t_rest):
    """Flagged pixels under the two-cutoff version."""
    return np.where(populated, scores >= t_pop, scores >= t_rest)
