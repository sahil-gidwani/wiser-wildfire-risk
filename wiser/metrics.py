"""Measure a cutoff on test crops.

- Realised miss rates: per-crop, pooled, capped, populated and new fire.
- Flagged area: flagged valid pixels as a share of all valid pixels.
- Average precision, for how well a model ranks pixels.

Unknown pixels (label -1) never count. See docs/evaluation.md.

Conventions used across the code:
- arrays have shape [n_crops, height, width];
- a pixel is flagged when its score >= the cutoff;
- `positives` is a boolean array marking the fire pixels a miss rate counts:
  all valid fire pixels, only those in populated cells, only new fire, and so on.
"""
import numpy as np
from sklearn.metrics import average_precision_score

CAP = 32  # a crop's weight in the capped miss rate stops growing at this many fire pixels


def crop_weights(P, kind, cap=CAP, n_px=1024):
    """Weight of each crop in a miss rate, given its number of counted fire pixels P.

    per_crop: every crop with fire counts once.
    pooled:   crops count in proportion to their fire pixels (divided by n_px, the pixels per crop).
    capped:   like pooled, but no crop counts more than `cap` fire pixels.
    """
    P = np.asarray(P, dtype=np.float64)
    if kind == "per_crop":
        return (P > 0).astype(np.float64)
    if kind == "pooled":
        return P / n_px
    if kind == "capped":
        return np.minimum(P, cap) / cap
    raise ValueError(f"unknown miss rate: {kind!r}")


def misses(flagged, positives):
    """Per crop: the number of counted fire pixels P and how many of them were missed (FN)."""
    n = len(positives)
    P = positives.reshape(n, -1).sum(axis=1)
    FN = (positives & ~flagged).reshape(n, -1).sum(axis=1)
    return P, FN


def miss_rate(P, FN, kind="capped", cap=CAP):
    """Realised miss rate over a set of crops.

    per_crop: the average over crops with fire of FN/P.
    pooled:   all missed fire pixels / all fire pixels.
    capped:   the pooled rate with each crop's weight capped at `cap` fire pixels.
    """
    P = np.asarray(P)
    w = crop_weights(P, kind, cap)  # n_px cancels out of the ratio, so the default is fine
    if w.sum() == 0:
        return float("nan")
    rho = np.divide(FN, P, out=np.zeros(len(P)), where=P > 0)
    return float((w * rho).sum() / w.sum())


def flagged_area(flagged, labels):
    """Share of valid pixels (label 0 or 1) that are flagged."""
    valid = labels >= 0
    return float((valid & flagged).sum() / valid.sum())


def average_precision(scores, labels):
    """Average precision of the scores over all valid pixels."""
    valid = labels >= 0
    return float(average_precision_score(labels[valid] == 1, scores[valid]))
