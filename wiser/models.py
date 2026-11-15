"""The scorers. Each gives every pixel of a crop a score: higher means more likely fire tomorrow.

- Persistence: tomorrow = today.
- Distance buffers: minus the distance to the nearest pixel burning today, measured either inside
  the window only (what the learned models see) or on the whole tile.
- Gradient-boosted trees on per-pixel features (scikit-learn, or LightGBM if installed). Not written yet.
- Tiny U-Net (about 470K parameters), also trained with today's fire mask as its only input. Not written yet.

Scores work like logits: a cutoff of 0 is the "probability 0.5" habit. For persistence that flags
exactly the pixels burning today, and so do the buffers (distance <= 0).

The baselines take today's fire on whole tiles [n, 64, 64] plus each tile's window position, and
return scores for the 32 x 32 windows [n, 32, 32].
"""
import numpy as np

from wiser.data import crop, distance_to_fire

NO_FIRE_DISTANCE = 1000.0  # km; stands in for "no fire in sight", far beyond a tile's 90 km diagonal


def windows(tiles, rows, cols):
    """Cut each tile's window: tiles [n, 64, 64] -> [n, 32, 32]."""
    return np.stack([crop(t, r, c) for t, r, c in zip(tiles, rows, cols)])


def persistence(fire_today, rows, cols):
    """Tomorrow = today: +1 where fire burns today, -1 everywhere else."""
    return np.where(windows(fire_today, rows, cols) == 1, 1.0, -1.0).astype(np.float32)


def buffer_window(fire_today, rows, cols):
    """Minus the distance in km to the nearest pixel burning today inside the window.

    A cutoff of -r flags every pixel within r km of today's fire, so the cutoff WISER picks is a radius.
    A window with no fire today gives every pixel the same score.
    """
    distance = np.minimum(distance_to_fire(windows(fire_today, rows, cols)), NO_FIRE_DISTANCE)
    return (-distance).astype(np.float32)


def buffer_tile(fire_today, rows, cols):
    """Like buffer_window, but the distance is measured on the whole tile, so fire outside the window counts."""
    distance = np.minimum(distance_to_fire(fire_today), NO_FIRE_DISTANCE)
    return (-windows(distance, rows, cols)).astype(np.float32)


BASELINES = {"persistence": persistence, "buffer_window": buffer_window, "buffer_tile": buffer_tile}
