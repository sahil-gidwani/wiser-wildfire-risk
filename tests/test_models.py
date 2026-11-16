"""Checks for the baseline scorers. Run: python -m pytest"""
import numpy as np

from wiser.calibrate import FIXED_CUTOFF
from wiser.models import NO_FIRE_DISTANCE, buffer_tile, buffer_window, persistence

ROWS, COLS = np.array([0]), np.array([0])  # window = top-left 32 x 32 of the tile


def tile_with_fire_at(r, c):
    today = np.zeros((1, 64, 64))
    today[0, r, c] = 1
    return today


def test_persistence_flags_todays_fire_at_the_fixed_cutoff():
    today = tile_with_fire_at(5, 5)
    today[0, 6, 6] = -1  # unknown today counts as not burning
    flagged = persistence(today, ROWS, COLS) >= FIXED_CUTOFF
    assert flagged.shape == (1, 32, 32)
    assert flagged[0, 5, 5] and flagged.sum() == 1


def test_buffer_window_scores_are_minus_the_distance():
    scores = buffer_window(tile_with_fire_at(10, 10), ROWS, COLS)
    assert scores[0, 10, 10] == 0 and scores[0, 10, 12] == -2
    assert ((scores[0] >= -1) == (np.hypot(*(np.indices((32, 32)) - 10)) <= 1)).all()  # cutoff -1 = within 1 km


def test_buffers_differ_when_todays_fire_is_outside_the_window():
    today = tile_with_fire_at(40, 10)  # burning below the window, 9 km past its bottom edge
    assert (buffer_window(today, ROWS, COLS) == -NO_FIRE_DISTANCE).all()  # the window sees nothing
    tile_scores = buffer_tile(today, ROWS, COLS)
    assert tile_scores[0, 31, 10] == -9 and tile_scores.max() == -9


def test_tiny_unet_gives_one_score_per_pixel():
    import torch

    from wiser.models import TinyUNet

    model = TinyUNet(12)
    assert model(torch.zeros(2, 12, 32, 32)).shape == (2, 32, 32)
    assert 450_000 < sum(p.numel() for p in model.parameters()) < 490_000  # "about 470K parameters"


def test_pixel_features_describe_todays_fire_nearby():
    from wiser.data import FIRE_TODAY, INPUTS
    from wiser.models import TREE_FEATURES, pixel_features

    x = np.zeros((1, len(INPUTS), 32, 32), dtype=np.float32)
    x[0, FIRE_TODAY, 10, 10] = 1  # one pixel burning today
    x[0, 0] = 500.0  # elevation everywhere
    f = pixel_features(x).reshape(32, 32, len(TREE_FEATURES))
    assert f.shape[-1] == 15 and (f[..., 0] == 500).all()
    assert f[10, 12, 12] == 2  # distance to fire
    assert f[10, 11, 13] == 1 and f[10, 12, 13] == 0  # burning within 3 x 3
    assert f[10, 13, 14] == 1 and f[10, 14, 14] == 0  # burning within 7 x 7
