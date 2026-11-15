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
