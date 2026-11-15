"""Checks for the small data helpers. Run: python -m pytest"""
import numpy as np

from wiser.data import count_records, crop, distance_to_fire


def test_crop_takes_the_right_window():
    tile = np.arange(64 * 64).reshape(64, 64)
    window = crop(tile, 5, 7)
    assert window.shape == (32, 32)
    assert window[0, 0] == tile[5, 7] and window[-1, -1] == tile[36, 38]


def test_distance_to_fire():
    today = np.zeros((2, 5, 5))
    today[0, 2, 2] = 1  # one burning pixel in the middle of the first crop; none in the second
    dist = distance_to_fire(today)
    assert dist[0, 2, 2] == 0 and dist[0, 2, 4] == 2
    assert np.isclose(dist[0, 0, 0], np.sqrt(8))
    assert np.isinf(dist[1]).all()


def test_count_records(tmp_path):
    from tfrecord import TFRecordWriter

    path = tmp_path / "three.tfrecord"
    writer = TFRecordWriter(str(path))
    for value in (1.0, 2.0, 3.0):
        writer.write({"x": ([value] * 10, "float")})
    writer.close()
    assert count_records(path) == 3
