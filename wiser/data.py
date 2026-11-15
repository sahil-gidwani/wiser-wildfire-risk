"""Prepare the NDWS data.

- Convert the Kaggle TFRecords once into NumPy arrays in outputs/data/:
  inputs.npy, labels.npy and tiles.csv (one row per tile).
- Assign the data roles: train, validation, calibration (official eval split) and test.
- Crops: a fresh random 32 x 32 crop each time for training; one fixed crop per tile otherwise.
- Derived features: distance to today's fire, new-fire mask, populated mask.
- Print the week-1 data checks (docs/data.md, section 6).

Run: python -m wiser.data
"""
