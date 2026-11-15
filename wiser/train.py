"""Train one model and save its scores for every validation, calibration and test crop.

Output: outputs/scores/<model>_seed<k>/ with scores.npy, tile_ids.npy and meta.json.
Early stopping uses the validation set only, never the calibration set.

Run: python -m wiser.train --model unet --seed 0
"""
