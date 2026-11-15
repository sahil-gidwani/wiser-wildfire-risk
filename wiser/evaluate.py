"""Run the experiments, using saved scores only.

For every model in outputs/scores/, pick cutoffs on the calibration crops with each rule and
measure them on held-out crops: miss rates (per-crop, pooled, capped, new fire), flagged area, AP.

The test crops stay untouched until the settings are written down; until then MEASURE_ON is
"validation". Validity checks, the miss breakdown and the populated-pixel comparison come later.

Output: outputs/results.csv
Run: python -m wiser.evaluate
"""
from pathlib import Path

import numpy as np
import pandas as pd

from wiser.calibrate import FIXED_CUTOFF, pixel_pooled_cutoff, wiser_cutoff
from wiser.data import FIRE_TODAY, load, new_fire, stored_crops
from wiser.metrics import average_precision, flagged_area, miss_rate, misses
from wiser.train import SCORES_DIR

ALPHAS = (0.05, 0.10)
MEASURE_ON = "validation"
RESULTS = Path("outputs") / "results.csv"
RULES = (
    ("WISER per-crop", "per_crop"),
    ("WISER pooled", "pooled"),
    ("WISER capped", "capped"),
    ("pixel-pooled (Dayan 2026)", "pixel"),
)


def main():
    inputs, labels, tiles = load()
    rows = []
    for folder in sorted(p for p in SCORES_DIR.glob("*") if (p / "scores.npy").exists()):
        rows += evaluate_model(folder, inputs, labels, tiles)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS, index=False)
    print(f"cutoffs picked on calibration crops, measured on {MEASURE_ON} crops\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def evaluate_model(folder, inputs, labels, tiles):
    """Every rule at every alpha for one model's saved scores."""
    scores, ids = np.load(folder / "scores.npy"), np.load(folder / "tile_ids.npy")
    row_of = {tile_id: i for i, tile_id in enumerate(ids)}

    def part(role):
        sub = tiles[tiles.role == role]
        s = scores[[row_of[t] for t in sub.tile_id]]
        return s, stored_crops(labels, sub), stored_crops(inputs[:, FIRE_TODAY], sub)

    cal_scores, cal_labels, _ = part("calibration")
    s, y, today = part(MEASURE_ON)
    fire, new = y == 1, new_fire(y, today)
    ap = average_precision(s, y)

    def measure(rule, alpha, t):
        flagged = s >= t
        P, FN = misses(flagged, fire)
        P_new, FN_new = misses(flagged, new)
        return {
            "model": folder.name, "rule": rule, "alpha": alpha, "cutoff": t,
            "missed_per_crop": miss_rate(P, FN, "per_crop"),
            "missed_pooled": miss_rate(P, FN, "pooled"),
            "missed_capped": miss_rate(P, FN, "capped"),
            "missed_new_fire": miss_rate(P_new, FN_new, "pooled"),
            "flagged_area": flagged_area(flagged, y),
            "ap": ap,
        }

    rows = [measure("fixed cutoff 0", np.nan, FIXED_CUTOFF)]
    for alpha in ALPHAS:
        for rule, kind in RULES:
            if kind == "pixel":
                t = pixel_pooled_cutoff(cal_scores, cal_labels == 1, alpha)
            else:
                t = wiser_cutoff(cal_scores, cal_labels == 1, alpha, kind)
            rows.append(measure(rule, alpha, t))
    return rows


if __name__ == "__main__":
    main()
