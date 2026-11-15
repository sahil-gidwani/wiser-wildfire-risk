"""Train one model and save its scores for every validation, calibration and test crop.

Output: outputs/scores/<model>_seed<k>/ with scores.npy, tile_ids.npy and meta.json.
Early stopping uses the validation set only, never the calibration set.
The baselines (persistence, buffer_window, buffer_tile) need no training: their scores come
straight from today's fire.

Run: python -m wiser.train --model buffer_tile
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from wiser.data import FIRE_TODAY, load
from wiser.models import BASELINES

SCORES_DIR = Path("outputs") / "scores"
SCORED_ROLES = ("validation", "calibration", "test")


def main():
    parser = argparse.ArgumentParser(description="Train a model and save its scores.")
    parser.add_argument("--model", required=True, choices=sorted(BASELINES))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    start = time.time()
    inputs, _, tiles = load()
    tiles = tiles[tiles.role.isin(SCORED_ROLES)]
    fire_today = np.stack([inputs[t, FIRE_TODAY] for t in tiles.tile_id])  # whole tiles
    scores = BASELINES[args.model](fire_today, tiles.crop_row.to_numpy(), tiles.crop_col.to_numpy())
    save(args.model, args.seed, scores, tiles.tile_id.to_numpy(), time.time() - start)


def save(model, seed, scores, tile_ids, seconds):
    """Write a model's scores, the tiles they belong to, and a short record of how they were made."""
    out = SCORES_DIR / f"{model}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "scores.npy", np.asarray(scores, dtype=np.float32))
    np.save(out / "tile_ids.npy", np.asarray(tile_ids))
    meta = {"model": model, "seed": seed, "commit": git_commit(), "seconds": round(seconds, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved scores for {len(tile_ids)} crops to {out}/")


def git_commit():
    """The current git commit, so every result can be traced to the code that made it."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


if __name__ == "__main__":
    main()
