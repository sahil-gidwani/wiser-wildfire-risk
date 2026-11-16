"""Train one model and save its scores for every validation, calibration and test crop.

Output: outputs/scores/<model>_seed<k>/ with scores.npy, tile_ids.npy and meta.json
(and model.pt for the learned models).

- Baselines (persistence, buffer_window, buffer_tile) need no training: their scores come
  straight from today's fire.
- Learned models (unet, unet_fire) train on a fresh random 32 x 32 window of each training tile
  every epoch, with no flips. The loss ignores unknown pixels. Training keeps the epoch with the
  best average precision on the validation crops and stops after PATIENCE epochs without
  improvement. The calibration and test crops are never used for training or for choices.

Run: python -m wiser.train --model buffer_tile
     python -m wiser.train --model unet --seed 0 --epochs 3     (a short smoke run)
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from wiser.data import CROP, FIRE_TODAY, INPUTS, TILE, load, windows_of
from wiser.metrics import average_precision
from wiser.models import BASELINES, NETWORKS

SCORES_DIR = Path("outputs") / "scores"
SCORED_ROLES = ("validation", "calibration", "test")
EPOCHS = 50
PATIENCE = 7  # stop after this many epochs without a better validation AP
BATCH = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def main():
    parser = argparse.ArgumentParser(description="Train a model and save its scores.")
    parser.add_argument("--model", required=True, choices=sorted(BASELINES) + sorted(NETWORKS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    start = time.time()
    inputs, labels, tiles = load()
    scored = tiles[tiles.role.isin(SCORED_ROLES)]
    if args.model in BASELINES:
        fire_today = np.stack([inputs[t, FIRE_TODAY] for t in scored.tile_id])  # whole tiles
        scores = BASELINES[args.model](fire_today, scored.crop_row.to_numpy(), scored.crop_col.to_numpy())
        extra, state = {}, None
    else:
        scores, extra, state = train_network(args.model, args.seed, args.epochs, inputs, labels, tiles, scored)
    save(args.model, args.seed, scores, scored.tile_id.to_numpy(), time.time() - start, extra, state)


def train_network(name, seed, epochs, inputs, labels, tiles, scored):
    """Train a TinyUNet; return its scores on the scored crops, a training record, and its weights."""
    import torch
    import torch.nn.functional as F

    from wiser.models import TinyUNet

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    channels = [INPUTS.index(k) for k in NETWORKS[name]]
    model = TinyUNet(len(channels)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    train_ids = tiles[tiles.role == "train"].tile_id.to_numpy().copy()  # a copy we can shuffle
    val = tiles[tiles.role == "validation"]
    x_val, y_val = windows_of(inputs, labels, val.tile_id, val.crop_row, val.crop_col)

    def predict(x):
        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(x), 512):
                out.append(model(torch.from_numpy(np.ascontiguousarray(x[s:s + 512, channels])).to(device)).cpu().numpy())
        return np.concatenate(out)

    best_ap, best_state, best_epoch, waited, history = -1.0, None, 0, 0, []
    for epoch in range(1, epochs + 1):
        tic = time.time()
        model.train()
        rng.shuffle(train_ids)
        total, count = 0.0, 0
        for s in range(0, len(train_ids), BATCH):
            ids = np.sort(train_ids[s:s + BATCH])
            corners = rng.integers(0, TILE - CROP + 1, size=(len(ids), 2))  # a new random window each time
            x, y = windows_of(inputs, labels, ids, corners[:, 0], corners[:, 1])
            x = torch.from_numpy(np.ascontiguousarray(x[:, channels])).to(device)
            y = torch.from_numpy(y.astype(np.float32)).to(device)
            known = (y >= 0).float()
            loss = F.binary_cross_entropy_with_logits(model(x), (y == 1).float(), reduction="none")
            loss = (loss * known).sum() / known.sum().clamp(min=1)  # unknown pixels never count
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(ids)
            count += len(ids)
        schedule.step()
        ap = average_precision(predict(x_val), y_val)
        history.append({"epoch": epoch, "loss": round(total / count, 5), "validation_ap": round(ap, 4), "seconds": round(time.time() - tic, 1)})
        print(f"epoch {epoch:2d}: training loss {total / count:.4f}, validation AP {ap:.4f}, {time.time() - tic:.0f} s")
        if ap > best_ap:
            best_ap, best_epoch, waited = ap, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                print(f"no better validation AP for {PATIENCE} epochs: stopping")
                break

    model.load_state_dict(best_state)
    x_all, _ = windows_of(inputs, labels, scored.tile_id, scored.crop_row, scored.crop_col)
    extra = {"device": device, "parameters": sum(p.numel() for p in model.parameters()),
             "best_epoch": best_epoch, "best_validation_ap": round(best_ap, 4), "history": history}
    return predict(x_all), extra, best_state


def save(model, seed, scores, tile_ids, seconds, extra=None, state=None):
    """Write a model's scores, the tiles they belong to, and a short record of how they were made."""
    out = SCORES_DIR / f"{model}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "scores.npy", np.asarray(scores, dtype=np.float32))
    np.save(out / "tile_ids.npy", np.asarray(tile_ids))
    meta = {"model": model, "seed": seed, "commit": git_commit(), "seconds": round(seconds, 1), **(extra or {})}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if state is not None:
        import torch

        torch.save(state, out / "model.pt")
    print(f"saved scores for {len(tile_ids)} crops to {out}/")


def git_commit():
    """The current git commit, so every result can be traced to the code that made it."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


if __name__ == "__main__":
    main()
