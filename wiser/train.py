"""Train one model and save its scores for every validation, calibration and test crop.

Output: outputs/scores/<model>_seed<k>/ with scores.npy, tile_ids.npy and meta.json
(and model.pt for the learned models).

- Baselines (persistence, buffer_window, buffer_tile) need no training: their scores come
  straight from today's fire.
- The tree model (trees) learns from single pixels: one random window per training tile, every
  fire pixel plus a NEGATIVE_SHARE sample of the others (weighted back up), with scikit-learn's
  gradient-boosted trees. Its score is the log-odds of fire, like a U-Net's logit.
- Learned networks (unet, unet_fire) train on a fresh random 32 x 32 window of each training tile
  every epoch, with no flips. The loss ignores unknown pixels. Training keeps the epoch that needs
  the least land to keep the promise on the validation crops (see land_needed) and stops after
  PATIENCE epochs without improvement. The calibration and test crops are never used for training
  or for choices.

Run: python -m wiser.train --model buffer_tile
     python -m wiser.train --model trees
     python -m wiser.train --model unet --seed 0 --epochs 3     (a short smoke run)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from wiser.data import CROP, FIRE_TODAY, INPUTS, TILE, load, windows_of
from wiser.calibrate import wiser_cutoff
from wiser.metrics import average_precision, flagged_area
from wiser.models import BASELINES, NETWORKS

SCORES_DIR = Path("outputs") / "scores"
SCORED_ROLES = ("validation", "calibration", "test")
EPOCHS = 15  # the learning rate shrinks to zero over this many epochs
PATIENCE = 7  # stop after this many epochs without a better validation AP
BATCH = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
# EPOCHS, LEARNING_RATE and WEIGHT_DECAY were picked by comparing the best validation AP of five
# settings (50 or 15 or 30 epochs; learning rate 1e-3 or 3e-4; weight decay 1e-4 or 1e-2).
NEGATIVE_SHARE = 0.05  # share of non-fire pixels the tree model learns from
SELECT_ALPHA = 0.10  # models are judged by the land WISER needs to keep this promise on validation
TREES, LEAVES = 500, 63  # at most 500 trees of up to 63 leaves; fewer if more trees stop helping


def main():
    parser = argparse.ArgumentParser(description="Train a model and save its scores.")
    parser.add_argument("--model", required=True, choices=sorted(BASELINES) + sorted(NETWORKS) + ["trees"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="starting learning rate")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--tag", default="", help="added to the output folder name, to keep tuning runs apart")
    args = parser.parse_args()

    start = time.time()
    inputs, labels, tiles = load()
    scored = tiles[tiles.role.isin(SCORED_ROLES)]
    if args.model in BASELINES:
        fire_today = np.stack([inputs[t, FIRE_TODAY] for t in scored.tile_id])  # whole tiles
        scores = BASELINES[args.model](fire_today, scored.crop_row.to_numpy(), scored.crop_col.to_numpy())
        extra, state = {}, None
    elif args.model == "trees":
        scores, extra = train_trees(args.seed, inputs, labels, tiles, scored)
        state = None
    else:
        scores, extra, state = train_network(args.model, args.seed, args.epochs, args.lr, args.weight_decay,
                                             inputs, labels, tiles, scored)
    name = f"{args.model}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    save(name, args.model, args.seed, scores, scored.tile_id.to_numpy(), time.time() - start, extra, state)


def land_needed(scores, labels, alpha=SELECT_ALPHA):
    """Share of known pixels WISER (capped) must flag to keep its promise at alpha on these crops.

    This is what the project cares about, so models are chosen on it: lower is better. The cutoff is
    picked and measured on the same validation crops, so no calibration data is used.
    """
    t = wiser_cutoff(scores, labels == 1, alpha, "capped")
    return flagged_area(scores >= t, labels)


def train_trees(seed, inputs, labels, tiles, scored):
    """Train the gradient-boosted trees; return their scores on the scored crops and a short record."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    from wiser.models import TREE_FEATURES, pixel_features

    rng = np.random.default_rng(seed)
    train = tiles[tiles.role == "train"]
    corners = rng.integers(0, TILE - CROP + 1, size=(len(train), 2))
    X, y = [], []
    for s in range(0, len(train), 1000):  # in chunks, to keep memory low
        part, c = train.iloc[s:s + 1000], corners[s:s + 1000]
        x, lab = windows_of(inputs, labels, part.tile_id, c[:, 0], c[:, 1], scaled=False)
        features, lab = pixel_features(x), lab.ravel()
        keep = (lab == 1) | ((lab == 0) & (rng.random(lab.size) < NEGATIVE_SHARE))  # unknown pixels never count
        X.append(features[keep])
        y.append(lab[keep])
    X, y = np.concatenate(X), np.concatenate(y)
    print(f"learning from {len(y):,} pixels ({int(y.sum()):,} fire)", flush=True)

    model = HistGradientBoostingClassifier(max_iter=TREES, max_leaf_nodes=LEAVES, early_stopping=True,
                                           validation_fraction=0.1, n_iter_no_change=20, random_state=seed)
    model.fit(X, y, sample_weight=np.where(y == 1, 1.0, 1 / NEGATIVE_SHARE))

    def predict(sub):
        out = []
        for s in range(0, len(sub), 500):
            part = sub.iloc[s:s + 500]
            x, _ = windows_of(inputs, labels, part.tile_id, part.crop_row, part.crop_col, scaled=False)
            out.append(model.decision_function(pixel_features(x)).reshape(len(part), CROP, CROP))
        return np.concatenate(out)

    val = tiles[tiles.role == "validation"]
    _, y_val = windows_of(inputs, labels, val.tile_id, val.crop_row, val.crop_col, scaled=False)
    s_val = predict(val)
    ap, land = average_precision(s_val, y_val), land_needed(s_val, y_val)
    print(f"{model.n_iter_} trees, validation land needed {land:.4f}, validation AP {ap:.4f}", flush=True)
    extra = {"trees": int(model.n_iter_), "leaves": LEAVES, "negative_share": NEGATIVE_SHARE,
             "training_pixels": int(len(y)), "features": list(TREE_FEATURES),
             "validation_land": round(land, 4), "validation_ap": round(ap, 4)}
    return predict(scored), extra


def train_network(name, seed, epochs, lr, weight_decay, inputs, labels, tiles, scored):
    """Train a TinyUNet; return its scores on the scored crops, a training record, and its weights."""
    import torch
    import torch.nn.functional as F

    from wiser.models import TinyUNet

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    channels = [INPUTS.index(k) for k in NETWORKS[name]]
    model = TinyUNet(len(channels)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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

    best_land, best_ap, best_state, best_epoch, waited, history = 2.0, 0.0, None, 0, 0, []
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
        s_val = predict(x_val)
        ap, land = average_precision(s_val, y_val), land_needed(s_val, y_val)
        history.append({"epoch": epoch, "loss": round(total / count, 5), "validation_land": round(land, 4),
                        "validation_ap": round(ap, 4), "seconds": round(time.time() - tic, 1)})
        print(f"epoch {epoch:2d}: training loss {total / count:.4f}, validation land needed {land:.4f}, "
              f"validation AP {ap:.4f}, {time.time() - tic:.0f} s", flush=True)
        if land < best_land:
            best_land, best_ap, best_epoch, waited = land, ap, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                print(f"no less validation land needed for {PATIENCE} epochs: stopping")
                break

    model.load_state_dict(best_state)
    x_all, _ = windows_of(inputs, labels, scored.tile_id, scored.crop_row, scored.crop_col)
    extra = {"device": device, "parameters": sum(p.numel() for p in model.parameters()),
             "epochs_max": epochs, "lr": lr, "weight_decay": weight_decay,
             "best_epoch": best_epoch, "best_validation_land": round(best_land, 4),
             "best_validation_ap": round(best_ap, 4), "history": history}
    return predict(x_all), extra, best_state


def save(name, model, seed, scores, tile_ids, seconds, extra=None, state=None):
    """Write a model's scores, the tiles they belong to, and a short record of how they were made."""
    out = SCORES_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "scores.npy", np.asarray(scores, dtype=np.float32))
    np.save(out / "tile_ids.npy", np.asarray(tile_ids))
    meta = {"model": model, "seed": seed, "seconds": round(seconds, 1), **(extra or {})}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if state is not None:
        import torch

        torch.save(state, out / "model.pt")
    print(f"saved scores for {len(tile_ids)} crops to {out}/")


if __name__ == "__main__":
    main()
