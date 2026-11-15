"""Prepare the NDWS data.

1. Download the dataset from Kaggle into data/ (skipped if it is already there; no login needed).
2. Convert the TFRecords once into NumPy arrays in outputs/data/ (skipped if done; delete
   outputs/data/ to redo it):
   - inputs.npy: float32 [N, 12, 64, 64], raw values, channels in the order of INPUTS;
   - labels.npy: int8 [N, 64, 64], tomorrow's fire: 1 fire, 0 no fire, -1 unknown;
   - tiles.csv: one row per tile with its file, split, data role, crop offset and counts.
     Tiles that exactly repeat an earlier tile are marked is_copy and left out by load().
3. Run the week-1 data checks (docs/data.md, section 6), print them and save them
   to outputs/data/checks.txt.

Run: python -m wiser.data
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

DATA_DIR = Path("data")
OUT_DIR = Path("outputs") / "data"
KAGGLE_DATASET = "fantineh/next-day-wildfire-spread"

# Channel order of inputs.npy: the order of INPUT_FEATURES in the NDWS code.
INPUTS = ("elevation", "pdsi", "NDVI", "pr", "sph", "th", "tmmn", "tmmx", "vs", "erc", "population", "PrevFireMask")
POPULATION = INPUTS.index("population")
FIRE_TODAY = INPUTS.index("PrevFireMask")

TILE = 64  # tile side in pixels; 1 pixel = 1 km
CROP = 32  # crop side in pixels
VALIDATION_SHARE = 0.10  # share of the official train split held out for making choices
CHUNK = 50  # validation is carved out in chunks of this many consecutive records
SEED = 0  # for the crop offsets and the validation carve-out
POPULATED = 10.0  # people per km² at which a pixel counts as populated (tau)
ROLES = {"train": "train", "eval": "calibration", "test": "test"}


def main():
    files = download()
    if (OUT_DIR / "tiles.csv").exists():
        print(f"{OUT_DIR}/ already exists, so the conversion is skipped (delete it to redo).")
    else:
        convert(files)
    report = data_checks()
    print(report)
    (OUT_DIR / "checks.txt").write_text(report + "\n")


# ---- Loading the converted data, and helpers the other modules use ----

def load(keep_copies=False):
    """The converted data: inputs and labels as read-only memory maps, plus the tiles table.

    Arrays are indexed by tile_id. Exact repeats of earlier tiles are left out of the table
    unless keep_copies is True.
    """
    inputs = np.load(OUT_DIR / "inputs.npy", mmap_mode="r")
    labels = np.load(OUT_DIR / "labels.npy", mmap_mode="r")
    tiles = pd.read_csv(OUT_DIR / "tiles.csv")
    if not keep_copies:
        tiles = tiles[~tiles.is_copy]
    return inputs, labels, tiles


def crop(a, row, col, size=CROP):
    """The size x size window of a tile (or a stack of layers) whose top-left pixel is (row, col)."""
    return a[..., row:row + size, col:col + size]


def stored_crops(a, tiles):
    """The stored crop of each tile in `tiles`, taken from an array indexed by tile_id."""
    return np.stack([crop(a[t.tile_id], t.crop_row, t.crop_col) for t in tiles.itertuples()])


def new_fire(labels, fire_today):
    """Fire tomorrow in pixels that are not burning today."""
    return (labels == 1) & (fire_today != 1)


def populated(population):
    """Pixels with at least POPULATED people per km²."""
    return population >= POPULATED


def distance_to_fire(fire_today):
    """Distance in pixels (= km) from each pixel to the nearest pixel burning today.

    Takes a stack of crops [n, h, w]. Crops with no fire today get infinity everywhere.
    """
    burning = np.asarray(fire_today) == 1
    out = np.full(burning.shape, np.inf)
    for i in range(len(burning)):
        if burning[i].any():
            out[i] = distance_transform_edt(~burning[i])
    return out


# ---- Download and conversion ----

def download():
    """Download NDWS from Kaggle into data/ unless it is already there. Returns the TFRecord files."""
    files = tfrecord_files()
    if not files:
        os.environ["KAGGLEHUB_CACHE"] = str((DATA_DIR / "kagglehub").resolve())
        import kagglehub  # imported here so that the cache location above applies

        kagglehub.dataset_download(KAGGLE_DATASET)
        files = tfrecord_files()
    if len(files) != 19:
        raise SystemExit(f"expected 19 NDWS TFRecord files under {DATA_DIR}/, found {len(files)}")
    return files


def tfrecord_files():
    """The NDWS files, sorted by name: eval_00-01, test_00-01, train_00-14."""
    return sorted(DATA_DIR.rglob("next_day_wildfire_spread_*.tfrecord"), key=lambda p: p.name)


def count_records(path):
    """Number of records in a TFRecord file, counted from the record headers only."""
    n = 0
    with open(path, "rb") as f:
        while header := f.read(12):  # 8-byte data length + 4-byte checksum
            f.seek(int.from_bytes(header[:8], "little") + 4, os.SEEK_CUR)  # skip data + its checksum
            n += 1
    return n


def convert(files):
    """Write inputs.npy, labels.npy and tiles.csv (see the module docstring)."""
    from tfrecord.reader import tfrecord_loader

    n = sum(count_records(f) for f in files)
    rng = np.random.default_rng(SEED)
    offsets = rng.integers(0, TILE - CROP + 1, size=(n, 2))  # one stored crop per tile
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = np.lib.format.open_memmap(OUT_DIR / "inputs.npy", "w+", np.float32, (n, len(INPUTS), TILE, TILE))
    labels = np.lib.format.open_memmap(OUT_DIR / "labels.npy", "w+", np.int8, (n, TILE, TILE))

    rows = []
    for path in files:
        split = path.stem.split("_")[-2]  # next_day_wildfire_spread_<split>_<nn>
        for record, example in enumerate(tfrecord_loader(str(path), None)):
            i = len(rows)
            x = np.stack([np.asarray(example[k], np.float32).reshape(TILE, TILE) for k in INPUTS])
            y = np.asarray(example["FireMask"], np.float32).reshape(TILE, TILE)
            inputs[i], labels[i] = x, y.astype(np.int8)
            row, col = offsets[i]
            y_crop, today_crop = crop(y, row, col), crop(x[FIRE_TODAY], row, col)
            rows.append({
                "tile_id": i,
                "file": path.name,
                "record": record,
                "split": split,
                "crop_row": row,
                "crop_col": col,
                # counts on the stored crop
                "fire_tomorrow": int((y_crop == 1).sum()),
                "new_fire": int(new_fire(y_crop, today_crop).sum()),
                "fire_today": int((today_crop == 1).sum()),
                "unknown": int((y_crop == -1).sum()),
                "labels_ok": bool(np.isin(y, (-1, 0, 1)).all() and np.isin(x[FIRE_TODAY], (-1, 0, 1)).all()),
                "hash": hashlib.blake2b(x.tobytes() + y.tobytes(), digest_size=8).hexdigest(),
            })
        print(f"converted {path.name} ({len(rows)} tiles so far)")
    inputs.flush()
    labels.flush()

    tiles = pd.DataFrame(rows)
    tiles.insert(4, "role", data_roles(tiles, rng))
    tiles["is_copy"] = tiles["hash"].duplicated(keep="first")  # exact repeat of an earlier tile
    tiles.to_csv(OUT_DIR / "tiles.csv", index=False)


def data_roles(tiles, rng):
    """Train, validation, calibration (official eval) and test (official test).

    Validation is about 10 % of the official train split, taken in chunks of consecutive
    records so that tiles of the same fire mostly stay on one side.
    """
    roles = tiles["split"].map(ROLES).to_numpy(dtype=object)
    train = np.flatnonzero(tiles["split"].to_numpy() == "train")
    chunks = np.array_split(train, int(np.ceil(len(train) / CHUNK)))
    for c in rng.choice(len(chunks), round(VALIDATION_SHARE * len(chunks)), replace=False):
        roles[chunks[c]] = "validation"
    return roles


# ---- Week-1 data checks ----

def data_checks():
    """The checks of docs/data.md, section 6, as printable text."""
    inputs, labels, tiles = load(keep_copies=True)
    y = np.asarray(labels)  # about 76 MB
    today = np.asarray(inputs[:, FIRE_TODAY]).astype(np.int8)
    pop = np.asarray(inputs[:, POPULATION])  # about 300 MB
    roles = ("train", "validation", "calibration", "test")
    lines = []

    def add(name, value):
        lines.append(f"{name:<72} {value}")

    def pct(x):
        return f"{100 * x:.2f} %"

    def precision_recall(truth, guess):
        ok = truth >= 0
        hits = ((guess == 1) & (truth == 1) & ok).sum()
        return f"{pct(hits / ((guess == 1) & ok).sum())} / {pct(hits / ((truth == 1) & ok).sum())}"

    # Sizes, label values and exact repeats
    add("Tiles per official split (train / eval / test)",
        " / ".join(str((tiles.split == s).sum()) for s in ("train", "eval", "test")))
    add("Tiles per data role (train / validation / calibration / test)",
        " / ".join(str((tiles.role == r).sum()) for r in roles))
    add("Tiles with label values other than -1, 0, 1", int((~tiles.labels_ok).sum()))
    repeated = tiles.hash.duplicated(keep=False)
    add("Tiles in groups of exact repeats / groups spanning two splits",
        f"{int(repeated.sum())} in {tiles[repeated].hash.nunique()} groups / "
        f"{int(tiles[repeated].groupby('hash').split.nunique().gt(1).sum())}")
    add("Repeats left out: train / validation / calibration / test",
        " / ".join(str(int(tiles.is_copy[tiles.role == r].sum())) for r in roles))

    # Labels on full tiles
    valid = y >= 0
    fire_tomorrow, fire_today = (y == 1).any(axis=(1, 2)), (today == 1).any(axis=(1, 2))
    add("Fire share of valid pixels (full tiles)", pct((y == 1).sum() / valid.sum()))
    add("Unknown share of all pixels (full tiles)", pct((y == -1).mean()))
    add("Tiles with fire today / tomorrow / both days",
        f"{pct(fire_today.mean())} / {pct(fire_tomorrow.mean())} / {pct((fire_today & fire_tomorrow).mean())}")

    # Is fire centred in the tile?
    c = TILE // 2
    centre = (slice(None), slice(c - 1, c + 1), slice(c - 1, c + 1))
    add("Tiles with fire tomorrow / today in the central 2 x 2 pixels",
        f"{pct((y[centre] == 1).any(axis=(1, 2)).mean())} / {pct((today[centre] == 1).any(axis=(1, 2)).mean())}")
    rate = (y == 1).mean(axis=0)
    add("Fire-tomorrow rate at the central 2 x 2 vs the whole tile",
        f"{pct(rate[c - 1:c + 1, c - 1:c + 1].mean())} vs {pct(rate.mean())}")

    # Stored crops of the validation, calibration and test tiles (repeats left out)
    held = tiles[(tiles.role != "train") & ~tiles.is_copy]
    fire = held.fire_tomorrow
    add("Stored crops (validation + calibration + test)", len(held))
    add("Fire share of valid pixels (stored crops)", pct(fire.sum() / (len(held) * CROP**2 - held.unknown.sum())))
    add("Fire pixels per crop: mean / median", f"{fire.mean():.1f} / {fire.median():.0f}")
    add("Crops with at least 32 fire pixels / with none", f"{pct((fire >= 32).mean())} / {pct((fire == 0).mean())}")
    add("New fire as a share of tomorrow's fire (stored crops)", pct(held.new_fire.sum() / fire.sum()))

    # People near fire
    fire_pop = pop[y == 1]
    add("Fire pixels with population > 0 / > 10 / > 100 per km²",
        " / ".join(pct((fire_pop > v).mean()) for v in (0, 10, 100)))
    cal = tiles[tiles.role == "calibration"]
    with_people = ((stored_crops(y, cal) == 1) & populated(stored_crops(pop, cal))).any(axis=(1, 2))
    add(f"Calibration crops with fire in populated cells (>= {POPULATED:g} per km²)",
        f"{int(with_people.sum())} of {len(cal)}")

    # Persistence (tomorrow = today) on the official test split. The paper scored full tiles,
    # repeats included, so that is the like-for-like test of our metric code.
    test = tiles[tiles.role == "test"]
    ids = test.tile_id.to_numpy()
    add("Persistence precision / recall, full test tiles (paper: 35.7 / 27.3)",
        precision_recall(y[ids], today[ids]))
    test = test[~test.is_copy]
    add("Persistence precision / recall, stored test crops",
        precision_recall(stored_crops(y, test), stored_crops(today, test)))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
