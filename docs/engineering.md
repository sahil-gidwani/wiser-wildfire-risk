# Engineering

The code is meant to stay small: one folder, one module per job, four commands.

No code has been written yet. Each module in `wiser/` holds only a short description of what goes in it.

## Layout

```
wiser-wildfire-risk/
├── README.md          # what the project is, and how to set it up
├── requirements.txt   # Python packages
├── docs/              # plan, method, data notes, proposal.pdf
├── wiser/             # all the code, one module per job
│   ├── data.py        # convert NDWS, crops, derived features, data checks
│   ├── models.py      # persistence, distance buffer, trees, U-Net
│   ├── train.py       # train a model and save its scores
│   ├── calibrate.py   # the WISER rule and the comparison rules
│   ├── metrics.py     # miss rates, flagged area, AP
│   ├── evaluate.py    # the experiments: checks, results table, figures
│   └── demo.py        # the HTML gallery
├── tests/             # tests, run with: python -m pytest
├── data/              # raw NDWS download (not in git)
└── outputs/           # converted data, scores, results, figures, demo (not in git)
```

## Running it

Run these from the repository root, in order:

| Step | Command | Writes |
|---|---|---|
| 1. Prepare the data | `python -m wiser.data` | `outputs/data/` |
| 2. Train a model | `python -m wiser.train --model unet --seed 0` | `outputs/scores/unet_seed0/` |
| 3. Run the experiments | `python -m wiser.evaluate` | `outputs/results.csv`, `outputs/figures/` |
| 4. Build the demo | `python -m wiser.demo` | `outputs/demo/index.html` |

- Step 2 runs once for each model and seed.
- Steps 3 and 4 only read saved files, so they are quick to re-run.
- Run the tests with `python -m pytest`.

## Keeping the code simple

- **One module per job.** Add a new file only when an existing one gets too long to read comfortably.
- **Plain functions** that take and return NumPy arrays. Use classes only where PyTorch needs them, i.e. the U-Net.
- **Settings are constants** at the top of the module that uses them. Only `train` takes command-line arguments (`--model`, `--seed`).
- **No frameworks beyond PyTorch:** no config systems and no experiment trackers.

## Files passed between steps

| File | Written by | Contents |
|---|---|---|
| `outputs/data/` | `wiser.data` | `inputs.npy`: float32 `[N, 12, 64, 64]`<br>`labels.npy`: int8 `[N, 64, 64]`<br>`tiles.csv`: one row per tile, with split, role, crop offset and counts |
| `outputs/scores/<model>_seed<k>/` | `wiser.train` | `scores.npy`: float32 logits for every validation, calibration and test crop<br>`tile_ids.npy`<br>`meta.json`: model, seed, git commit, training time |
| `outputs/results.csv` | `wiser.evaluate` | One row per (model, rule, α, check, metric), with the value and its interval |

Rules for these files:
- Calibration and metrics are computed only from saved scores, never inside training code.
- That computation runs in NumPy, in float64, on the CPU.
- Figures and the demo read only these files.

## Environment

The setup commands are in the root [README](../README.md).

- **Python 3.12.**
- **Versions:** `requirements.txt` lists minimum versions. Once everything installs on one laptop, pin the exact versions with `pip freeze`, so all three laptops match.
- **LightGBM** is optional and commented out in `requirements.txt`. scikit-learn's `HistGradientBoostingClassifier` does the same job.

## Pitfalls

| Pitfall | How to avoid it |
|---|---|
| PyTorch and LightGBM clash over OpenMP when both are imported in one process on macOS | Run them in separate processes, or use scikit-learn's `HistGradientBoostingClassifier` |
| MPS has no float64, and some operations fall back to the CPU | Train in float32; compute metrics on the CPU in float64 |
| DataLoader workers hang on macOS | Use `num_workers=0` with memory-mapped arrays |
| TFRecord parsing errors | Check the GZIP compression and the 64 × 64 feature spec on one record first |
| Per-pixel tables for the trees get large (about 2.9 GB) | Train on a 5–10 % sample of pixels, and score in chunks |
| Disk space | Allow for the raw data (about 4 GB), the converted data (about 3.7 GB) and the scores |

## Reproducibility

- Seeds are command-line arguments, and they are saved in `meta.json`.
- The four commands above rebuild everything.
- Before submitting, run them from a fresh clone.

## Licences

| Material | Licence | What we may do |
|---|---|---|
| This repository | GPL-3.0 | — |
| NDWS data | CC BY 4.0 | Use it; cite Huot et al. 2022 |
| NDWS code | Apache-2.0 | Adapt snippets, with attribution |
| The anchor paper's repository | None | Read it only; don't copy its code |
| MAPIE | BSD-3-Clause | Use it as a dependency |
