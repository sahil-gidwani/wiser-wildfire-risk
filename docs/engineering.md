# Engineering

No code exists yet; this is the plan.

## Layout

```
wiser-wildfire-risk/
├── docs/
├── src/wiser/
│   ├── data/        # TFRecord conversion, crops, derived features
│   ├── models/      # baselines, trees, U-Nets
│   ├── calibrate/   # the WISER rule and the comparison rules
│   ├── evaluate/    # metrics, validity checks, results table
│   └── demo/        # gallery builder
├── scripts/         # one entry point per step: convert, check, train, score, calibrate, evaluate, demo
├── tests/           # calibration tests on synthetic data, metric tests
├── data/            # raw download (not in git)
├── outputs/         # converted data, scores, results (not in git)
└── pyproject.toml, uv.lock
```

## Files passed between steps

| File | Made by | Contents |
|---|---|---|
| Converted data (`outputs/data/`) | Data and baselines | `inputs.npy`: float32 `[N, 12, 64, 64]`<br>`labels.npy`: int8 `[N, 64, 64]`<br>`tiles.parquet`: one row per tile, with split, role, crop offset and counts |
| Scores (`outputs/scores/<model>/<run>/`) | Whoever trains the model | `scores.npy`: float32 logits for every validation, calibration and test crop<br>`tile_ids.npy`<br>`meta.json`: model, seed, git commit, training time |
| Results (`outputs/results.csv`) | Calibration and evaluation | One row per (run, rule, α, check, metric), with the value and its interval |

Rules:
- Calibration and metrics are computed only from saved scores, never inside training code.
- They run in NumPy, in float64, on the CPU.
- Figures and the demo read only these files.

## Environment

- **Python:** 3.12, managed with `uv` and a lockfile.
- **Core libraries:** PyTorch (MPS on Apple Silicon, CPU fallback), NumPy, SciPy, scikit-learn, pandas with pyarrow, matplotlib.
- **Extras:**
  - LightGBM (optional);
  - `tfrecord`, for the one-time conversion;
  - MAPIE, for cross-checking the calibration code.

## Pitfalls

| Pitfall | How to avoid it |
|---|---|
| PyTorch and LightGBM in one process clash over OpenMP on macOS | Run them in separate processes, or use scikit-learn's `HistGradientBoostingClassifier` |
| MPS has no float64, and some operations fall back to the CPU | Train in float32; compute metrics on the CPU in float64 |
| DataLoader workers hang on macOS | Use `num_workers=0` with memory-mapped arrays |
| TFRecord parsing errors | Check GZIP compression and the 64 × 64 feature spec on one record first |
| Per-pixel tables for the trees get large (about 2.9 GB) | Train on a 5–10 % sample of pixels; score in chunks |
| Disk space | Budget for the raw data (about 4 GB), the converted data (about 3.7 GB) and the scores |

## Reproducibility

- Keep seeds in config files and in `meta.json`.
- Provide one command per step, documented in the root README once the code exists.
- Before submitting, run everything from a fresh clone.

## Licences

| Material | Licence | What we may do |
|---|---|---|
| This repository | GPL-3.0 | — |
| NDWS data | CC BY 4.0 | Use it; cite Huot et al. 2022 |
| NDWS code | Apache-2.0 | Adapt snippets, with attribution |
| The anchor paper's repository | None | Read it only; don't copy its code |
| MAPIE | BSD-3-Clause | Use it as a dependency |

Before the first download, add `data/`, `outputs/` and `.DS_Store` to `.gitignore`.
