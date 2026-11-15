# Glossary

## Terms

| Term | Meaning here |
|---|---|
| **Active-fire detection** | A satellite flag that a 1 km pixel was burning at the time of the overpass. NDWS uses MODIS Terra (MOD14A1). It is not a perimeter or a burned area. |
| **Anchor paper** | Dayan 2026 (arXiv:2603.22331), the CRC-on-NDWS preprint the proposal built on ([proposal-review](proposal-review.md)) |
| **AP (average precision)** | The area under the precision–recall curve. Our main measure of how well a model ranks fire pixels. |
| **AUROC** | The area under the ROC curve. It looks good for almost any model when 99 % of pixels are non-fire, so we use it only to compare with the anchor paper. |
| **Calibration set** | Crops used only to choose the cutoff: the official NDWS eval split |
| **Capped miss rate** | A pooled miss rate in which a crop with more than `c` fire pixels counts as if it had `c`. WISER's main target. |
| **Conformal risk control (CRC)** | A method (Angelopoulos et al. 2024) that picks a cutoff from calibration data so that the expected loss on a new example is at most α |
| **Crop** | A 32 × 32 window of a tile. Our unit of analysis. |
| **Cutoff (threshold)** | Pixels that score at or above it are flagged |
| **Distance buffer** | A no-learning baseline: flag everything within r km of today's fire, with r chosen by the WISER rule |
| **Exchangeable** | Calibration and future examples are alike, in the sense that shuffling their order changes nothing. The one assumption CRC needs. |
| **Exposure** | Here: population density from the NDWS channel (GPWv4, 1 km). Not people at risk. |
| **Flagged area** | The share of valid pixels that are flagged. The cost of a cutoff. |
| **Miss rate** | The share of tomorrow's fire pixels left unflagged. The per-crop, pooled, capped and populated versions are defined in [method](method.md). |
| **New fire** | Fire tomorrow in a pixel that is not burning today |
| **NDWS** | The Next Day Wildfire Spread dataset (Huot et al. 2022) |
| **Official split** | NDWS's train/eval/test split by 8-day windows |
| **Persistence** | A baseline that predicts tomorrow's fire = today's fire |
| **Populated pixel** | A pixel with population density ≥ τ |
| **Realised miss rate** | The miss rate actually measured on test crops |
| **RCPS** | A variant of CRC whose guarantee holds with probability ≥ 1 − δ over the calibration set |
| **Tile** | One NDWS sample: 64 × 64 pixels at 1 km |
| **Unknown pixel** | A pixel labelled −1 (cloud, or not processed). It is never counted. |
| **Validation set** | Crops carved from the training data, used for every choice |
| **Window** | The 8-day block used by the official NDWS split: 7 sampled days plus 1 buffer day |
| **WISER rule** | Our calibration rule ([method §4](method.md)) |

## Symbols

| Symbol | Meaning |
|---|---|
| `s` | A pixel's model score (higher = more likely fire) |
| `t`, `t̂` | A cutoff; the calibrated cutoff |
| `α` | The target miss rate, e.g. 0.10 |
| `α_pop` | The stricter target for populated pixels |
| `P_i` | Tomorrow's fire pixels in crop `i` (valid pixels only) |
| `FN_i(t)` | Fire pixels in crop `i` left unflagged at cutoff `t` |
| `ρ_i(t)` | The miss fraction of crop `i` |
| `ω_i` | The weight of crop `i` in the WISER rule (0 to 1) |
| `n_eff` | The total weight `Σ ω_i`, which must be at least (1 − α)/α |
| `c` | The cap on fire pixels per crop (start: 32) |
| `τ` | The population density at which a pixel counts as "populated" |
