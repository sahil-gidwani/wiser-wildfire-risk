# Method

The pipeline has three steps:
1. score each pixel;
2. choose a cutoff on calibration data with the WISER rule;
3. measure misses and flagged area on test data.

## 1. Task

| | |
|---|---|
| Input | 12 day-t layers on a 32 × 32 km crop of an NDWS tile ([data](data.md)) |
| Output | A score `s` for each 1 km pixel. Higher means more likely to be a fire detection tomorrow. |
| Decision | Flag a pixel if `s ≥ t`. Lowering the cutoff `t` only ever adds pixels, and this nesting is what makes calibration work. |
| Truth | Tomorrow's MODIS fire mask. Unknown pixels (−1) are skipped. |

Scores don't need to be probabilities, because calibration only uses their ranking. A model that ranks fire pixels higher can keep a higher cutoff for the same miss rate, so it flags less area. That is why model quality shows up as **flagged area**.

## 2. Counting misses

**The crop is the unit.** Pixels in the same crop are strongly correlated, so they can't be treated as separate examples. That was the anchor paper's main error ([proposal-review](proposal-review.md)).

For crop `i`, `P_i` is its number of fire pixels tomorrow, and `FN_i(t)` is how many of them cutoff `t` leaves unflagged. Misses can be averaged in different ways, and the choice changes the map:

| Miss rate | How it averages | Role |
|---|---|---|
| **Per-crop** | The average over crops of `FN_i / P_i`. A 2-pixel fire counts as much as a 100-pixel one. | Baseline: standard CRC |
| **Pooled** | All missed fire pixels ÷ all fire pixels. Big fires dominate. | The anchor paper's target. We run the valid version as a comparison. |
| **Capped** | Pooled, but a crop with more than `c` fire pixels counts as if it had `c` (start with `c` = 32). | **Main WISER rule** |
| **Populated** | Per-crop, counting only fire pixels in populated cells (population ≥ τ) | Exposure-aware version (§7) |

**Example.** Crop A has 2 fire pixels and misses 1 (50 %). Crop B has 100 and misses 5 (5 %).

| Miss rate | Calculation | Result |
|---|---|---|
| Per-crop | (50 % + 5 %) / 2 | 27.5 % |
| Pooled | 6 / 102 | 5.9 % |
| Capped at 32 | (1 + 32 × 5 %) / (2 + 32) | 7.6 % |

We also report the **new-fire miss rate**: misses among fire pixels that were not burning today. Persistence catches continuing fire; new fire is the hard part.

## 3. The promise and its limits

> If the calibration crops and a new crop are exchangeable, the **expected** miss rate on the new crop is at most α.

It does **not** say:
1. that every crop misses at most α. It is an average, and single crops can miss everything;
2. that the cutoff from our one calibration set hits α. About half of calibration sets land slightly above; what is controlled is the average over calibration sets;
3. anything about fire that the satellite did not detect;
4. anything under change: other years, regions, sensors or resolutions;
5. anything about places with no fire today, because NDWS contains only ongoing fires;
6. anything about people's safety.

**Assumptions:**
1. Crops are exchangeable. This is exactly true for random re-splits, and only roughly true across the official time windows.
2. The truth is MODIS detections, with unknown pixels skipped.
3. Crops come from NDWS's sampling: fire on both days, centred on tomorrow's fire.
4. Every setting (`c`, τ, α, α_pop) is fixed before calibration, and the calibration data is used for nothing else.
5. Flagging more never increases misses.

## 4. The WISER rule

Each calibration crop `i` gets a weight `ω_i` between 0 and 1 and a miss fraction `ρ_i(t)`:

| Miss rate | Weight `ω_i` | Miss fraction `ρ_i(t)` | Cost of missing one fire pixel |
|---|---|---|---|
| Per-crop | 1 if the crop has fire, else 0 | `FN_i / P_i` | `1 / P_i` |
| Pooled | `P_i / 1024` | `FN_i / P_i` | `1 / 1024` |
| Capped | `min(P_i, c) / c` | `FN_i / P_i` | `min(P_i, c) / (c · P_i)` |
| Populated | 1 if the crop has populated fire, else 0 | Missed populated fire pixels ÷ populated fire pixels | `1 / (populated fire pixels in the crop)` |

**Rule.** Choose the largest cutoff `t` with

```
Σ_i ω_i · (ρ_i(t) − α)  +  (1 − α)  ≤  0
```

In words: the weighted misses on the calibration crops, plus a margin of one worst-case crop, must stay within α times the total weight. The largest such cutoff flags the least area. If no cutoff qualifies, flag everything; that is useless, but still honest.

**Guarantee.** For a new crop, `E[ω · ρ(t̂)] ≤ α · E[ω]`.
- For the per-crop and populated rates this means "the expected miss rate is at most α".
- For the pooled and capped rates it means "the pooled (or capped) miss rate is at most α".

**Why it holds.** It is Theorem 1 of the CRC paper (Angelopoulos et al. 2024), applied to the loss `ω_i(ρ_i(t) − α)`. That loss is bounded by `1 − α` and can only fall as more is flagged.

So this is a **direct corollary**, not a new theorem. Our contribution is:
- putting several miss rates under one rule;
- the exact solver;
- the feasibility check;
- the evaluation.

**Enough data?** A useful cutoff needs a total weight `n_eff = Σ ω_i ≥ (1 − α) / α`. That is at least 19 at α = 5 % and 9 at α = 10 %. The safety margin then uses up `(1 − α) / n_eff` of the miss budget.

| Miss rate | `n_eff` on about 1,800 calibration crops | Consequence |
|---|---|---|
| Per-crop | About 1,800 | The margin is tiny. |
| Pooled | About 55 (*estimate*) | At α = 5 %, calibration misses must stay below about 3.3 %. Valid but expensive, because the rule must allow for a crop with up to 1,024 fire pixels. |
| Capped | Hundreds or more (*estimate*) | The margin is small. This is why capped is the main rule. |

**Exact solver.**
1. Take every fire pixel of the calibration crops, with its score and its cost (table above).
2. Sort them from the lowest score up.
3. Walk up the list, adding costs while they still fit in the budget `α · n_eff − (1 − α)`.
4. The cutoff is the score of the first pixel you can't afford to miss.

```python
import numpy as np

def calibrate_threshold(scores, costs, n_eff, alpha):
    """Largest t such that the summed cost of fire pixels with score < t is at most
    alpha * n_eff - (1 - alpha). scores, costs: 1-D float64 arrays over all valid
    fire pixels of the calibration crops. Flag a pixel iff score >= t."""
    budget = alpha * n_eff - (1.0 - alpha)
    if budget < 0:
        return -np.inf                       # not enough data: flag everything
    order = np.argsort(scores, kind="stable")
    s, cum = scores[order], np.cumsum(costs[order])
    k = int(np.searchsorted(cum, budget * (1 - 1e-12), side="right"))  # we can miss the k lowest
    return np.inf if k == len(s) else s[k]
```

Compute in float64 on the CPU. Pixels tied at the cutoff are flagged, which errs on the safe side.

**A distance buffer for free.** Score each pixel by minus its distance to the nearest pixel burning today. The same rule then returns a radius, "flag everything within r km of today's fire", with the same guarantee.

**The anchor paper's rule.** Dayan 2026 sorts all calibration fire pixels by score and leaves the lowest ≈ α·m unflagged, where m is the number of fire pixels. That is our pooled rule **without** the worst-case margin. It would be justified only if pixels were independent, and they are not. At the paper's scale, the missing margin is about 0.8 percentage points of miss rate.

## 5. Rules compared

All rules use the same U-Net scores and the same calibration crops.

| Rule | Guarantee |
|---|---|
| Fixed cutoff 0.5 | None |
| Anchor's pixel-pooled rule | None (it assumes independent pixels) |
| Standard per-crop CRC | Per-crop miss rate ≤ α |
| WISER pooled | Pooled miss rate ≤ α |
| **WISER capped** (main) | Capped miss rate ≤ α |
| **WISER + populated cutoff** (§7) | Capped ≤ α for other pixels, and per-crop ≤ α_pop for populated pixels |

## 6. Models

| Model | What it is | Why it's here |
|---|---|---|
| Flag everything | — | Upper bound on area, and the fallback |
| Persistence | Tomorrow = today | The standard reference. Reproducing its published precision and recall checks our metric code. |
| Distance buffer | Flag within r km of today's fire, with r set by the WISER rule | Shows how much learning adds over "near today's fire" |
| Gradient-boosted trees | LightGBM (Ke 2017), with scikit-learn's `HistGradientBoostingClassifier` as fallback; about 500 trees. Per-pixel features: the 12 inputs, distance to today's fire, and fire counts in the 3 × 3 and 7 × 7 neighbourhoods. | A simple learned baseline |
| **Tiny U-Net** (main) | A U-Net (Ronneberger 2015) with about 470K parameters: 12 → 32 → 64 channels, 128 at the bottleneck, two decoder blocks with skip connections. The same size as the anchor paper's. | The main model |
| Fire-only U-Net | The same U-Net with today's fire mask as its only input | Tests whether the other 11 inputs help |

**Training.**
- Loss: binary cross-entropy on valid pixels.
- Optimiser: AdamW, learning rate 1e-3, cosine schedule.
- Length: up to 50 epochs, with early stopping on validation average precision. Never stop on calibration data.
- Inputs: random 32 × 32 crops, in batches of 64–256.
- Hardware: PyTorch on MPS or CPU.
- Seeds: three for the main U-Net if time allows, and at least one.
- Output: float32 scores for every validation, calibration and test crop. All calibration runs afterwards, on the saved scores.

**Laptop cost (*estimate*).** A Tiny U-Net run takes minutes on MPS and under an hour on CPU. All core training is under 10 laptop-hours. If a run takes more than three times its estimate, shrink the model first, then try free cloud notebooks.

**Not used:**
- Swin-UNet, unless the instructor asks for it. The evidence says it won't beat a tuned U-Net ([proposal-review](proposal-review.md)).
- The ResGNN-U-Net, which gave no gain in the anchor paper.
- Pretrained models.

## 7. Exposure-aware version

- **Populated pixel:** population density ≥ τ (start with τ = 10 people/km²). τ is fixed on training data before calibration.
- **Two cutoffs:**
  - Populated pixels get their own cutoff, calibrated so that their per-crop miss rate is at most a stricter α_pop (start with α / 2). We use per-crop here because most crops have only a few populated fire pixels.
  - All other pixels use the main capped rule at α.

  Both cutoffs come from the WISER rule, each run on its own pixels. Extra area is spent only where people are.
- **Compared with:**
  - the single standard cutoff, for which we measure how many populated fire pixels it misses;
  - flagging every populated pixel plus the standard cutoff elsewhere. This is the trivial way to never miss in a populated cell, and the two-cutoff version must flag less.
- **Check first (week 1):** count how many calibration crops contain populated fire pixels. If there are fewer than about 200, the populated cutoff is weak. In that case, shrink the experiment and report the shortage as a finding.

**Limits.**
- "Exposure" here means GPWv4 population density at 1 km, a static census proxy. It is not people at risk, homes or lives.
- Density-based rules give rural and tribal communities less protection. So we report results by density class (0, 0–10, 10–100, > 100 people/km²), and every other fire pixel still stays under the α target.
- Population is also one of the model's inputs. That doesn't affect validity: the populated/not split is computed from the inputs and fixed before calibration.

## 8. If time allows

| Extra | What it adds |
|---|---|
| Block re-splitting | A validity check that keeps chunks of consecutive records together. A more honest test when crops are dependent. |
| Anchor replication | Re-run the anchor paper's protocol (random split, full tiles) to explain its high AUROC. At most 2 days. |
| Population-weighted single cutoff | One cutoff for a miss rate in which each fire pixel counts by log population, with a floor (e.g. 0.2). This is the proposal's "population-weighted calibration". |
| Cutoffs by fire size | A separate guarantee for small, medium and large fires |
| "With 90 % confidence" version | RCPS (Bates 2021): holds for the calibration set actually used, with probability ≥ 90 %, at a small cost in area |
| Risk zones in the demo | Priority / review / not flagged, never "safe". Built from two separately calibrated cutoffs. |
| Medium U-Net, more input ablations | Model questions |

**Not pursued:**
- Online or time-based methods: NDWS has no dates.
- Reweighting for an estimated shift: there is no proven guarantee.
- Worst-case shift balls: there is no defensible radius.
