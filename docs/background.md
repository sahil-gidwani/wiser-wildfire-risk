# Background

What the literature says, in four parts. Sources are listed in [references](references.md).

## 1. Next-day wildfire spread prediction

**Different tasks are not interchangeable.** "Spread prediction" can mean several things:
- next-day active fire (NDWS, WildfireSpreadTS);
- daily newly burned area;
- final burned area from ignition;
- sub-daily perimeters;
- physics simulation.

Results don't transfer between these tasks. We work on next-day active fire.

**What the evidence says** on next-day active-fire data:

1. **Today's fire dominates.** Removing today's fire mask hurts NDWS models the most, and a U-Net given only the previous fire mask matches 12-input models (Fitzgerald 2023, *reported*).
2. **Extra inputs add little.** NDWS ablations move AUC-PR by at most about 2 points per input. On other datasets, fuel and land-cover layers add about +0.01–0.08 AP.
3. **Architecture matters less than tuning.** Swin variants tie with or lose to tuned U-Nets ([proposal-review](proposal-review.md)).
4. **Metrics change the rankings.**
   - The highest-AP model can flag 4–5× the burned area at a 0.5 cutoff (Gopakumar 2026).
   - Counting continuing fire as a success and relabelling unknown pixels inflates F1 by over 44 % (Han 2026).
5. **Labels are noisy.** MODIS daily fire masks are "highly stochastic" (Karlsson 2025). Terra passes over at about 10:30 and 22:30, so it misses the afternoon peak, and it misses fire under cloud.
6. **Transfer is weak.** In DailyWildfireCAN, U-Net AUC-PR drops from 0.285 to 0.218 on unseen regions (Ettahri 2026, *reported*).

**NDWS benchmark numbers:**

| Model | AUC-PR | Source |
|---|---|---|
| Persistence | 11.5 % | Huot 2022 |
| Original convolutional autoencoder | 28.4 % | Huot 2022 |
| Best official-split single models | About 0.35 (U-Net: 0.346) | Esparza 2026 |

More detail is in [data](data.md).

**Uncertainty methods without guarantees.** MC dropout, Bayesian networks and ensembles have been applied to spread maps (e.g. Chakravarty 2025). None of them gives a finite-sample bound on missed fire; that is what conformal risk control adds.

**What this means for us:**
- compare against persistence and a distance buffer;
- judge models by flagged area at a fixed miss rate;
- report misses on new fire separately.

## 2. Conformal risk control

**Conformal risk control (CRC)** (Angelopoulos, Bates, Fisch, Lei & Schuster, ICLR 2024) picks a cutoff from calibration data so that the **expected** loss on a new example is at most α. It needs three things:
- **exchangeable** examples: calibration and future examples are alike, in the sense that shuffling their order changes nothing;
- a loss that can only go down as more is flagged;
- a known upper bound on the loss.

**In image segmentation the example is the image,** with one loss per image, such as the share of the object's pixels that were missed. Treating each pixel as a separate example is not covered, because pixels in an image are correlated.

**Related methods, and whether we use them:**

| Method | What it adds | Use here |
|---|---|---|
| RCPS (Bates 2021), Learn then Test (Angelopoulos 2025) | "With probability ≥ 1 − δ over the calibration set" instead of "on average" | Optional |
| Group-conditional CRC (Aldirawi 2026) | A separate guarantee within each group, e.g. small versus large fires | Optional |
| Hierarchical conformal (Lee, Barber & Willett 2023) | Handles dependent examples by treating groups as the units | Optional (block re-splitting) |
| Weighted conformal (Tibshirani 2019) | Handles a known distribution shift | No: the shift isn't known |
| Online methods (adaptive conformal inference, rolling risk control) | Adapt over time | No: NDWS has no dates |
| Beyond exchangeability (Barber 2023) | Bounds the damage from a mild shift | Background only |

**A useful fact.** CRC's guarantee is an average over calibration sets. For any one calibration set, the realised risk can land slightly above α. Hultberg et al. (2026) report this happens about half the time at n = 500.

**Precedents in conformal segmentation:**
- image-level CRC losses (Mossina 2024);
- calibrated dilation of a predicted mask (Mossina & Friedrich 2025). This is the precedent for our calibrated distance buffer.

**Conformal work on wildfire:**
- prediction sets for wildfire magnitude (Xu 2022);
- the pixel-pooled CRC study on NDWS (Dayan 2026; see [proposal-review](proposal-review.md));
- calibrated alarm thresholds for fire detectors (Sandireddy 2026).

We found no per-crop, exposure-aware CRC study for spread maps. We searched in November 2026; check again before claiming novelty.

## 3. Exposure

- **Exposure is growing.**
  - About 595,000 people lived inside large-fire perimeters in the contiguous US over 2000–2019, and that exposure rose 125 % (Modaresi Rad 2023).
  - Homes in the wildland–urban interface rose from 30.8 to 43.4 million between 1990 and 2010 (Radeloff 2018).
- **Losses concentrate at the interface.**
  - In California, interface areas held 50 % of destroyed buildings but only 2 % of the burned area (Kramer 2019).
  - 95 % of building losses happened within 100 m of wildland vegetation (Caggiano 2020).
- **No exposure-weighted metrics** for machine-learning spread models were found in the literature.
- **On NDWS, we can only use its own population channel** (GPWv4 density, 1 km). Better layers exist, such as housing density, WUI maps and building footprints, but they can't be joined because NDWS has no coordinates.

## 4. Why these are not evacuation maps

- **How evacuations work.** US evacuations are lawful orders for pre-defined zones. Emergency managers set them using live observations, fire-behaviour analysts and trigger buffers. Operational fire models run at tens of metres and hourly time steps.
- **What our target is.** It is "the Terra satellite detects fire in this 1 km cell tomorrow". Building losses happen within about 100 m of vegetation, and wind-driven runs unfold within hours.
- **The data's own limits.** The NDWS authors say the data "does not provide the fine-scale information required for tactical decision-making".
- **"Safe" labels are avoided in practice.**
  - Australia's Neighbourhood Safer Places "will not guarantee safety".
  - Its national fire-danger system says "No Rating" rather than "low".

So the maps are, at most, retrospective decision-support material.

**Equity.** Weighting by population density gives rural and tribal communities less weight. Social vulnerability to wildfire is unevenly spread and is not captured by density (Davies 2018; Wigtil 2016). We report results by density class and keep every fire pixel under a miss target ([method §7](method.md)).
