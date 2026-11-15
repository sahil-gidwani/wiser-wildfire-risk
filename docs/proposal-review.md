# Proposal review

We fact-checked the initial proposal ([proposal.pdf](proposal.pdf), 8 slides) claim by claim and audited the paper it builds on. Short version: **the problem is right, but the foundation and the three proposed improvements don't hold up as written.**

Of 31 claims checked:

| Verdict | Claims |
|---|---|
| Accurate | 3 |
| Partly accurate | 10 |
| Wrong | 3 |
| Unsupported | 3 |
| Not feasible as written | 5 |
| Overclaimed (needs rewording) | 7 |

## What the proposal said

- **Goal:** combine modern wildfire models, conformal risk control (CRC) and exposure-aware evaluation to produce "more reliable evacuation maps across different datasets and environmental conditions".
- **Problem:** judge models not only by accuracy, but by *where* they miss fire, *who* is exposed, and whether their guarantees *generalise*. This is the strongest part of the proposal, and we keep it.
- **Approach:**
  - start from an existing CRC setup on NDWS (Dayan 2026, "the anchor paper");
  - add Swin-UNet as a stronger model;
  - add richer vegetation and fuel datasets;
  - add population density as an input;
  - output SAFE / MONITOR / EVACUATE zones.
- **Data:** NDWS, extended with WildfireSpreadTS, PyroStack and the RCDA-Net dataset "if they can be aligned".
- **Experiments:**
  - a model comparison: LightGBM, Tiny U-Net, ResGNN-U-Net and Swin-UNet;
  - standard versus population-weighted CRC;
  - generalisation across datasets.
- **Plan:**
  - week 1: reproduce the anchor paper;
  - weeks 2–3: Swin;
  - week 4: exposure;
  - week 5: external datasets;
  - week 6: report.

## Verdicts in brief

| Claim (slide) | Verdict | Why |
|---|---|---|
| Models can miss where fire spreads (2) | Accurate | NDWS persistence recall is 27 %, and the anchor paper's models miss 28–93 % of fire pixels at a 0.5 cutoff |
| Models behave differently across regions and datasets (2) | Accurate | Large drops when tested on new regions or datasets; e.g. AP falls from 0.354 to 0.238 on AFSeq (Guillen 2026) |
| Swin-UNet is a stronger model (4) | Unsupported | Swin ties with or loses to a tuned U-Net on NDWS and two other datasets (table below) |
| Richer vegetation and fuel data will help (4–5) | Not feasible | Fuel and land-cover layers add only about +0.01–0.08 AP on other datasets. The candidate datasets don't line up with NDWS, and WildfireSpreadTS alone is a 48 GB download. |
| Add population density as an input (4) | Wrong | NDWS already has it. Removing it moves AUC-PR from 28.4 to 28.1 (Huot 2022, *reported*). |
| "Wildfire Spread Forecasting with Deep Learning" shows environmental features drive quality (3) | Wrong | That paper (Anastasiou 2025) predicts final burned area from ignition in the Mediterranean, and only ablates the time window |
| Start from the existing CRC setup (4) | Unsupported | Its calibration and split have serious problems (next section), and its code has no licence |
| The output is "newly burned area" (5) | Wrong | The NDWS target is *all* fire detections on the next day, not newly burned area |
| "Perimeter FNR" and event-level metrics (6) | Overclaimed / not feasible | NDWS has detections, not perimeters, and no fire ids or dates |
| Evacuation maps; a SAFE zone (1, 7) | Overclaimed | 1 km daily satellite detections can't support evacuation decisions; the NDWS authors say the data lacks the detail needed for tactical decisions; and operational systems avoid "safe" labels |
| Six-week roadmap (8) | Not feasible | It has no data check or validation step, puts Swin before the baselines, puts 48 GB of external data in week 5, and names no owners |

All four papers cited in the proposal exist. FireBench (Ho 2026) is real but is about physics simulations, so it is background only.

**Swin versus U-Net:**

| Source | Data | U-Net | Swin variant |
|---|---|---|---|
| Zhou et al. 2025 | NDWS, 32 × 32 crops | AUC-PR 0.274 | Swin-UNet 0.280, a tie within seed noise |
| Esparza et al. 2026 | NDWS, official split | AUC-PR 0.346 | SwinUNETR 0.334 |
| Lahrichi et al. 2026 (*reported*) | WildfireSpreadTS | AP 0.468 | SwinUnet 0.437 |
| DailyWildfireCAN, Ettahri et al. 2026 (*reported*) | Canada, 90 m | AUC-PR 0.285 | Swin-U-Net 0.185 |

## The anchor paper (Dayan 2026)

*Conformal Risk Control for Safety-Critical Wildfire Evacuation Mapping*, arXiv:2603.22331, is a single-author preprint that has not been peer reviewed.
- It trains LightGBM, a Tiny U-Net and a ResGNN-U-Net on NDWS.
- It then calibrates a cutoff so that about 95 % of fire pixels are flagged.

Its reported results at α = 5 %:

| Model | AUROC | Realised miss rate | Flagged area |
|---|---|---|---|
| LightGBM | 0.854 | 5.9 % | 62.6 % |
| Tiny U-Net | 0.969 | 5.3 % | 14.9 % |
| ResGNN-U-Net | 0.964 | 5.2 % | 15.1 % |

**What's wrong with it.** We checked each of these in the paper and its code.

1. **Pixels are treated as independent.**
   - It pools all calibration pixels and takes a quantile.
   - CRC needs exchangeable units, such as images or crops, and pixels in the same crop are strongly correlated. The author concedes this.
   - The valid version of the same rule needs a margin of one extra worst-case crop: about 0.8 percentage points at the paper's scale ([method §4](method.md)).
2. **Misses are above target, yet called "guaranteed".** All three realised miss rates (5.2–5.9 %) exceed 5 %, while the text promises that "≥95% of fires will be detected".
3. **A random re-split.** It ignores the official NDWS split, so overlapping tiles of the same fire on consecutive days land in train, calibration and test.
4. **Full tiles.**
   - NDWS tiles are centred on next-day fire, so a full-tile model can learn "fire is in the middle".
   - The NDWS authors crop 32 × 32 for this reason; the anchor paper doesn't.
   - Together with point 3, this likely explains its AUROC of 0.969, against about 0.85 elsewhere.
5. **Calibration data is reused.** The best checkpoint is picked on the calibration split, although the paper says the calibration set is never used in training.
6. **The SAFE zone is empty.** Its three-zone scheme uses a loss that isn't monotone, so the CRC theory doesn't apply, and it labels 0 % of pixels SAFE for every model.

There are smaller issues too:
- an off-by-one quantile index, immaterial at its scale;
- wrong author lists for the CRC and NDWS papers;
- no persistence baseline;
- AUROC as the headline metric.

**What we keep:**
- its framing: a calibrated miss rate is the safety target, and flagged area is the cost;
- its comparison of a tree model with a Tiny U-Net;
- its own future-work list: image-level CRC, shift, and transformers.

**What we do with it:**
- We use its pixel-pooled rule as a comparison row on our own scores.
- We replicate its full protocol only if time allows.
- Its repository has no licence, so we write our own code.

## What we changed

| Proposal | Now | Why |
|---|---|---|
| "More reliable evacuation maps" | Risk-controlled next-day fire maps, framed as decision support | The labels and resolution can't support evacuation claims |
| Build on the anchor paper's CRC setup | Valid per-crop and capped calibration; the anchor paper's rule becomes a comparison | Pixel pooling has no guarantee, and the split and cropping have problems |
| Swin-UNet as the stronger model | A Tiny U-Net plus simple baselines; Swin only if the instructor asks | Swin doesn't beat tuned U-Nets |
| Richer fuel datasets | One input test: a U-Net on today's fire only, versus all inputs | The expected gain is small and the datasets don't align |
| Population as an input | Population decides which misses get a stricter target | It is already an input, with no predictive value |
| External datasets | NDWS only | Size, different targets, not enough time |
| "Perimeter FNR" | Miss rate by distance from today's fire | NDWS has no perimeters |
| SAFE / MONITOR / EVACUATE | Optional neutral zones in the demo: priority / review / not flagged | The anchor paper's scheme is invalid, and "safe" is never promised |
| ResGNN-U-Net | Dropped | No gain over the U-Net, even in the anchor paper |
| — | Added: a data check, validity checks, persistence and a no-learning distance buffer | Needed for any honest guarantee |
| Demo: "choose an event" | A gallery of test crops | NDWS has no event names or locations |

## If the instructor expects the original plan

- **Swin-UNet:** run a small one, with a single seed, after the core results.
- **Cross-dataset results:** the cheapest credible version is to run the no-learning baselines on one small external dataset with fire ids ([data §7](data.md)). Do this only if it is required.
- **"Evacuation maps":** use the phrase only in quotation marks, as the proposal's original framing.
