# Evaluation

## Metrics

- **Realised miss rate** on test crops, using the same definition the cutoff was calibrated for. It is always shown **next to the flagged area**, meaning flagged valid pixels as a share of all valid pixels.
- **New-fire miss rate:** the same, counting only fire pixels that were not burning today.
- **Average precision (AP)** for how well a model ranks pixels.
- **AUROC** only for comparison with the anchor paper. With about 99 % non-fire pixels it looks good for almost any model.
- **Precision, recall and F1** only for comparison with published numbers.
- **Every results table has a persistence row.**
- **Unknown pixels** never count.
- **Uncertainty:**
  - a bootstrap over test crops with 1,000 resamples (optimistic, because crops from one fire are related);
  - the spread across training seeds.

Not used:
- accuracy, which is meaningless at 1 % fire;
- the false-positive rate, which is almost the same as flagged area here.

## Validity checks

1. **Random re-splits (sanity check).**
   - Pool the calibration and test crops.
   - 1,000 times: draw a random calibration set (from 100 crops up to the full set), calibrate, and measure on the rest.
   - For every CRC-based rule, the **average** realised miss rate must be at most α, within simulation error. If it isn't, the code has a bug.
   - This check says nothing about new fires, because one fire can sit on both sides.
2. **Official split (realistic check).**
   - Calibrate on the calibration half of the official eval split and report on the official test split, which covers different time windows.
   - Then swap the two.
   - Each is a single draw, so a small excess over α can be chance.
3. ***If time:* block re-splits.** Like check 1, but move whole chunks of consecutive records together, so that crops of the same fire stay on one side.

**Test the calibration code before touching NDWS:**
- on synthetic crops with known answers;
- against a brute-force search over cutoffs;
- against MAPIE's risk-control functions (Taquet 2022).

A teammate who didn't write the rule reviews it.

## Experiments

### 1 · Data check (week 1)

- **Do:**
  - convert the data;
  - run the checks in [data §6](data.md);
  - reproduce persistence precision and recall (35.7 % / 27.3 %);
  - run the distance buffer through calibration end to end.
- **Done when** the numbers are in and persistence matches within about 1.5 points, or the gap is explained.
- **Status:** done on 2026-11-15. Persistence matches exactly on full tiles; the numbers are in [data §6](data.md).
- **Buffer end to end** (cutoffs picked on calibration crops, measured on validation crops; the test crops stay untouched):
  - 25 % of the calibration crops with fire tomorrow have no fire today inside the window, and they hold 14 % of tomorrow's fire. The window-only buffer can't rank anything in them, so every rule flags everything at 5 % and at 10 %.
  - Measuring distance on the whole tile instead, WISER picks a radius of 21–31 km and flags 62–83 % of the land.
  - Persistence also has to flag everything. At the fixed cutoff 0 (today's fire only) it flags 1.5 % of the land and misses 74–78 % of tomorrow's fire, including all new fire.
  - AP on validation crops: buffer 0.218, persistence 0.108.
- **U-Net smoke run** (3 epochs, seed 0, Mac GPU, 10–22 s per epoch, 470,401 parameters):
  - With our own input scaling: validation AP 0.188 → 0.226 → 0.233, already above both buffers (0.22). With WISER capped it flags 84 % of the land at 5 % and 74 % at 10 %.
  - An earlier 3-epoch run with the NDWS authors' scaling reached the same AP (0.235) but flagged 76 % and 64 %. Flagged area after 3 epochs is noisy; only full training with several seeds can compare settings.
  - At the fixed cutoff 0 (probability 0.5) it flags nothing and misses all fire: with fire at about 3 % of pixels, no pixel reaches 50 %.
- **First full U-Net run, and a leak in our validation set** (50 epochs, 11 min, validation then carved from the train split):
  - Validation AP rose from 0.19 to 0.385 and was still creeping up at epoch 50, so early stopping never triggered.
  - On the calibration crops, from weeks the model never trained on, its AP was only 0.219 (whole-tile buffer: 0.222 on validation, 0.183 on calibration).
  - Validation tiles came from the same weeks as the training tiles, so they shared fires and places with them, and the model was partly memorising those. Validation over-rated it, early stopping had a misleading yardstick, and WISER's cutoff (picked on the harder calibration crops) missed only 2.4 % on validation against a 5 % target.
  - Fix (decision 22): validation is now half of the official eval split, so validation, calibration and test all come from weeks the model never trains on. The earlier validation numbers are kept here only as the record of this finding.
- **Retraining and tuning with the new validation half** (seed 0):
  - The original settings now stop early: best validation AP 0.245 at epoch 11, stopped at 18. On weeks it never saw, the model learns its useful patterns in about 10 epochs.
  - Five settings compared on validation AP: 50 planned epochs 0.245; 15 epochs 0.242; 15 epochs with learning rate 3e-4 0.229; **15 epochs with weight decay 0.01: 0.250** (steady after its peak, 3 minutes); 30 epochs with learning rate 3e-4 and weight decay 0.01: 0.236. Differences under about 0.01 are within noise; the smaller learning rate clearly hurts.
  - Chosen (decision 23): 15 epochs, learning rate 1e-3, weight decay 0.01.
- **U-Net against the buffers** (cutoff picked on the calibration half, measured on the validation half; AP 0.250 vs 0.180 for the whole-tile buffer):

  | Target, capped | buffer_window | buffer_tile | U-Net |
  |---|---|---|---|
  | 5 % | flags 100 % | missed 6.4 %, flags 76 % | missed 4.7 %, flags 71 % |
  | 10 % | flags 100 % | missed 11.1 %, flags 63 % | missed 9.3 %, flags 58 % |

  The U-Net flags about 5 points less land while seeing only the window. The whole-tile buffer landed above its targets on this single validation draw, and the older pixel-pooled rule overshot most (14.1 % missed against a 10 % target); experiment 2 will check averages over many draws. At the fixed cutoff 0 the U-Net flags almost nothing and misses 98 % of fire.
- **Three U-Net seeds** (tuned settings; same calibration and validation halves):

  | Seed | Validation AP | 5 %: missed / flagged | 10 %: missed / flagged |
  |---|---|---|---|
  | 0 | 0.250 | 4.7 % / 70.8 % | 9.3 % / 58.0 % |
  | 1 | 0.242 | 5.5 % / 72.5 % | 9.9 % / 58.6 % |
  | 2 | 0.237 | 4.5 % / 73.0 % | 8.8 % / 60.7 % |
  | Mean ± spread | 0.243 ± 0.007 | 4.9 % / 72.1 % ± 1.2 | 9.3 % / 59.1 % ± 1.5 |

  The U-Net flags about 4 points less land than the whole-tile buffer at 5 % (72 % vs 76 %) and about 3.5 points less at 10 % (59 % vs 63 %): roughly three times the seed-to-seed spread, so the gap looks real but modest. Seed 0 won the tuning, so its numbers are slightly flattering; seeds 1 and 2 are fresh.

### 2 · Does it hold? (RQ1)

- **Do:** run the rules of [method §5](method.md) on the Tiny U-Net's scores at α = 5 % and 10 %. Use validity checks 1 and 2.
- **Output:**
  - for each rule: realised per-crop, pooled, capped and new-fire miss rates, and flagged area;
  - a plot of the average realised miss rate minus α against calibration-set size.
- **Expect:**
  - the CRC rules pass the sanity check;
  - the pixel-pooled rule may exceed α, especially with small calibration sets;
  - per-crop control flags much more area than capped at the same α.

### 3 · What does it cost? (RQ2)

- **Do:** apply WISER capped at α = 5 % and 10 % to each model: flag everything, persistence, the two distance buffers (window only and whole tile), gradient-boosted trees, fire-only U-Net and Tiny U-Net.
- **Output:**
  - flagged area for each model, plus the buffer radius in km;
  - AP for each model;
  - a plot of flagged area against α.
- **Expect:** the U-Net flags less than the buffer and the trees; the fire-only U-Net comes close to the full one.

### 4 · Where are the misses, and who is near them? (RQ3)

**Miss breakdown** for the U-Net and the buffer, at the WISER cutoff:
- misses by distance from today's fire (0, 1, 2–3, > 3 km);
- misses on new versus continuing fire;
- four example crops, picked by a rule fixed in advance:
  - the median-miss crop;
  - a 90th-percentile failure;
  - a crop with populated fire;
  - the crop where the buffer and the U-Net differ most in flagged area.

**Exposure.** Compare three options:
- the standard cutoff;
- two cutoffs, stricter in populated pixels;
- flagging every populated pixel.

For each option, report the miss rate in populated pixels, the overall miss rate and the flagged area. Break all of them down by density class.

**Expect:**
- most misses are new fire 2 km or more from today's fire;
- the two-cutoff version meets its stricter target while flagging less than "flag every populated pixel".

### 5 · Demo (weeks 4–5)

**Required:** a static HTML gallery of test crops, built by one command from the saved scores and results.
- Each crop has four panels: observed fire, flagged area, missed fire and population.
- You can switch between WISER and at least two baselines: standard per-crop CRC and the pixel-pooled rule.
- You can filter by distance band, density class and fire size.

**Optional:** a small local viewer with an α slider.

The wording rules below apply to the demo too.

## Fix before experiment 2

Before the first calibration that touches test data, record these in the decisions table in [plan](plan.md):
- the main model;
- the α values;
- the cap `c`;
- τ and α_pop;
- the crop seed;
- the validation carve-out;
- the example-selection rule.

Anything changed after that is labelled *exploratory*.

## How to word results

| Avoid | Say instead |
|---|---|
| evacuation map, EVACUATE zone | risk-controlled next-day fire map; decision-support map |
| safe zone, low-risk zone | not flagged (not an all-clear) |
| "guarantees 95 % of fires are detected" | "keeps the expected miss rate of MODIS-detected next-day fire pixels at most 5 %, for crops like the calibration crops" |
| fire front, perimeter | next-day fire detections; distance from today's detected fire |
| people at risk, lives protected | miss rate in populated pixels (population-density proxy, GPWv4, 1 km) |
| leakage-free split | window-disjoint split |
| our theorem, novel guarantee | follows directly from the CRC theorem (Angelopoulos et al. 2024) |
| outperforms, state of the art | flags x % less area than … at the same α |
| real-time, operational | retrospective benchmark on NDWS |

Also:
- **State the guarantee in full once.** It must say what is averaged, "in expectation", "MODIS-detected", and "for crops like the calibration crops (NDWS crops with fire on both days)". After that, a short form is fine.
- **Always give the flagged area** next to a miss rate.
- **Hedge novelty claims** with "to our knowledge, as of <date>", and make them only after a fresh literature check.
