# Data

We use one dataset: **Next Day Wildfire Spread (NDWS)** (Huot et al., IEEE TGRS 2022).

## 1. The dataset

| | |
|---|---|
| Size | 18,545 tiles, each 64 × 64 pixels at 1 km |
| Coverage | Contiguous US, 2012–2020 |
| Inputs | 12 day-t layers (below) |
| Label | Next-day fire mask: 1 = fire, 0 = no fire, −1 = unknown (cloud, or not processed) |
| Fire share | 1.14 % of valid pixels are fire, and 2.39 % of all pixels are unknown (our week-1 count; an independent audit found 0.98 % and 1.77 %, Esparza 2026) |
| Access | Kaggle `fantineh/next-day-wildfire-spread`, CC BY 4.0, about 4 GB of TFRecords |
| Code | google-research `next_day_wildfire_spread` (Apache-2.0), including the export scripts |

**Inputs:**

| Key | Variable | Source |
|---|---|---|
| `elevation` | Elevation (m) | SRTM |
| `th`, `vs` | Wind direction (°), wind speed (m/s) | GRIDMET, about 4 km, daily |
| `tmmn`, `tmmx` | Minimum and maximum temperature (K) | GRIDMET |
| `sph` | Specific humidity | GRIDMET |
| `pr` | Precipitation (mm/day) | GRIDMET |
| `erc` | Energy Release Component, a fuel-dryness index | GRIDMET |
| `pdsi` | Palmer Drought Severity Index | GRIDMET drought |
| `NDVI` | Vegetation index | VIIRS, 500 m, 8-day |
| `population` | Population density (people/km²) | GPWv4.11, about 1 km |
| `PrevFireMask` | Today's fire mask, values {−1, 0, 1} | MODIS MOD14A1 |

The target `FireMask` is tomorrow's MODIS mask. The NDWS code clips each input to fixed bounds and standardises it with fixed means and standard deviations (`DATA_STATS`). We use the same values.

**Population** is heavily skewed (mean 26, 99.9th percentile 2,534 people/km²). It is also static: a place has the same value in every year.

## 2. Labels

- **Fire** means a MODIS Terra active-fire detection at 1 km, at low, nominal or high confidence.
- **What MODIS misses.** Terra passes over at about 10:30 and 22:30, so it misses the afternoon peak, fire under cloud, and short fires.
- **Unknown pixels (−1)** are left out of every loss, metric and area count.
- **Most next-day fire is new.** Persistence (tomorrow = today) has precision 35.7 % and recall 27.3 % on full test tiles (Huot 2022; our code gives the same numbers). On our crops, 77 % of tomorrow's fire pixels were not burning today.
- **What the label is not.** It is not a perimeter, a burned area or a "fire front". Every miss we count is a miss of a satellite detection.

## 3. Three quirks that shape the design

All three come from the official export code.

1. **Tiles are centred on next-day fire.** Each tile is cut around a pixel where fire is detected tomorrow, so a model trained on full tiles can learn "fire is in the middle". In 85 % of tiles the central 2 × 2 pixels burn tomorrow, against 1.1 % of pixels overall. The NDWS authors use random 32 × 32 crops for this reason, and so do we.
2. **Only ongoing fires.** Every tile has fire today. 89 % also have fire detected tomorrow; the other 11 % have none, and they are much cloudier (14 % unknown pixels, against 1 %). Our results apply to "ongoing fire situations as sampled by NDWS", not to any landscape.
3. **No coordinates, dates or fire ids.** We can't join other layers (housing, WUI maps), name fires, or test across years or regions.

Several overlapping tiles are also often cut from the same fire on the same day, so tiles are not independent. Some are even identical: 860 tiles fall into 396 groups of exact repeats, each group within one split. We keep the first tile of each group and leave out the other 464 (`is_copy` in `tiles.csv`).

## 4. Splits

**The official split** is by 8-day *windows*, not by tile:
- Windows are shuffled and assigned to train, eval and test at about 8:1:1.
- The last day of each window is left out as a buffer.
- The sizes are 14,979 / 1,877 / 1,689 tiles (confirmed in week 1).
- Fires that burn for weeks can still span windows in different splits. So call the split "window-disjoint", not "leakage-free".

**Our data roles:**

| Role | Data | Used for |
|---|---|---|
| Train | Official train minus validation | Fitting models |
| Validation | About 10 % of official train (1,498 tiles), taken in chunks of 50 consecutive records so that same-fire tiles mostly stay together | Every choice: early stopping, model choice, settings, the exposure settings |
| Calibration | Official eval | Choosing the cutoff, and nothing else |
| Test | Official test | Reporting, and nothing else |

Never use calibration or test data to make a choice. Anything decided after looking at test numbers is labelled *exploratory*.

## 5. Preprocessing

`python -m wiser.data` takes under a minute: it downloads and converts the data, draws the stored crops and runs the week-1 checks. The other steps happen at training and evaluation time; the helpers for step 4 are already in `wiser/data.py`.

1. **Convert once.**
   - Read the TFRecords with the `tfrecord` package, which works without TensorFlow. The files are not compressed.
   - Write NumPy arrays: inputs `[N, 12, 64, 64]` float32 (about 3.7 GB) and labels `[N, 64, 64]` int8.
   - Write a table with one row per tile: source file, record index, split, data role, crop offset, simple counts, and an `is_copy` flag for exact repeats.
2. **Normalise** the inputs with the official statistics, in the loader.
3. **Crop.**
   - Training: a fresh random 32 × 32 crop each time a tile is drawn.
   - Validation, calibration and test: one random crop per tile, drawn once with a fixed seed and saved.
4. **Derive features** inside each crop:
   - the distance from each pixel to the nearest pixel burning today, used by the buffer baseline and the miss breakdown;
   - a new-fire mask: fire tomorrow, not today;
   - a populated mask: population ≥ τ.
5. **Handle wind under augmentation.** Flips and rotations change the wind direction. Either convert wind to (u, v) components and rotate those too, or skip flips and rotations. Decide in week 1.

## 6. Week-1 data check (done 2026-11-15)

`python -m wiser.data` prints these numbers and saves them to `outputs/data/checks.txt`. Crop numbers cover the validation, calibration and test crops, with repeats left out.

| Check | Expected | Measured |
|---|---|---|
| Tiles per split | 14,979 / 1,877 / 1,689 | The same |
| Fire share of valid pixels | About 1 %; the anchor paper says about 5 % | 1.14 % on full tiles and 2.69 % on crops. The anchor paper's 5 % is wrong. |
| Unknown-pixel share | About 1.8 % | 2.39 % |
| Fire pixels per crop | Mean about 30 (*estimate*) | Mean 26.9, median 11; a quarter of crops have 32 or more. The cap `c` = 32 stays. |
| Fire at the tile centre | About 100 % if tiles are centred | 85 % of tiles burn tomorrow in the central 2 × 2 pixels, where the fire rate is 54 % against 1.1 % over the whole tile. Tiles are centred, so the crops stay. |
| Tiles with fire on both days | All | Fire today: 100 %. Fire tomorrow: 89 %. The 11 % without are much cloudier. |
| Crops with no fire tomorrow | At most about 6 % | 15.8 %, mostly from the tiles with no fire tomorrow. Such crops carry no weight in the WISER rule; they only add flagged area. |
| New fire as a share of tomorrow's fire | About 73 % | 77 % |
| Exact duplicate tiles | Unknown | 860 tiles in 396 groups, each group within one split. The 464 repeats after the first are left out. |
| Fire pixels with population > 0 / > 10 / > 100 per km² | Unknown | 50.6 % / 3.8 % / 0.6 % |
| Calibration crops with fire in populated cells (≥ 10 per km²) | At least about 200 needed | 366 of 1,877, so the exposure experiment can go ahead |
| Persistence precision / recall | 35.7 % / 27.3 % | 35.7 % / 27.3 % on full test tiles, so our metric code matches the paper. On our test crops: 37.9 % / 24.7 %. |

## 7. Other datasets (not used)

The proposal's three extra datasets all exist, but none of them fits a six-week laptop project:

| Dataset | Why not |
|---|---|
| WildfireSpreadTS (Gerard 2023) | One 48 GB zip; VIIRS at 375 m; no population layer |
| PyroStack (Kondur 2026) | A different target (12-hourly perimeters at 300 m); 31.7 GB |
| RCDA-Net data (Huang 2026) | Canada and Alaska at 50 m; the target is newly burned area |
| NDWS North America extension (Li & Rad 2024) | Overlaps the NDWS years, so it is not an independent test |

If a cross-dataset result is ever required, the cheapest credible option is this: run the no-learning baselines (persistence and the distance buffer) on one PyroStack year or on the RCDA-Net data. Both have fire ids.

## 8. Limitations to state

- Labels are 1 km daily Terra detections, with omissions.
- Tiles are selected on the outcome and centred on tomorrow's fire.
- Weather is 4 km and daily.
- Population is a static census proxy at 1 km.
- There is no geography and no time.
