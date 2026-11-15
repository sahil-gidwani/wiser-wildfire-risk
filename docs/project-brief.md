# Project brief

## Problem

Next-day wildfire models output a fire probability for each 1 km pixel. To act on it, someone has to choose a cutoff. Cutoffs picked by habit (0.5, or the one that maximises F1) come with no statement about how much fire is missed. On the NDWS benchmark, the only prior conformal study reports that a 0.5 cutoff misses 28–93 % of next-day fire pixels across its three models (Dayan 2026).

Some misses matter more than others:
- Homes in the wildland–urban interface grew from 30.8 to 43.4 million between 1990 and 2010 (Radeloff 2018).
- In California, interface areas held half of the destroyed buildings but only 2 % of the burned area (Kramer 2019).

So a useful map needs two things:
1. a checkable bound on missed fire;
2. a way to treat misses near people more strictly, with the cost in flagged area shown.

## Approach: WISER

WISER is a calibration layer that works on top of any model.

1. **Score.** A model (by default a small U-Net) gives each pixel of a 32 × 32 km crop a fire score.
2. **Calibrate.** On a held-out calibration set, the WISER rule picks the cutoff so that the expected miss rate is at most α (e.g. 10 %). This is an application of conformal risk control (Angelopoulos et al. 2024).
3. **Exposure-aware version.** Populated pixels get their own, stricter cutoff.
4. **Measure.** On a separate test set, record the realised miss rate and the share of the map that was flagged.

Details are in [method](method.md).

## Research questions

| | Question | Experiment ([evaluation](evaluation.md)) |
|---|---|---|
| RQ1 | **Does the promise hold?** Is the realised miss rate at or below α on held-out data? Does the prior pixel-pooled rule (Dayan 2026) keep its target? | 2 |
| RQ2 | **What does it cost?** How much of the map has to be flagged at a given α? Compared across a no-learning distance buffer, gradient-boosted trees and a U-Net. | 3 |
| RQ3 | **Where are the misses, and who is near them?** Are misses mostly new fire away from today's fire? Can misses in populated pixels be held to a stricter target, and for how much extra area? | 4 |

## Contributions

1. **One calibration rule** that controls the per-crop miss rate, a capped pooled miss rate and the miss rate in populated pixels. It comes with an exact solver and a check for whether there is enough calibration data. It follows directly from the conformal risk control theorem; we do not claim a new theorem.
2. **An honest evaluation on NDWS:**
   - the official split;
   - random crops, which remove a centring shortcut in the data;
   - validity checked by repeated re-splitting;
   - flagged area compared with simple baselines;
   - a look at where the misses fall.

   It includes a check of the prior pixel-pooled rule.
3. **An exposure-aware version.** It is compared with the obvious alternative, flagging every populated pixel, and its equity limits are stated.

## What success looks like

- **Valid.** Averaged over repeated re-splits, the realised miss rate is at or below α, within simulation error. On the official test split it lands near α; a single draw can sit slightly above by chance.
- **Useful.** The flagged area is well below 100 %, and the U-Net flags less than the distance buffer at the same α.
- **Exposure-aware.** The stricter cutoff for populated pixels meets its target while flagging less than "flag every populated pixel".

If one of these fails, that is still a result: we report it and explain why.

## Scope

| | Contents |
|---|---|
| **Core** | Data check; persistence, distance buffer, gradient-boosted trees, Tiny U-Net and a fire-only U-Net; calibration rules compared; validity checks; miss breakdown; exposure-aware version; demo |
| **If time** | Replicate the anchor paper; block re-splitting; a population-weighted single cutoff; separate cutoffs by fire size; a "with 90 % confidence" version; risk zones in the demo; a medium U-Net and more input ablations |
| **Out** | Other datasets; Swin-UNet, unless the instructor asks; evacuation claims; online or time-based methods (NDWS has no dates) |

## Deliverables

1. **Code** that reproduces every result from a fresh clone.
2. **A demo:** a gallery of test crops showing observed fire, flagged area and misses under WISER and the baselines.
3. **A final report.** Its format will be planned later.

## Constraints

- Laptops only (Apple M4, 16 GB RAM), with PyTorch on MPS or CPU.
- About six weeks.
- Three people: Jaiwin Mehta, Nikita Goswami and Sahil Gidwani.
- Nothing from the proposal is locked.

## Not goals

- Not an evacuation tool, and not a forecast of fire perimeters or structure loss.
- Not a new state-of-the-art model. Model quality matters only through flagged area.
- No claims about named fires: NDWS tiles have no coordinates or dates.
