# Plan

The project runs for six weeks, with three people, on laptops only. Dates are indicative until the due date is confirmed.

## Who does what

| Area | Owns | Person |
|---|---|---|
| **Data and baselines** | Conversion, data check, crops, persistence, distance buffer, gradient-boosted trees | to assign |
| **Models and demo** | U-Nets, training runs, saved scores, demo gallery | to assign |
| **Calibration and evaluation** | Calibration code and its tests, metrics, validity checks, experiments 2–4, results table, figures | to assign |

The areas hand over files, not code ([engineering](engineering.md)): converted data → saved scores for each model → one results table. The distance buffer produces scores on day 2, so calibration work can start before any network is trained.

## Timeline

| Week | Dates | Work | Checkpoint |
|---|---|---|---|
| 1 | 16–22 Nov | Download and convert; data check; calibration code with synthetic tests; distance buffer end to end; U-Net smoke run | Data numbers in; persistence reproduced; calibration tests pass |
| 2 | 23–29 Nov | Train the trees, the Tiny U-Net (with seeds) and the fire-only U-Net; teammate review of the WISER rule; fix the settings listed in [evaluation](evaluation.md) | All core scores saved |
| 3 | 30 Nov–6 Dec | Experiments 2 and 3 | Validity and cost results |
| 4 | 7–13 Dec | Experiment 4; demo gallery | Core results done |
| 5 | 14–20 Dec | Finish the demo; "if time" extras; run everything from a fresh clone | Demo and code reproducible |
| 6 | 21–27 Dec | Write-up and polish | Submit |

Hold a 30-minute weekly sync: progress, blockers, and decisions to write down.

**Calendar:** if US holidays apply, week 2 includes 26 Nov and week 6 includes 25 Dec. Treat week 5 as the last full working week.

## If we fall behind

Cut in this order:
1. the optional demo viewer;
2. the "if time" extras;
3. the extra U-Net seeds;
4. the gradient-boosted trees, but only if they cause trouble.

Never cut:
- the data check;
- the persistence and buffer baselines;
- the validity checks;
- the exposure comparison (shrink it if the data is thin);
- the static demo.

## Decisions

| # | Decision | Why | Status |
|---|---|---|---|
| 1 | Laptops only; six weeks; three people | Course constraints | Confirmed by the user (2026-11-14) |
| 2 | Nothing in the proposal is locked | It was an initial draft | Confirmed by the user (2026-11-14) |
| 3 | Deliverables: code, demo, final report | Course requirements | Confirmed by the user (2026-11-15); report format planned later |
| 4 | The headline is what a calibrated next-day fire map can honestly promise, and at what flagged-area cost. Not "evacuation maps". | Labels and resolution can't support evacuation claims | Proposed |
| 5 | NDWS only | Other datasets are too big or don't line up | Proposed |
| 6 | Official split: validation is carved from train, calibration is the official eval split, test is the official test split | Avoids the anchor paper's leakage and data reuse | Proposed |
| 7 | Random 32 × 32 crops | Tiles are centred on tomorrow's fire | Proposed; confirm in the data check |
| 8 | The crop, not the pixel, is the calibration unit | Pixels in a crop are correlated | Proposed |
| 9 | Main rule: WISER capped, starting with `c` = 32. Standard per-crop CRC is the baseline. | Balances small and big fires, with enough data for a small margin | Proposed |
| 10 | Population decides which misses get a stricter target. It is not added as a new input. | It is already an NDWS input, with no predictive value | Proposed |
| 11 | Persistence and the distance buffer appear in every comparison | Today's fire carries most of the signal | Proposed |
| 12 | Models: trees, Tiny U-Net (main), fire-only U-Net. No Swin unless the instructor asks; no ResGNN. | Swin evidence; ResGNN added nothing | Proposed |
| 13 | Write our own code; copy nothing from the anchor paper's repository | It has no licence | Proposed |
| 14 | The wording rules in [evaluation](evaluation.md) apply to the code README and the demo | Avoids overclaiming | Proposed |

## Open questions

| Question | Why it matters | By |
|---|---|---|
| For the instructor: the due date; how code and demo are submitted; the AI-tool policy; whether the change of scope from the proposal is acceptable | Plan and grading | Week 1 |
| Who owns each area? | Accountability | Week 1 |
| Data-check numbers: fire share (about 1 %, versus 5 % in the anchor paper), fire pixels per crop, centring, duplicates, populated fire pixels | These settle the crops, the cap `c` and the size of the exposure experiment | Week 1 |
| Wind direction under flips and rotations: rotate (u, v) components, or skip those augmentations? | Correct augmentation | Week 1 |
| Commit `proposal.pdf` (12.9 MB) to git? | Repository size | Week 1 |
| τ and α_pop for the exposure version | They must be fixed on validation data before calibration | Week 2 |
| Has anyone published per-crop or exposure-aware CRC for wildfire maps since November 2026? | Novelty wording | Week 2 |

## Risks

| Risk | Warning sign | Response |
|---|---|---|
| Per-crop control at α = 5 % flags almost the whole map | Flagged area above 50 % | Lead with α = 10 % and the capped rate. The cost is a finding in itself. |
| Too few populated fire pixels | Fewer than about 200 calibration crops with populated fire | Shrink the exposure experiment and report the shortage |
| Laptop training too slow | Runs take more than three times the estimate | Use smaller models, then free cloud notebooks |
| Bug in the calibration code | The sanity check fails | Synthetic tests, a brute-force check, a MAPIE cross-check, peer review |
| Overclaiming | Banned words in the README or demo | The wording table in [evaluation](evaluation.md) |
| The instructor expects the original plan (Swin, other datasets) | Feedback on scope | Ask in week 1; add a small Swin run if required |
| The pixel-pooled rule looks fine on the full calibration set | Its realised miss rate stays near α | The story doesn't depend on it failing. It still has no guarantee, and small calibration sets may show the gap. |
| Scope creep | Extras started before the core results | Follow the cut order |
