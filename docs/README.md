# WISER docs

**WISER** (Wildfire Intelligence with Spatial Evaluation and Risk-control) is a six-week course project for three people, run on laptops only.

**The idea.** A next-day wildfire model gives every 1 km pixel a fire score. To turn those scores into a map, someone has to pick a cutoff. The usual cutoffs (0.5, or the one with the best F1) say nothing about how much fire gets missed.

WISER picks the cutoff with *conformal risk control*. Using held-out data, it finds the cutoff that keeps the expected share of missed next-day fire pixels at or below a target, such as 10 %. A second version holds misses in populated pixels to a stricter target.

We test whether this promise holds on the NDWS dataset, and how much of the map has to be flagged to keep it.

**Status (2026-11-15):** planning is done; no code or experiments yet. The next step is the week-1 data check ([plan](plan.md)).

## Files

| File | Read it for |
|---|---|
| [project-brief](project-brief.md) | Problem, approach, research questions, scope, deliverables |
| [method](method.md) | The task, the promise and its limits, the WISER rule, the models, the exposure-aware version |
| [evaluation](evaluation.md) | Metrics, validity checks, the five experiments, how to word results |
| [plan](plan.md) | Timeline, who does what, decisions, open questions, risks |
| [data](data.md) | The NDWS dataset, its quirks, splits, preprocessing, the week-1 checks |
| [engineering](engineering.md) | Code layout, the files passed between steps, environment, pitfalls, licences |
| [proposal-review](proposal-review.md) | What held up in the original proposal, what didn't, and what changed |
| [background](background.md) | What the literature says |
| [glossary](glossary.md) | Terms and symbols in plain words |
| [references](references.md) | Sources |

New to the project? Read this page, then [project-brief](project-brief.md), [method](method.md) and [plan](plan.md). That takes about 20 minutes.

## Conventions

- Each fact lives in one file. Other files link to it instead of repeating it.
- Numbers we have not measured yet are marked *estimate*.
- Numbers from sources we have not re-checked are marked *(reported)*.
- To change the plan, update the decisions table in [plan](plan.md) first.
