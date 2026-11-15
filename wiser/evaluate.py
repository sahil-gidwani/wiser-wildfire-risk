"""Run the experiments in docs/evaluation.md, using saved scores only.

- Validity checks: random re-splits of calibration and test crops, and the official split.
- Cost: flagged area for each model at the same target.
- Where misses fall, and the populated-pixel comparison.

Output: outputs/results.csv and figures in outputs/figures/.

Run: python -m wiser.evaluate
"""
