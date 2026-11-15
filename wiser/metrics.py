"""Measure a cutoff on test crops.

- Realised miss rates: per-crop, pooled, capped, populated and new fire.
- Flagged area: flagged valid pixels as a share of all valid pixels.
- Average precision, for how well a model ranks pixels.

Unknown pixels (label -1) never count. See docs/evaluation.md.
"""
