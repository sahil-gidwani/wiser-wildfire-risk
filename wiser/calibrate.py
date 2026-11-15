"""Choose the cutoff.

- The WISER rule for the per-crop, pooled, capped and populated miss rates,
  solved exactly (docs/method.md, section 4).
- The comparison rules: a fixed 0.5 cutoff and the pixel-pooled rule of Dayan 2026.

Works only on saved scores, in float64 on the CPU.
"""
