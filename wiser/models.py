"""The scorers. Each gives every pixel of a crop a score: higher means more likely fire tomorrow.

- Persistence: tomorrow = today.
- Distance buffer: minus the distance to the nearest pixel burning today.
- Gradient-boosted trees on per-pixel features (scikit-learn, or LightGBM if installed).
- Tiny U-Net (about 470K parameters), also trained with today's fire mask as its only input.
"""
