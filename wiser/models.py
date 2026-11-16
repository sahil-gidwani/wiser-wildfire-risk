"""The scorers. Each gives every pixel of a crop a score: higher means more likely fire tomorrow.

- Persistence: tomorrow = today.
- Distance buffers: minus the distance to the nearest pixel burning today, measured either inside
  the window only (what the learned models see) or on the whole tile.
- Gradient-boosted trees on per-pixel features (scikit-learn, or LightGBM if installed). Not written yet.
- Tiny U-Net (about 470K parameters), also trained with today's fire mask as its only input.

Scores work like logits: a cutoff of 0 is the "probability 0.5" habit. For persistence that flags
exactly the pixels burning today, and so do the buffers (distance <= 0).

The baselines take today's fire on whole tiles [n, 64, 64] plus each tile's window position, and
return scores for the 32 x 32 windows [n, 32, 32].
"""
import numpy as np
import torch
from torch import nn

from wiser.data import crop, distance_to_fire

NO_FIRE_DISTANCE = 1000.0  # km; stands in for "no fire in sight", far beyond a tile's 90 km diagonal


def windows(tiles, rows, cols):
    """Cut each tile's window: tiles [n, 64, 64] -> [n, 32, 32]."""
    return np.stack([crop(t, r, c) for t, r, c in zip(tiles, rows, cols)])


def persistence(fire_today, rows, cols):
    """Tomorrow = today: +1 where fire burns today, -1 everywhere else."""
    return np.where(windows(fire_today, rows, cols) == 1, 1.0, -1.0).astype(np.float32)


def buffer_window(fire_today, rows, cols):
    """Minus the distance in km to the nearest pixel burning today inside the window.

    A cutoff of -r flags every pixel within r km of today's fire, so the cutoff WISER picks is a radius.
    A window with no fire today gives every pixel the same score.
    """
    distance = np.minimum(distance_to_fire(windows(fire_today, rows, cols)), NO_FIRE_DISTANCE)
    return (-distance).astype(np.float32)


def buffer_tile(fire_today, rows, cols):
    """Like buffer_window, but the distance is measured on the whole tile, so fire outside the window counts."""
    distance = np.minimum(distance_to_fire(fire_today), NO_FIRE_DISTANCE)
    return (-windows(distance, rows, cols)).astype(np.float32)


BASELINES = {"persistence": persistence, "buffer_window": buffer_window, "buffer_tile": buffer_tile}


def double_conv(c_in, c_out):
    """Two 3 x 3 convolutions, each followed by batch normalisation and ReLU."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    """A small U-Net, about 470K parameters.

    Down: 32 x 32 with 32 channels -> 16 x 16 with 64 -> 8 x 8 with 128 in the middle.
    Up: back to 16 x 16 and 32 x 32, joining the matching down-path layers (skip connections).
    Output: one logit per pixel, [n, 32, 32].
    """

    def __init__(self, in_channels=12):
        super().__init__()
        self.enc1, self.enc2, self.mid = double_conv(in_channels, 32), double_conv(32, 64), double_conv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, stride=2), double_conv(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, stride=2), double_conv(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        m = self.mid(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(m), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1).squeeze(1)


# The learned models: which input layers each one sees.
NETWORKS = {
    "unet": ("elevation", "pdsi", "NDVI", "pr", "sph", "th", "tmmn", "tmmx", "vs", "erc", "population", "PrevFireMask"),
    "unet_fire": ("PrevFireMask",),
}
