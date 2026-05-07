from __future__ import annotations

from typing import Any, Dict

import torch


class IdentityAugmenter:
    """Default no-op augmenter."""

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform.float()


from pedalboard import Pedalboard, Reverb, Gain
import numpy as np

class ReverbNoiseAugmenter:
    """Pedalboard-based reverb + gain + background noise augmentation."""

    def __init__(self, noise_std: float = 0.0025):
        self.noise_std = noise_std

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        sample_rate = 16000  # must match your config
        arr = waveform.numpy().astype(np.float32)

        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False

        board = Pedalboard([
            Reverb(
                room_size=np.random.uniform(0.1, 0.6),
                damping=np.random.uniform(0.3, 0.7),
                wet_level=np.random.uniform(0.1, 0.4),
                dry_level=np.random.uniform(0.6, 0.9),
                width=np.random.uniform(0.5, 1.0),
            ),
            Gain(gain_db=np.random.uniform(-3.0, 3.0)),
        ])

        effected = board(arr, sample_rate)
        noise = np.random.normal(0, self.noise_std, effected.shape).astype(np.float32)
        effected = np.clip(effected + noise, -1.0, 1.0)

        if squeeze:
            effected = effected.squeeze(0)
        return torch.from_numpy(effected).float()


def build_augmenter(cfg):
    if cfg is None:
        return IdentityAugmenter()
    aug_type = cfg.get("type")
    if aug_type is None:          # ← handles {"type": null} from JSON
        return IdentityAugmenter()
    if aug_type == "reverb_noise":
        return ReverbNoiseAugmenter(noise_std=cfg.get("noise_std", 0.0025))
    raise ValueError(f"Unknown augmenter type: {aug_type}")
