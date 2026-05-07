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
        sample_rate = 16000  # must match your config to receive the waveform
        arr = waveform.numpy().astype(np.float32)  # Convert to numpy

        if arr.ndim == 1:  # add a channel dimension
            arr = arr[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False

        # Build the effect chain
        board = Pedalboard([
            #  Simulate a room
            Reverb(
                room_size=np.random.uniform(0.1, 0.6),  # how large the room sounds
                damping=np.random.uniform(0.3, 0.7),  # how quickly high frequencies decay
                wet_level=np.random.uniform(0.1, 0.4),  # how much reverb is added
                dry_level=np.random.uniform(0.6, 0.9),  # how much of the original signal is preserved
                width=np.random.uniform(0.5, 1.0),  # controls stereo spread
            ),
            Gain(gain_db=np.random.uniform(-3.0, 3.0)),  # randomly shift the volume
        ])

        # Apply the effects
        effected = board(arr, sample_rate)

        # Add background noise
        noise = np.random.normal(0, self.noise_std, effected.shape).astype(np.float32)

        # Confine the amplitude to be between -1.0 and 1.0
        effected = np.clip(effected + noise, -1.0, 1.0)

        if squeeze:
            effected = effected.squeeze(0)  # back to (num_samples,)
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
