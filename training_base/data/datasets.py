from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import audio
import torch
from torch.utils.data import Dataset
import os
import yaml
import random
import numpy as np
import torch
import torchaudio
from pathlib import Path
from pedalboard import Pedalboard, Reverb, Gain
from pedalboard.io import AudioFile

# General MIDI program number -> instrument family mapping
# May have to modify based on the classes we choose
PROGRAM_TO_CLASS = {
    range(0, 8): 0,  # Piano
    range(8, 16): 1,  # Chromatic Percussion
    range(16, 24): 2,  # Organ
    range(24, 32): 3,  # Guitar
    range(32, 40): 4,  # Bass
    range(40, 48): 5,  # Strings
    range(48, 56): 6,  # Ensemble
    range(56, 64): 7,  # Brass
    range(64, 72): 8,  # Reed
    range(72, 80): 9,  # Pipe
    range(80, 88): 10,  # Synth Lead
    range(88, 96): 11,  # Synth Pad
    range(96, 104): 12,  # Synth Effects
    range(104, 112): 13,  # Ethnic
    range(112, 120): 14,  # Percussive
    range(120, 128): 15,  # Sound Effects
}


def program_to_class_id(program: int) -> int:
    for r, cls in PROGRAM_TO_CLASS.items():
        if program in r:
            return cls
    return -1


class AudioAugmenter:
    """
    Applies realistic-sounding augmentations to a waveform using Pedalboard.
    All effects are randomized within reasonable ranges each call
    """

    def __init__(self, sample_rate: int, noise_std: float = 0.005):
        self.sample_rate = sample_rate
        self.noise_std = noise_std

    def apply(self, waveform: np.ndarray) -> np.ndarray:
        """
        waveform: np.ndarray of shape (channels, samples) or (samples,)
        returns: np.ndarray of the same shape
        """
        import numpy as np
        from pedalboard import (
            Pedalboard,
            Reverb,
            Gain,
            Compressor,
            HighpassFilter,
            LowpassFilter,
            Distortion,
        )

        # Ensure shape = (channels, samples)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False

        waveform = waveform.astype(np.float32)

        board = Pedalboard([
            # 🎛️ EQ (simulate mic / recording chain)
            HighpassFilter(
                cutoff_frequency_hz=np.random.uniform(40, 120)
            ),
            LowpassFilter(
                cutoff_frequency_hz=np.random.uniform(6000, 12000)
            ),

            # 🎚️ Compression (very important for realism)
            Compressor(
                threshold_db=np.random.uniform(-24, -12),
                ratio=np.random.uniform(2.0, 5.0),
                attack_ms=np.random.uniform(5, 30),
                release_ms=np.random.uniform(50, 200),
            ),

            # 🌊 Reverb (space)
            Reverb(
                room_size=np.random.uniform(0.1, 0.6),
                damping=np.random.uniform(0.3, 0.7),
                wet_level=np.random.uniform(0.1, 0.4),
                dry_level=np.random.uniform(0.6, 0.9),
                width=np.random.uniform(0.5, 1.0),
            ),

            # 🔥 Mild saturation (subtle realism)
            Distortion(
                drive_db=np.random.uniform(1.0, 6.0)
            ),

            # 🔊 Gain variation
            Gain(
                gain_db=np.random.uniform(-3.0, 3.0)
            ),
        ])

        effected = board(waveform, self.sample_rate)

        # 🔉 Add light noise
        noise = np.random.normal(0, self.noise_std, effected.shape).astype(np.float32)
        effected += noise

        # 🔒 Clip to valid range
        effected = np.clip(effected, -1.0, 1.0)

        return effected.squeeze(0) if squeeze else effected


class SlakhDataset(Dataset):
    """
    PyTorch Dataset for the Slakh2100 dataset.

    Supports instrument classification.
        - "instrument_classification": returns (audio_clip, label) where label is
          the instrument class index of a single stem.

    Set manipulate=True to apply random reverb, gain variation, and background
    noise via Pedalboard — useful for training more robust models.

    Expected directory layout:
        root/
            Track00001/
                metadata.yaml
                mix.wav
                stems/
                    S00.wav
                    S01.wav
                    ...
    """

    def __init__(
            self,
            split_cfg: Any,
            task_type: str,
            num_classes: int,
            sample_rate: int,
            clip_num_samples: int,
            train_mode: bool,
            manipulate: bool = False,
            noise_std: float = 0.005,
    ) -> None:
        self.task_type = task_type
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.clip_num_samples = clip_num_samples
        self.train_mode = train_mode
        self.manipulate = manipulate
        self.augmenter = AudioAugmenter(sample_rate, noise_std) if manipulate else None

        root = Path(split_cfg["root"])
        split = split_cfg.get("split", "train")

        split_dir_map = {
            "train": "train",
            "validation": "validation",
            "val": "validation",
            "test": "test",
            "none": "",
        }
        split_dir = root / split_dir_map[split]
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.samples: List[Dict] = []
        self._build_index(split_dir)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self, split_dir: Path) -> None:
        """Walk split_dir and collect one entry per usable stem for classification."""

        for track_dir in sorted(split_dir.iterdir()):
            if not track_dir.is_dir():
                continue

            meta_path = track_dir / "metadata.yaml"
            mix_path = track_dir / "mix.wav"
            stems_dir = track_dir / "stems"

            if not (meta_path.exists() and mix_path.exists() and stems_dir.exists()):
                continue

            with open(meta_path) as f:
                meta = yaml.safe_load(f)

            stems_meta = meta.get("stems", {})

            if self.task_type == "instrument_classification":
                for stem_id, stem_info in stems_meta.items():
                    stem_path = stems_dir / f"{stem_id}.wav"
                    if not stem_path.exists():
                        continue
                    if stem_info.get("is_drum", False):
                        continue
                    program = stem_info.get("program_num", -1)
                    class_id = program_to_class_id(program)
                    if class_id < 0 or class_id >= self.num_classes:
                        continue
                    self.samples.append({
                        "audio_path": stem_path,
                        "label": class_id,
                    })
            else:
                raise ValueError(f"Unknown task_type: {self.task_type!r}")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]

        if self.task_type == "instrument_classification":
            excerpt = audio.load_waveform(item["audio_path"], self.sample_rate)
            excerpt = audio.trim_or_pad(excerpt, self.clip_num_samples, self.train_mode)

            if self.manipulate and self.augmenter is not None:
                excerpt = self.augmenter.apply(excerpt.numpy())
                excerpt = torch.from_numpy(excerpt).float()

            label = torch.tensor(item["label"], dtype=torch.long)
            return excerpt, label


class OpenMicDataset(Dataset):
    def __init__(
            self,
            split_cfg: Any,
            task_type: str,
            num_classes: int,
            sample_rate: int,
            clip_num_samples: int,
            train_mode: bool,
    ) -> None:
        raise NotImplementedError(
            "We need to implement the Dataset object for the openmic dataset here"
        )


def build_dataset(
        split_cfg: Any,
        task_type: str,
        num_classes: int,
        sample_rate: int,
        clip_num_samples: int,
        train_mode: bool,
) -> Dataset:
    ds_type = split_cfg.type

    if ds_type == "slakh":
        return SlakhDataset(
            split_cfg=split_cfg,
            task_type=task_type,
            num_classes=num_classes,
            sample_rate=sample_rate,
            clip_num_samples=clip_num_samples,
            train_mode=train_mode,
        )

    if ds_type == "openmic":
        return OpenMicDataset(
            split_cfg=split_cfg,
            task_type=task_type,
            num_classes=num_classes,
            sample_rate=sample_rate,
            clip_num_samples=clip_num_samples,
            train_mode=train_mode,
        )

    raise ValueError(f"Unsupported dataset type: {ds_type}")