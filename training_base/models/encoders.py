from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

import torch
from torch import nn


def _freeze_module(module: nn.Module, freeze: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = not freeze


def _load_factory(path: str):
    module_name, attr_name = path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


class BaseEncoder(nn.Module):
    def __init__(self, output_dim: int, input_kind: str) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_kind = input_kind

    def forward(self, inputs: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class IdentityEncoder(BaseEncoder):
    def __init__(self, output_dim: int) -> None:
        super().__init__(output_dim=output_dim, input_kind="embedding")

    def forward(self, inputs: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        return inputs.float()


class ConvWaveformEncoder(BaseEncoder):
    """
    Tiny example waveform encoder.
    This is only here so the scaffold can run without a full pretrained model.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__(output_dim=output_dim, input_kind="waveform")
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
        )
        self.proj = nn.Linear(128, output_dim)

    def forward(self, inputs: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = inputs.float().unsqueeze(1)
        feats = self.net(x)
        pooled = feats.mean(dim=-1)
        return self.proj(pooled)


import torchaudio

class MelSpectrogramEncoder(BaseEncoder):
    """
    A strong baseline 2D CNN operating on log-Mel Spectrograms.
    Similar in architecture to VGGish or PANNs CNN10.
    """
    def __init__(
        self, 
        output_dim: int, 
        sample_rate: int = 22050, 
        n_fft: int = 1024, 
        hop_length: int = 256, 
        n_mels: int = 64
    ) -> None:
        super().__init__(output_dim=output_dim, input_kind="waveform")
        
        # 1. On-the-fly Spectrogram Extraction
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=50.0,
            f_max=sample_rate / 2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        
        # 2. 2D CNN Backbone
        # Input shape: [Batch, 1, n_mels, time_frames]
        self.features = nn.Sequential(
            self._conv_block(1, 32, pool=True),      # [B, 32, n_mels/2, time/2]
            self._conv_block(32, 64, pool=True),     # [B, 64, n_mels/4, time/4]
            self._conv_block(64, 128, pool=True),    # [B, 128, n_mels/8, time/8]
            self._conv_block(128, 256, pool=True),   # [B, 256, n_mels/16, time/16]
            self._conv_block(256, 512, pool=False),  # [B, 512, n_mels/16, time/16]
        )
        
        # 3. Projection Head
        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )

    def _conv_block(self, in_channels: int, out_channels: int, pool: bool) -> nn.Module:
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # inputs shape: [Batch, Samples]
        
        # 1. Create Spectrogram
        x = self.mel_transform(inputs)          # [Batch, n_mels, time]
        x = self.amplitude_to_db(x)             # Log-mel scale [Batch, n_mels, time]
        
        # 2. Per-Instance Z-Score Normalization
        # We calculate the mean and std for EACH item in the batch independently
        # Keep dims=True so we can broadcast the subtraction/division back to the original shape
        mean = x.mean(dim=[1, 2], keepdim=True)
        std = x.std(dim=[1, 2], keepdim=True)
        
        # Add a tiny epsilon (1e-5) to prevent division by zero in pure silence
        x = (x - mean) / (std + 1e-5)
        
        # Add channel dimension for 2D Conv
        x = x.unsqueeze(1)                      # [Batch, 1, n_mels, time]
        
        # 3. Extract features
        x = self.features(x)                    # [Batch, 512, freq, time]
        
        # 4. Global Pooling (mean across time and frequency)
        x = x.mean(dim=[2, 3])                  # [Batch, 512]
        
        # 5. Project
        return self.proj(x)                     # [Batch, output_dim]

class ExternalEncoder(BaseEncoder):
    """
    Adapter for encoders created elsewhere, e.g. CLAP or AudioMAE.

    Expected factory contract:
    - config supplies encoder.factory = "your_module:build_encoder"
    - the factory returns an nn.Module
    - the module forward accepts either:
      1. forward(inputs)
      2. forward(inputs, lengths=lengths)
    """

    def __init__(
        self,
        output_dim: int,
        input_kind: str,
        factory: str,
        factory_kwargs: Optional[Dict[str, Any]] = None,
        freeze: bool = True,
    ) -> None:
        super().__init__(output_dim=output_dim, input_kind=input_kind)
        builder = _load_factory(factory)
        self.module = builder(**(factory_kwargs or {}))
        if not isinstance(self.module, nn.Module):
            raise TypeError(f"Encoder factory {factory} did not return an nn.Module")
        _freeze_module(self.module, freeze)

    def forward(self, inputs: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        try:
            outputs = self.module(inputs, lengths=lengths)
        except TypeError:
            outputs = self.module(inputs)
        if isinstance(outputs, dict):
            for key in ("embedding", "embeddings", "x"):
                if key in outputs:
                    outputs = outputs[key]
                    break
        return outputs.float()


def build_encoder(cfg: Any) -> BaseEncoder:
    #here depending on our configuration file we are building the corresponding encoder
    if cfg.type == "identity":
        return IdentityEncoder(output_dim=cfg.output_dim)
    if cfg.type == "conv_waveform":
        return ConvWaveformEncoder(output_dim=cfg.output_dim)
    if cfg.type == "mel_cnn": 
        return MelSpectrogramEncoder(
            output_dim=cfg.output_dim,
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            n_mels=cfg.n_mels
        )
    if cfg.type == "external":
        if not cfg.factory:
            raise ValueError("External encoder requires encoder.factory")
        return ExternalEncoder(
            output_dim=cfg.output_dim,
            input_kind=cfg.input_kind,
            factory=cfg.factory,
            factory_kwargs=cfg.factory_kwargs,
            freeze=cfg.freeze,
        )
    raise ValueError(f"Unsupported encoder type: {cfg.type}")
