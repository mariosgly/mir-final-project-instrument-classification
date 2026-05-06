from __future__ import annotations

try:  # pragma: no cover - depends on installed package
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
except ImportError:  # pragma: no cover - depends on installed package
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    try:
        from pytorch_lightning.loggers import WandbLogger
    except ImportError:  # pragma: no cover - depends on installed package
        WandbLogger = None

try:  # pragma: no cover - depends on installed package
    import wandb
except ImportError:  # pragma: no cover - depends on installed package
    wandb = None

__all__ = ["pl", "ModelCheckpoint", "WandbLogger", "wandb"]
