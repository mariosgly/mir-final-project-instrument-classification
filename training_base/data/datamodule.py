from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from ..lightning_imports import pl
from .datasets import build_dataset


def _basic_collate(batch: Any) -> Dict[str, Any]:
    inputs = [item["inputs"] for item in batch]
    targets = [item["target"] for item in batch]
    domains = [item["domain"] for item in batch]
    lengths = torch.tensor([int(sample.shape[-1]) for sample in inputs], dtype=torch.long)

    if inputs[0].ndim == 1 and any(sample.shape[-1] != inputs[0].shape[-1] for sample in inputs):
        stacked_inputs = pad_sequence(inputs, batch_first=True)
    else:
        stacked_inputs = torch.stack(inputs)

    first_target = targets[0]
    if isinstance(first_target, torch.Tensor) and first_target.ndim > 0:
        stacked_targets = torch.stack(targets).float()
    else:
        stacked_targets = torch.stack([torch.as_tensor(target, dtype=torch.long) for target in targets])

    return {
        "inputs": stacked_inputs,
        "targets": stacked_targets,
        "lengths": lengths,
        "domains": domains,
    }


def collate_slakh_batch(batch: Any) -> Dict[str, Any]:
    return _basic_collate(batch)


def collate_openmic_batch(batch: Any) -> Dict[str, Any]:
    return _basic_collate(batch)


class InstrumentDataModule(pl.LightningDataModule):
    """Thin DataModule that mostly wires datasets into dataloaders."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        data_cfg = self.cfg.data
        if stage in (None, "fit"):
            self.train_dataset = build_dataset(
                data_cfg.train,
                task_type=data_cfg.task_type,
                num_classes=data_cfg.num_classes,
                sample_rate=data_cfg.sample_rate,
                clip_num_samples=data_cfg.clip_num_samples,
                train_mode=True,
                min_activity_ratio=data_cfg.min_activity_ratio,
                label_names=data_cfg.label_names,
                relevance_threshold=data_cfg.relevance_threshold,
            )
            if data_cfg.val is not None:
                self.val_dataset = build_dataset(
                    data_cfg.val,
                    task_type=data_cfg.task_type,
                    num_classes=data_cfg.num_classes,
                    sample_rate=data_cfg.sample_rate,
                    clip_num_samples=data_cfg.clip_num_samples,
                    train_mode=False,
                    min_activity_ratio=data_cfg.min_activity_ratio,
                    label_names=data_cfg.label_names,
                    relevance_threshold=data_cfg.relevance_threshold,
                )
        if stage in (None, "test"):
            if data_cfg.test is not None:
                self.test_dataset = build_dataset(
                    data_cfg.test,
                    task_type=data_cfg.task_type,
                    num_classes=data_cfg.num_classes,
                    sample_rate=data_cfg.sample_rate,
                    clip_num_samples=data_cfg.clip_num_samples,
                    train_mode=False,
                    min_activity_ratio=data_cfg.min_activity_ratio,
                    label_names=data_cfg.label_names,
                    relevance_threshold=data_cfg.relevance_threshold,
                )

    def train_dataloader(self) -> DataLoader:
        return self._build_loader(self.train_dataset, self.cfg.data.train, shuffle=True, is_train=True)

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return self._build_loader(self.val_dataset, self.cfg.data.val, shuffle=False, is_train=False)

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return self._build_loader(self.test_dataset, self.cfg.data.test, shuffle=False, is_train=False)

    def _build_loader(self, dataset: Any, split_cfg: Any, shuffle: bool, is_train: bool) -> DataLoader:
        loader_cfg = split_cfg.loader
        persistent_workers = loader_cfg.persistent_workers and loader_cfg.num_workers > 0
        return DataLoader(
            dataset,
            batch_size=loader_cfg.batch_size,
            shuffle=shuffle,
            num_workers=loader_cfg.num_workers,
            pin_memory=loader_cfg.pin_memory,
            persistent_workers=persistent_workers,
            drop_last=loader_cfg.drop_last if shuffle else False,
            collate_fn=self._get_collate_fn(split_cfg.type),
        )

    def _get_collate_fn(self, dataset_type: str):
        if dataset_type == "slakh":
            return collate_slakh_batch
        if dataset_type == "openmic":
            return collate_openmic_batch
        raise ValueError(f"Unsupported dataset type for collate selection: {dataset_type}")
