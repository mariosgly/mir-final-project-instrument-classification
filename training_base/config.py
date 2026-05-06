from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _merge_dict(defaults: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(defaults)
    if override:
        merged.update(override)
    return merged


@dataclass
class TrainerConfig:
    seed: int = 0
    accelerator: str = "auto"
    devices: Any = "auto"
    strategy: str = "auto"
    precision: str = "bf16-mixed"
    max_epochs: int = 5
    log_every_n_steps: int = 25
    gradient_clip_val: float = 0.0
    accumulate_grad_batches: int = 1
    deterministic: bool = False
    benchmark: bool = True
    limit_train_batches: Any = 1.0
    limit_val_batches: Any = 1.0
    limit_test_batches: Any = 1.0
    num_sanity_val_steps: int = 0
    fast_dev_run: bool = False
    save_dir: Optional[str] = None


@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = "none"
    warmup_steps: int = 0


@dataclass
class LoaderConfig:
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = False


@dataclass
class SplitConfig:
    type: str
    domain: str
    input_kind: str = "waveform"
    params: Dict[str, Any] = field(default_factory=dict)
    augmentations: Dict[str, Any] = field(default_factory=dict)
    loader: Optional[LoaderConfig] = None


@dataclass
class DataConfig:
    task_type: str = "multiclass"
    num_classes: int = 10
    sample_rate: int = 16_000
    clip_duration_sec: float = 5.0
    label_names: List[str] = field(default_factory=list)
    threshold: float = 0.5
    relevance_threshold: float = 0.5
    min_activity_ratio: float = 0.1
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    train: Optional[SplitConfig] = None
    val: Optional[SplitConfig] = None
    test: Optional[SplitConfig] = None

    @property
    def clip_num_samples(self) -> int:
        return int(self.sample_rate * self.clip_duration_sec)


@dataclass
class EncoderConfig:
    type: str = "identity"
    input_kind: str = "embedding"
    output_dim: int = 128
    freeze: bool = True
    factory: Optional[str] = None
    factory_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    #lines below are for the mel cnn    
    sample_rate: int = 22050
    n_fft: int = 1024
    n_mels: int = 64


@dataclass
class ClassifierConfig:
    type: str = "linear"
    hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class ExperimentConfig:
    name: str = "synthetic_to_real"
    train_domain: str = "synthetic"
    eval_domain: str = "real"
    run_test_after_fit: bool = True


@dataclass
class WandbConfig:
    enabled: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None
    name: Optional[str] = None
    mode: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    log_model: bool = False


@dataclass
class ValDemoConfig:
    enabled: bool = True
    count: int = 0
    seed: Optional[int] = None
    every_n_epochs: int = 1


@dataclass
class LoggingConfig:
    wandb: WandbConfig = field(default_factory=WandbConfig)
    val_demos: ValDemoConfig = field(default_factory=ValDemoConfig)


@dataclass
class ProjectConfig:
    experiment: ExperimentConfig
    trainer: TrainerConfig
    optimizer: OptimizerConfig
    data: DataConfig
    encoder: EncoderConfig
    classifier: ClassifierConfig
    logging: LoggingConfig


def _parse_loader_config(cfg: Optional[Dict[str, Any]], fallback: Optional[LoaderConfig] = None) -> LoaderConfig:
    base = fallback if fallback is not None else LoaderConfig()
    merged = _merge_dict(base.__dict__, cfg)
    return LoaderConfig(**merged)


def _parse_split_config(cfg: Optional[Dict[str, Any]], loader_fallback: LoaderConfig) -> Optional[SplitConfig]:
    if cfg is None:
        return None
    payload = dict(cfg)
    payload["loader"] = _parse_loader_config(payload.get("loader"), fallback=loader_fallback)
    return SplitConfig(**payload)


def parse_project_config(raw_cfg: Dict[str, Any]) -> ProjectConfig:
    trainer = TrainerConfig(**raw_cfg.get("trainer", {}))
    optimizer = OptimizerConfig(**raw_cfg.get("optimizer", {}))
    base_loader = _parse_loader_config(raw_cfg.get("data", {}).get("loader"))

    data_section = raw_cfg.get("data", {})
    data = DataConfig(
        task_type=data_section.get("task_type", "multiclass"),
        num_classes=int(data_section.get("num_classes", 10)),
        sample_rate=int(data_section.get("sample_rate", 16_000)),
        clip_duration_sec=float(data_section.get("clip_duration_sec", 5.0)),
        label_names=list(data_section.get("label_names", [])),
        threshold=float(data_section.get("threshold", 0.5)),
        relevance_threshold=float(data_section.get("relevance_threshold", 0.5)),
        min_activity_ratio=float(data_section.get("min_activity_ratio", 0.1)),
        loader=base_loader,
        train=_parse_split_config(data_section.get("train"), base_loader),
        val=_parse_split_config(data_section.get("val"), base_loader),
        test=_parse_split_config(data_section.get("test"), base_loader),
    )

    experiment = ExperimentConfig(**raw_cfg.get("experiment", {}))
    encoder = EncoderConfig(**raw_cfg.get("encoder", {}))
    classifier = ClassifierConfig(**raw_cfg.get("classifier", {}))
    logging_section = raw_cfg.get("logging", {})
    logging = LoggingConfig(
        wandb=WandbConfig(**logging_section.get("wandb", {})),
        val_demos=ValDemoConfig(**logging_section.get("val_demos", {})),
    )

    if data.train is None:
        raise ValueError("Config must define data.train")
    if data.task_type not in {"multiclass", "multilabel"}:
        raise ValueError(f"Unsupported task_type: {data.task_type}")
    if encoder.input_kind != data.train.input_kind:
        raise ValueError(
            "Encoder input_kind must match the training split input_kind "
            f"({encoder.input_kind} != {data.train.input_kind})"
        )

    return ProjectConfig(
        experiment=experiment,
        trainer=trainer,
        optimizer=optimizer,
        data=data,
        encoder=encoder,
        classifier=classifier,
        logging=logging,
    )


def resolve_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((Path(base_dir) / candidate).resolve())
