#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_base.cli import build_trainer
from training_base.config import load_json, parse_project_config
from training_base.data.datamodule import InstrumentDataModule
from training_base.lightning_imports import pl
from training_base.models.system import DomainTransferSystem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved checkpoint on the configured test split using the same test path as training."
    )
    parser.add_argument("--config", required=True, help="Project config JSON")
    parser.add_argument("--ckpt-path", required=True, help="Checkpoint path to evaluate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = parse_project_config(load_json(args.config))
    pl.seed_everything(cfg.trainer.seed, workers=True)

    datamodule = InstrumentDataModule(cfg)
    model = DomainTransferSystem(cfg)
    trainer = build_trainer(cfg)

    trainer.test(model, datamodule=datamodule, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
