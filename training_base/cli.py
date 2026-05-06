from __future__ import annotations

import argparse
import os

from .config import load_json, parse_project_config
from .data.datamodule import InstrumentDataModule
from .lightning_imports import ModelCheckpoint, WandbLogger, pl
from .models.system import DomainTransferSystem


def build_logger(cfg):
    if not cfg.logging.wandb.enabled:
        return True
    if WandbLogger is None:
        raise ImportError(
            "W&B logging was requested, but WandbLogger is unavailable. "
            "Install wandb and a Lightning version with WandbLogger support."
        )

    save_dir = None
    if cfg.trainer.save_dir:
        save_dir = os.path.join(cfg.trainer.save_dir, cfg.experiment.name)

    return WandbLogger(
        project=cfg.logging.wandb.project,
        entity=cfg.logging.wandb.entity,
        name=cfg.logging.wandb.name or cfg.experiment.name,
        save_dir=save_dir,
        mode=cfg.logging.wandb.mode,
        tags=cfg.logging.wandb.tags,
        log_model=cfg.logging.wandb.log_model,
    )


def build_trainer(cfg):
    checkpoint_dir = None
    if cfg.trainer.save_dir:
        checkpoint_dir = os.path.join(cfg.trainer.save_dir, cfg.experiment.name, "checkpoints")

    if cfg.data.val is not None:
        callbacks = [ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=1, monitor="val/loss", mode="min")]
    else:
        callbacks = [ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=1, monitor=None, save_last=True)]
    logger = build_logger(cfg)
    return pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        deterministic=cfg.trainer.deterministic,
        benchmark=cfg.trainer.benchmark,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        fast_dev_run=cfg.trainer.fast_dev_run,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a synthetic-vs-real audio classifier with Lightning.")
    parser.add_argument("--config", required=True, help="Path to a project JSON config")
    parser.add_argument("--test-only", action="store_true", help="Skip fit and only run trainer.test")
    parser.add_argument("--ckpt-path", default=None, help="Optional checkpoint path for test or resumed fit")
    args = parser.parse_args()


    # this is where we read the configuration file of the whole training and parse the 
    # json into the class objects that are defined inside config.py 
    cfg = parse_project_config(load_json(args.config))
    
    #seeding everything
    pl.seed_everything(cfg.trainer.seed, workers=True)

    # the following data module is to build the dataloader for training, validation and testing
    datamodule = InstrumentDataModule(cfg)
    
    #here we are building the model based on out configuration file 
    model = DomainTransferSystem(cfg)
    
    #again build our trainer based on the parameters defined in the configuration file
    trainer = build_trainer(cfg)

    if not args.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)

    if cfg.experiment.run_test_after_fit or args.test_only:
        if args.test_only:
            test_ckpt_path = args.ckpt_path
        elif cfg.data.val is not None:
            test_ckpt_path = "best"
        else:
            test_ckpt_path = "last"
        trainer.test(model, datamodule=datamodule, ckpt_path=test_ckpt_path)


if __name__ == "__main__":
    main()
