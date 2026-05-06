from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from ..evaluation.metrics import compute_epoch_metrics, format_test_report, init_metric_state, update_metric_state
from ..lightning_imports import WandbLogger, pl, wandb
from .classifiers import build_classifier
from .encoders import build_encoder


class DomainTransferSystem(pl.LightningModule):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        #building the encoder here
        self.encoder = build_encoder(cfg.encoder)
        #building what will be on top of the encoder
        self.classifier = build_classifier(cfg.classifier, self.encoder.output_dim, cfg.data.num_classes)
        self.task_type = cfg.data.task_type
        #threshold is being used for predicting or not predicting a specific instrument
        self.threshold = cfg.data.threshold
        self.optimizer_cfg = cfg.optimizer
        self.val_state: Dict[str, torch.Tensor] = {}
        self.test_state: Dict[str, torch.Tensor] = {}
        self._val_demo_indices: Optional[List[int]] = None
        self._val_demo_lookup: set[int] = set()
        self._val_demo_rows: List[Dict[str, Any]] = []
        self._val_items_seen = 0
        self.save_hyperparameters(ignore=["cfg"])

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        #this is where we pass the inputs inside the ecoder and then into the classification head
        embeddings = self.encoder(inputs, lengths=lengths)
        return self.classifier(embeddings)

    def training_step(self, batch, batch_idx):
        #the shared step is to calculate the loss function
        logits, loss = self._shared_step(batch)
        batch_size = batch["targets"].shape[0]
        # Step loss is useful for debugging, but with shuffled random crops it is
        # naturally noisy. We also log the epoch-average so convergence is easier to read.
        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=False, batch_size=batch_size)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_state = init_metric_state(self.task_type, self.device, self.cfg.data.num_classes)
        self._val_demo_rows = []
        self._val_items_seen = 0
        self._maybe_initialize_val_demo_indices()

    def validation_step(self, batch, batch_idx):
        logits, loss = self._shared_step(batch)
        update_metric_state(self.val_state, self.task_type, logits, batch["targets"], loss, self.threshold)
        self._collect_validation_demos(batch, logits)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics(self.val_state, "val")
        self._log_validation_demos()

    def on_test_epoch_start(self) -> None:
        self.test_state = init_metric_state(self.task_type, self.device, self.cfg.data.num_classes)

    def test_step(self, batch, batch_idx):
        logits, loss = self._shared_step(batch)
        update_metric_state(self.test_state, self.task_type, logits, batch["targets"], loss, self.threshold)

    def on_test_epoch_end(self) -> None:
        self._print_test_report(self.test_state)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.optimizer_cfg.lr, weight_decay=self.optimizer_cfg.weight_decay)
        if self.optimizer_cfg.scheduler == "none":
            return optimizer
        if self.optimizer_cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.trainer.max_epochs),
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        raise ValueError(f"Unsupported scheduler: {self.optimizer_cfg.scheduler}")

    def _shared_step(self, batch):
        #Shared forward + loss path. Good place to customize task logic.
        logits = self(batch["inputs"], batch.get("lengths"))
        if self.task_type == "multiclass":
            targets = batch["targets"].long()
            loss = F.cross_entropy(logits, targets)
            return logits, loss

        targets = batch["targets"].float()
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return logits, loss

    def _log_epoch_metrics(self, state: Dict[str, torch.Tensor], prefix: str) -> None:
        if not state:
            return
        global_state = self._reduce_metric_state(state)
        metrics = compute_epoch_metrics(
            self.task_type,
            global_state,
            label_names=self.cfg.data.label_names,
        )
        metrics = self._select_logged_metrics(metrics, prefix)
        if not metrics:
            return
        self.log_dict(
            {f"{prefix}/{name}": value for name, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=(prefix == "val"),
            sync_dist=False,
        )

    def _print_test_report(self, state: Dict[str, torch.Tensor]) -> None:
        if not getattr(self.trainer, "is_global_zero", True):
            return
        if not self.cfg.data.label_names:
            return
        global_state = self._reduce_metric_state(state)
        report = format_test_report(
            self.task_type,
            global_state,
            label_names=self.cfg.data.label_names,
        )
        self.print("")
        self.print(report)

    def _reduce_metric_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # We accumulate metric counts locally in each process, then pack everything
        # into one vector so distributed reduction works for both scalar and vector
        # metrics (for example per-class TP/FP/FN tensors).
        state_shapes = {key: tensor.shape for key, tensor in state.items()}
        state_sizes = {key: tensor.numel() for key, tensor in state.items()}
        packed = torch.cat([tensor.reshape(-1) for tensor in state.values()])
        if self.trainer.world_size > 1:
            packed = self.all_gather(packed).sum(dim=0)

        global_state: Dict[str, torch.Tensor] = {}
        offset = 0
        for key in state:
            size = state_sizes[key]
            shape = state_shapes[key]
            global_state[key] = packed[offset : offset + size].reshape(shape)
            offset += size
        return global_state

    def _select_logged_metrics(self, metrics: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        # Keep W&B focused: train logs train loss, validation logs only the
        # aggregate metrics we actually want to monitor, and test metrics stay
        # in the terminal report instead of being uploaded.
        if prefix == "test":
            return {}
        if prefix != "val":
            return metrics

        allowed_names = {
            "loss",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_macro_weighted",
            "recall_macro_weighted",
            "f1_macro_weighted",
        }
        return {name: value for name, value in metrics.items() if name in allowed_names}

    def _maybe_initialize_val_demo_indices(self) -> None:
        if self._val_demo_indices is not None:
            return
        if not self._should_log_val_demos():
            return
        datamodule = getattr(self.trainer, "datamodule", None)
        val_dataset = getattr(datamodule, "val_dataset", None)
        if val_dataset is None:
            return

        num_items = len(val_dataset)
        if num_items == 0:
            return

        demo_cfg = self.cfg.logging.val_demos
        demo_count = min(demo_cfg.count, num_items)
        if demo_count <= 0:
            return

        seed = demo_cfg.seed if demo_cfg.seed is not None else self.cfg.trainer.seed
        generator = torch.Generator().manual_seed(seed)
        chosen = torch.randperm(num_items, generator=generator)[:demo_count].tolist()
        self._val_demo_indices = sorted(chosen)
        self._val_demo_lookup = set(self._val_demo_indices)

    def _should_log_val_demos(self) -> bool:
        demo_cfg = self.cfg.logging.val_demos
        if not demo_cfg.enabled or demo_cfg.count <= 0:
            return False
        if self.trainer is None or self.logger is None:
            return False
        if not getattr(self.trainer, "is_global_zero", True):
            return False
        if demo_cfg.every_n_epochs <= 0:
            return False
        if (self.current_epoch + 1) % demo_cfg.every_n_epochs != 0:
            return False
        return self._get_wandb_experiment() is not None

    def _collect_validation_demos(self, batch, logits: torch.Tensor) -> None:
        batch_size = int(batch["inputs"].shape[0])
        start_index = self._val_items_seen
        self._val_items_seen += batch_size
        if not self._should_log_val_demos() or not self._val_demo_lookup:
            return

        probabilities = self._compute_probabilities(logits).detach().cpu()
        targets = batch["targets"].detach().cpu()
        inputs = batch["inputs"].detach().cpu()

        for local_idx in range(batch_size):
            global_idx = start_index + local_idx
            if global_idx not in self._val_demo_lookup:
                continue

            pred_labels = self._decode_prediction(probabilities[local_idx])
            target_labels = self._decode_target(targets[local_idx])
            top_scores = self._format_top_scores(probabilities[local_idx], top_k=5)

            self._val_demo_rows.append(
                {
                    "epoch": self.current_epoch,
                    "dataset_index": global_idx,
                    "audio": wandb.Audio(
                        inputs[local_idx].float().numpy(),
                        sample_rate=self.cfg.data.sample_rate,
                        caption=f"epoch={self.current_epoch} val_idx={global_idx}",
                    ),
                    "prediction": ", ".join(pred_labels) if pred_labels else "(none)",
                    "ground_truth": ", ".join(target_labels) if target_labels else "(none)",
                    "top_scores": top_scores,
                }
            )

    def _compute_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        if self.task_type == "multiclass":
            return torch.softmax(logits, dim=-1)
        return torch.sigmoid(logits)

    def _decode_prediction(self, probabilities: torch.Tensor) -> List[str]:
        label_names = self.cfg.data.label_names
        if self.task_type == "multiclass":
            predicted_index = int(probabilities.argmax().item())
            return [label_names[predicted_index]]

        predicted_indices = (probabilities >= self.threshold).nonzero(as_tuple=False).flatten().tolist()
        return [label_names[idx] for idx in predicted_indices]

    def _decode_target(self, targets: torch.Tensor) -> List[str]:
        label_names = self.cfg.data.label_names
        if self.task_type == "multiclass":
            return [label_names[int(targets.item())]]

        target_indices = (targets >= 0.5).nonzero(as_tuple=False).flatten().tolist()
        return [label_names[idx] for idx in target_indices]

    def _format_top_scores(self, probabilities: torch.Tensor, top_k: int) -> str:
        top_values, top_indices = torch.topk(probabilities, k=min(top_k, probabilities.numel()))
        parts = []
        for value, index in zip(top_values.tolist(), top_indices.tolist()):
            parts.append(f"{self.cfg.data.label_names[index]}={value:.3f}")
        return ", ".join(parts)

    def _log_validation_demos(self) -> None:
        if not self._val_demo_rows:
            return
        experiment = self._get_wandb_experiment()
        if experiment is None:
            return

        table = wandb.Table(columns=["epoch", "dataset_index", "audio", "prediction", "ground_truth", "top_scores"])
        for row in self._val_demo_rows:
            table.add_data(
                row["epoch"],
                row["dataset_index"],
                row["audio"],
                row["prediction"],
                row["ground_truth"],
                row["top_scores"],
            )
        experiment.log({"val/demos": table}, step=self.global_step)

    def _get_wandb_experiment(self):
        if wandb is None:
            return None
        logger = self.logger
        if logger is None:
            return None
        # Only a real WandbLogger exposes an experiment object with `.log(...)`.
        # The default Lightning logger when W&B is off is usually TensorBoard, whose
        # `experiment` is a SummaryWriter. Treating that as W&B caused the crash.
        if WandbLogger is None or not isinstance(logger, WandbLogger):
            return None
        return logger.experiment
