from __future__ import annotations

from typing import Dict, List

import torch


def init_metric_state(task_type: str, device: torch.device, num_classes: int) -> Dict[str, torch.Tensor]:
    state = {
        "loss_sum": torch.zeros(1, device=device),
        "num_items": torch.zeros(1, device=device),
    }
    if task_type == "multiclass":
        state["correct"] = torch.zeros(1, device=device)
        state["confusion_matrix"] = torch.zeros(num_classes, num_classes, device=device)
    else:
        state["tp"] = torch.zeros(num_classes, device=device)
        state["fp"] = torch.zeros(num_classes, device=device)
        state["fn"] = torch.zeros(num_classes, device=device)
        state["tn"] = torch.zeros(num_classes, device=device)
        state["exact_match"] = torch.zeros(1, device=device)
    return state


def update_metric_state(
    state: Dict[str, torch.Tensor],
    task_type: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss: torch.Tensor,
    threshold: float,
) -> None:
    batch_size = targets.shape[0]
    state["loss_sum"] += loss.detach() * batch_size
    state["num_items"] += batch_size

    if task_type == "multiclass":
        preds = logits.argmax(dim=-1)
        state["correct"] += (preds == targets).sum()
        num_classes = state["confusion_matrix"].shape[0]
        flat_indices = targets.long() * num_classes + preds.long()
        counts = torch.bincount(flat_indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        state["confusion_matrix"] += counts
        return

    preds = (torch.sigmoid(logits) >= threshold).float()
    state["tp"] += ((preds == 1) & (targets == 1)).sum(dim=0)
    state["fp"] += ((preds == 1) & (targets == 0)).sum(dim=0)
    state["fn"] += ((preds == 0) & (targets == 1)).sum(dim=0)
    state["tn"] += ((preds == 0) & (targets == 0)).sum(dim=0)
    state["exact_match"] += (preds == targets).all(dim=-1).sum()


def _safe_precision(tp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    return tp / (tp + fp).clamp_min(1.0)


def _safe_recall(tp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    return tp / (tp + fn).clamp_min(1.0)


def _safe_f1(precision: torch.Tensor, recall: torch.Tensor) -> torch.Tensor:
    return (2 * precision * recall) / (precision + recall).clamp_min(1e-6)


def compute_epoch_metrics(
    task_type: str,
    state: Dict[str, torch.Tensor],
    label_names: List[str] | None = None,
) -> Dict[str, torch.Tensor]:
    num_items = state["num_items"].clamp_min(1.0)
    metrics = {"loss": state["loss_sum"] / num_items}
    if task_type == "multiclass":
        metrics["accuracy"] = state["correct"] / num_items
        confusion = state["confusion_matrix"]
        tp = confusion.diag()
        support = confusion.sum(dim=1)
        predicted = confusion.sum(dim=0)
        precision_per_class = _safe_precision(tp, predicted - tp)
        recall_per_class = _safe_recall(tp, support - tp)
        f1_per_class = _safe_f1(precision_per_class, recall_per_class)
        support_sum = support.sum().clamp_min(1.0)
        metrics["precision_macro"] = precision_per_class.mean()
        metrics["recall_macro"] = recall_per_class.mean()
        metrics["f1_macro"] = f1_per_class.mean()
        metrics["f1_macro_weighted"] = (f1_per_class * support).sum() / support_sum
        metrics["precision_macro_weighted"] = (precision_per_class * support).sum() / support_sum
        metrics["recall_macro_weighted"] = (recall_per_class * support).sum() / support_sum
        if label_names:
            for class_name, precision, recall, f1, class_support in zip(
                label_names,
                precision_per_class,
                recall_per_class,
                f1_per_class,
                support,
            ):
                metrics[f"precision_{class_name}"] = precision
                metrics[f"recall_{class_name}"] = recall
                metrics[f"f1_{class_name}"] = f1
                metrics[f"support_{class_name}"] = class_support
        return metrics

    tp = state["tp"]
    fp = state["fp"]
    fn = state["fn"]
    support = tp + fn
    precision_per_class = _safe_precision(tp, fp)
    recall_per_class = _safe_recall(tp, fn)
    f1_per_class = _safe_f1(precision_per_class, recall_per_class)
    support_sum = support.sum().clamp_min(1.0)

    precision_micro = _safe_precision(tp.sum(), fp.sum())
    recall_micro = _safe_recall(tp.sum(), fn.sum())
    f1_micro = _safe_f1(precision_micro, recall_micro)

    metrics["precision_micro"] = precision_micro
    metrics["recall_micro"] = recall_micro
    metrics["f1_micro"] = f1_micro
    metrics["exact_match"] = state["exact_match"] / num_items
    metrics["precision_macro"] = precision_per_class.mean()
    metrics["recall_macro"] = recall_per_class.mean()
    metrics["f1_macro"] = f1_per_class.mean()
    metrics["precision_macro_weighted"] = (precision_per_class * support).sum() / support_sum
    metrics["recall_macro_weighted"] = (recall_per_class * support).sum() / support_sum
    metrics["f1_macro_weighted"] = (f1_per_class * support).sum() / support_sum

    if label_names:
        for class_name, precision, recall, f1, class_support in zip(
            label_names,
            precision_per_class,
            recall_per_class,
            f1_per_class,
            support,
        ):
            metrics[f"precision_{class_name}"] = precision
            metrics[f"recall_{class_name}"] = recall
            metrics[f"f1_{class_name}"] = f1
            metrics[f"support_{class_name}"] = class_support
    return metrics


def format_test_report(
    task_type: str,
    state: Dict[str, torch.Tensor],
    label_names: List[str],
) -> str:
    if task_type == "multiclass":
        confusion = state["confusion_matrix"].detach().cpu()
        tp = confusion.diag()
        support = confusion.sum(dim=1)
        predicted = confusion.sum(dim=0)
        precision_per_class = _safe_precision(tp, predicted - tp)
        recall_per_class = _safe_recall(tp, support - tp)
        f1_per_class = _safe_f1(precision_per_class, recall_per_class)
        support_sum = support.sum().clamp_min(1.0)
        weighted_precision = (precision_per_class * support).sum() / support_sum
        weighted_recall = (recall_per_class * support).sum() / support_sum
        weighted_f1 = (f1_per_class * support).sum() / support_sum
        macro_precision = precision_per_class.mean()
        macro_recall = recall_per_class.mean()
        macro_f1 = f1_per_class.mean()
        accuracy = tp.sum() / support_sum

        lines = [
            "Test Summary",
            f"accuracy: {accuracy.item():.4f}",
            f"precision_macro: {macro_precision.item():.4f}",
            f"recall_macro: {macro_recall.item():.4f}",
            f"f1_macro: {macro_f1.item():.4f}",
            f"precision_macro_weighted: {weighted_precision.item():.4f}",
            f"recall_macro_weighted: {weighted_recall.item():.4f}",
            f"f1_macro_weighted: {weighted_f1.item():.4f}",
            "",
            "Classification Report",
            "class | precision | recall | f1 | support",
        ]
        for class_name, precision, recall, f1, class_support in zip(
            label_names,
            precision_per_class.tolist(),
            recall_per_class.tolist(),
            f1_per_class.tolist(),
            support.tolist(),
        ):
            lines.append(
                f"{class_name:>16} | {precision:>9.3f} | {recall:>6.3f} | {f1:>5.3f} | {int(class_support):>7}"
            )

        lines.append("")
        lines.append("Confusion Matrix")
        header = "pred->".ljust(16) + " ".join(f"{name[:8]:>8}" for name in label_names)
        lines.append(header)
        for class_name, row in zip(label_names, confusion.tolist()):
            lines.append(f"{class_name[:16]:>16} " + " ".join(f"{int(value):>8}" for value in row))
        return "\n".join(lines)

    tp = state["tp"].detach().cpu()
    fp = state["fp"].detach().cpu()
    fn = state["fn"].detach().cpu()
    tn = state["tn"].detach().cpu()
    support = tp + fn
    precision_per_class = _safe_precision(tp, fp)
    recall_per_class = _safe_recall(tp, fn)
    f1_per_class = _safe_f1(precision_per_class, recall_per_class)
    support_sum = support.sum().clamp_min(1.0)
    precision_micro = _safe_precision(tp.sum(), fp.sum())
    recall_micro = _safe_recall(tp.sum(), fn.sum())
    f1_micro = _safe_f1(precision_micro, recall_micro)
    exact_match = state["exact_match"].detach().cpu() / state["num_items"].detach().cpu().clamp_min(1.0)
    weighted_precision = (precision_per_class * support).sum() / support_sum
    weighted_recall = (recall_per_class * support).sum() / support_sum
    weighted_f1 = (f1_per_class * support).sum() / support_sum
    macro_precision = precision_per_class.mean()
    macro_recall = recall_per_class.mean()
    macro_f1 = f1_per_class.mean()

    lines = [
        "Test Summary",
        f"exact_match: {exact_match.item():.4f}",
        f"precision_micro: {precision_micro.item():.4f}",
        f"recall_micro: {recall_micro.item():.4f}",
        f"f1_micro: {f1_micro.item():.4f}",
        f"precision_macro: {macro_precision.item():.4f}",
        f"recall_macro: {macro_recall.item():.4f}",
        f"f1_macro: {macro_f1.item():.4f}",
        f"precision_macro_weighted: {weighted_precision.item():.4f}",
        f"recall_macro_weighted: {weighted_recall.item():.4f}",
        f"f1_macro_weighted: {weighted_f1.item():.4f}",
        "",
        "Classification Report",
        "class | precision | recall | f1 | support",
    ]
    for class_name, precision, recall, f1, class_support in zip(
        label_names,
        precision_per_class.tolist(),
        recall_per_class.tolist(),
        f1_per_class.tolist(),
        support.tolist(),
    ):
        lines.append(
            f"{class_name:>16} | {precision:>9.3f} | {recall:>6.3f} | {f1:>5.3f} | {int(class_support):>7}"
        )

    lines.append("")
    lines.append("Confusion Matrix (one-vs-rest per class)")
    lines.append("class | TP | FP | FN | TN")
    for class_name, tp_value, fp_value, fn_value, tn_value in zip(
        label_names,
        tp.tolist(),
        fp.tolist(),
        fn.tolist(),
        tn.tolist(),
    ):
        lines.append(
            f"{class_name:>16} | {int(tp_value):>2} | {int(fp_value):>2} | {int(fn_value):>2} | {int(tn_value):>2}"
        )
    return "\n".join(lines)
