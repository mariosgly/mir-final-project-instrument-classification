#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


# Keep this script independent from the training package imports so it is easy to run
# even in a lightweight environment. We duplicate the tiny normalization tables here
# on purpose so the split builder does not depend on the training package import path.
OPENMIC_NAME_ALIASES = {
    "acoustic_guitar": "guitar",
    "electric_guitar": "guitar",
    "bass_guitar": "bass",
    "violin__fiddle": "violin",
    "fiddle": "violin",
    "mallet": "mallet_percussion",
    "mallet_percussion": "mallet_percussion",
    "synth": "synthesizer",
    "synthesizer": "synthesizer",
}

CORRUPTED_SAMPLE_KEYS = {
    "071826",
    "071827",
    "087435",
    "095253",
    "095259",
    "095263",
    "102144",
    "113025",
    "113604",
    "138485",
}


def canonicalize_instrument_name(name: str) -> str:
    normalized = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    return OPENMIC_NAME_ALIASES.get(normalized, normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reproducible OpenMIC train/val/test split files for the chosen label set."
    )
    parser.add_argument("--root", required=True, help="OpenMIC dataset root")
    parser.add_argument("--labels-json", default=None, help="Optional JSON config file to read data.label_names and data.relevance_threshold from")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional explicit label list. Overrides --labels-json")
    parser.add_argument("--relevance-threshold", type=float, default=None, help="Optional explicit relevance threshold. Overrides config")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fraction of samples assigned to train")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of samples assigned to val")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of samples assigned to test")
    parser.add_argument(
        "--ignore-official-partitions",
        action="store_true",
        help="Ignore OpenMIC's shipped train/test partition files and split the whole usable pool yourself.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", required=True, help="Directory where train.txt / val.txt / test.txt will be written")
    return parser.parse_args()


def load_config_labels(config_path: Path) -> Tuple[List[str], float]:
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    data_cfg = payload.get("data", {})
    labels = list(data_cfg.get("label_names", []))
    relevance_threshold = float(data_cfg.get("relevance_threshold", 0.5))
    return labels, relevance_threshold


def load_openmic_samples(root: Path, chosen_labels: Set[str], relevance_threshold: float) -> Dict[str, Set[str]]:
    labels_path = root / "openmic-2018-aggregated-labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"OpenMIC aggregated labels CSV not found: {labels_path}")

    sample_to_labels: Dict[str, Set[str]] = defaultdict(set)
    with open(labels_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"sample_key", "instrument", "relevance"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "OpenMIC aggregated labels CSV must contain at least "
                f"{sorted(required_columns)}"
            )

        for row in reader:
            sample_key = str(row["sample_key"]).zfill(6)
            if sample_key in CORRUPTED_SAMPLE_KEYS:
                continue

            class_name = canonicalize_instrument_name(row["instrument"])
            if class_name not in chosen_labels:
                continue

            relevance = float(row["relevance"])
            if relevance >= relevance_threshold:
                sample_to_labels[sample_key].add(class_name)

    # Drop samples that do not have any positive labels in the chosen target space.
    return {sample_key: labels for sample_key, labels in sample_to_labels.items() if labels}


def read_partition_file(path: Path) -> Set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_official_openmic_partitions(root: Path) -> Dict[str, Set[str]]:
    partitions_dir = root / "partitions"
    train_path = partitions_dir / "split01_train.csv"
    test_path = partitions_dir / "split01_test.csv"
    if not (train_path.exists() and test_path.exists()):
        return {}
    return {
        "train": read_partition_file(train_path),
        "test": read_partition_file(test_path),
    }


def score_assignment(
    labels: Set[str],
    current_counts: Counter[str],
    target_counts: Dict[str, float],
    current_size: int,
    target_size: float,
) -> float:
    # Lower score is better. We prefer putting a sample into the split that is
    # currently most under-filled for the labels this sample carries.
    label_score = 0.0
    for label in labels:
        label_score += current_counts[label] / max(target_counts[label], 1.0)
    size_score = current_size / max(target_size, 1.0)
    return label_score + size_score


def build_splits(
    sample_to_labels: Dict[str, Set[str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    sample_items = list(sample_to_labels.items())
    rng = random.Random(seed)
    rng.shuffle(sample_items)

    # Start from the hardest samples first: many labels means they are more useful for
    # keeping the splits label-diverse.
    sample_items.sort(key=lambda item: (-len(item[1]), item[0]))

    total_samples = len(sample_items)
    target_sizes = {
        "train": total_samples * train_ratio,
        "val": total_samples * val_ratio,
        "test": total_samples * test_ratio,
    }

    global_label_counts = Counter()
    for _, labels in sample_items:
        global_label_counts.update(labels)

    target_label_counts = {
        split_name: {label: count * ratio for label, count in global_label_counts.items()}
        for split_name, ratio in [("train", train_ratio), ("val", val_ratio), ("test", test_ratio)]
    }

    split_samples = {"train": [], "val": [], "test": []}
    split_label_counts = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    for sample_key, labels in sample_items:
        best_split = min(
            ("train", "val", "test"),
            key=lambda split_name: score_assignment(
                labels=labels,
                current_counts=split_label_counts[split_name],
                target_counts=target_label_counts[split_name],
                current_size=len(split_samples[split_name]),
                target_size=target_sizes[split_name],
            ),
        )
        split_samples[best_split].append(sample_key)
        split_label_counts[best_split].update(labels)

    return split_samples


def build_train_val_from_fixed_test(
    sample_to_labels: Dict[str, Set[str]],
    official_train_keys: Set[str],
    official_test_keys: Set[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    # We keep the official OpenMIC test partition intact. That is the safest choice
    # if we care about benchmarking on the intended held-out distribution.
    #
    # The only custom split we create is a train/val split inside the official train pool.
    train_pool = {
        sample_key: labels
        for sample_key, labels in sample_to_labels.items()
        if sample_key in official_train_keys
    }
    test_pool = sorted(sample_key for sample_key in sample_to_labels if sample_key in official_test_keys)

    if not train_pool:
        raise ValueError("No usable target-label samples were found inside the official OpenMIC train partition")
    if not test_pool:
        raise ValueError("No usable target-label samples were found inside the official OpenMIC test partition")

    # Only the relative train/val proportion matters inside the train pool, because test is fixed.
    relative_total = train_ratio + val_ratio
    if relative_total <= 0:
        raise ValueError("train_ratio + val_ratio must be positive when respecting official test partitions")

    sample_items = list(train_pool.items())
    rng = random.Random(seed)
    rng.shuffle(sample_items)
    sample_items.sort(key=lambda item: (-len(item[1]), item[0]))

    total_train_pool = len(sample_items)
    train_fraction = train_ratio / relative_total
    val_fraction = val_ratio / relative_total
    target_sizes = {
        "train": total_train_pool * train_fraction,
        "val": total_train_pool * val_fraction,
    }

    global_label_counts = Counter()
    for _, labels in sample_items:
        global_label_counts.update(labels)

    target_label_counts = {
        "train": {label: count * train_fraction for label, count in global_label_counts.items()},
        "val": {label: count * val_fraction for label, count in global_label_counts.items()},
    }

    split_samples = {"train": [], "val": []}
    split_label_counts = {
        "train": Counter(),
        "val": Counter(),
    }

    for sample_key, labels in sample_items:
        best_split = min(
            ("train", "val"),
            key=lambda split_name: score_assignment(
                labels=labels,
                current_counts=split_label_counts[split_name],
                target_counts=target_label_counts[split_name],
                current_size=len(split_samples[split_name]),
                target_size=target_sizes[split_name],
            ),
        )
        split_samples[best_split].append(sample_key)
        split_label_counts[best_split].update(labels)

    return {
        "train": split_samples["train"],
        "val": split_samples["val"],
        "test": test_pool,
    }


def summarize_split_labels(
    split_samples: Dict[str, List[str]],
    sample_to_labels: Dict[str, Set[str]],
    chosen_labels: List[str],
) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for split_name, sample_keys in split_samples.items():
        label_counts = Counter()
        labels_per_sample: List[int] = []
        for sample_key in sample_keys:
            labels = sample_to_labels[sample_key]
            label_counts.update(labels)
            labels_per_sample.append(len(labels))

        summary[split_name] = {
            "num_samples": len(sample_keys),
            "avg_labels_per_sample": (sum(labels_per_sample) / len(labels_per_sample)) if labels_per_sample else 0.0,
            "label_counts": {label: int(label_counts.get(label, 0)) for label in chosen_labels},
            "label_presence_ratio": {
                label: (float(label_counts.get(label, 0)) / max(len(sample_keys), 1))
                for label in chosen_labels
            },
        }
    return summary


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_labels: List[str] = []
    config_threshold = 0.5
    if args.labels_json is not None:
        config_labels, config_threshold = load_config_labels(Path(args.labels_json).expanduser())

    chosen_labels = list(args.labels) if args.labels else config_labels
    if not chosen_labels:
        raise ValueError("You must provide labels either through --labels or through --labels-json")

    relevance_threshold = args.relevance_threshold if args.relevance_threshold is not None else config_threshold
    sample_to_labels = load_openmic_samples(root, set(chosen_labels), relevance_threshold)
    if not sample_to_labels:
        raise ValueError("No OpenMIC samples remained after applying the chosen labels and relevance threshold")

    official_partitions = {} if args.ignore_official_partitions else load_official_openmic_partitions(root)
    if official_partitions:
        split_samples = build_train_val_from_fixed_test(
            sample_to_labels=sample_to_labels,
            official_train_keys=official_partitions["train"],
            official_test_keys=official_partitions["test"],
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        split_mode = "official_test_fixed_train_val_split"
    else:
        split_samples = build_splits(
            sample_to_labels=sample_to_labels,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        split_mode = "custom_full_pool_split"

    split_stats = summarize_split_labels(
        split_samples=split_samples,
        sample_to_labels=sample_to_labels,
        chosen_labels=chosen_labels,
    )

    for split_name, sample_keys in split_samples.items():
        output_path = output_dir / f"{split_name}.txt"
        with open(output_path, "w", encoding="utf-8") as handle:
            for sample_key in sorted(sample_keys):
                handle.write(f"{sample_key}\n")

    summary = {
        "num_samples": len(sample_to_labels),
        "labels": chosen_labels,
        "relevance_threshold": relevance_threshold,
        "seed": args.seed,
        "split_mode": split_mode,
        "splits": {name: len(keys) for name, keys in split_samples.items()},
        "split_stats": split_stats,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote OpenMIC split files to {output_dir}")
    print(f"split_mode: {split_mode}")
    for split_name, sample_keys in split_samples.items():
        print(f"  {split_name}: {len(sample_keys)}")
        split_info = split_stats[split_name]
        print(f"    avg_labels_per_sample: {split_info['avg_labels_per_sample']:.3f}")
        print("    label_counts:")
        for label in chosen_labels:
            print(f"      {label}: {split_info['label_counts'][label]}")


if __name__ == "__main__":
    main()
