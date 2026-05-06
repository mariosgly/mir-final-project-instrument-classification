from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from . import audio
from .utils import OPENMIC_CLASS_TO_INDEX, get_split_cfg_params, get_split_cfg_value


# OpenMIC mostly already uses the vocabulary we want, but in practice there are a
# few naming variants that are worth normalizing so the dataset can still line up
# with the same canonical label names we use everywhere else in the repo.
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


def _resolve_openmic_split_file(partitions_dir: Path, split: str) -> Path:
    # Different local copies of OpenMIC are not always packaged with exactly the
    # same partition filenames. Some use names like `split01_train.csv`, while
    # other scripts in the wild expect simpler names like `train.txt`.
    #
    # Instead of making the whole dataset loader depend on one exact convention,
    # we try the small set of partition filenames that are commonly seen.
    split_candidates = {
        "train": [
            "split01_train.csv",
            "train.csv",
            "train.txt",
        ],
        "test": [
            "split01_test.csv",
            "test.csv",
            "test.txt",
        ],
        # OpenMIC does not really ship an official validation split in the same
        # way Slakh does. If somebody wants a separate validation set, the clean
        # way is to pass `split_file` explicitly in the config.
        #
        # Still, if they write "val" or "validation" without a custom file, the
        # least surprising fallback is to use the official test partition.
        "val": [
            "split01_test.csv",
            "test.csv",
            "test.txt",
        ],
        "validation": [
            "split01_test.csv",
            "test.csv",
            "test.txt",
        ],
        "vallidation": [
            "split01_test.csv",
            "test.csv",
            "test.txt",
        ],
    }

    if split not in split_candidates:
        raise ValueError(
            "OpenMIC split must be one of train/test/val/validation/vallidation, "
            "or split_cfg.params.split_file must be provided explicitly"
        )

    for candidate_name in split_candidates[split]:
        candidate_path = partitions_dir / candidate_name
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        f"Could not find an OpenMIC partition file for split={split!r} under {partitions_dir}. "
        f"Tried: {split_candidates[split]}"
    )


class OpenMicDataset(Dataset):
    """
    Config-driven OpenMIC dataset that mirrors the rest of this repo.

    The important design choice here is that we do not hardcode a separate OpenMIC
    target space. Instead, we take `label_names` from the project config and treat
    that as the real output space, just like we do for Slakh.

    That means:
    - if the config asks for all 20 OpenMIC classes, we use all 20
    - if the config asks for a smaller subset, we keep only that subset
    - the relevance threshold used to turn aggregated annotations into binary labels
      also comes from config, so it is explicit and reproducible
    """

    def __init__(
        self,
        split_cfg: Any,
        task_type: str,
        num_classes: int,
        sample_rate: int,
        clip_num_samples: int,
        train_mode: bool,
        label_names: Optional[List[str]] = None,
        relevance_threshold: float = 0.5,
    ) -> None:
        if task_type != "multilabel":
            raise ValueError(f"OpenMIC currently supports only multilabel mode (got {task_type!r})")

        self.domain = get_split_cfg_value(split_cfg, "domain", "real")
        self.task_type = task_type
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.clip_num_samples = clip_num_samples
        self.train_mode = train_mode

        params = get_split_cfg_params(split_cfg)
        root = Path(params["root"]).expanduser()
        split = str(params.get("split", "train")).lower()
        self.relevance_threshold = float(params.get("relevance_threshold", relevance_threshold))

        # The target space comes from config.label_names, not from a hardcoded OpenMIC list.
        # That keeps OpenMIC aligned with the rest of the training/evaluation pipeline.
        if label_names:
            chosen_label_names = list(label_names)
        else:
            chosen_label_names = list(OPENMIC_CLASS_TO_INDEX.keys())

        unknown_label_names = [name for name in chosen_label_names if name not in OPENMIC_CLASS_TO_INDEX]
        if unknown_label_names:
            raise ValueError(f"Unsupported OpenMIC label names in config: {unknown_label_names}")
        if len(chosen_label_names) != num_classes:
            raise ValueError(
                "data.num_classes must match len(data.label_names) for OpenMIC "
                f"({num_classes} != {len(chosen_label_names)})"
            )

        self.label_names = chosen_label_names
        self.class_name_to_index = {class_name: idx for idx, class_name in enumerate(self.label_names)}

        aggregated_labels_path = root / "openmic-2018-aggregated-labels.csv"
        if not aggregated_labels_path.exists():
            raise FileNotFoundError(f"OpenMIC labels CSV not found: {aggregated_labels_path}")
        partitions_dir = root / "partitions"

        # We keep the split resolution explicit. OpenMIC normally uses official train/test
        # partitions, and for evaluation we should use the full official test split rather
        # than some balanced subset, because test should reflect the real benchmark.
        split_file = params.get("split_file")
        if split_file is None:
            if not partitions_dir.exists():
                raise FileNotFoundError(
                    f"OpenMIC partitions directory not found: {partitions_dir}. "
                    "If your OpenMIC copy does not ship official partition files, "
                    "please provide split_cfg.params.split_file explicitly."
                )
            split_path = _resolve_openmic_split_file(partitions_dir, split)
        else:
            split_path = Path(split_file)
            if not split_path.is_absolute():
                split_path = partitions_dir / split_path
            if not split_path.exists():
                raise FileNotFoundError(f"OpenMIC split file not found: {split_path}")

        split_keys = {
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if not split_keys:
            raise ValueError(f"OpenMIC split file is empty: {split_path}")

        df = pd.read_csv(aggregated_labels_path)
        if "sample_key" not in df.columns or "instrument" not in df.columns or "relevance" not in df.columns:
            raise ValueError(
                "OpenMIC aggregated labels CSV must contain at least "
                "'sample_key', 'instrument', and 'relevance' columns"
            )

        df["sample_key"] = df["sample_key"].astype(str).str.zfill(6)
        df = df[df["sample_key"].isin(split_keys)]
        df = df[~df["sample_key"].isin(CORRUPTED_SAMPLE_KEYS)]

        def canonicalize_instrument_name(name: Any) -> str:
            normalized = str(name).strip().lower().replace(" ", "_").replace("-", "_")
            return OPENMIC_NAME_ALIASES.get(normalized, normalized)

        df["instrument"] = df["instrument"].apply(canonicalize_instrument_name)

        # Keep only the rows that correspond to the classes we are actually training on.
        df = df[df["instrument"].isin(self.class_name_to_index)]

        self.samples: List[Dict[str, Any]] = []
        for sample_key, clip_df in df.groupby("sample_key"):
            # OpenMIC stores the audio by three-digit subfolder prefix.
            audio_path = root / "audio" / sample_key[:3] / f"{sample_key}.ogg"
            if not audio_path.exists():
                continue

            target = torch.zeros(self.num_classes, dtype=torch.float32)
            for _, row in clip_df.iterrows():
                if float(row["relevance"]) < self.relevance_threshold:
                    continue
                class_name = str(row["instrument"])
                class_index = self.class_name_to_index.get(class_name)
                if class_index is not None:
                    target[class_index] = 1.0

            # OpenMIC is weak-label multilabel tagging, so clips with no positive labels
            # for the chosen target vocabulary are not useful training/eval examples here.
            if target.sum().item() <= 0:
                continue

            self.samples.append(
                {
                    "sample_key": sample_key,
                    "audio_path": str(audio_path),
                    "target": target,
                }
            )

        if not self.samples:
            raise ValueError(
                "OpenMIC dataset produced zero usable samples. "
                "Please check the root path, split file, label_names, and relevance_threshold."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        waveform = audio.load_waveform(item["audio_path"], self.sample_rate)
        excerpt = audio.trim_or_pad(waveform, self.clip_num_samples, random_crop=self.train_mode)
        return {
            "inputs": excerpt,
            "target": item["target"].clone(),
            "domain": self.domain,
        }
