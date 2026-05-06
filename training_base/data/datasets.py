from __future__ import annotations

from collections import defaultdict
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm

from . import audio
from .augmentations import build_augmenter
from .openmic_dataset_loader import OpenMicDataset
from .utils import (
    OPENMIC_CLASS_TO_INDEX,
    build_slakh_track_class_intervals,
    build_slakh_track_stem_entries,
    crop_waveform,
    get_audio_duration_sec,
    get_index_paths,
    get_root_paths,
    get_split_cfg_params,
    get_split_cfg_value,
    materialize_slakh_label_view,
    midi_note_intervals,
    normalize_slakh_index_payload,
    normalize_split_list,
    overlap_duration_sec,
    program_to_openmic_class_name,
    resolve_slakh_split_dir,
    union_intervals,
)


class SlakhDataset(Dataset):
    """
    Slakh dataset aligned to the future OpenMIC label space.

    Supported modes:
        - multiclass: one example per mapped stem
        - multilabel: one example per mix with crop-level MIDI-derived targets
        
        The whole idea for the multilabel is this, we are going through all the tracks
        in each different split and we randomly cropping a clip_duration_sec (defined in the config file) segment.
        We need to return the active instrument labels in this segment, the way we do it is check the midi file,
        and if one instrument is active for more than min_activity_ratio * clip_duration_sec in this segment we consider it active.
    """

    def __init__(
        self,
        split_cfg: Any,
        task_type: str,
        num_classes: int,
        sample_rate: int,
        clip_num_samples: int,
        train_mode: bool,
        min_activity_ratio: float = 0.1,
        label_names: Optional[List[str]] = None,
        relevance_threshold: float = 0.5,  # added
    ) -> None:
        #we basically at the moment allow 2 tasks multilabel and multiclass, but we are mainly training for 
        #multilabel throughout
        if task_type not in {"multiclass", "multilabel"}:
            raise ValueError(
                "SlakhDataset currently supports only multiclass or multilabel classification "
                f"(got task_type={task_type!r})"
            )
            
        #next line is to read some free-form dataset parameters from the config.
        params = get_split_cfg_params(split_cfg)

        # Keep the split/domain name so the batch still knows where the sample came from
        self.domain = get_split_cfg_value(split_cfg, "domain", "synthetic")
        # Store the task mode so __getitem__ knows whether to produce a scalar label or a multi-hot vector

        self.task_type = task_type
        # Number of output classes for the current experiment.
        self.num_classes = num_classes
        # Audio sample rate used everywhere in this dataset instance
        self.sample_rate = sample_rate
        # Target crop length, already converted from seconds to number of samples
        self.clip_num_samples = clip_num_samples
        
        #true for training, false for val/test.
        self.train_mode = train_mode
        
        # min activity ration is the minimum fraction of the crop that must be active for a class to count.
        self.min_activity_ratio = min_activity_ratio
        
        # The configured experiment label list, typically an OpenMIC subset
        self.label_names = list(label_names or [])
        
        # Map from class name to the final target index used by the model.
        self.class_name_to_index: Dict[str, int] = {}
        
        
        # optional augmentation config this is being defined in the config of the training. We split it into stem-level and mix-level later.
        augmentations_cfg = get_split_cfg_value(split_cfg, "augmentations", {}) or {}

        # Support one root or multiple roots in the config.
        roots = get_root_paths(params)
        # Support one split or a list of split folders such as ["train", "ommited"] the idea here using the ommited part of slakh is that
        #it makes sense to have the same tracks with different instrument sounds
        requested_splits = [split_name.lower() for split_name in normalize_split_list(params.get("split", "train"))]
        
        # Optional cached index files. If present we load them instead of rescanning disk. The index files contain actually information
        #about instrument activity. Basically they are preprocessed and extracted information from the midi files so that we do not have 
        # to read midis and extract instrument activity on the fly
        index_paths = get_index_paths(params)

        #the real target space (classes) for training comes from config.label_names. These names refer
        # to the OpenMIC-style vocabulary above. We only train on the subset selected there.
        # Example: if config.label_names = ["piano", "guitar"], then all other mapped Slakh
        # classes are ignored even if they are present in the track. That means that we can still keep a trumpet in a track
        #but we won't include it in the hot labels, which make sense because usually when you are building a classifier on instrument recognition
        #you want it to be able to identify the instruments you are interested in even in a mix that has other instruments too. 
        if self.label_names:
            # If the user provided an explicit subset, that subset is the real target space.
            chosen_label_names = self.label_names
        else:
            #fall back to the full supported vocabulary
            chosen_label_names = list(OPENMIC_CLASS_TO_INDEX.keys())

        # Fail early if the config contains names we do not know how to map to.
        unknown_label_names = [name for name in chosen_label_names if name not in OPENMIC_CLASS_TO_INDEX]
        if unknown_label_names:
            raise ValueError(f"Unsupported label names in config: {unknown_label_names}")
        if len(chosen_label_names) != num_classes:
            raise ValueError(
                "data.num_classes must match len(data.label_names) for Slakh/OpenMIC-aligned training "
                f"({num_classes} != {len(chosen_label_names)})"
            )

        # Store the finalized label order used by the model
        self.label_names = chosen_label_names
        
        # Build the class-name -> index lookup once, instead of repeating it during sampling
        self.class_name_to_index = {class_name: idx for idx, class_name in enumerate(self.label_names)}
        
        # Fast membership check for "is this class allowed in this experiment?"
        self.allowed_class_names = set(self.class_name_to_index)
        
        # Enable the on-the-fly stem remix path only when requested
        self.stem_remix_enabled = bool(params.get("stem_remix", self.train_mode and task_type == "multilabel"))

        # We optionally allow extra non-target stems to be added into the training submix.
        #
        # The reason for this is robustness: the model should not only hear "clean"
        # mixes made from the labels we care about. Real music will still contain other
        # instruments, and we want the classifier to learn the target labels in the
        # presence of distractors too.
        #
        # Very important semantic rule:
        # - target stems can affect both the audio and the labels
        # - context stems can affect only the audio, never the labels
        self.include_context_stems = bool(params.get("include_context_stems", self.train_mode and task_type == "multilabel"))
        # Probability of keeping an active non-target stem as background context.
        self.context_stem_keep_prob = float(params.get("context_stem_keep_prob", 0.35))
        # Hard cap so context does not overwhelm the target submix.
        self.max_context_stems = max(0, int(params.get("max_context_stems", 2)))
        
        #The following parameters refer to the fact that slakh is inbalanced, the idea here is that
        # on the fly we can create submixes of the track so that we try to have balance in instrument and how often they appear 
        # as much as possible
        
        # Limit how many stems from the same class survive into one synthetic submix
        self.max_stems_per_class = max(1, int(params.get("max_stems_per_class", 1)))
        # Controls how aggressively common classes are downweighted
        self.class_keep_exponent = float(params.get("class_keep_exponent", 0.5))
        # Lower bound so very common classes are not removed all the time
        self.class_keep_prob_min = float(params.get("class_keep_prob_min", 0.25))
        # We want at least this many classes in a synthetic training crop
        self.min_classes_per_crop = max(1, int(params.get("min_classes_per_crop", 1)))
        # Optional upper bound on the number of classes in one crop
        max_classes_per_crop = int(params.get("max_classes_per_crop", 0))
        # Convert "0" into "no upper bound"
        self.max_classes_per_crop = max_classes_per_crop if max_classes_per_crop > 0 else None
        # If a random crop is unusable, we can retry a few times
        self.remix_resample_attempts = max(1, int(params.get("remix_resample_attempts", 5)))
        # Peak-normalize the final synthetic submix if requested
        self.normalize_submix = bool(params.get("normalize_submix", True))
        
        # The following variables refer to the augmentations that we are going to add
        # Stem-level augmenter runs before we sum stems together
        self.stem_augmenter = build_augmenter(augmentations_cfg.get("stem"))
        # Mix-level augmenter runs after stems have been summed
        self.mix_augmenter = build_augmenter(augmentations_cfg.get("mix"))

        # This is the final list of samples used by __getitem__.
        self.samples: List[Dict[str, Any]] = []
        if index_paths:
            # Fast path: load the prebuilt MIDI index and then keep only the tracks that
            # belong to the requested split.
            for index_path in index_paths:
                # Each index file can contribute samples to this split.
                self.samples.extend(self._load_index(index_path, requested_splits=requested_splits))
        else:
            for root in roots:
                for split_name in requested_splits:
                    # Without a cache we walk the Slakh folders directly and build sample metadata.
                    #This is slow we are not using it actually in our training
                    split_dir = self._resolve_split_dir(root, split_name)
                    self._build_index(split_dir)

        # Count how many training tracks contain each class at all.
        self.class_track_counts = self._compute_class_track_counts()
        # Convert those counts into keep probabilities used by the remix policy.
        self.class_keep_probabilities = self._compute_class_keep_probabilities()


    @staticmethod
    def _resolve_split_dir(root: Path, split: str) -> Path:
        # Keep split-name normalization in one shared helper
        return resolve_slakh_split_dir(root, split)

    def _load_index(self, index_path: Path, requested_splits: List[str]) -> List[Dict[str, Any]]:
        # Cached training is only possible if the index file actually exists.
        if not index_path.exists():
            raise FileNotFoundError(f"Slakh index file not found: {index_path}")

        print(f"loading cached Slakh index from {index_path}")
        payload = torch.load(index_path, map_location="cpu")
        samples = normalize_slakh_index_payload(payload)
        #keep only the entries belonging to the requested split or split list
        filtered = self._filter_index_samples_for_split(samples, requested_splits)
        if not filtered:
            raise ValueError(
                f"No cached Slakh tracks matched splits={requested_splits!r} in index {index_path}"
            )
        # The cache now stores raw stem facts. We rebuild the current mapped view here so
        # that changing SLAKH_MIDI_TO_OPENMIC_CLASS does not force a cache rebuild.
        remapped_samples: List[Dict[str, Any]] = []
        for item in tqdm(
            filtered,
            desc=f"remapping cached tracks ({','.join(requested_splits)})",
            leave=False,
        ):
            all_mapped_stems, _ = materialize_slakh_label_view(
                item.get("raw_stems", []),
                allowed_class_names=None,
            )
            mapped_stems, class_intervals = materialize_slakh_label_view(
                item.get("raw_stems", []),
                allowed_class_names=self.allowed_class_names,
            )
            if self.task_type == "multilabel" and not class_intervals:
                continue
            remapped_item = dict(item)
            remapped_item["stems"] = mapped_stems
            remapped_item["context_stems"] = [
                stem for stem in all_mapped_stems if stem["class_name"] not in self.allowed_class_names
            ]
            remapped_item["class_intervals"] = class_intervals
            remapped_samples.append(remapped_item)
        if not remapped_samples:
            raise ValueError(
                f"No cached Slakh tracks remained after applying the current mapping for splits={requested_splits!r} "
                f"in index {index_path}"
            )
        return remapped_samples

    def _filter_index_samples_for_split(
        self,
        samples: List[Dict[str, Any]],
        requested_splits: List[str],
    ) -> List[Dict[str, Any]]:
        # The current cached index is global, so every item must tell us which split
        # it came from. We canonicalize names so "val", "validation", and
        # "vallidation" all match the same logical split.
        canonical_requested = {self._canonical_split_name(split_name) for split_name in requested_splits}
        filtered = []
        for item in samples:
            item_split = item["split"]
            if self._canonical_split_name(str(item_split)) in canonical_requested:
                filtered.append(item)
        return filtered

    @staticmethod
    def _canonical_split_name(split: str) -> str:
        return {
            "train": "train",
            "validation": "validation",
            "val": "validation",
            "vallidation": "validation",
            "test": "test",
        }.get(split.lower(), split.lower())

    def _build_index(self, split_dir: Path) -> None:
        # This is the slow path used when we do not provide a prebuilt cache file.
        print(f"building index for split {split_dir}")
        #reading all the needed files for one specific track
        track_dirs = sorted(split_dir.iterdir())
        for track_dir in tqdm(track_dirs, desc=f"indexing {split_dir.name}", leave=False):
            # Skip any non-directory entries.
            if not track_dir.is_dir():
                continue

            # Standard Slakh files we need for this track.
            meta_path = track_dir / "metadata.yaml"
            mix_path = track_dir / "mix.flac"
            stems_dir = track_dir / "stems"
            midi_dir = track_dir / "MIDI"
            # If essential metadata or stems are missing, skip the track.
            if not (meta_path.exists() and stems_dir.exists()):
                continue

            # Read the YAML once so we can inspect the stem metadata.
            with open(meta_path, "r", encoding="utf-8") as handle:
                metadata = yaml.safe_load(handle) or {}

            # Slakh stores per-stem metadata under the "stems" key.
            stems_metadata = metadata.get("stems", {})
            if self.task_type in {"multiclass"}:
                # In multiclass mode each stem becomes its own dataset sample.
                self._add_multiclass_samples(stems_dir, stems_metadata)
                continue

            # In multilabel mode we need both the mix and the MIDI directory.
            if not (mix_path.exists() and midi_dir.exists()):
                continue

            # here we are calculating for each instrument for how long it is active
            raw_stem_entries = build_slakh_track_stem_entries(
                track_dir=track_dir,
                stems_metadata=stems_metadata,
            )
            all_mapped_stems, _ = materialize_slakh_label_view(
                raw_stem_entries,
                allowed_class_names=None,
            )
            stem_entries, class_intervals = materialize_slakh_label_view(
                raw_stem_entries,
                allowed_class_names=self.allowed_class_names,
            )
            if not class_intervals:
                # If nothing from this track maps into our configured label set, drop it.
                continue

            # Store enough information to build either a full-mix crop or a remixed submix later.
            self.samples.append(
                {
                    "track_id": track_dir.name,
                    "sample_id": str(mix_path.resolve()),
                    "split": split_dir.name,
                    "audio_path": mix_path,
                    "duration_sec": get_audio_duration_sec(mix_path),
                    "raw_stems": raw_stem_entries,
                    "class_intervals": class_intervals,
                    "stems": stem_entries,
                    "context_stems": [
                        stem for stem in all_mapped_stems if stem["class_name"] not in self.allowed_class_names
                    ],
                }
            )

    def _build_track_class_intervals(
        self,
        track_dir: Path,
        stems_metadata: Dict[str, Any],
    ) -> Dict[str, List[Tuple[float, float]]]:
        # Delegate to the shared helper so the dataset path and the standalone index-building
        # script always use exactly the same MIDI-to-interval logic.
        return build_slakh_track_class_intervals(
            track_dir=track_dir,
            stems_metadata=stems_metadata,
            num_classes=self.num_classes,
            allowed_class_names=self.allowed_class_names,
        )

    def _compute_class_track_counts(self) -> Dict[str, int]:
        # Count track presence, not note count and not total active duration.
        counts = {class_name: 0 for class_name in self.label_names}
        for item in self.samples:
            # We only count a class once per track, even if the track has many stems of that class.
            present_classes = set(item.get("class_intervals", {}).keys())
            for class_name in present_classes:
                if class_name in counts:
                    counts[class_name] += 1
        return counts

    def _compute_class_keep_probabilities(self) -> Dict[str, float]:
        # Ignore classes that never appear at all when computing the minimum presence count.
        nonzero_counts = [count for count in self.class_track_counts.values() if count > 0]
        if not nonzero_counts:
            # Degenerate case: if nothing is counted, keep everything.
            return {class_name: 1.0 for class_name in self.label_names}

        # The rarest present class defines the reference count.
        min_count = min(nonzero_counts)
        keep_probs: Dict[str, float] = {}
        for class_name in self.label_names:
            count = self.class_track_counts.get(class_name, 0)
            if count <= 0:
                # If a class never appears, its keep probability is irrelevant, so keep 1.0.
                keep_probs[class_name] = 1.0
                continue
            # Common classes get a smaller keep probability than rare classes.
            keep_prob = (min_count / float(count)) ** self.class_keep_exponent
            # Never go below the configured floor, and never exceed 1.
            keep_probs[class_name] = max(self.class_keep_prob_min, min(1.0, keep_prob))
        return keep_probs

    def _get_track_num_samples(self, item: Dict[str, Any]) -> int:
        # The current cache always stores duration, so we can convert it directly into
        # a sample count without reopening the audio file here.
        duration_sec = float(item["duration_sec"])
        return max(1, int(round(duration_sec * self.sample_rate)))

    def _sample_crop_offset(self, item: Dict[str, Any]) -> int:
        # Convert cached duration into the crop-start search range.
        total_num_samples = self._get_track_num_samples(item)
        if total_num_samples <= self.clip_num_samples:
            # Short tracks always start at zero and get padded later if needed.
            return 0
        max_offset = total_num_samples - self.clip_num_samples
        if self.train_mode:
            # Training uses a random crop start.
            return int(torch.randint(0, max_offset + 1, size=(1,)).item())
        # Validation/test use a deterministic center crop.
        return max_offset // 2

    def _class_passes_activity_rule(
        self,
        class_name: str,
        active_sec: float,
        crop_duration_sec: float,
    ) -> bool:
        # We treat drums a bit differently from the pitched instruments.
        #
        # The reason is that drum MIDI is often represented as many tiny note hits.
        # If we apply the same "minimum active ratio" rule that we use for sustained
        # instruments, drums can look artificially inactive even though there are
        # clearly drum events happening in the crop.
        #
        # So the rule is:
        # - for drums: any overlap at all inside the crop is enough
        # - for every other class: keep using min_activity_ratio * crop_duration
        if class_name == "drums":
            return active_sec > 0.0

        min_active_sec = crop_duration_sec * self.min_activity_ratio
        return active_sec >= min_active_sec

    def _collect_active_stems(
        self,
        stems: List[Dict[str, Any]],
        crop_start_sec: float,
        crop_end_sec: float,
    ) -> Dict[str, List[Dict[str, Any]]]:
        # Group active stems by class for this exact crop.
        active_by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        crop_duration_sec = crop_end_sec - crop_start_sec

        for stem in stems:
            class_name = stem["class_name"]
            # Measure how much this stem is active inside the crop window.
            active_sec = overlap_duration_sec(stem["intervals"], crop_start_sec, crop_end_sec)
            if not self._class_passes_activity_rule(class_name, active_sec, crop_duration_sec):
                # Ignore stems that are only weakly present in this crop.
                # Drums are the exception: for them, any overlap is enough.
                continue
            # Copy the cached stem metadata so we can attach crop-specific active duration.
            active_stem = dict(stem)
            active_stem["active_sec"] = active_sec
            active_by_class[class_name].append(active_stem)
        return active_by_class

    def _choose_context_stems(
        self,
        active_context_by_class: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        # Context stems are optional background distractors. They should make the audio
        # more realistic, but they should not dominate the mixture.
        if not self.include_context_stems:
            return []

        candidate_stems: List[Dict[str, Any]] = []
        for stems in active_context_by_class.values():
            for stem in stems:
                if torch.rand(1).item() <= self.context_stem_keep_prob:
                    candidate_stems.append(stem)

        if not candidate_stems:
            return []

        if self.max_context_stems > 0 and len(candidate_stems) > self.max_context_stems:
            permutation = torch.randperm(len(candidate_stems)).tolist()
            candidate_stems = [candidate_stems[index] for index in permutation[: self.max_context_stems]]

        return candidate_stems

    def _choose_classes_for_submix(self, active_by_class: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        # Start from the classes that are genuinely active in this crop.
        active_classes = list(active_by_class.keys())
        if not active_classes:
            return []

        kept_classes = []
        dropped_classes = []
        for class_name in active_classes:
            # Frequent classes like piano/guitar/drums usually get a lower keep probability.
            keep_prob = self.class_keep_probabilities.get(class_name, 1.0)
            if torch.rand(1).item() <= keep_prob:
                kept_classes.append(class_name)
            else:
                dropped_classes.append(class_name)

        if len(kept_classes) < self.min_classes_per_crop:
            # If we dropped too much, add back the rarer classes first.
            dropped_classes = sorted(
                dropped_classes,
                key=lambda class_name: self.class_track_counts.get(class_name, 10**9),
            )
            while dropped_classes and len(kept_classes) < self.min_classes_per_crop:
                kept_classes.append(dropped_classes.pop(0))

        if not kept_classes:
            # Fallback: keep the rarest active class so we never generate an empty target.
            kept_classes = [
                min(active_classes, key=lambda class_name: self.class_track_counts.get(class_name, 10**9))
            ]

        if self.max_classes_per_crop is not None and len(kept_classes) > self.max_classes_per_crop:
            # If there are too many classes, again prioritize the rarer ones.
            kept_classes = sorted(
                kept_classes,
                key=lambda class_name: self.class_track_counts.get(class_name, 10**9),
            )[: self.max_classes_per_crop]
        return kept_classes

    def _choose_stems_for_class(self, stems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # If the class already has few stems, keep them all.
        if len(stems) <= self.max_stems_per_class:
            return stems
        # Otherwise randomly keep only a subset.
        permutation = torch.randperm(len(stems)).tolist()
        return [stems[index] for index in permutation[: self.max_stems_per_class]]

    def _load_stem_excerpt(self, stem: Dict[str, Any], offset: int) -> torch.Tensor:
        # Load the stem waveform from disk.
        waveform = audio.load_waveform(str(stem["audio_path"]), self.sample_rate)
        # Slice out the same crop window we are using for the track-level example.
        excerpt = audio.trim_or_pad(
            waveform[offset : offset + self.clip_num_samples],
            self.clip_num_samples,
            random_crop=False,
        )
        # Apply optional stem-level augmentation before mixing.
        excerpt = self.stem_augmenter(excerpt)
        return excerpt

    def _mix_selected_stems(
        self,
        selected_stems: List[Dict[str, Any]],
        offset: int,
    ) -> torch.Tensor:
        # Empty submix should still return a valid waveform tensor.
        if not selected_stems:
            return torch.zeros(self.clip_num_samples, dtype=torch.float32)

        # Start from silence, then add each kept stem.
        mixture = torch.zeros(self.clip_num_samples, dtype=torch.float32)
        for stem in selected_stems:
            mixture = mixture + self._load_stem_excerpt(stem, offset)

        # Apply optional mix-level augmentation after summing stems.
        mixture = self.mix_augmenter(mixture)
        if self.normalize_submix:
            # Peak-normalize to avoid very large amplitudes when many stems are summed.
            peak = mixture.abs().max().clamp_min(1e-6)
            mixture = mixture / peak
        return mixture.float()

    def _build_submix_target(self, kept_classes: List[str]) -> torch.Tensor:
        # Build the final multi-hot target from the classes that survived the remix.
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for class_name in kept_classes:
            class_id = self.class_name_to_index.get(class_name)
            if class_id is not None:
                target[class_id] = 1.0
        return target

    def _build_multilabel_excerpt_and_target(
        self,
        item: Dict[str, Any],
        *,
        export_debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # We return debug metadata too, because the preview/export tool reuses this path.
        debug_payload: Dict[str, Any] = {}
        # During training we can retry a few different random crops if the first one is bad.
        attempts = self.remix_resample_attempts if self.stem_remix_enabled and self.train_mode else 1

        for _ in range(attempts):
            # Pick a crop start in samples, then convert it to seconds for MIDI overlap checks.
            offset = self._sample_crop_offset(item)
            crop_start_sec = offset / self.sample_rate
            crop_end_sec = crop_start_sec + (self.clip_num_samples / self.sample_rate)

            if not (self.stem_remix_enabled and self.train_mode):
                # Val/test, or training without remix, uses the original Slakh full mix crop.
                waveform = audio.load_waveform(str(item["audio_path"]), self.sample_rate)
                excerpt = audio.trim_or_pad(
                    waveform[offset : offset + self.clip_num_samples],
                    self.clip_num_samples,
                    random_crop=False,
                )
                # Labels are computed from all classes active in the original mix crop.
                target = self._build_multilabel_target(item["class_intervals"], crop_start_sec, crop_end_sec)
                debug_payload = {
                    "offset": offset,
                    "crop_start_sec": crop_start_sec,
                    "crop_end_sec": crop_end_sec,
                    "mode": "full_mix",
                    "kept_classes": [
                        self.label_names[class_idx] for class_idx, value in enumerate(target.tolist()) if value >= 0.5
                    ],
                    "selected_stems": [],
                }
                return excerpt.float(), target, debug_payload

            # Training with remix first determines which stems are active in the crop.
            active_by_class = self._collect_active_stems(item.get("stems", []), crop_start_sec, crop_end_sec)
            if not active_by_class:
                # Retry if none of the configured classes is active in this crop.
                continue

            # Drop some classes probabilistically to reduce dominance of common classes.
            kept_classes = self._choose_classes_for_submix(active_by_class)
            selected_target_stems: List[Dict[str, Any]] = []
            for class_name in kept_classes:
                # Even inside one class, limit how many stems survive.
                selected_target_stems.extend(self._choose_stems_for_class(active_by_class[class_name]))

            if not selected_target_stems:
                # If everything was removed, try another crop.
                continue

            active_context_by_class = self._collect_active_stems(
                item.get("context_stems", []),
                crop_start_sec,
                crop_end_sec,
            )
            selected_context_stems = self._choose_context_stems(active_context_by_class)

            selected_stems = list(selected_target_stems) + list(selected_context_stems)

            # Render the final training waveform from the target stems plus optional
            # background context stems.
            excerpt = self._mix_selected_stems(selected_stems, offset)
            # The target is built from the kept target classes only, never from the
            # extra context stems.
            target = self._build_submix_target(kept_classes)
            debug_payload = {
                "offset": offset,
                "crop_start_sec": crop_start_sec,
                "crop_end_sec": crop_end_sec,
                "mode": "stem_submix",
                "kept_classes": kept_classes,
                "num_context_stems": len(selected_context_stems),
                "selected_stems": [
                    {
                        "stem_id": stem["stem_id"],
                        "class_name": stem["class_name"],
                        "program_num": stem["program_num"],
                        "active_sec": round(float(stem["active_sec"]), 4),
                        "role": "target" if stem in selected_target_stems else "context",
                    }
                    for stem in selected_stems
                ],
            }
            return excerpt, target, debug_payload

        # Final fallback if every retry landed on empty configured activity.
        # This path should be rare, but it prevents the dataset from crashing.
        waveform = audio.load_waveform(str(item["audio_path"]), self.sample_rate)
        excerpt = audio.trim_or_pad(waveform, self.clip_num_samples, random_crop=False)
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        debug_payload = {
            "offset": 0,
            "crop_start_sec": 0.0,
            "crop_end_sec": self.clip_num_samples / self.sample_rate,
            "mode": "fallback_empty",
            "kept_classes": [],
            "selected_stems": [],
        }
        return excerpt.float(), target, debug_payload

    def _add_multiclass_samples(self, stems_dir: Path, stems_metadata: Dict[str, Any]) -> None:
        for stem_id, stem_info in stems_metadata.items():
            # In multiclass mode each stem waveform becomes one dataset item.
            stem_path = stems_dir / f"{stem_id}.flac"
            if not stem_path.exists():
                continue

            # Map the stem program into the experiment label space.
            program = int(stem_info.get("program_num", -1))
            is_drum = bool(stem_info.get("is_drum", False))
            class_name = program_to_openmic_class_name(program, is_drum=is_drum)
            if class_name is None or class_name not in self.class_name_to_index:
                continue

            self.samples.append(
                {
                    "audio_path": stem_path,
                    "label": self.class_name_to_index[class_name],
                }
            )

    def __len__(self) -> int:
        # Lightning/Dataloader use this to know how many samples the split contains.
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Fetch the cached metadata for this sample.
        item = self.samples[idx]

        if self.task_type in {"multiclass"}:
            # Multiclass mode loads one stem waveform and one scalar class id.
            waveform = audio.load_waveform(str(item["audio_path"]), self.sample_rate)
            excerpt, _ = crop_waveform(waveform, self.clip_num_samples, self.train_mode)
            target = torch.tensor(item["label"], dtype=torch.long)
        else:
            # Multilabel mode either uses the original full mix crop or builds a synthetic
            # submix from a subset of active stems, depending on the remix settings.
            excerpt, target, _ = self._build_multilabel_excerpt_and_target(item)

        # The datamodule collate function expects this exact dictionary structure.
        return {
            "inputs": excerpt,
            "target": target,
            "domain": self.domain,
        }

    def _build_multilabel_target(
        self,
        class_intervals: Dict[str, List[Tuple[float, float]]],
        crop_start_sec: float,
        crop_end_sec: float,
    ) -> torch.Tensor:
        crop_duration_sec = crop_end_sec - crop_start_sec
        # Start from an all-zero multi-hot vector.
        target = torch.zeros(self.num_classes, dtype=torch.float32)

        for class_name, intervals in class_intervals.items():
            # Look up where this class lives in the final output vector.
            class_id = self.class_name_to_index.get(class_name)
            if class_id is None:
                continue
            # Measure how long this class is active inside the crop.
            active_sec = overlap_duration_sec(intervals, crop_start_sec, crop_end_sec)
            if self._class_passes_activity_rule(class_name, active_sec, crop_duration_sec):
                # Mark the class present if it crosses the activity rule.
                # For drums, that rule is "any activity at all".
                target[class_id] = 1.0
        return target


def build_dataset(
    split_cfg: Any,
    task_type: str,
    num_classes: int,
    sample_rate: int,
    clip_num_samples: int,
    train_mode: bool,
    min_activity_ratio: float = 0.1,
    label_names: Optional[List[str]] = None,
    relevance_threshold: float = 0.5,
) -> Dataset:
    ds_type = get_split_cfg_value(split_cfg, "type")
    # The dataset needs access to the config label names because those names define which
    # subset of the OpenMIC vocabulary we are actually training on.
    if ds_type == "slakh":
        print('building the Slakh Dataset')
        return SlakhDataset(
            split_cfg=split_cfg,
            task_type=task_type,
            num_classes=num_classes,
            sample_rate=sample_rate,
            clip_num_samples=clip_num_samples,
            train_mode=train_mode,
            min_activity_ratio=min_activity_ratio,
            label_names=label_names,
            relevance_threshold=relevance_threshold,
        )

    if ds_type == "openmic":
        return OpenMicDataset(
            split_cfg=split_cfg,
            task_type=task_type,
            num_classes=num_classes,
            sample_rate=sample_rate,
            clip_num_samples=clip_num_samples,
            train_mode=train_mode,
            label_names=label_names,
            relevance_threshold=relevance_threshold,
        )

    raise ValueError(f"Unsupported dataset type: {ds_type}")


def _build_positive_stem_reports(
    track_dir: Path,
    crop_start_sec: float,
    crop_end_sec: float,
    sample_rate: int,
    clip_num_samples: int,
    min_activity_ratio: float,
    allowed_class_names: Set[str],
) -> List[Dict[str, Any]]:
    with open(track_dir / "metadata.yaml", "r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle) or {}

    stems_metadata = metadata.get("stems", {})
    min_active_sec = (crop_end_sec - crop_start_sec) * min_activity_ratio
    stem_reports: List[Dict[str, Any]] = []

    for stem_id, stem_info in stems_metadata.items():
        stem_path = track_dir / "stems" / f"{stem_id}.flac"
        midi_path = track_dir / "MIDI" / f"{stem_id}.mid"
        if not (stem_path.exists() and midi_path.exists()):
            continue

        program = int(stem_info.get("program_num", -1))
        is_drum = bool(stem_info.get("is_drum", False))
        class_name = program_to_openmic_class_name(program, is_drum=is_drum)
        if class_name is None or class_name not in allowed_class_names:
            continue

        intervals = union_intervals(midi_note_intervals(midi_path))
        active_sec = overlap_duration_sec(intervals, crop_start_sec, crop_end_sec)
        if active_sec < min_active_sec:
            continue

        waveform = audio.load_waveform(str(stem_path), sample_rate)
        start_sample = int(round(crop_start_sec * sample_rate))
        excerpt = audio.trim_or_pad(
            waveform[start_sample : start_sample + clip_num_samples],
            clip_num_samples,
            random_crop=False,
        )
        stem_reports.append(
            {
                "stem_id": stem_id,
                "class_name": class_name,
                "program_num": program,
                "audio": excerpt,
                "active_sec": active_sec,
            }
        )
    return stem_reports


def _save_audio_with_mp3_fallback(audio_tensor: torch.Tensor, sample_rate: int, output_path: Path) -> Path:
    if audio.torchaudio is None:
        raise ImportError("torchaudio is required to export preview audio")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        audio.torchaudio.save(str(output_path), audio_tensor.unsqueeze(0), sample_rate, format="mp3")
        return output_path
    except Exception:
        fallback_path = output_path.with_suffix(".wav")
        audio.torchaudio.save(str(fallback_path), audio_tensor.unsqueeze(0), sample_rate)
        return fallback_path


def export_random_slakh_previews(
    dataset: SlakhDataset,
    output_dir: Path,
    num_examples: int,
    seed: int,
) -> Path:
    if dataset.task_type not in {"multilabel"}:
        raise ValueError("Preview export currently supports Slakh multilabel mode only.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    chosen_indices = list(range(len(dataset)))
    rng.shuffle(chosen_indices)
    chosen_indices = chosen_indices[: min(num_examples, len(chosen_indices))]

    summary_examples = []
    for export_idx, sample_idx in enumerate(chosen_indices):
        item = dataset.samples[sample_idx]
        track_dir = Path(item["audio_path"]).parent
        excerpt, target, debug_payload = dataset._build_multilabel_excerpt_and_target(item)
        crop_start_sec = float(debug_payload["crop_start_sec"])
        crop_end_sec = float(debug_payload["crop_end_sec"])
        clip_duration_sec = crop_end_sec - crop_start_sec
        active_labels = [
            dataset.label_names[class_idx]
            for class_idx, value in enumerate(target.tolist())
            if value >= 0.5
        ]

        example_dir = output_dir / f"example_{export_idx:02d}_{track_dir.name}"
        mix_path = _save_audio_with_mp3_fallback(excerpt, dataset.sample_rate, example_dir / "mix.mp3")

        stem_entries = []
        stem_lookup = {stem["stem_id"]: stem for stem in item.get("stems", [])}
        stem_lookup.update({stem["stem_id"]: stem for stem in item.get("context_stems", [])})
        for stem_report in debug_payload.get("selected_stems", []):
            stem = stem_lookup.get(stem_report["stem_id"])
            if stem is None:
                continue
            stem_audio = dataset._load_stem_excerpt(stem, int(debug_payload["offset"]))
            stem_path = _save_audio_with_mp3_fallback(
                stem_audio,
                dataset.sample_rate,
                example_dir / f"{stem_report.get('role', 'target')}_{stem_report['class_name']}_{stem_report['stem_id']}.mp3",
            )
            stem_entries.append(
                {
                    "stem_id": stem_report["stem_id"],
                    "class_name": stem_report["class_name"],
                    "program_num": stem_report["program_num"],
                    "active_sec": round(float(stem_report["active_sec"]), 4),
                    "role": stem_report.get("role", "target"),
                    "audio_path": str(stem_path),
                }
            )

        payload = {
            "dataset_index": sample_idx,
            "sample_id": item.get("sample_id"),
            "track_id": item.get("track_id"),
            "source_mix_path": str(item["audio_path"]),
            "saved_mix_path": str(mix_path),
            "mode": debug_payload.get("mode"),
            "crop_start_sec": round(crop_start_sec, 4),
            "crop_end_sec": round(crop_end_sec, 4),
            "clip_duration_sec": round(clip_duration_sec, 4),
            "labels": {label_name: int(label_name in active_labels) for label_name in dataset.label_names},
            "active_labels": active_labels,
            "positive_stems": stem_entries,
        }

        json_path = example_dir / "labels.json"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        summary_examples.append(payload)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({"num_examples": len(summary_examples), "examples": summary_examples}, handle, indent=2)
    return summary_path


def _run_preview_export_from_cli() -> None:
    parser = argparse.ArgumentParser(description="Export random Slakh dataset previews with crop labels and stems.")
    parser.add_argument("--config", required=True, help="Project config JSON")
    parser.add_argument("--split", default="train", choices=["train", "val", "validation", "vallidation", "test"])
    parser.add_argument("--output-dir", required=True, help="Folder where previews should be written")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of random 10-second crops to export")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    args = parser.parse_args()

    from training_base.config import load_json, parse_project_config

    cfg = parse_project_config(load_json(args.config))
    split_cfg = {
        "train": cfg.data.train,
        "val": cfg.data.val,
        "validation": cfg.data.val,
        "vallidation": cfg.data.val,
        "test": cfg.data.test,
    }.get(args.split)
    if split_cfg is None:
        raise ValueError(f"Config does not define data.{args.split}")

    dataset = build_dataset(
        split_cfg=split_cfg,
        task_type=cfg.data.task_type,
        num_classes=cfg.data.num_classes,
        sample_rate=cfg.data.sample_rate,
        clip_num_samples=cfg.data.clip_num_samples,
        train_mode=True,
        min_activity_ratio=cfg.data.min_activity_ratio,
        label_names=cfg.data.label_names,
        relevance_threshold=cfg.data.relevance_threshold,
    )
    summary_path = export_random_slakh_previews(
        dataset=dataset,
        output_dir=Path(args.output_dir).expanduser(),
        num_examples=args.num_examples,
        seed=args.seed,
    )
    print(f"Saved preview artifacts to {summary_path.parent}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    _run_preview_export_from_cli()


"""
python -m training_base.data.datasets \
  --config configs/slakh_multilabel_conv_6_classes.json \
  --split train \
  --output-dir /scratch/jnl9728/slakh_preview \
  --num-examples 100 \
  --seed 42

"""
