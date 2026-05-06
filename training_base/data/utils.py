from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import mido
import torch

from . import audio


#these are the openmic target classes
OPENMIC_CLASS_TO_INDEX = {
    "accordion": 0,
    "banjo": 1,
    "bass": 2,
    "cello": 3,
    "clarinet": 4,
    "cymbals": 5,
    "drums": 6,
    "flute": 7,
    "guitar": 8,
    "mallet_percussion": 9,
    "mandolin": 10,
    "organ": 11,
    "piano": 12,
    "saxophone": 13,
    "synthesizer": 14,
    "trombone": 15,
    "trumpet": 16,
    "ukulele": 17,
    "violin": 18,
    "voice": 19,
}

# This is how we group the slakh program numbers into the OpenMIC label names.
SLAKH_MIDI_TO_OPENMIC_CLASS = {
    # Piano-like
    0: "piano", 1: "piano", 2: "piano", 3: "piano",
    4: "piano", 5: "piano", 6: "piano", 7: "piano",

    # Mallet / pitched percussion
    8: "mallet_percussion", 9: "mallet_percussion",
    10: "mallet_percussion", 11: "mallet_percussion",
    12: "mallet_percussion", 13: "mallet_percussion",
    14: "mallet_percussion", 15: "mallet_percussion",

    # Organ / accordion
    16: "organ", 17: "organ", 18: "organ", 19: "organ", 20: "organ",
    21: "accordion", 23: "accordion",

    # Guitar
    24: "guitar", 25: "guitar", 26: "guitar", 27: "guitar",
    28: "guitar", 29: "guitar", 30: "guitar", 31: "guitar",

    # Bass
    32: "bass", 33: "bass", 34: "bass", 35: "bass",
    36: "bass", 37: "bass", 38: "bass", 39: "bass",
    43: "bass",

    # Strings mapped to closest OpenMIC classes
    40: "violin",
    41: "violin",
    42: "cello",
    44: "violin",
    45: "violin",
    48: "violin",
    49: "violin",
    50: "violin",
    51: "violin",

    # Voice / choir
    52: "voice",
    53: "voice",
    54: "voice",
    85: "voice",
    91: "voice",

    # Brass
    56: "trumpet",
    57: "trombone",
    58: "trombone",
    59: "trumpet",
    60: "trumpet",
    61: "trumpet",
    62: "synthesizer",
    63: "synthesizer",

    # Reed / sax
    64: "saxophone", 65: "saxophone", 66: "saxophone", 67: "saxophone",
    68: "clarinet",
    69: "clarinet",
    70: "clarinet",
    71: "clarinet",

    # Flutes / pipe
    72: "flute", 73: "flute", 74: "flute", 75: "flute",
    76: "flute", 77: "flute", 78: "flute", 79: "flute",

    # Synth
    80: "synthesizer", 81: "synthesizer", 82: "synthesizer",
    83: "synthesizer", 84: "synthesizer", 86: "synthesizer",
    87: "synthesizer", 88: "synthesizer", 89: "synthesizer",
    90: "synthesizer", 92: "synthesizer", 93: "synthesizer",
    94: "synthesizer", 95: "synthesizer", 96: "synthesizer",
    97: "synthesizer", 98: "synthesizer", 99: "synthesizer",
    100: "synthesizer", 101: "synthesizer",
    102: "synthesizer", 103: "synthesizer",

    # Ethnic / misc OpenMIC-compatible
    105: "banjo",
    108: "mallet_percussion",
    110: "violin",

    # Percussive / drums / cymbals
    112: "mallet_percussion",
    113: "mallet_percussion",
    114: "mallet_percussion",
    115: "mallet_percussion",
    116: "drums",
    117: "drums",
    118: "drums",
    119: "cymbals",

    # Drums
    128: "drums",
}


INDEX_TO_OPENMIC_CLASS = {index: name for name, index in OPENMIC_CLASS_TO_INDEX.items()}


# Lots of helper functions on the way. The reason they live here now is that the
# dataset file was starting to become too crowded: when one file has both the real
# dataset logic and all the small support utilities, it becomes hard to know what
# is "the main story" and what is just plumbing.


def program_to_openmic_class_name(program: int, is_drum: bool) -> Optional[str]:
    # The idea of this helper function is this: besides each midi program corresponding
    # to a specific midi instrument, in the Slakh metadata there is also a bool value
    # telling us whether this stem should be treated as drums.
    #
    # So if this bool is true, we do not care about the midi program number anymore.
    # We directly decide that this stem belongs to the "drums" class.
    #
    # If the bool is false, then we look up the midi program number in our mapping table.
    if is_drum:
        return "drums"
    return SLAKH_MIDI_TO_OPENMIC_CLASS.get(program)


def program_to_openmic_class_id(program: int, is_drum: bool) -> int:
    # Most of the training code works with class names first, because names are easier
    # to read and they let us select arbitrary subsets from the config.
    #
    # Still, a few scripts want the old fixed integer OpenMIC ids, so this wrapper keeps
    # that interface around.
    class_name = program_to_openmic_class_name(program, is_drum=is_drum)
    if class_name is None:
        return -1
    return OPENMIC_CLASS_TO_INDEX[class_name]


# the following helpers are just for reading the config values and passing them into
# usable parameters
def get_split_cfg_value(split_cfg: Any, key: str, default: Any = None) -> Any:
    # Some places pass the split config as a dict, while other places pass a dataclass-like
    # object. This helper lets us read values without caring which representation we got.
    if isinstance(split_cfg, dict):
        return split_cfg.get(key, default)
    return getattr(split_cfg, key, default)


def get_split_cfg_params(split_cfg: Any) -> Dict[str, Any]:
    # The "params" field is the open-ended bucket where we store dataset-specific things
    # such as paths, split names, index paths, remix knobs, and so on.
    params = get_split_cfg_value(split_cfg, "params", {}) or {}
    if not isinstance(params, dict):
        raise TypeError(f"Expected split_cfg.params to be a dict, got {type(params)!r}")
    return params


# helpers for the paths of the subsets
def normalize_path_list(raw_value: Any, *, field_name: str) -> List[Path]:
    # We allow both one path and a list of paths in the config.
    # That way the user can write either:
    #   "root": "/path/to/slakh"
    # or:
    #   "roots": ["/path/a", "/path/b"]
    if raw_value is None:
        return []
    if isinstance(raw_value, (str, Path)):
        return [Path(raw_value).expanduser()]
    if isinstance(raw_value, (list, tuple)):
        return [Path(value).expanduser() for value in raw_value]
    raise TypeError(f"Expected {field_name} to be a path or list of paths, got {type(raw_value)!r}")


def get_root_paths(params: Dict[str, Any]) -> List[Path]:
    # The user may pass one root or multiple roots.
    if "roots" in params:
        roots = normalize_path_list(params.get("roots"), field_name="roots")
    else:
        roots = normalize_path_list(params.get("root"), field_name="root")
    if not roots:
        raise ValueError("SlakhDataset requires split_cfg.params.root or split_cfg.params.roots")
    return roots


# this is for reading the index, we will talk more about what the index is later
def get_index_paths(params: Dict[str, Any]) -> List[Path]:
    # Same idea as roots: allow one cached index or multiple cached indexes.
    if "index_paths" in params:
        return normalize_path_list(params.get("index_paths"), field_name="index_paths")
    return normalize_path_list(params.get("index_path"), field_name="index_path")


def normalize_split_list(raw_value: Any) -> List[str]:
    # A split can be a single folder name like "train", or a list like
    # ["train", "ommited"] if we want to merge multiple folders into one logical split.
    if raw_value is None:
        return ["train"]
    if isinstance(raw_value, str):
        return [raw_value]
    if isinstance(raw_value, (list, tuple)):
        return [str(value) for value in raw_value]
    raise TypeError(f"Expected split to be a string or list of strings, got {type(raw_value)!r}")


def resolve_slakh_split_dir(root: Path, split: str) -> Path:
    # This helper is shared by both the dataset and the standalone index-building script.
    # We normalize a few common spellings so the config can keep using "val"
    # or the misspelled historical "vallidation" without breaking anything.
    normalized = {
        "train": "train",
        "validation": "validation",
        "val": "validation",
        "test": "test",
    }.get(split, split)

    # Try a few variants because different copies of the dataset may use slightly
    # different folder naming conventions.
    candidate_names = [
        normalized,
        "validation" if normalized == "vallidation" else "vallidation",
        normalized.capitalize(),
        normalized.upper(),
        split,
        split.capitalize(),
        split.upper(),
    ]

    seen = set()
    for candidate_name in candidate_names:
        if candidate_name in seen:
            continue
        seen.add(candidate_name)
        candidate = root / candidate_name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find Slakh split directory for split={split!r} under {root}"
    )


def union_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    # The point of this helper is to merge overlapping note/activity intervals so
    # we do not double count time when multiple notes of the same stem or class overlap.
    if not intervals:
        return []

    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]
    for start_sec, end_sec in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start_sec <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end_sec))
        else:
            merged.append((start_sec, end_sec))
    return merged


def overlap_duration_sec(
    intervals: List[Tuple[float, float]],
    crop_start_sec: float,
    crop_end_sec: float,
) -> float:
    # Once we know an instrument/stem is active during certain time intervals,
    # this helper tells us how much of that activity falls inside the current crop.
    total = 0.0
    for start_sec, end_sec in intervals:
        overlap_start = max(start_sec, crop_start_sec)
        overlap_end = min(end_sec, crop_end_sec)
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
    return total


def midi_note_intervals(midi_path: Path) -> List[Tuple[float, float]]:
    # We read the midi file with mido.
    midi_file = mido.MidiFile(midi_path)
    active_notes: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    intervals: List[Tuple[float, float]] = []
    absolute_time = 0.0

    # We go through all midi messages and build note-on / note-off intervals in seconds.
    for message in midi_file:
        absolute_time += float(message.time)
        if message.type == "note_on" and message.velocity > 0:
            key = (getattr(message, "channel", 0), int(message.note))
            active_notes[key].append(absolute_time)
        elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
            key = (getattr(message, "channel", 0), int(message.note))
            starts = active_notes.get(key)
            if not starts:
                continue
            start_time = starts.pop()
            if absolute_time > start_time:
                intervals.append((start_time, absolute_time))

    # If some note_on events were never explicitly closed, we close them at the end
    # of the file instead of silently dropping them.
    for starts in active_notes.values():
        for start_time in starts:
            if absolute_time > start_time:
                intervals.append((start_time, absolute_time))

    return intervals


def build_slakh_track_class_intervals(
    track_dir: Path,
    stems_metadata: Dict[str, Any],
    num_classes: int,
    allowed_class_names: Optional[Set[str]] = None,
) -> Dict[str, List[Tuple[float, float]]]:
    # This is the reusable "expensive" piece we may want to precompute into an index file.
    # For each track we convert all mapped stem MIDIs into class-level active intervals.
    class_to_intervals: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    midi_dir = track_dir / "MIDI"

    for stem_id, stem_info in stems_metadata.items():
        midi_path = midi_dir / f"{stem_id}.mid"
        if not midi_path.exists():
            continue

        # getting the program number for each stem
        program = int(stem_info.get("program_num", -1))
        # there is also the is_drum bool in the metadata so we can either consider something
        # as a drum from the is_drum bool or the midi program
        is_drum = bool(stem_info.get("is_drum", False))
        # We first map to the semantic class name. Later we keep only the names that belong
        # to the currently configured target space.
        class_name = program_to_openmic_class_name(program, is_drum=is_drum)
        if class_name is None:
            continue
        if allowed_class_names is not None and class_name not in allowed_class_names:
            continue

        # here calculating for how long each instrument is active
        note_intervals = midi_note_intervals(midi_path)
        if not note_intervals:
            continue

        class_to_intervals[class_name].extend(note_intervals)

    merged: Dict[str, List[Tuple[float, float]]] = {}
    for class_name, intervals in class_to_intervals.items():
        united_intervals = union_intervals(intervals)
        if united_intervals:
            merged[class_name] = united_intervals
    return merged


def build_slakh_track_stem_entries(
    track_dir: Path,
    stems_metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    # The important design choice here is that the cache should stay as raw as possible.
    #
    # So instead of deciding the final OpenMIC class right here and baking that decision
    # into the cached file forever, we only store the raw stem-level facts:
    # - where the audio lives
    # - where the midi lives
    # - program number
    # - is_drum flag
    # - note-active intervals
    #
    # Then later, when the dataset is loaded, we can apply whatever the *current*
    # SLAKH_MIDI_TO_OPENMIC_CLASS mapping is at that time.
    #
    # This means that if we narrow or broaden a mapping later, we do not need to
    # rebuild the expensive MIDI interval cache again just for that semantic change.
    stem_entries: List[Dict[str, Any]] = []
    midi_dir = track_dir / "MIDI"
    stems_dir = track_dir / "stems"

    for stem_id, stem_info in stems_metadata.items():
        midi_path = midi_dir / f"{stem_id}.mid"
        audio_path = stems_dir / f"{stem_id}.flac"
        if not (midi_path.exists() and audio_path.exists()):
            continue

        program = int(stem_info.get("program_num", -1))
        is_drum = bool(stem_info.get("is_drum", False))

        note_intervals = midi_note_intervals(midi_path)
        if not note_intervals:
            continue

        united_intervals = union_intervals(note_intervals)
        stem_entries.append(
            {
                "stem_id": stem_id,
                "audio_path": str(audio_path),
                "midi_path": str(midi_path),
                "program_num": program,
                "is_drum": is_drum,
                "intervals": [(float(start_sec), float(end_sec)) for start_sec, end_sec in united_intervals],
            }
        )
    return stem_entries


def materialize_slakh_label_view(
    raw_stems: List[Dict[str, Any]],
    allowed_class_names: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Tuple[float, float]]]]:
    # This helper is the place where raw cached stems become the *current*
    # label-space view of the world.
    #
    # In other words:
    # - the cache stores raw stem facts
    # - this helper applies today's midi->class mapping
    # - and only then do we get class names and merged class intervals
    #
    # That is the key trick that lets us change SLAKH_MIDI_TO_OPENMIC_CLASS without
    # having to rebuild the MIDI interval cache every time.
    mapped_stems: List[Dict[str, Any]] = []
    class_to_intervals: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    for raw_stem in raw_stems:
        program = int(raw_stem.get("program_num", -1))
        is_drum = bool(raw_stem.get("is_drum", False))
        class_name = program_to_openmic_class_name(program, is_drum=is_drum)
        if class_name is None:
            continue
        if allowed_class_names is not None and class_name not in allowed_class_names:
            continue

        mapped_stem = dict(raw_stem)
        mapped_stem["class_name"] = class_name
        mapped_stem["intervals"] = [tuple(interval) for interval in raw_stem.get("intervals", [])]
        mapped_stems.append(mapped_stem)
        class_to_intervals[class_name].extend(mapped_stem["intervals"])

    merged_class_intervals: Dict[str, List[Tuple[float, float]]] = {}
    for class_name, intervals in class_to_intervals.items():
        united_intervals = union_intervals(intervals)
        if united_intervals:
            merged_class_intervals[class_name] = united_intervals
    return mapped_stems, merged_class_intervals


def get_audio_duration_sec(audio_path: Path) -> float:
    # We cache track duration because later, during training, we only need to sample a crop start.
    # If the duration is already known we do not need to reopen the whole file.
    if audio.torchaudio is not None and hasattr(audio.torchaudio, "info"):
        info = audio.torchaudio.info(str(audio_path))
        return float(info.num_frames) / float(info.sample_rate)

    waveform = audio.load_waveform(str(audio_path), target_sample_rate=16000)
    return float(waveform.shape[-1]) / 16000.0


def normalize_slakh_index_payload(payload: Any) -> List[Dict[str, Any]]:
    # We now assume one single cache format:
    # a global payload with a top-level "tracks" dictionary keyed by track id.
    if not isinstance(payload, dict) or "tracks" not in payload:
        raise ValueError("Slakh index file must be a dict with a top-level 'tracks' field")

    samples = list(payload["tracks"].values())

    normalized_samples: List[Dict[str, Any]] = []
    for item in samples:
        if "split" not in item:
            raise ValueError("Each cached Slakh track must store its originating split")
        if "duration_sec" not in item:
            raise ValueError("Each cached Slakh track must store duration_sec")

        raw_stems = []
        for stem in item.get("stems", []):
            raw_stems.append(
                {
                    "stem_id": stem.get("stem_id"),
                    "audio_path": Path(stem["audio_path"]),
                    "midi_path": Path(stem["midi_path"]) if stem.get("midi_path") else None,
                    "program_num": int(stem.get("program_num", -1)),
                    "is_drum": bool(stem.get("is_drum", False)),
                    "intervals": [tuple(interval) for interval in stem.get("intervals", [])],
                }
            )

        # We also materialize the current mapping view immediately for convenience,
        # but the important thing is that it is derived from raw_stems, not trusted
        # from whatever class names an old cache may have stored.
        stems, class_intervals = materialize_slakh_label_view(raw_stems)
        normalized_samples.append(
            {
                "track_id": item.get("track_id"),
                "sample_id": item.get("sample_id", str(Path(item["audio_path"]))),
                "split": str(item["split"]),
                "audio_path": Path(item["audio_path"]),
                "duration_sec": float(item["duration_sec"]),
                "raw_stems": raw_stems,
                "class_intervals": class_intervals,
                "stems": stems,
            }
        )
    return normalized_samples


def crop_waveform(
    waveform: torch.Tensor,
    target_num_samples: int,
    random_crop: bool,
) -> tuple[torch.Tensor, int]:
    # This helper takes a waveform and returns exactly target_num_samples samples.
    # If the clip is too short, we pad it.
    # If the clip is too long, we crop it and also return the starting offset that was used.
    if waveform.shape[-1] <= target_num_samples:
        return audio.trim_or_pad(waveform, target_num_samples, random_crop=False), 0

    max_offset = waveform.shape[-1] - target_num_samples
    if random_crop:
        offset = torch.randint(0, max_offset + 1, size=(1,)).item()
    else:
        # For deterministic evaluation we take the center crop.
        offset = max_offset // 2
    return waveform[offset : offset + target_num_samples], offset
