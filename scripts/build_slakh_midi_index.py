#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml

# When this file is run as `python scripts/build_slakh_midi_index.py`, Python puts
# `scripts/` on sys.path, not the repo root. We add the repo root explicitly so the
# `training_base` package can be imported reliably.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_base.data.utils import (
    build_slakh_track_stem_entries,
    get_audio_duration_sec,
    materialize_slakh_label_view,
)


CANONICAL_SPLIT_NAMES = {
    "train": "train",
    "validation": "validation",
    "val": "validation",
    "vallidation": "validation",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reusable global Slakh MIDI interval index."
    )
    parser.add_argument("--root", required=True, help="Path to the Slakh root directory")
    parser.add_argument("--output", required=True, help="Output .pt path")
    parser.add_argument("--num-classes", type=int, default=20, help="Target label-space size")
    parser.add_argument("--print-every", type=int, default=100, help="Progress print frequency")
    return parser.parse_args()


def read_track_metadata(track_dir: Path) -> Dict[str, Any]:
    with open(track_dir / "metadata.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def find_available_splits(root: Path) -> List[Tuple[str, Path]]:
    discovered: List[Tuple[str, Path]] = []
    seen_dirs = set()
    for split_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        resolved_dir = split_dir.resolve()
        if resolved_dir in seen_dirs:
            continue
        seen_dirs.add(resolved_dir)
        split_name = CANONICAL_SPLIT_NAMES.get(split_dir.name.lower(), split_dir.name.lower())
        discovered.append((split_name, split_dir))
    if not discovered:
        raise FileNotFoundError(f"No Slakh split directories found under {root}")
    return discovered


def build_index_samples_for_split(
    split_name: str,
    split_dir: Path,
    num_classes: int,
    print_every: int,
) -> Dict[str, Dict[str, Any]]:
    track_dirs = sorted(track_dir for track_dir in split_dir.iterdir() if track_dir.is_dir())
    tracks: Dict[str, Dict[str, Any]] = {}

    for idx, track_dir in enumerate(track_dirs, start=1):
        if idx == 1 or idx % print_every == 0 or idx == len(track_dirs):
            print(f"[{split_name} {idx}/{len(track_dirs)}] indexing {track_dir.name}", flush=True)

        mix_path = track_dir / "mix.flac"
        midi_dir = track_dir / "MIDI"
        meta_path = track_dir / "metadata.yaml"
        if not (mix_path.exists() and midi_dir.exists() and meta_path.exists()):
            continue

        metadata = read_track_metadata(track_dir)
        raw_stem_entries = build_slakh_track_stem_entries(
            track_dir=track_dir,
            stems_metadata=metadata.get("stems", {}),
        )
        if not raw_stem_entries:
            continue

        # We also compute the currently mapped class intervals for reporting/debugging,
        # but the important thing is that the cache stores the raw stems. Later the
        # dataset can remap those stems again using whatever mapping is current then.
        _, class_intervals = materialize_slakh_label_view(raw_stem_entries)

        tracks[track_dir.name] = {
            "track_id": track_dir.name,
            "sample_id": str(mix_path.resolve()),
            "split": split_name,
            "audio_path": str(mix_path),
            "duration_sec": get_audio_duration_sec(mix_path),
            "class_intervals": {
                str(class_name): [(float(start_sec), float(end_sec)) for start_sec, end_sec in intervals]
                for class_name, intervals in class_intervals.items()
            },
            "stems": [
                {
                    "stem_id": stem["stem_id"],
                    "audio_path": str(stem["audio_path"]),
                    "midi_path": str(stem["midi_path"]) if stem.get("midi_path") is not None else None,
                    "program_num": int(stem["program_num"]),
                    "is_drum": bool(stem["is_drum"]),
                    "intervals": [(float(start_sec), float(end_sec)) for start_sec, end_sec in stem["intervals"]],
                }
                for stem in raw_stem_entries
            ],
        }

    return tracks


def main() -> None:
    args = parse_args()

    root = Path(args.root).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    available_splits = find_available_splits(root)
    all_tracks: Dict[str, Dict[str, Any]] = {}
    split_counts: Dict[str, int] = {}

    for split_name, split_dir in available_splits:
        split_tracks = build_index_samples_for_split(
            split_name=split_name,
            split_dir=split_dir,
            num_classes=args.num_classes,
            print_every=args.print_every,
        )
        overlap = set(all_tracks).intersection(split_tracks)
        if overlap:
            raise ValueError(
                f"Duplicate track ids across splits detected: {sorted(overlap)[:5]}"
            )
        all_tracks.update(split_tracks)
        split_counts[split_name] = len(split_tracks)

    payload = {
        "root": str(root),
        "num_classes": args.num_classes,
        "split_counts": split_counts,
        "tracks": all_tracks,
    }
    torch.save(payload, output_path)

    print(f"Saved global Slakh index with {len(all_tracks)} tracks to {output_path}")
    for split_name, count in split_counts.items():
        print(f"  {split_name}: {count}")


if __name__ == "__main__":
    main()
