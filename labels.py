"""
rec_to_npy_structured.py
------------------------
Convert .rec annotation files into dense binary label arrays (.npy),
sized to EDF duration, sampled at 256 Hz with 22 channels.

- .rec under: edf/train/**, edf/eval/**, edf/test/**
- .edf under: raw/train/**, raw/eval/**, raw/test/** (same stem as .rec)
- Output: npy/<split>/<same relative path under split>/<stem>.npy

So if EDF is raw/test/patient1/session/file.edf,
labels are saved to npy/test/patient1/session/file.npy
"""

from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

try:
    import pyedflib
except ImportError:
    print("Error: pyedflib is required. Install with: pip install pyedflib", file=sys.stderr)
    raise

POSITIVE_LABELS = {1, 2, 3}   # spsw, gped, pled
FS = 256                      # fixed label sampling rate
N_CHANNELS = 22                # fixed channel count


def read_rec(rec_path: Path) -> List[Tuple[int, float, float, int]]:
    """Parse .rec into list of (channel, start_sec, stop_sec, label_int)."""
    events: List[Tuple[int, float, float, int]] = []
    with rec_path.open("r", newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [c.strip() for c in line.split(",") if c.strip()]
            if len(parts) < 4:
                continue
            try:
                ch = int(parts[0]); start = float(parts[1]); stop = float(parts[2]); label = int(parts[3])
            except Exception:
                continue
            if stop < start:
                start, stop = stop, start
            events.append((ch, start, stop, label))
    return events


def edf_duration_seconds(edf_path: Path) -> float:
    """Read EDF header to get full duration in seconds."""
    with pyedflib.EdfReader(str(edf_path)) as f:
        return float(f.getFileDuration())


def build_binary_array(events: List[Tuple[int, float, float, int]], duration_sec: float) -> np.ndarray:
    """Dense (samples, channels) array at 256 Hz across EDF duration."""
    n_samples = max(1, int(round(duration_sec * FS)))
    arr = np.zeros((n_samples, N_CHANNELS), dtype=np.uint8)
    for ch, start, stop, label in events:
        if label not in POSITIVE_LABELS:
            continue
        if not (0 <= ch < N_CHANNELS):
            continue
        start_idx = max(0, int(math.floor(start * FS)))
        stop_idx  = min(n_samples, int(math.ceil(stop * FS)))
        if stop_idx > start_idx:
            arr[start_idx:stop_idx, ch] = 1
    return arr


def infer_split(rec_path: Path, edf_root: Path) -> Optional[str]:
    """Infer split (train/eval/test) from rec_path under edf_root."""
    try:
        rel = rec_path.relative_to(edf_root)
    except ValueError:
        return None
    return rel.parts[0].lower() if rel.parts else None


def find_matching_edf(stem: str, raw_split_root: Path) -> Optional[Path]:
    """Find <stem>.edf under raw/<split> recursively (case-insensitive)."""
    candidates = list(raw_split_root.rglob(f"{stem}.edf")) + list(raw_split_root.rglob(f"{stem}.EDF"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: len(p.relative_to(raw_split_root).parts))
    return candidates[0]


def process_all(edf_root: Path, raw_root: Path, out_root: Path) -> None:
    rec_files = list(edf_root.rglob("*.rec"))
    if not rec_files:
        print(f"No .rec files found under {edf_root}", file=sys.stderr)
        return

    print(f"Found {len(rec_files)} .rec files under {edf_root}")
    processed = 0

    for rec_path in rec_files:
        split = infer_split(rec_path, edf_root)
        if split not in {"train", "eval", "test"}:
            print(f"Skipping {rec_path}, could not infer split.", file=sys.stderr)
            continue

        raw_split_root = raw_root / split
        if not raw_split_root.exists():
            print(f"Missing raw split folder {raw_split_root}, skipping {rec_path}", file=sys.stderr)
            continue

        stem = rec_path.stem
        edf_path = find_matching_edf(stem, raw_split_root)
        if edf_path is None:
            print(f"No EDF found for {rec_path.name} in {raw_split_root}", file=sys.stderr)
            continue

        events = read_rec(rec_path)
        try:
            duration_sec = edf_duration_seconds(edf_path)
        except Exception as e:
            print(f"Failed to read EDF {edf_path}: {e}, skipping.", file=sys.stderr)
            continue

        arr = build_binary_array(events, duration_sec)

        # Save under npy/<split>/<relative subpath from raw/<split>>/<stem>.npy
        rel_subpath = edf_path.relative_to(raw_split_root).with_suffix(".npy")
        out_path = out_root / split / rel_subpath
        out_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(out_path, arr)
        print(f"Saved {out_path}  shape={arr.shape}  positives={arr.sum()}")
        processed += 1

    print(f"Done. Processed {processed} files.")


def main():
    parser = argparse.ArgumentParser(description="Convert .rec to dense binary .npy (22 ch @ 256 Hz) sized to EDF duration, preserving raw folder structure under split.")
    parser.add_argument("--edf-dir", type=Path, default=Path("edf"), help="Root with .rec files (edf/train|eval|test).")
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"), help="Root with .edf files (raw/train|eval|test).")
    parser.add_argument("--out-dir", type=Path, default=Path("npy"), help="Output root directory ('npy').")
    args = parser.parse_args()
    process_all(args.edf_dir, args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
