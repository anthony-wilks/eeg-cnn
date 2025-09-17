from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.signal import resample_poly, butter, filtfilt

try:
    import pyedflib
except ImportError as e:
    print("Error: pyedflib is required. Install with: pip install pyedflib", file=sys.stderr)
    raise

# Paths
RAW_ROOT = Path("./raw")
OUT_ROOT = Path("./preprocessed")
SPLITS = ["train", "eval", "test"]

# Constants
TARGET_FS = 256
TRIM_SECONDS = 60
TRIM_POINTS = TARGET_FS * TRIM_SECONDS
SEGMENT_LENGTH = 2560
CLIP_UV = 800.0
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 100.0

# Montage table
def build_montage_df() -> pd.DataFrame:
    rows = [
        ( 0, "FP1", "F7"),
        ( 1, "F7",  "T3"),
        ( 2, "T3",  "T5"),
        ( 3, "T5",  "O1"),
        ( 4, "FP2", "F8"),
        ( 5, "F8",  "T4"),
        ( 6, "T4",  "T6"),
        ( 7, "T6",  "O2"),
        ( 8, "A1",  "T3"),
        ( 9, "T3",  "C3"),
        (10, "C3",  "CZ"),
        (11, "CZ",  "C4"),
        (12, "C4",  "T4"),
        (13, "T4",  "A2"),
        (14, "FP1", "F3"),
        (15, "F3",  "C3"),
        (16, "C3",  "P3"),
        (17, "P3",  "O1"),
        (18, "FP2", "F4"),
        (19, "F4",  "C4"),
        (20, "C4",  "P4"),
        (21, "P4",  "O2"),
    ]
    return pd.DataFrame(rows, columns=["trace", "ch1", "ch2"])

# Preprocessing functions
def normalize_label(label: str) -> str:
    s = label.upper().strip()
    for token in ["EEG", "REF", "LE", "RE"]:
        s = s.replace(token, "")
    s = s.replace("-", "").replace("_", "").replace(" ", "")
    synonyms = {
        "T7": "T3",
        "T8": "T4",
        "P7": "T5",
        "P8": "T6",
        "M1": "A1",
        "M2": "A2",
    }
    return synonyms.get(s, s)

def build_label_index_map(raw_labels: List[str]) -> Dict[str, int]:
    idx_map = {}
    for i, lab in enumerate(raw_labels):
        idx_map[normalize_label(lab)] = i
    return idx_map

def _safe_band_edges(fs: float, low=BANDPASS_LOW, high=BANDPASS_HIGH):
    nyq = 0.5 * fs
    lo = max(1e-6, low / nyq)
    hi = min(0.999, high / nyq)
    if hi <= lo:
        lo = 1e-6
        hi = min(0.999, max(0.1, hi))
    return lo, hi

def bandpass_filter_1d(x: np.ndarray, fs: float) -> np.ndarray:
    lo, hi = _safe_band_edges(fs)
    b, a = butter(4, [lo, hi], btype="band")
    return filtfilt(b, a, x, axis=0)

def rational_resample_1d(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    ratio = Fraction.from_float(fs_out / fs_in).limit_denominator(1000)
    p, q = ratio.numerator, ratio.denominator
    y = resample_poly(x, p, q)
    target_len = int(round(len(x) * fs_out / fs_in))
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    return y.astype(np.float32, copy=False)

def standardize_ct(data_ct: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = data_ct.mean(axis=1, keepdims=True)
    std = data_ct.std(axis=1, keepdims=True)
    return ((data_ct - mean) / (std + eps)).astype(np.float32, copy=False)

def segment_ct(
    data_ct: np.ndarray,
    seg_len: int,
    pad: bool = True,
    pad_mode: str = "constant",
) -> np.ndarray:
    C, T = data_ct.shape[0], data_ct.shape[1]
    if T < seg_len:
        if not pad:
            return np.zeros((0, C, seg_len), dtype=data_ct.dtype)
        pad_amt = seg_len - T
        padded = np.pad(data_ct, ((0, 0), (0, pad_amt)), mode=pad_mode)
        return padded[None, :, :seg_len].astype(data_ct.dtype, copy=False)
    n_full = T // seg_len
    remainder = T % seg_len
    if remainder == 0 or not pad:
        return np.stack([data_ct[:, i*seg_len:(i+1)*seg_len] for i in range(n_full)], axis=0)
    pad_amt = seg_len - remainder
    padded = np.pad(data_ct, ((0, 0), (0, pad_amt)), mode=pad_mode)
    n = n_full + 1
    return np.stack([padded[:, i*seg_len:(i+1)*seg_len] for i in range(n)], axis=0)

# ---------- EDF reading ----------
def read_all_signals(reader: pyedflib.EdfReader) -> Tuple[List[np.ndarray], List[float]]:
    sigs, srs = [], []
    for i in range(reader.signals_in_file):
        fs = float(reader.getSampleFrequency(i))
        x = reader.readSignal(i).astype(np.float32, copy=False)
        sigs.append(x)
        srs.append(fs)
    return sigs, srs

# ---------- Core per-file processing ----------
def process_one_edf(
    edf_path: Path,
    montage_df: pd.DataFrame,
    out_root: Path,
    strict_channels: bool = False,
    standardize: bool = True,
) -> None:
    out_path = (out_root / edf_path.relative_to(RAW_ROOT)).with_suffix(".npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read EDF
    reader = pyedflib.EdfReader(str(edf_path))
    try:
        labels = reader.getSignalLabels()
        sigs, srs = read_all_signals(reader)
    finally:
        reader.close()

    n_ch = len(sigs)
    if n_ch == 0:
        np.save(out_path, np.zeros((0, 22, SEGMENT_LENGTH), dtype=np.float32))
        print(f"[WARN] Empty EDF: {edf_path.name} → saved empty array {out_path}")
        return

    # --- Align by TIME, not by raw sample count ---
    # Compute per-channel durations (seconds) and choose the minimum
    durations_sec = [len(x)/float(fs) for x, fs in zip(sigs, srs)]
    min_dur_sec = min(durations_sec)
    T_target = int(round(min_dur_sec * TARGET_FS))  # common time length after resampling

    # 1) Clip each raw channel independently
    sigs = [np.clip(x, -CLIP_UV, CLIP_UV) for x in sigs]

    # 2) Bandpass each channel at its OWN sample rate
    sigs_bp = [bandpass_filter_1d(x, fs) for x, fs in zip(sigs, srs)]

    # 3) Resample each channel to TARGET_FS and enforce common time length
    sigs_256 = [rational_resample_1d(x, fs_in=fs, fs_out=TARGET_FS) for x, fs in zip(sigs_bp, srs)]
    if T_target > 0:
        sigs_256 = [y[:T_target] if len(y) >= T_target else np.pad(y, (0, T_target - len(y))) for y in sigs_256]
    data = np.stack(sigs_256, axis=0)  # (C_raw, T_target)

    # 4) Optional trim (apply consistently with labels if you enable it)
    # if data.shape[1] > 2 * TRIM_POINTS:
    #     data = data[:, TRIM_POINTS:-TRIM_POINTS]

    # --- Build montage differences on preprocessed data ---
    idx_map = build_label_index_map(labels)
    required = set(montage_df["ch1"]) | set(montage_df["ch2"])
    missing = [ch for ch in sorted(required) if ch not in idx_map]
    if missing and strict_channels:
        raise RuntimeError(
            f"Missing channels for montage in {edf_path.name}: {missing}\n"
            f"EDF labels: {labels}"
        )

    # Gather channels present
    channel_cache: Dict[str, np.ndarray] = {}
    for ch in required:
        i = idx_map.get(ch, None)
        if i is not None:
            channel_cache[ch] = data[i]  # already preprocessed, common length

    # Montage array (22, T_target)
    T = data.shape[1]
    montage = np.zeros((len(montage_df), T), dtype=np.float32)
    for _, row in montage_df.iterrows():
        t = int(row["trace"])
        a = str(row["ch1"])
        b = str(row["ch2"])
        xa = channel_cache.get(a, None)
        xb = channel_cache.get(b, None)
        if xa is not None and xb is not None:
            montage[t] = xa - xb
        # else keep zeros for this trace

    # 5) Standardize per channel (optional)
    if standardize:
        montage = standardize_ct(montage)

    # 6) Segment into 10 s windows → (N,22,2560) with padding
    segs = segment_ct(montage, seg_len=SEGMENT_LENGTH, pad=True, pad_mode="constant")

    # Save
    np.save(out_path, segs.astype(np.float32, copy=False))
    print(f"[OK] {edf_path} → {out_path}  shape={segs.shape}  (min_dur_sec={min_dur_sec:.3f}, T_target={T_target})")

# Creating EDF list
def find_edfs(root: Path) -> List[Path]:
    paths = []
    for split in SPLITS:
        for p in (root / split).rglob("*.edf"):
            paths.append(p)
    return sorted(paths)

# CLI
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Preprocess EDFs → bipolar montage (N,22,2560)")
    ap.add_argument("--raw-root", type=Path, default=RAW_ROOT, help="Root containing {train,eval,test}")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT, help="Output root for .npy")
    ap.add_argument("--save-montage-csv", action="store_true",
                    help="Also save montage table as montage.csv in out-root")
    ap.add_argument("--strict-channels", action="store_true",
                    help="Error if any montage channel is missing in an EDF")
    ap.add_argument("--standardize", action="store_true",
                    help="Apply per-channel standardisation before segmentation")
    return ap.parse_args()

def main():
    args = parse_args()
    montage_df = build_montage_df()

    if args.save_montage_csv:
        args.out_root.mkdir(parents=True, exist_ok=True)
        montage_df.to_csv(args.out_root / "montage.csv", index=False)

    edfs = find_edfs(args.raw_root)
    if not edfs:
        print(f"No EDFs found under {args.raw_root}/{{train,eval,test}}", file=sys.stderr)
        sys.exit(1)

    for edf_path in edfs:
        try:
            process_one_edf(
                edf_path=edf_path,
                montage_df=montage_df,
                out_root=args.out_root,
                strict_channels=args.strict_channels,
                standardize=args.standardize,
            )
        except Exception as e:
            print(f"[FAIL] {edf_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
