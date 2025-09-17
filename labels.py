from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np

try:
    import pyedflib
except ImportError:
    pyedflib = None  # We'll warn and fall back if unavailable

# Paths
NPY_ROOT = Path("./npy")
OUT_ROOT = Path("./label")
RAW_ROOT = Path("./raw")
SPLITS = ("train", "eval", "test")

# Constants
TARGET_FS_DEFAULT = 256
TRIM_SECONDS_DEFAULT = 60
SEGMENT_SECONDS_DEFAULT = 10

# ----------------- Resampling -----------------
def _resample_1d_binary(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Nearest-neighbor resample of a 1D binary signal to avoid label bleeding."""
    if fs_in == fs_out or len(x) == 0:
        return x.astype(np.float32, copy=False)
    T_out = int(round(len(x) * fs_out / fs_in))
    scale = fs_in / fs_out
    idx = np.clip((np.arange(T_out) * scale).round().astype(np.int64), 0, len(x) - 1)
    return x[idx].astype(np.float32, copy=False)

def resample_labels_TxC(arr_TxC: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if fs_in == fs_out:
        return arr_TxC
    T_out = int(round(arr_TxC.shape[0] * fs_out / fs_in))
    out = np.empty((T_out, arr_TxC.shape[1]), dtype=np.float32)
    for c in range(arr_TxC.shape[1]):
        out[:, c] = _resample_1d_binary(arr_TxC[:, c], fs_in, fs_out)
    return out

# ----------------- Trimming & Segmenting -----------------
def trim_edges_TxC(arr_TxC: np.ndarray, fs: int, trim_seconds: int) -> np.ndarray:
    trim_pts = int(fs * trim_seconds)
    if arr_TxC.shape[0] > 2 * trim_pts:
        return arr_TxC[trim_pts:-trim_pts]
    return arr_TxC

def segment_TxC_to_NCT(
    arr_TxC: np.ndarray,
    fs: int | float,
    segment_seconds: int,
    n_channels_expected: int = 22,
    pad: bool = True,
) -> np.ndarray:
    """(T, C) -> (N, C, Tseg) with optional tail padding."""
    if arr_TxC.ndim != 2:
        raise ValueError(f"Expected 2D (T,C), got {arr_TxC.shape}")
    T, C = arr_TxC.shape
    if C != n_channels_expected:
        print(f"[WARN] Channels = {C} (expected {n_channels_expected}); proceeding.", file=sys.stderr)
    Tseg = int(round(float(fs) * float(segment_seconds)))
    if Tseg <= 0:
        raise ValueError(f"Computed non-positive segment length: {Tseg}")
    if T < Tseg:
        if pad:
            pad_amt = Tseg - T
            arr_TxC = np.pad(arr_TxC, ((0, pad_amt), (0, 0)), mode='constant')
            T = arr_TxC.shape[0]
        else:
            return np.zeros((0, C, Tseg), dtype=arr_TxC.dtype)

    N = T // Tseg
    remainder = T % Tseg
    if remainder != 0 and pad:
        pad_amt = Tseg - remainder
        arr_TxC = np.pad(arr_TxC, ((0, pad_amt), (0, 0)), mode='constant')
        T = arr_TxC.shape[0]
        N = T // Tseg

    trimmed = arr_TxC[: N * Tseg]
    out = trimmed.reshape(N, Tseg, C).transpose(0, 2, 1)  # (N, C, Tseg)
    return out

# ----------------- IO & EDF duration discovery -----------------
def load_TxC(path: Path) -> Tuple[np.ndarray, bool]:
    arr = np.load(path)
    transposed = False
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected 2D label array, got {arr.shape}")
    # Auto (22,T) -> (T,22)
    if arr.shape[0] == 22 and arr.shape[1] != 22:
        arr = arr.T
        transposed = True
    elif arr.shape[1] != 22 and arr.shape[0] != 22:
        # Ambiguous; assume (T,C)
        pass
    return arr.astype(np.float32, copy=False), transposed

def save_NCT(path: Path, arr_NCT: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr_NCT.dtype != np.uint8:
        arr_NCT = (arr_NCT > 0.5).astype(np.uint8, copy=False)
    np.save(path, arr_NCT)

def npy_to_edf_path(npy_root: Path, raw_root: Path, npy_path: Path) -> Path:
    """Map npy/<split>/.../file.npy -> raw/<split>/.../file.edf"""
    rel = npy_path.relative_to(npy_root).with_suffix(".edf")
    return raw_root / rel

def read_min_duration_sec_from_edf(edf_path: Path) -> Optional[float]:
    """
    Return the minimum channel duration in seconds from the EDF.
    This is the common time base used by the EDF pipeline.
    """
    if pyedflib is None:
        print("[ERROR] pyedflib not installed; cannot read EDF duration.", file=sys.stderr)
        return None
    if not edf_path.exists():
        print(f"[WARN] EDF not found for labels: {edf_path}", file=sys.stderr)
        return None
    try:
        reader = pyedflib.EdfReader(str(edf_path))
        try:
            n_sig = reader.signals_in_file
            if n_sig <= 0:
                return None
            durations = []
            for i in range(n_sig):
                fs = float(reader.getSampleFrequency(i))
                n = int(reader.getNSamples()[i]) if hasattr(reader, "getNSamples") else len(reader.readSignal(i))
                durations.append(n / max(fs, 1e-9))
            return float(min(durations)) if durations else None
        finally:
            reader.close()
    except Exception as e:
        print(f"[WARN] Failed to read EDF {edf_path}: {e}", file=sys.stderr)
        return None

# ----------------- Per-file processing -----------------
def process_one_file(
    src: Path,
    dst_root: Path,
    npy_root: Path,
    raw_root: Path,
    cli_fs_in: float,
    fs_out: float,
    trim_seconds: int,
    segment_seconds: int,
) -> None:
    # Destination path
    rel = src.relative_to(npy_root)
    dst = (dst_root / rel).with_suffix('.npy')

    try:
        arr_TxC, transposed = load_TxC(src)  # (T_in_labels, C)
    except Exception as e:
        print(f"[FAIL] {src}: {e}", file=sys.stderr)
        return

    T_in_labels = int(arr_TxC.shape[0])

    # Discover EDF min duration (seconds)
    edf_path = npy_to_edf_path(npy_root=npy_root, raw_root=raw_root, npy_path=src)
    min_dur_sec = read_min_duration_sec_from_edf(edf_path)

    if min_dur_sec is not None and min_dur_sec > 0:
        # Infer labels' actual input fs from duration
        fs_in_labels = float(T_in_labels) / float(min_dur_sec)
        # Resample labels to target fs
        arr_TxC = resample_labels_TxC(arr_TxC, fs_in=fs_in_labels, fs_out=fs_out)
        # Hard-enforce exact time alignment to EDF duration at target fs
        T_target = int(round(min_dur_sec * fs_out))
        if arr_TxC.shape[0] < T_target:
            pad_amt = T_target - arr_TxC.shape[0]
            arr_TxC = np.pad(arr_TxC, ((0, pad_amt), (0, 0)), mode='constant')
        elif arr_TxC.shape[0] > T_target:
            arr_TxC = arr_TxC[:T_target]
        debug_tag = f"(fs_in≈{fs_in_labels:.6g} Hz, min_dur={min_dur_sec:.3f}s, T_target={T_target})"
    else:
        # Fallback: use CLI fs_in (older behavior)
        fs_in_labels = float(cli_fs_in)
        arr_TxC = resample_labels_TxC(arr_TxC, fs_in=fs_in_labels, fs_out=fs_out)
        debug_tag = f"(EDF missing → fallback fs_in={fs_in_labels} Hz)"

    # Optional trim (keep consistent with EDF pipeline)
    # arr_TxC = trim_edges_TxC(arr_TxC, fs=fs_out, trim_seconds=trim_seconds)

    # Segment (with padding)
    segs_NCT = segment_TxC_to_NCT(
        arr_TxC, fs=fs_out, segment_seconds=segment_seconds, n_channels_expected=22, pad=True
    )

    # Save
    save_NCT(dst, segs_NCT)
    print(f"[OK] {src} -> {dst}  shape={segs_NCT.shape}  {debug_tag}")

# ----------------- Discovery -----------------
def list_label_npy_files(npy_root: Path) -> List[Path]:
    files: List[Path] = []
    for split in SPLITS:
        root = npy_root / split
        if not root.exists():
            continue
        files.extend(root.rglob('*.npy'))
    return sorted(files)

# ----------------- CLI -----------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Preprocess dense label arrays: resample/trim/segment -> (N,22,2560)")
    ap.add_argument('--npy-root', type=Path, default=NPY_ROOT, help='Root of dense labels (npy/{train,eval,test})')
    ap.add_argument('--raw-root', type=Path, default=RAW_ROOT, help='Root of raw EDFs (raw/{train,eval,test})')
    ap.add_argument('--out-root', type=Path, default=OUT_ROOT, help='Root to save segmented labels (label/{train,eval,test})')
    ap.add_argument('--input-fs', type=float, default=TARGET_FS_DEFAULT, help='Fallback input label sampling rate (Hz) if EDF missing')
    ap.add_argument('--target-fs', type=float, default=TARGET_FS_DEFAULT, help='Target label sampling rate (Hz), usually 256')
    ap.add_argument('--trim-seconds', type=int, default=TRIM_SECONDS_DEFAULT, help='Trim this many seconds at both start and end (if long enough)')
    ap.add_argument('--segment-seconds', type=int, default=SEGMENT_SECONDS_DEFAULT, help='Segment window length in seconds (10 -> 2560 samples @256 Hz)')
    return ap.parse_args()

def main():
    args = parse_args()

    # Ensure split roots exist in the output
    for split in SPLITS:
        (args.out_root / split).mkdir(parents=True, exist_ok=True)

    files = list_label_npy_files(args.npy_root)
    if not files:
        print(f"No .npy label files found under {args.npy_root}/{{train,eval,test}}", file=sys.stderr)
        sys.exit(1)

    if pyedflib is None:
        print("[WARN] pyedflib not installed; will use --input-fs fallback for all files (no EDF duration sync).", file=sys.stderr)

    for src in files:
        try:
            process_one_file(
                src=src,
                dst_root=args.out_root,
                npy_root=args.npy_root,
                raw_root=args.raw_root,
                cli_fs_in=float(args.input_fs),
                fs_out=float(args.target_fs),
                trim_seconds=int(args.trim_seconds),
                segment_seconds=int(args.segment_seconds),
            )
        except Exception as e:
            print(f"[FAIL] {src}: {e}", file=sys.stderr)

if __name__ == '__main__':
    main()
