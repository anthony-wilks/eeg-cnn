"""
EEG augmentation using max safe shift per file (no fixed shift list):
- Inputs:
  * Signals: ./data/preprocessed/train/<name>.npy of shape (S, 22, 2560)
  * Labels : ./data/labels/train/<name>.npy      of shape (S, 22, 2560) (binary)
- For each label file containing ≥1 event:
  * Compute per-segment event bounds (across channels); ignore segments with no events
  * Max safe left shift (samples)  = min(start_idx across event segments)
  * Max safe right shift (samples) = min((T-1)-end_idx across event segments)
  * Create augments for:
      - left shift by max_left (if > 0)
      - right shift by max_right (if > 0)
    and for each, scale the signal by each factor in SCALES (labels are not scaled)
  * Save signal & label pairs with matching suffixes
"""

from pathlib import Path
import numpy as np

FS_HZ = 256
T = 10 * FS_HZ  # 2560 samples per 10s segment
SCALES = [0.9, 0.95, 1.0, 1.05, 1.1]

LABELS_DIR = Path("./data/labels/train")
PREPROC_DIR = Path("./data/preprocessed/train")
OVERWRITE = False

def load_npy(p: Path) -> np.ndarray:
    return np.load(p, allow_pickle=False)

def save_npy(p: Path, arr: np.ndarray):
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, arr)

def any_event(labels_file: np.ndarray) -> bool:
    return np.any(labels_file == 1)

def event_bounds_per_segment(lbl_seg: np.ndarray):
    """
    lbl_seg: (22, T) binary
    Returns (start_idx, end_idx) for this segment if any event exists, else None.
    Event is defined as any channel active at a time sample (union across channels).
    """
    time_any = lbl_seg.any(axis=0)  # (T,)
    if not time_any.any():
        return None
    idx = np.flatnonzero(time_any)
    return int(idx[0]), int(idx[-1])

def safe_shift_caps(labels_file: np.ndarray):
    """
    Compute maximum safe left/right shift (samples) across all segments with events.
    Segments without events do not constrain the caps.
    """
    left_caps, right_caps = [], []
    for s in range(labels_file.shape[0]):
        bounds = event_bounds_per_segment(labels_file[s])
        if bounds is None:
            continue
        start, end = bounds
        left_caps.append(start)            # ≤ start to avoid wrapping at t=0
        right_caps.append((T - 1) - end)   # ≤ T-1-end to avoid wrapping at t=T-1
    if not left_caps:
        return 0, 0  # no events found (caller skips such files anyway)
    return int(min(left_caps)), int(min(right_caps))

def roll_time(arr: np.ndarray, k: int) -> np.ndarray:
    """Roll along time axis (last axis). Positive k = shift right; negative = left."""
    return np.roll(arr, k, axis=-1)

def names_with_suffix(sig_path: Path, lab_path: Path, direction: str, shift_samples: int, scale: float):
    stem_sig, suffix_sig = sig_path.stem, sig_path.suffix
    stem_lab, suffix_lab = lab_path.stem, lab_path.suffix
    shift_sec = shift_samples / FS_HZ
    # direction: 'l' (left) or 'r' (right); include seconds in name for clarity
    new_sig = sig_path.with_name(f"{stem_sig}_shift{direction}{shift_sec:g}s_scale{scale:.2f}{suffix_sig}")
    new_lab = lab_path.with_name(f"{stem_lab}_shift{direction}{shift_sec:g}s_scale{scale:.2f}{suffix_lab}")
    return new_sig, new_lab

def apply_and_save_shift(direction, shift_samples, signal, labels, sig_path, lab_path,
                         saved_pairs_ref, skipped_existing_ref):
    """
    Apply a given shift (samples) left or right, then save all scale variants.
    Updates counters by reference (ints wrapped in list).
    """
    if shift_samples <= 0:
        return
    k = -shift_samples if direction == 'l' else shift_samples
    shifted_sig = roll_time(signal, k)
    shifted_lab = roll_time(labels, k)
    for scale in SCALES:
        out_sig, out_lab = names_with_suffix(sig_path, lab_path, direction, shift_samples, scale)
        if not OVERWRITE and out_sig.exists() and out_lab.exists():
            skipped_existing_ref[0] += 1
            continue
        save_npy(out_sig, (shifted_sig * scale).astype(signal.dtype, copy=False))
        save_npy(out_lab, shifted_lab.astype(labels.dtype, copy=False))
        saved_pairs_ref[0] += 1

def main():
    if not LABELS_DIR.exists() or not PREPROC_DIR.exists():
        raise FileNotFoundError("Expected ./data/labels/train and ./data/preprocessed/train")

    label_files = sorted(LABELS_DIR.glob("*.npy"))
    scanned = len(label_files)
    skipped_all_zero = 0
    skipped_missing_signal = 0
    saved_pairs = [0]        # use list as mutable ref for helper
    skipped_existing = [0]
    skipped_zero_caps = 0

    for lab_path in label_files:
        labels = load_npy(lab_path)
        if not any_event(labels):
            skipped_all_zero += 1
            continue

        sig_path = PREPROC_DIR / lab_path.name
        if not sig_path.exists():
            skipped_missing_signal += 1
            print(f"[WARN] Missing signal for labels: {lab_path.name}")
            continue
        signal = load_npy(sig_path)

        # Sanity checks
        if labels.shape != signal.shape:
            print(f"[WARN] Shape mismatch for {lab_path.name}: labels {labels.shape} vs signal {signal.shape}")
            continue
        if labels.shape[-1] != T:
            print(f"[WARN] Unexpected time length for {lab_path.name}: got {labels.shape[-1]}, expected {T}")
            continue

        max_left_samp, max_right_samp = safe_shift_caps(labels)

        did_any = False

        # LEFT: apply max-left and half-left (if positive)
        if max_left_samp > 0:
            apply_and_save_shift('l', max_left_samp, signal, labels, sig_path, lab_path,
                                 saved_pairs, skipped_existing)
            half_left = max_left_samp // 2
            if half_left > 0 and half_left != max_left_samp:
                apply_and_save_shift('l', half_left, signal, labels, sig_path, lab_path,
                                     saved_pairs, skipped_existing)
            did_any = True

        # RIGHT: apply max-right and half-right (if positive)
        if max_right_samp > 0:
            apply_and_save_shift('r', max_right_samp, signal, labels, sig_path, lab_path,
                                 saved_pairs, skipped_existing)
            half_right = max_right_samp // 2
            if half_right > 0 and half_right != max_right_samp:
                apply_and_save_shift('r', half_right, signal, labels, sig_path, lab_path,
                                     saved_pairs, skipped_existing)
            did_any = True

        if not did_any:
            skipped_zero_caps += 1  # nothing to shift safely for this file

    print("\n=== Max+Half-Shift Augmentation Summary ===")
    print(f"Label files scanned           : {scanned}")
    print(f"Skipped (all-zero labels)     : {skipped_all_zero}")
    print(f"Skipped (no signal match)     : {skipped_missing_signal}")
    print(f"Saved signal/label pairs      : {saved_pairs[0]}  (each pair = one signal + one label)")
    print(f"Skipped existing (no overwrite): {skipped_existing[0]}")
    print(f"Files with zero safe shift    : {skipped_zero_caps}")
    print("Done.")

if __name__ == "__main__":
    main()
