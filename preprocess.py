"""
preprocess.py
-------------
- Reads EDF, bandpass filters, resamples to 256 Hz, trims first/last 60 s,
  and segments into 10 s windows (2560 samples).
- Loads label .npy (already 256 Hz, channels-last), trims/segments identically,
  and transposes to (segments, channels, time) to match data.
- Saves per-record dict: {'data': (N, C, T), 'labels': (N, C, T)}

Assumes:
- EDFs at ./data/raw/{train,eval,test} (non-recursive; adjust to rglob if needed)
- Labels at ./data/npy/{train,eval} with same basename (.npy)
"""

import os
from pathlib import Path
from fractions import Fraction

import numpy as np
import pyedflib
from scipy.signal import resample_poly, butter, filtfilt

# -----------------------------
# Constants
# -----------------------------
TARGET_FS = 256            # Hz
TRIM_SECONDS = 60          # seconds to trim from start and end
TRIM_POINTS = TARGET_FS * TRIM_SECONDS
SEGMENT_LENGTH = 2560      # 10 s @ 256 Hz
CLIP_UV = 800
BANDPASS_LOW = 0.5         # Hz
BANDPASS_HIGH = 100.0      # Hz

# -----------------------------
# Helpers
# -----------------------------
def _safe_band_edges(fs, low=BANDPASS_LOW, high=BANDPASS_HIGH):
    nyq = 0.5 * fs
    lo = max(1e-6, low / nyq)
    hi = min(0.999, high / nyq)
    if hi <= lo:  # fallback to lowpass if nyquist is tight
        lo = 1e-6
        hi = min(0.999, max(0.1, hi))
    return lo, hi

def bandpass_filter(data_ch_by_t, fs):
    """data shape: (C, T)."""
    lo, hi = _safe_band_edges(fs)
    b, a = butter(4, [lo, hi], btype='band')
    return filtfilt(b, a, data_ch_by_t, axis=1)

def _rational_resample(signal, fs_in, fs_out):
    """
    Polyphase resample with robust P/Q from a float ratio.
    signal: (T,) -> (T_out,)
    """
    # Ensure floats
    fs_in = float(fs_in)
    fs_out = float(fs_out)

    # Build a rational approximation of the ratio (fs_out / fs_in)
    # (avoid Fraction(fs_out, fs_in) because the 2-arg ctor requires ints)
    ratio = Fraction.from_float(fs_out / fs_in).limit_denominator(1000)
    p, q = ratio.numerator, ratio.denominator

    y = resample_poly(signal, p, q)

    # Snap to expected length to avoid +/-1 sample drift
    target_len = int(round(len(signal) * fs_out / fs_in))
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    return y

def _resample_2d(data, fs_in, fs_out):
    """
    Resample (C, T) -> (C, T_out) along axis=1 with consistent lengths.
    """
    fs_in = float(fs_in)
    fs_out = float(fs_out)
    target_len = int(round(data.shape[1] * fs_out / fs_in))
    out = np.empty((data.shape[0], target_len), dtype=np.float32)
    for c in range(data.shape[0]):
        out[c] = _rational_resample(data[c], fs_in, fs_out)
    return out

def _segment_ct(data_ct, seg_len):
    """Segment (C, T) into (N, C, seg_len). Drops leftover tail."""
    T = data_ct.shape[1]
    n = (T - seg_len) // seg_len + 1 if T >= seg_len else 0
    if n <= 0:
        return np.zeros((0, data_ct.shape[0], seg_len), dtype=data_ct.dtype)
    segs = np.stack([data_ct[:, i*seg_len:(i+1)*seg_len] for i in range(n)], axis=0)
    return segs

# -----------------------------
# Core
# -----------------------------
def preprocess_edf_file(edf_file_path, segment_length=SEGMENT_LENGTH, target_fs=TARGET_FS):
    try:
        with pyedflib.EdfReader(str(edf_file_path)) as f:
            n_channels = f.signals_in_file
            # Confirm (or assume) uniform fs across channels; if not, use channel 0 and proceed
            fs_ch0 = float(f.getSampleFrequency(0))
            n_samples_ch0 = f.getNSamples()[0]

            # Read signals into (C, T0); if channels differ in length, pad/truncate to min
            # (Most datasets keep equal lengths; this keeps it safe.)
            T_list = [f.getNSamples()[i] for i in range(n_channels)]
            T0 = min(T_list)
            data = np.empty((n_channels, T0), dtype=np.float32)
            for i in range(n_channels):
                sig = f.readSignal(i).astype(np.float32)
                if len(sig) != T0:
                    sig = sig[:T0] if len(sig) > T0 else np.pad(sig, (0, T0 - len(sig)))
                data[i] = sig

        # Clip
        data = np.clip(data, -CLIP_UV, CLIP_UV)

        # Bandpass @ original fs
        data = bandpass_filter(data, fs=fs_ch0)

        # Resample to target_fs
        if abs(fs_ch0 - target_fs) > 1e-9:
            data = _resample_2d(data, fs_in=fs_ch0, fs_out=target_fs)

        # Trim 60 s at start/end if long enough
        if data.shape[1] > 2 * TRIM_POINTS:
            data = data[:, TRIM_POINTS:-TRIM_POINTS]

        # Segment to (N, C, Tseg)
        segments = _segment_ct(data, seg_len=segment_length)
        return segments

    except Exception as e:
        print(f"Error processing {edf_file_path}: {e}")
        return None

def load_labels(label_file_path, segment_length=SEGMENT_LENGTH):
    try:
        labels = np.load(label_file_path)  # shape (T, C) expected from your generator

        # Trim 60 s
        if labels.shape[0] > 2 * TRIM_POINTS:
            labels = labels[TRIM_POINTS:-TRIM_POINTS, :]

        # Segment into (N, Tseg, C)
        T = labels.shape[0]
        n = (T - segment_length) // segment_length + 1 if T >= segment_length else 0
        if n <= 0:
            return np.zeros((0, labels.shape[1], segment_length), dtype=labels.dtype)
        segs = np.stack([labels[i*segment_length:(i+1)*segment_length, :] for i in range(n)], axis=0)

        # Transpose to (N, C, Tseg) to match data
        segs = np.transpose(segs, (0, 2, 1))
        return segs

    except Exception as e:
        print(f"Error loading labels from {label_file_path}: {e}")
        return None

def process_train_eval(edf_folder_path, label_folder_path, save_folder, segment_length=SEGMENT_LENGTH):
    # If you only want eval, set subfolders = ['eval']
    subfolders = ['train', 'eval']
    for sub in subfolders:
        print(f"Processing {sub} folder...")
        edf_subfolder = Path(edf_folder_path) / sub
        label_subfolder = Path(label_folder_path) / sub
        save_subfolder = Path(save_folder) / sub
        save_subfolder.mkdir(parents=True, exist_ok=True)

        edf_files = sorted([p for p in edf_subfolder.glob("*.edf")])
        for edf_file in edf_files:
            print(f"Processing {edf_file}...")
            data_segments = preprocess_edf_file(edf_file, segment_length)
            if data_segments is None or data_segments.shape[0] == 0:
                print(f"Skipping {edf_file} (no segments).")
                continue

            label_file = label_subfolder / (edf_file.stem + '.npy')
            label_segments = load_labels(label_file, segment_length)
            if label_segments is None:
                print(f"Skipping {edf_file} (labels missing).")
                continue

            # # Align segment counts (drop extra from the longer one)
            # n = min(data_segments.shape[0], label_segments.shape[0])
            # if n == 0:
            #     print(f"Skipping {edf_file} (no overlapping segments).")
            #     continue
            # data_segments = data_segments[:n]
            # label_segments = label_segments[:n]

            print(f"Shapes for {edf_file.stem}: data {data_segments.shape}, labels {label_segments.shape}")


            # Save dict
            save_file = save_subfolder / f"{edf_file.stem}_data_labels.npy"
            np.save(save_file, {'data': data_segments, 'labels': label_segments})
            print(f"Saved {save_file}  data{data_segments.shape}  labels{label_segments.shape}")

def process_test(edf_folder_path, save_folder, segment_length=SEGMENT_LENGTH):
    edf_subfolder = Path(edf_folder_path) / 'test'
    save_subfolder = Path(save_folder) / 'test'
    save_subfolder.mkdir(parents=True, exist_ok=True)

    edf_files = sorted([p for p in edf_subfolder.glob("*.edf")])
    for edf_file in edf_files:
        print(f"Processing {edf_file} for test...")
        data_segments = preprocess_edf_file(edf_file, segment_length)
        if data_segments is None or data_segments.shape[0] == 0:
            print(f"Skipping {edf_file} (no segments).")
            continue

        save_file = save_subfolder / f"{edf_file.stem}_data.npy"
        np.save(save_file, data_segments)
        print(f"Saved {save_file}  data{data_segments.shape}")

def process_test_labels(npy_test_folder, segment_length=SEGMENT_LENGTH):
    # Define paths
    label_files = sorted([f for f in os.listdir(npy_test_folder) if f.endswith('.npy')])

    # Ensure output folder for processed labels
    processed_labels_folder = './data/processed_data/test_labels'
    Path(processed_labels_folder).mkdir(parents=True, exist_ok=True)

    # Process each label file
    for label_file in label_files:
        label_file_path = os.path.join(npy_test_folder, label_file)
        print(f"Processing {label_file_path}...")
        
        # Load and segment labels
        label_segments = load_labels(label_file_path, segment_length)
        if label_segments is None:
            continue
        
        # Save the segmented labels to the processed folder
        save_path = os.path.join(processed_labels_folder, label_file)
        np.save(save_path, label_segments)
        print(f"Saved processed labels to {save_path}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    raw_folder = './data/raw'
    npy_folder = './data/npy'
    save_folder = './data/processed_data'
    npy_test_folder = './data/npy/test'
    os.makedirs(save_folder, exist_ok=True)

    process_train_eval(raw_folder, npy_folder, save_folder)
    process_test(raw_folder, save_folder)
    process_test_labels(npy_test_folder)
