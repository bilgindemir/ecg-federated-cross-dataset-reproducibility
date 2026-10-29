import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt
FS = 200

def preprocess_signal(x, fs_original):
    """Native-rate 0.5-40 Hz filter -> PCHIP -> record-wise z-score.

    PCHIP is algebraically the cubic-Hermite/Bezier implementation in the
    uploaded code. Preserve native filtering before downsampling.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != 12 or not np.isfinite(x).all(): raise ValueError("Invalid 12-lead input")
    b, a = butter(4, [0.5 / (fs_original / 2), 40 / (fs_original / 2)], btype="band")
    x = filtfilt(b, a, x, axis=1).astype(np.float32)
    old = np.arange(x.shape[1], dtype=np.float64) / fs_original
    new = np.arange(round(x.shape[1] * FS / fs_original), dtype=np.float64) / FS
    x = PchipInterpolator(old, x, axis=1)(new).astype(np.float32)
    x = ((x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    if not np.isfinite(x).all(): raise ValueError("Nonfinite preprocessed signal")
    return x
