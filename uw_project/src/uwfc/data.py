"""Windowing, splitting and normalisation for channel forecasting.

Task: from the last `L` steps of measured channel descriptors at one receiver (plus mean of the other receivers),
forecast the CHANGE of [gain_dB, rms_delay_ms] at horizons HZ (seconds ahead).
All recordings are resampled (block-mean) to a common RATE Hz grid.
"""
from pathlib import Path
import numpy as np

RATE = 16          # Hz common grid
L = 48             # input steps (3 s)
HZ = (0.5, 1.0, 2.0)   # horizons in seconds
HSTEPS = tuple(int(h * RATE) for h in HZ)
TGT = (0, 1)       # target feature indices: gain_db, rms_delay_ms
NF = 5


def load_recording(path):
    z = np.load(path, allow_pickle=True)
    F, fs = z["F"], float(z["fs_time"])
    b = max(int(round(fs / RATE)), 1)
    T = (F.shape[0] // b) * b
    F = F[:T].reshape(T // b, b, *F.shape[1:]).mean(1)       # (T', R, NF)
    return F, str(z["desc"])


def make_windows(F, t_lo, t_hi, stride=2):
    """Windows whose input AND target all lie inside [t_lo, t_hi) -> no cross-boundary leakage."""
    T, R, _ = F.shape
    X, C, Y, last = [], [], [], []
    hmax = max(HSTEPS)
    for t in range(t_lo + L, t_hi - hmax, stride):          # t = first future index
        w = F[t - L:t]                                        # (L,R,NF)
        for r in range(R):
            own = w[:, r]
            others = np.delete(w, r, axis=1)
            ctx = others.mean(1) if others.shape[1] else np.zeros_like(own)
            y = np.stack([F[t - 1 + h, r, list(TGT)] - own[-1, list(TGT)] for h in HSTEPS])  # (H,2)
            X.append(own); C.append(ctx); Y.append(y.ravel()); last.append(own[-1])
    if not X:
        return None
    return np.stack(X).astype(np.float32), np.stack(C).astype(np.float32), np.stack(Y).astype(np.float32), np.stack(last).astype(np.float32)


class Norm:
    """Fit on TRAIN only. Inputs are made window-relative (site-invariant) + normalised absolute last value."""
    def fit(self, X, C, Y, last):
        rel = X - X[:, -1:, :]
        self.rel_s = rel.reshape(-1, NF).std(0) + 1e-6
        self.abs_m, self.abs_s = last.mean(0), last.std(0) + 1e-6
        self.y_s = Y.std(0) + 1e-6
        return self

    def x(self, X, C, last, use_abs=True, use_ctx=True):
        a = (X - X[:, -1:, :]) / self.rel_s
        parts = [a]
        if use_ctx:
            parts.append((C - C[:, -1:, :]) / self.rel_s)
        if use_abs:
            ab = ((last - self.abs_m) / self.abs_s)[:, None, :].repeat(X.shape[1], 1)
            parts.append(ab)
        return np.concatenate(parts, -1).astype(np.float32)
