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
MAXR = 6           # max receivers kept per recording


def load_recording(path):
    z = np.load(path, allow_pickle=True)
    F, fs = z["F"], float(z["fs_time"])
    b = max(int(round(fs / RATE)), 1)
    T = (F.shape[0] // b) * b
    F = F[:T].reshape(T // b, b, *F.shape[1:]).mean(1)       # (T', R, NF)
    if F.shape[1] > MAXR:                                      # balance sites: evenly spaced subset of receivers
        F = F[:, np.linspace(0, F.shape[1] - 1, MAXR).round().astype(int)]
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
    """Fit on TRAIN only. Scale-invariant (RevIN-style) pipeline shared by ALL models:
    inputs are window-relative and divided by the window's own volatility (std of first differences);
    targets are divided by the same local volatility; log-volatility and normalised absolute level are extra inputs."""
    def fit(self, X, C, Y, last):
        self.abs_m, self.abs_s = last.mean(0), last.std(0) + 1e-6
        s = self.local_scale(X, raw=True)
        self.floor = 0.2 * np.median(s, 0) + 1e-6
        return self

    @staticmethod
    def local_scale(X, raw=False):
        s = np.diff(X, axis=1).std(1) + 1e-9
        return s

    def scale(self, X):
        return np.maximum(np.diff(X, axis=1).std(1), self.floor)            # (N,NF)

    def tgt_scale(self, X):
        s = self.scale(X)[:, list(TGT)]                                       # (N,2) gain, delay
        return np.tile(s, (1, len(HSTEPS)))                                   # (N,6) matching [g,d]*H ordering

    def x(self, X, C, last, use_abs=True, use_ctx=True):
        s = self.scale(X)
        parts = [(X - X[:, -1:, :]) / s[:, None, :]]
        if use_ctx:
            parts.append((C - C[:, -1:, :]) / (np.maximum(np.diff(C, axis=1).std(1), self.floor)[:, None, :]))
        parts.append(np.log(s / self.floor)[:, None, :].repeat(X.shape[1], 1) * 0.3)      # volatility regime
        if use_abs:
            parts.append(((last - self.abs_m) / self.abs_s)[:, None, :].repeat(X.shape[1], 1))
        return np.concatenate(parts, -1).astype(np.float32)
