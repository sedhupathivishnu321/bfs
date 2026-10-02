"""Feature extraction from real measured underwater acoustic channel impulse responses.

Input : h_hat (T, R, L) complex, time-varying CIR for R receivers, L delay taps (Zenodo 10.5281/zenodo.21287414).
Output: per-frame, per-receiver channel descriptors computed ONLY from the measured CIR (no simulation).
"""
import h5py
import numpy as np

FEATURES = ["gain_db", "rms_delay_ms", "peak_ratio", "spec_flat_db", "peak_tap_ms"]


def load_cir(path):
    with h5py.File(path, "r") as f:
        h = f["h_hat"][()]
        h = h["real"] + 1j * h["imag"]
        m = f["meta"]
        txt = lambda k: "".join(chr(c) for c in m[k][()].ravel())
        info = dict(codename=txt("codename"), description=txt("description"),
                    fc=float(f["params/fc"][()].ravel()[0]), fs_delay=float(f["params/fs_delay"][()].ravel()[0]),
                    fs_time=float(f["params/fs_time"][()].ravel()[0]))
    return h, info


def describe(h, fs_delay, eps=1e-20):
    """h: (T,R,L) -> features (T,R,F)."""
    p = np.abs(h) ** 2
    tot = p.sum(-1) + eps
    tau = np.arange(h.shape[-1]) / fs_delay * 1e3  # ms
    mean_tau = (p * tau).sum(-1) / tot
    rms = np.sqrt(np.maximum((p * tau ** 2).sum(-1) / tot - mean_tau ** 2, 0))
    peak = p.max(-1) / tot
    H = np.abs(np.fft.fft(h, axis=-1)) ** 2 + eps
    flat = 10 * np.log10(np.exp(np.log(H).mean(-1)) / H.mean(-1))  # spectral flatness (<=0 dB)
    ptap = tau[p.argmax(-1)]
    return np.stack([10 * np.log10(tot), rms, peak, flat, ptap], -1).astype(np.float32)
