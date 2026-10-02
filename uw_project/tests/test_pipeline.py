import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uwfc import data as D, baselines as B, metrics as M
from uwfc.models import make_model
import torch


def synth(T=400, R=3, seed=0):
    r = np.random.default_rng(seed); return np.cumsum(r.normal(size=(T, R, D.NF)), 0).astype(np.float32)


def test_windows_do_not_cross_segment_boundary():
    F = synth(); lo, hi = 100, 300
    X, C, Y, last = D.make_windows(F, lo, hi, stride=1)
    # reconstruct: every input/target index must lie in [lo, hi)
    n_per_rec = hi - max(D.HSTEPS) - (lo + D.L)
    assert len(X) == n_per_rec * F.shape[1]
    first_t = lo + D.L
    assert np.allclose(X[0], F[first_t - D.L:first_t, 0])
    assert np.allclose(Y[0][2], F[first_t - 1 + D.HSTEPS[1], 0, 0] - F[first_t - 1, 0, 0])
    last_t = hi - max(D.HSTEPS) - 1
    assert last_t - 1 + max(D.HSTEPS) < hi


def test_norm_uses_train_only_and_is_scale_invariant():
    F = synth(); tr = D.make_windows(F, 0, 300); te = D.make_windows(F, 300, 400)
    n = D.Norm().fit(*tr); floor0 = n.floor.copy()
    n.x(te[0], te[1], te[3]); assert np.array_equal(floor0, n.floor)      # transforming test data does not refit
    a = n.x(tr[0], tr[1], tr[3], use_abs=False)
    b = n.x(tr[0] * 7.0, tr[1] * 7.0, tr[3], use_abs=False)               # amplitude-scaled copy of the same windows
    assert np.allclose(a[..., :D.NF], b[..., :D.NF], atol=1e-3)


def test_models_forward_and_shapes():
    x = torch.randn(4, D.L, 3 * D.NF)
    for nme in ("LSTM", "GRU", "TCN", "Transformer", "PCT"):
        mu, lv = make_model(nme, x.shape[-1], 6, L=D.L, n_own=D.NF)(x); assert mu.shape == (4, 6)
    assert make_model("PCT", x.shape[-1], 6, L=D.L, n_own=D.NF)(x)[1].shape == (4, 6)


def test_persistence_and_metrics():
    Y = np.random.randn(50, 6).astype(np.float32); S = np.ones((50, 6), np.float32)
    P = B.Persistence().fit(np.zeros((50, 3, 3)), Y).predict(np.zeros((50, 3, 3)))
    m = M.reg_metrics(P, Y, S, [f"t{i}" for i in range(6)]); assert m["MAE_t0"] > 0
