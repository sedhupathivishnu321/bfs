import os, time, copy, numpy as np, torch, torch.nn as nn
from .models import make_model
from .data import NF

torch.set_num_threads(int(os.environ.get('UWFC_THREADS', 4)))


def seed_all(s):
    np.random.seed(s); torch.manual_seed(s)


def nll(mu, lv, y):
    return (0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv))).mean()


def fit_torch(name, Xtr, Ytr, Xva, Yva, seed, epochs=40, bs=256, lr=2e-3, wd=1e-3, patience=8, hetero_w=0.3, loss='huber', **kw):
    """Normalised targets. Early-stop on val Huber loss. Returns best model (CPU, eval)."""
    seed_all(seed)
    kw.setdefault('n_own', NF)
    m = make_model(name, Xtr.shape[-1], Ytr.shape[1], L=Xtr.shape[1], **kw)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * int(np.ceil(len(Xtr) / bs)); sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    Xt, Yt, Xv, Yv = map(torch.from_numpy, (Xtr, Ytr, Xva, Yva))
    best, bs_, wait = None, 1e9, 0
    hl = nn.SmoothL1Loss(beta=0.5) if loss == 'huber' else nn.L1Loss()
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]; mu, lv = m(Xt[idx]); loss = hl(mu, Yt[idx])
            if lv is not None: loss = loss + hetero_w * nll(mu.detach(), lv, Yt[idx])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        m.eval()
        with torch.no_grad(): v = hl(m(Xv)[0], Yv).item()
        if v < bs_ - 1e-5: bs_, best, wait = v, copy.deepcopy(m.state_dict()), 0
        else:
            wait += 1
            if wait >= patience: break
    m.load_state_dict(best); m.eval(); return m


@torch.no_grad()
def predict_torch(m, X):
    mu, lv = m(torch.from_numpy(X)); return mu.numpy(), (lv.numpy() if lv is not None else None)


def count_params(m): return sum(p.numel() for p in m.parameters())


def flops(m, x):
    from torch.utils.flop_counter import FlopCounterMode
    m.eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad(): m(x)
    return fc.get_total_flops()


def latency_ms(m, x, n=200):
    m.eval()
    with torch.no_grad():
        for _ in range(20): m(x)
        t = time.perf_counter()
        for _ in range(n): m(x)
    return (time.perf_counter() - t) / n * 1e3
