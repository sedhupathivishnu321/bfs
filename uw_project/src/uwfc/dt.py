"""Graph-temporal Digital Twin (PIGT-DT) and baselines: forecast per-node acoustic/optical SNR 1 and 4 control steps ahead.
Everything here is trained/evaluated on SIMULATED link states (real measured acoustic gain dynamics + assumed link budget; see linksim)."""
import copy, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from . import linksim as S
from .models import DSBlock, Head, make_model

H_STEPS = (1, 4); NOUT = 4          # outputs: [ac h1, op h1, ac h4, op h4]
FIN = 7
CLIP = (-30.0, 60.0)


def _tl(d, p): return 10 * p.spreading_k * np.log10(d) + S.thorp_db_per_km(p.fc_khz) * d / 1000.0


def build(sc, p, rng, mismatch=None, lag=None, drop=0.0, extra_noise=0.0, fail=None):
    """Windows over every (env, time). Truth uses `mismatch` (the 'real world'); observations are noisy truth; the physics prior uses NOMINAL params."""
    lag = p.obs_delay - 1 if lag is None else lag
    s_ac, s_op = S.snr_db(sc, p, mismatch); B, N, T = s_ac.shape
    o_ac = s_ac + rng.normal(0, p.obs_noise_db + extra_noise, s_ac.shape); o_op = np.clip(s_op, *CLIP) + rng.normal(0, p.obs_noise_db + extra_noise, s_op.shape)
    o_op = np.clip(o_op, *CLIP)
    if drop > 0:                                                    # sensor dropout: hold last value
        m = rng.random(o_ac.shape) < drop
        for t in range(1, T): o_ac[..., t] = np.where(m[..., t], o_ac[..., t - 1], o_ac[..., t]); o_op[..., t] = np.where(m[..., t], o_op[..., t - 1], o_op[..., t])
    d = sc["dist"]; L = S.HIST; ts = list(range(L + lag, T - max(H_STEPS) + 1))   # t0: first unobserved index (before lag)
    X, PRI, TGT, ENV, TT, DTG, ADJ = [], [], [], [], [], [], []
    for t0 in ts:
        e = t0 - 1 - lag                                           # last observed index
        sl = slice(e - L + 1, e + 1); a = o_ac[..., sl]; o = o_op[..., sl]; dd = d[..., sl]
        c_hat = np.clip((p.snr_ref_op_db - 40 * np.log10(dd) - o) / (8.686 * dd), 0, 2.0)
        f = np.stack([(a - a[..., -1:]) / 3.0, (o - o[..., -1:]) / 6.0, (np.log10(dd) - np.log10(dd[..., -1:])) * 5, a / 30.0, o / 30.0, c_hat, np.zeros_like(a) + lag / 4.0], -1)   # (B,N,L,F)
        pri, tgt, dtg = [], [], []
        for h in H_STEPS:
            ti = t0 - 1 + h; dt_ = d[..., ti]; d_e = d[..., e]
            pri.append(o_ac[..., e] - (_tl(dt_, p) - _tl(d_e, p)))                                           # physics: geometry/absorption update, persistent noise & gain
            ch = c_hat[..., -1]; pri.append(np.clip(o_op[..., e] - 20 * ch * (dt_ - d_e) * np.log10(np.e) - 40 * np.log10(dt_ / d_e), *CLIP))
            tgt += [s_ac[..., ti], np.clip(s_op[..., ti], *CLIP)]; dtg.append(dt_)
        order = [0, 1, 2, 3]
        X.append(f); PRI.append(np.stack(pri, -1)); TGT.append(np.stack(tgt, -1)); ENV.append(np.arange(B)); TT.append(np.full(B, t0)); DTG.append(np.stack(dtg, -1)); ADJ.append(sc["adj"])
    cat = lambda z: np.concatenate(z, 0)
    X = cat(X).astype(np.float32); out = dict(X=X, prior=cat(PRI).astype(np.float32), tgt=cat(TGT).astype(np.float32), env=cat(ENV), t=cat(TT), d_tgt=cat(DTG).astype(np.float32), adj=cat(ADJ).astype(np.float32),
                                                   last_ac=X[:, :, -1, 3] * 30.0, last_op=X[:, :, -1, 4] * 30.0, near=cat([sc["near"]] * len(ts)), dist_last=np.exp(0) * cat([d[..., t0 - 1 - lag] for t0 in ts]).astype(np.float32))
    out["persist"] = np.stack([out["last_ac"], out["last_op"], out["last_ac"], out["last_op"]], -1).astype(np.float32)
    return out


class DTNet(nn.Module):
    """PIGT-DT residual network. flags: graph, temporal (DS-conv encoder vs last-step MLP), hetero."""
    def __init__(s, fin=FIN, out=NOUT, h=32, graph=True, temporal=True, hetero=True):
        super().__init__(); s.graph, s.temporal, s.h = graph, temporal, h
        if temporal: s.inp = nn.Conv1d(fin, h, 1); s.blocks = nn.ModuleList([DSBlock(h, dl) for dl in (1, 2, 4, 8)])
        else: s.mlp = nn.Sequential(nn.Linear(fin, h), nn.GELU(), nn.Linear(h, h))
        if graph: s.msg = nn.Linear(h, h); s.mix = nn.Linear(2 * h, h)
        s.head = Head(h, out, hetero); nn.init.zeros_(s.head.mu.weight); nn.init.zeros_(s.head.mu.bias)
    def forward(s, x, adj):
        B, N, L, F = x.shape
        if s.temporal:
            z = s.inp(x.reshape(B * N, L, F).transpose(1, 2))
            for b in s.blocks: z = b(z)
            z = z[:, :, -1].reshape(B, N, s.h)
        else: z = s.mlp(x[:, :, -1])
        if s.graph: z = z + Fn.gelu(s.mix(torch.cat([z, torch.bmm(adj, s.msg(z))], -1)))
        mu, lv = s.head(z); return mu, lv


class Wrap(nn.Module):
    """Temporal-only baselines (LSTM/GRU/TCN/Transformer): per node, no graph, no physics."""
    def __init__(s, name, fin=FIN, out=NOUT): super().__init__(); s.m = make_model(name, fin, out, L=S.HIST, n_own=fin)
    def forward(s, x, adj):
        B, N, L, F = x.shape; mu, lv = s.m(x.reshape(B * N, L, F)); return mu.reshape(B, N, -1), None


SPECS = {  # name -> (net factory, prior type)
    "PIGT-DT": (lambda: DTNet(), "physics"), "Physics-only": (None, "physics"),
    "Data-only": (lambda: DTNet(graph=False, hetero=False), "persist"),
    "LSTM": (lambda: Wrap("LSTM"), "persist"), "GRU": (lambda: Wrap("GRU"), "persist"), "TCN": (lambda: Wrap("TCN"), "persist"), "Transformer": (lambda: Wrap("Transformer"), "persist"),
    "GNN": (lambda: DTNet(temporal=False, hetero=False), "persist"),
    "- no physics prior": (lambda: DTNet(), "persist"), "- no graph": (lambda: DTNet(graph=False), "physics"),
    "- no temporal": (lambda: DTNet(temporal=False), "physics"), "- no uncertainty": (lambda: DTNet(hetero=False), "physics"),
}
YS = 5.0   # dB scale for residual targets
def prior_of(d, t): return d['prior'] if t == 'physics' else d['persist']


def fit(name, tr, va, seed, epochs=30, bs=128, lr=2e-3, factory=None):
    torch.manual_seed(seed); np.random.seed(seed)
    mk, pri = SPECS[name]; mk = factory or mk
    if mk is None: return None
    m = mk(); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-3); n = len(tr["X"]); steps = epochs * int(np.ceil(n / bs)); sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    T = lambda d, k: torch.from_numpy(d[k]); P_ = lambda d: prior_of(d, SPECS[name][1])
    Xt, At, Yt = T(tr, "X"), None, torch.from_numpy((tr["tgt"] - P_(tr)) / YS)
    Av = torch.from_numpy(va["adj"]); Xv, Yv = T(va, "X"), torch.from_numpy((va["tgt"] - P_(va)) / YS); At = T(tr, "adj")
    best, bv, wait = None, 1e9, 0; hl = nn.SmoothL1Loss(beta=0.3)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; mu, lv = m(Xt[idx], At[idx]); loss = hl(mu, Yt[idx])
            if lv is not None: loss = loss + 0.2 * (0.5 * (lv + (Yt[idx] - mu.detach()) ** 2 * torch.exp(-lv))).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        m.eval()
        with torch.no_grad(): v = hl(m(Xv, Av)[0], Yv).item()
        if v < bv - 1e-5: bv, best, wait = v, copy.deepcopy(m.state_dict()), 0
        else:
            wait += 1
            if wait >= 7: break
    m.load_state_dict(best); m.eval(); return m


@torch.no_grad()
def predict(m, d, prior_type, bs=512):
    """-> (mean dB (S,N,4), sd dB (S,N,4) or None)"""
    pri = prior_of(d, prior_type)
    if m is None: return pri.copy(), None
    mu, lv = [], []
    for i in range(0, len(d["X"]), bs):
        a, b = m(torch.from_numpy(d["X"][i:i + bs]), torch.from_numpy(d["adj"][i:i + bs])); mu.append(a.numpy()); lv.append(b.numpy() if b is not None else None)
    mu = np.concatenate(mu) * YS + pri
    sd = np.exp(0.5 * np.concatenate(lv)) * YS if lv[0] is not None else None
    return mu, sd


def ens_predict(members, d, prior_type):
    outs = [predict(m, d, prior_type) for m in members]; mus = np.stack([o[0] for o in outs])
    var = mus.var(0) + (np.mean([o[1] ** 2 for o in outs], 0) if outs[0][1] is not None else 0)
    return mus.mean(0), np.sqrt(var) if (outs[0][1] is not None or len(members) > 1) else None


def derived(snr_ac, snr_op, p=S.Params()):
    """BER, PDR, throughput implied by (predicted or true) SNRs for each link at LOW power."""
    ba, bo = S.ber_bpsk(snr_ac), S.ber_qam64(snr_op); pa, po = S.pdr_from_ber(ba, p.pkt_bits), S.pdr_from_ber(bo, p.pkt_bits)
    return dict(ber_ac=ba, ber_op=bo, pdr_ac=pa, pdr_op=po, thr_ac=p.r_ac_kbps * pa, thr_op=p.r_op_kbps * po)
