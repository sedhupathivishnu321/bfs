import torch, torch.nn as nn, torch.nn.functional as Fn


class Head(nn.Module):
    def __init__(self, d, out, hetero):
        super().__init__(); self.mu = nn.Linear(d, out); self.lv = nn.Linear(d, out) if hetero else None
    def forward(self, h):
        return self.mu(h), (self.lv(h).clamp(-6, 4) if self.lv is not None else None)


class LSTMNet(nn.Module):
    def __init__(s, fin, out, h=64, **k):
        super().__init__(); s.r = nn.LSTM(fin, h, batch_first=True, num_layers=1); s.h = Head(h, out, False)
    def forward(s, x): return s.h(s.r(x)[0][:, -1])

class GRUNet(LSTMNet):
    def __init__(s, fin, out, h=64, **k):
        super().__init__(fin, out, h); s.r = nn.GRU(fin, h, batch_first=True)

class TCNNet(nn.Module):
    """Plain dilated causal TCN (standard dense convs), no gating / prior / uncertainty."""
    def __init__(s, fin, out, h=48, **k):
        super().__init__(); s.inp = nn.Conv1d(fin, h, 1)
        s.cs = nn.ModuleList([nn.Conv1d(h, h, 3, dilation=d) for d in (1, 2, 4, 8)]); s.ds = (1, 2, 4, 8); s.h = Head(h, out, False)
    def forward(s, x):
        z = s.inp(x.transpose(1, 2))
        for c, d in zip(s.cs, s.ds): z = z + Fn.relu(c(Fn.pad(z, (2 * d, 0))))
        return s.h(z[:, :, -1])

class TransformerNet(nn.Module):
    def __init__(s, fin, out, d=32, **k):
        super().__init__(); s.p = nn.Linear(fin, d); s.pos = nn.Parameter(torch.zeros(1, 64, d))
        s.t = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 64, 0.1, batch_first=True, norm_first=True), 2, enable_nested_tensor=False)
        s.h = Head(d, out, False)
    def forward(s, x): return s.h(s.t(s.p(x) + s.pos[:, :x.shape[1]])[:, -1])


class DSBlock(nn.Module):
    """Gated depthwise-separable causal conv block (efficient convolution)."""
    def __init__(s, h, dil, gate=True, sep=True):
        super().__init__(); s.dil, s.gate = dil, gate
        k = 3
        if sep:
            s.dw = nn.Conv1d(h, h, k, dilation=dil, groups=h)
        else:
            s.dw = nn.Conv1d(h, h, k, dilation=dil)
        s.pw = nn.Conv1d(h, 2 * h if gate else h, 1); s.n = nn.GroupNorm(1, h); s.hid = h
    def forward(s, z):
        y = s.pw(s.dw(Fn.pad(s.n(z), (2 * s.dil, 0))))
        y = y[:, :s.hid] * torch.sigmoid(y[:, s.hid:]) if s.gate else Fn.gelu(y)
        return z + y


class PCTNet(nn.Module):
    """Physics-prior Causal Temporal network (proposed).
    mean = gated( linear AR/extrapolation prior over own history )  +  gate * nonlinear residual (dilated DS-conv encoder)
    + heteroscedastic log-variance head for calibrated uncertainty.
    Flags allow controlled ablation of each component."""
    def __init__(s, fin, out, L=48, h=32, dil=(1, 2, 4, 8), prior=True, gate=True, sep=True, hetero=True, pool=True, n_own=None, **k):
        super().__init__()
        s.prior_on, s.pool_on = prior, pool
        s.inp = nn.Conv1d(fin, h, 1)
        s.blocks = nn.ModuleList([DSBlock(h, d, gate, sep) for d in dil])
        s.att = nn.Linear(h, 1)
        s.head = Head(2 * h if pool else h, out, hetero)
        n_own = n_own or fin
        s.n_own = n_own
        s.ar = nn.Linear(L * n_own, out) if prior else None     # linear prior on own relative history
        s.alpha = nn.Parameter(torch.tensor(0.0))               # residual scale, starts at 0 => begins as the AR prior
        if prior: nn.init.zeros_(s.ar.weight); nn.init.zeros_(s.ar.bias)
    def forward(s, x):
        z = s.inp(x.transpose(1, 2))
        for b in s.blocks: z = b(z)
        z = z.transpose(1, 2)
        last = z[:, -1]
        h = torch.cat([last, (torch.softmax(s.att(z), 1) * z).sum(1)], -1) if s.pool_on else last
        mu, lv = s.head(h)
        if s.prior_on:
            mu = s.ar(x[:, :, :s.n_own].flatten(1)) + torch.tanh(s.alpha + 1.0) * mu
        return mu, lv


def make_model(name, fin, out, **kw):
    reg = dict(LSTM=LSTMNet, GRU=GRUNet, TCN=TCNNet, Transformer=TransformerNet, PCT=PCTNet)
    return reg[name](fin, out, **kw)
