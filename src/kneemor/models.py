"""Study-level heads operating on frozen slice tokens [B, P, D, G, C].

Proposed: MV-MoR  (Multi-View Mixture-of-Recursions)
    tokens -> LN+Linear(C->d) + plane/slice/cell embeddings
           -> ONE shared pre-LN transformer block applied R times
              with expert-choice token routing (MoR, Bae et al. 2025):
              at recursion r a sigmoid router keeps the top ceil(cap_r * T)
              still-active tokens; the rest exit early (identity).
              h <- h + g * f(h)   for routed tokens, g = sigmoid(router)
           -> label-query cross-attention decoder (12 queries, Query2Label/ML-Decoder)
           -> per-label group-wise linear logits

Baselines on identical tokens:
    MeanPoolMLP, ABMIL (gated attention MIL), TransformerHead (R unshared blocks).
Ablations are configurations of MVMoR (see ``build``).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_LABELS = 12


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int, drop: float):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(drop), nn.Linear(ffn, d))
        self.drop = nn.Dropout(drop)

    def residual(self, h, pad):
        """f(h): the block's residual branch, so Block(h) = h + f(h)."""
        x = self.n1(h)
        a = self.attn(x, x, x, key_padding_mask=pad, need_weights=False)[0]
        u = h + self.drop(a)
        return u + self.drop(self.ffn(self.n2(u))) - h


class Tokenizer(nn.Module):
    def __init__(self, c_in: int, d: int, P: int, D: int, G: int, plane_emb: bool = True):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(c_in), nn.Linear(c_in, d))
        self.plane = nn.Parameter(torch.zeros(P, 1, 1, d)) if plane_emb else None
        self.slice = nn.Parameter(torch.zeros(P, D, 1, d))
        self.cell = nn.Parameter(torch.zeros(1, 1, G, d))
        for p in (self.plane, self.slice, self.cell):
            if p is not None:
                nn.init.trunc_normal_(p, std=0.02)

    def forward(self, x, plane_mask):
        B, P, D, G, _ = x.shape
        h = self.proj(x) + self.slice + self.cell
        if self.plane is not None:
            h = h + self.plane
        pad = (~plane_mask)[:, :, None, None].expand(B, P, D, G).reshape(B, P * D * G)
        return h.reshape(B, P * D * G, -1), pad


class LabelQueryDecoder(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int, drop: float, n_labels: int = N_LABELS):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.nq, self.nk, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.xattn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(drop), nn.Linear(ffn, d))
        self.w = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_labels))
        self.last_attn = None

    def forward(self, h, pad):
        B = h.shape[0]
        q = self.nq(self.q).unsqueeze(0).expand(B, -1, -1)
        k = self.nk(h)
        a, w = self.xattn(q, k, k, key_padding_mask=pad, need_weights=True, average_attn_weights=True)
        self.last_attn = w.detach()
        z = q + a
        z = z + self.ffn(self.n2(z))
        return (z * self.w).sum(-1) + self.b


class MeanDecoder(nn.Module):
    """Ablation: replace label queries by masked mean pooling + linear."""

    def __init__(self, d: int, n_labels: int = N_LABELS):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.fc = nn.Linear(d, n_labels)

    def forward(self, h, pad):
        m = (~pad).float().unsqueeze(-1)
        return self.fc(self.n((h * m).sum(1) / m.sum(1).clamp(min=1)))


class MVMoR(nn.Module):
    def __init__(self, c_in=512, d=128, heads=4, ffn_mult=2, P=3, D=24, G=4, recursions=3,
                 capacity=(1.0, 0.5, 0.25), shared=True, routing=True, decoder="query",
                 plane_emb=True, drop=0.1):
        super().__init__()
        self.tok = Tokenizer(c_in, d, P, D, G, plane_emb)
        n_blocks = 1 if shared else recursions
        self.blocks = nn.ModuleList([Block(d, heads, ffn_mult * d, drop) for _ in range(n_blocks)])
        self.R, self.shared, self.routing = recursions, shared, routing
        self.capacity = list(capacity) if routing else [1.0] * recursions
        self.routers = nn.ModuleList([nn.Linear(d, 1) for _ in range(recursions)]) if routing else None
        self.norm = nn.LayerNorm(d)
        self.dec = LabelQueryDecoder(d, heads, ffn_mult * d, drop) if decoder == "query" else MeanDecoder(d)
        self.last_depth = None  # [B, T] number of recursions each token received (for analysis)

    def forward(self, x, plane_mask):
        h, pad = self.tok(x, plane_mask)
        B, T, d = h.shape
        depth = torch.zeros(B, T, device=h.device)
        active = ~pad  # tokens still eligible for further recursion
        for r in range(self.R):
            blk = self.blocks[0 if self.shared else r]
            if not self.routing:
                h = h + blk.residual(h, pad)
                depth += (~pad).float()
                continue
            # expert-choice routing among still-active tokens
            score = self.routers[r](h).squeeze(-1)                         # [B, T]
            k = max(1, math.ceil(self.capacity[r] * T))
            masked = score.masked_fill(~active, float("-inf"))
            idx = masked.topk(k, dim=1).indices                            # [B, k]
            sel_valid = torch.gather(active, 1, idx)                       # padding may be picked if few valid
            hs = torch.gather(h, 1, idx[..., None].expand(-1, -1, d))
            g = torch.sigmoid(torch.gather(score, 1, idx)).unsqueeze(-1)
            upd = blk.residual(hs, ~sel_valid) * g * sel_valid.unsqueeze(-1).float()
            h = h.scatter_add(1, idx[..., None].expand(-1, -1, d), upd)
            new_active = torch.zeros_like(active).scatter(1, idx, sel_valid)
            depth += new_active.float()
            active = new_active
        self.last_depth = depth.detach()
        return self.dec(self.norm(h), pad)


class TransformerHead(MVMoR):
    """Baseline: standard (unshared, unrouted) transformer with the same decoder."""

    def __init__(self, **kw):
        kw.update(shared=False, routing=False)
        super().__init__(**kw)


class MeanPoolMLP(nn.Module):
    def __init__(self, c_in=512, hidden=256, P=3, drop=0.2, **_):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(P * c_in), nn.Dropout(drop), nn.Linear(P * c_in, hidden),
                                 nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, N_LABELS))

    def forward(self, x, plane_mask):
        B, P, D, G, C = x.shape
        z = x.float().mean((2, 3)) * plane_mask[..., None].float()   # [B, P, C] (missing plane -> 0)
        return self.net(z.reshape(B, P * C))


class ABMIL(nn.Module):
    """Gated attention MIL (Ilse et al., ICML 2018) over all slice tokens, one attention map per label."""

    def __init__(self, c_in=512, d=128, P=3, D=24, G=4, drop=0.1, **_):
        super().__init__()
        self.tok = Tokenizer(c_in, d, P, D, G, plane_emb=True)
        self.V, self.U = nn.Linear(d, d), nn.Linear(d, d)
        self.w = nn.Linear(d, N_LABELS)
        self.cls = nn.Parameter(torch.randn(N_LABELS, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(N_LABELS))
        self.drop = nn.Dropout(drop)

    def forward(self, x, plane_mask):
        h, pad = self.tok(x, plane_mask)
        h = self.drop(h)
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))       # [B, T, L]
        a = a.masked_fill(pad[..., None], float("-inf")).softmax(1)
        z = torch.einsum("btl,btd->bld", a, h)
        return (z * self.cls).sum(-1) + self.b


def build(name: str, **kw) -> nn.Module:
    presets = {
        # proposed
        "mvmor": dict(),
        # ablations of MV-MoR (one factor at a time)
        "mvmor_norouting": dict(routing=False),                 # shared recursive transformer, full depth
        "mvmor_r1": dict(recursions=1, capacity=(1.0,)),        # single pass (no recursion)
        "mvmor_meandec": dict(decoder="mean"),                  # no label queries
        "mvmor_noplane": dict(plane_emb=False),                 # no view identity
        "mvmor_unshared": dict(shared=False),                   # routing but unshared weights
    }
    if name in presets:
        cfg = {**presets[name], **kw}
        return MVMoR(**cfg)
    if name == "transformer":
        return TransformerHead(**kw)
    if name == "meanmlp":
        return MeanPoolMLP(**kw)
    if name == "abmil":
        return ABMIL(**kw)
    raise KeyError(name)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
