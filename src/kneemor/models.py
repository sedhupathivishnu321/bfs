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

Extension: MV-MoRE  (Multi-View Mixture-of-Recursive-Experts)
    Same MoR depth-routing loop, but the block's single FFN is replaced by a sparse
    top-k Mixture-of-Experts FFN (``MoEFFN``): several small expert FFNs, a learned
    router sends each token to its top_k experts, and only those experts run on that
    token (true sparse dispatch, not a dense weighted sum -- see MoEFFN docstring).
    This adds label-specialisation capacity (e.g. an expert for focal tears vs. one for
    diffuse OA/effusion) to the depth-routed tokens without materially increasing FLOPs,
    since the frozen/fine-tuned image backbone still dominates study-level compute by
    ~99.9% (results/efficiency.json, README §5.8). Two auxiliary losses keep the experts
    *reliable*: a switch-style load-balancing loss (Fedus et al. 2022) against routing
    collapse onto one expert, and an ST-MoE router z-loss (Zoph et al. 2022) against the
    router-logit blow-up that is the documented cause of MoE training instability. Unlike
    Switch Transformer, tokens are never dropped for a fixed per-expert capacity: with
    only ~60-600 tokens per study, capacity-based dropping would be an avoidable accuracy
    cost for no hardware-batching benefit here, so every token is served by its top_k
    experts. See docs/literature_review.md §4-5 for the research gap and design note, and
    README "Proposed extension: MV-MoRE" for parameter/FLOPs accounting and ablations.

Baselines on identical tokens:
    MeanPoolMLP, ABMIL (gated attention MIL), TransformerHead (R unshared blocks).
Ablations are configurations of MVMoR / MVMoRE (see ``build``).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_LABELS = 12


class MoEFFN(nn.Module):
    """Sparse top-k Mixture-of-Experts FFN (see module docstring for MV-MoRE).

    Each token is dispatched to exactly ``top_k`` of ``n_experts`` small FFNs (true
    sparse compute: an expert only ever sees the tokens routed to it, via boolean-mask
    gather, mirroring the gather/scatter pattern MVMoR already uses for MoR depth
    routing) and its output is the softmax-renormalised weighted sum of those experts.
    Exposes ``aux_loss`` (switch-style load-balancing, Fedus et al. 2022 / Mixtral) and
    ``z_loss`` (ST-MoE router z-loss, Zoph et al. 2022, penalises large router logits --
    the documented cause of MoE training instability) after every forward call, for the
    caller to add to the task loss with small weights.
    """

    def __init__(self, d: int, ffn: int, drop: float, n_experts: int = 4, top_k: int = 2,
                 expert_drop: float = 0.0):
        super().__init__()
        assert 1 <= top_k <= n_experts
        self.n_experts, self.top_k, self.expert_drop = n_experts, top_k, expert_drop
        self.router = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(drop), nn.Linear(ffn, d))
            for _ in range(n_experts)
        ])
        self.aux_loss = torch.zeros(())
        self.z_loss = torch.zeros(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])                                   # [N, d]
        logits = self.router(flat)                                       # [N, E]
        probs = logits.softmax(-1)
        topv, topi = probs.topk(self.top_k, dim=-1)                       # [N, k]
        topv = topv / topv.sum(-1, keepdim=True).clamp_min(1e-9)          # renormalise over the kept experts
        if self.training and self.expert_drop > 0:                       # expert dropout: forces redundancy,
            keep = torch.rand_like(topv) > self.expert_drop               # not relying on a single expert
            keep[..., 0] = True                                          # never drop every expert for a token
            topv = topv * keep / keep.float().mean(-1, keepdim=True).clamp_min(1e-9)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            for slot in range(self.top_k):
                sel = topi[:, slot] == e
                if sel.any():
                    out[sel] = out[sel] + topv[sel, slot:slot + 1] * expert(flat[sel])
        importance = probs.mean(0)                                        # P_i: mean router probability
        frac = torch.stack([(topi == e).any(-1).float().mean() for e in range(self.n_experts)])  # f_i: usage share
        self.aux_loss = self.n_experts * (importance * frac).sum()        # switch load-balance loss
        self.z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()       # ST-MoE router z-loss
        return out.reshape(shape)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int, drop: float, moe: dict | None = None,
                 aux_weight: float = 0.01, z_weight: float = 0.001):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ffn = MoEFFN(d, ffn, drop, **moe) if moe else nn.Sequential(
            nn.Linear(d, ffn), nn.GELU(), nn.Dropout(drop), nn.Linear(ffn, d))
        self.drop = nn.Dropout(drop)
        self.aux_weight, self.z_weight = aux_weight, z_weight
        self.last_moe_loss = None  # set on every residual() call when ffn is a MoEFFN

    def residual(self, h, pad):
        """f(h): the block's residual branch, so Block(h) = h + f(h)."""
        x = self.n1(h)
        a = self.attn(x, x, x, key_padding_mask=pad, need_weights=False)[0]
        u = h + self.drop(a)
        y = self.ffn(self.n2(u))
        if isinstance(self.ffn, MoEFFN):
            self.last_moe_loss = self.aux_weight * self.ffn.aux_loss + self.z_weight * self.ffn.z_loss
        return u + self.drop(y) - h


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
                 plane_emb=True, drop=0.1, moe: dict | None = None,
                 moe_aux_weight: float = 0.01, moe_z_weight: float = 0.001):
        super().__init__()
        self.tok = Tokenizer(c_in, d, P, D, G, plane_emb)
        n_blocks = 1 if shared else recursions
        self.blocks = nn.ModuleList([Block(d, heads, ffn_mult * d, drop, moe=moe, aux_weight=moe_aux_weight,
                                           z_weight=moe_z_weight) for _ in range(n_blocks)])
        self.R, self.shared, self.routing = recursions, shared, routing
        self.capacity = list(capacity) if routing else [1.0] * recursions
        self.routers = nn.ModuleList([nn.Linear(d, 1) for _ in range(recursions)]) if routing else None
        self.norm = nn.LayerNorm(d)
        self.dec = LabelQueryDecoder(d, heads, ffn_mult * d, drop) if decoder == "query" else MeanDecoder(d)
        self.last_depth = None      # [B, T] number of recursions each token received (for analysis)
        self.last_moe_aux = None    # scalar: mean MoE (load-balance + z-loss) over recursion steps run, or None

    def forward(self, x, plane_mask):
        h, pad = self.tok(x, plane_mask)
        B, T, d = h.shape
        depth = torch.zeros(B, T, device=h.device)
        active = ~pad  # tokens still eligible for further recursion
        moe_losses = []
        for r in range(self.R):
            blk = self.blocks[0 if self.shared else r]
            if not self.routing:
                h = h + blk.residual(h, pad)
                depth += (~pad).float()
                if blk.last_moe_loss is not None:
                    moe_losses.append(blk.last_moe_loss)
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
            if blk.last_moe_loss is not None:
                moe_losses.append(blk.last_moe_loss)
            h = h.scatter_add(1, idx[..., None].expand(-1, -1, d), upd)
            new_active = torch.zeros_like(active).scatter(1, idx, sel_valid)
            depth += new_active.float()
            active = new_active
        self.last_depth = depth.detach()
        self.last_moe_aux = torch.stack(moe_losses).mean() if moe_losses else None
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


class HybridMVMoR(nn.Module):
    """Hybrid global-local head (MV-MoR-H).

    Motivation (measured in results/): the per-plane mean-pool MLP is the strongest
    head on OOF AUC, while token attention (MV-MoR) is competitive on gold and on
    localised labels. The two capture different evidence:
        global branch  z_g = MLP([mean_{d,g} x_p]_{p=1..3})      diffuse findings (effusion, OA)
        local branch   z_l = MV-MoR(x)                             focal findings (tears, cysts)
    Fusion is a learned per-label convex gate on the logits:
        z = a * z_g + (1 - a) * z_l,  a = sigmoid(g_label)  (initialised 0.5)
    so each label chooses its own mix and the gate is directly interpretable.
    Deep supervision: during training each branch also gets its own BCE loss
    (weight `aux`), which keeps both branches predictive instead of one branch
    dominating through the gate.
    """

    def __init__(self, aux: float = 0.5, gate: bool = True, **kw):
        super().__init__()
        self.glob = MeanPoolMLP()
        self.loc = MVMoR(**kw)
        self.g = nn.Parameter(torch.zeros(N_LABELS), requires_grad=gate)
        self.aux = aux
        self.aux_logits = None
        self.last_moe_aux = None  # forwarded from the local (MVMoR/MVMoRE) branch, if it uses MoE

    def alpha(self):
        return torch.sigmoid(self.g)

    def forward(self, x, plane_mask):
        zg, zl = self.glob(x, plane_mask), self.loc(x, plane_mask)
        a = self.alpha()
        self.aux_logits = (zg, zl) if self.training and self.aux > 0 else None
        self.last_moe_aux = self.loc.last_moe_aux
        return a * zg + (1 - a) * zl


def build(name: str, **kw) -> nn.Module:
    if name == "hybrid":
        return HybridMVMoR(**kw)
    if name == "hybrid_noaux":                      # ablation: no deep supervision
        return HybridMVMoR(aux=0.0, **kw)
    if name == "hybrid_fixedgate":                  # ablation: fixed 0.5/0.5 logit average, trained jointly
        return HybridMVMoR(gate=False, **kw)
    if name == "hybrid_more":                       # hybrid with an MV-MoRE (MoE) local branch
        return HybridMVMoR(moe=dict(n_experts=4, top_k=2), **kw)
    presets = {
        # proposed (MoR only)
        "mvmor": dict(),
        # ablations of MV-MoR (one factor at a time)
        "mvmor_norouting": dict(routing=False),                 # shared recursive transformer, full depth
        "mvmor_r1": dict(recursions=1, capacity=(1.0,)),        # single pass (no recursion)
        "mvmor_meandec": dict(decoder="mean"),                  # no label queries
        "mvmor_noplane": dict(plane_emb=False),                 # no view identity
        "mvmor_unshared": dict(shared=False),                   # routing but unshared weights
        # proposed extension: MV-MoRE = MV-MoR + sparse top-k Mixture-of-Experts FFN
        "mvmore": dict(moe=dict(n_experts=4, top_k=2)),
        # ablations of MV-MoRE (one factor at a time, mirroring the MV-MoR ablation table)
        "mvmore_e2": dict(moe=dict(n_experts=2, top_k=1)),                    # fewer, less redundant experts
        "mvmore_e8": dict(moe=dict(n_experts=8, top_k=2)),                    # more experts, same top_k
        "mvmore_top1": dict(moe=dict(n_experts=4, top_k=1)),                  # hard top-1 (Switch-style) routing
        "mvmore_expdrop": dict(moe=dict(n_experts=4, top_k=2, expert_drop=0.1)),  # + expert dropout
        "mvmore_unshared": dict(shared=False, moe=dict(n_experts=4, top_k=2)),    # routing+MoE, unshared weights
        "mvmore_nozloss": dict(moe=dict(n_experts=4, top_k=2), moe_z_weight=0.0),  # drop the stability term
        "mvmore_nobalance": dict(moe=dict(n_experts=4, top_k=2), moe_aux_weight=0.0),  # drop load-balancing
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
