"""Controllers on the SIMULATED hybrid link (see linksim): fixed / heuristic / reactive-greedy / DT-based (PIGT-DT) / oracle / MAPPO.
Per control step each node picks one of 6 actions (AC|OP|HYB x low|high power). Rewards trade goodput vs energy, latency, reliability violations,
switching, and a battery budget that couples time steps."""
import copy, time
import numpy as np, torch, torch.nn as nn
from . import linksim as S, dt as DT

A_FIX = {"AC-only": 1, "OP-only": 3}   # (power high) fixed baselines


class Episode:
    """Per-step arrays (T,B,N,...) from dt.build output; truth for step k is the h=1 target."""
    def __init__(self, d, B, p, dt_pred=None, fail=None):
        self.p, self.B = p, B; T = len(d["X"]) // B; self.T = T; self.N = d["X"].shape[1]
        rs = lambda a: a.reshape(T, B, *a.shape[1:])
        self.tr_ac, self.tr_op = rs(d["tgt"])[..., 0], rs(d["tgt"])[..., 1]; self.dist = rs(d["d_tgt"])[..., 0]
        self.obs_ac, self.obs_op = rs(d["persist"])[..., 0], rs(d["persist"])[..., 1]
        self.near = rs(d["near"])[0]; self.adj = d["adj"][:B]
        self.dt_mu = rs(dt_pred[0])[..., [0, 1]] if dt_pred is not None else None                 # (T,B,N,2) h=1 prediction (dB)
        self.dt_sd = rs(dt_pred[1])[..., [0, 1]] if (dt_pred is not None and dt_pred[1] is not None) else None
        self.alive_node = np.ones((B, self.N), bool) if fail is None else ~fail
        self.batt0 = p.battery_frac * T * S.STEP_S * p.p_ac[1]


def obs_features(ep, k, batt_frac, prev, with_dt):
    f = [ep.obs_ac[k] / 30.0, ep.obs_op[k] / 30.0, np.log10(ep.dist[k]) / 3.0, batt_frac, np.full_like(batt_frac, k / ep.T)]
    f = np.stack(f, -1); oh = np.eye(S.N_ACT, dtype=np.float32)[prev]; f = np.concatenate([f, oh], -1)
    if with_dt and ep.dt_mu is not None:
        mu = ep.dt_mu[k]; sd = ep.dt_sd[k] if ep.dt_sd is not None else np.zeros_like(mu)
        pa = S.pdr_from_ber(S.ber_bpsk(mu[..., 0]), ep.p.pkt_bits); po = S.pdr_from_ber(S.ber_qam64(mu[..., 1]), ep.p.pkt_bits)
        f = np.concatenate([f, np.stack([mu[..., 0] / 30, mu[..., 1] / 30, sd[..., 0] / 3, sd[..., 1] / 3, pa, po], -1)], -1)
    return f.astype(np.float32)


class Fixed:
    def __init__(self, a): self.a = a; self.name = f"fixed{a}"
    def act(self, ctx): return np.full(ctx["shape"], self.a)


class Heuristic:
    """Threshold rules on observed SNR (what a practitioner would hand-code)."""
    def __init__(self, thr_op=24.0, thr_hyb=14.0, thr_ac=11.0): self.t = (thr_op, thr_hyb, thr_ac)
    def act(self, ctx):
        a_, o_ = ctx["obs_ac"], ctx["obs_op"]; ac_pw = (a_ < self.t[2]).astype(int)
        return np.where(o_ > self.t[0], 2 + 0, np.where(o_ > self.t[1], 4 + 1, ac_pw))


class Greedy:
    """argmax of expected one-step reward given an SNR estimate: est='obs' (reactive), 'dt' (PIGT-DT forecast, optional uncertainty/risk), 'oracle' (true)."""
    def __init__(self, est="obs", use_unc=True, k_mc=8, risk=0.0, pace=True, seed=0):
        self.est, self.use_unc, self.k, self.risk, self.pace = est, use_unc, k_mc, risk, pace; self.rng = np.random.default_rng(seed)
    def act(self, ctx):
        ep, k, p = ctx["ep"], ctx["k"], ctx["p"]
        if self.est == "obs": sa, so = ep.obs_ac[k][None], ep.obs_op[k][None]
        elif self.est == "oracle": sa, so = ep.tr_ac[k][None], ep.tr_op[k][None]
        else:
            mu = ep.dt_mu[k]; sa, so = mu[..., 0][None], mu[..., 1][None]
            if self.use_unc and ep.dt_sd is not None:
                sd = ep.dt_sd[k]; z = self.rng.normal(size=(self.k,) + mu.shape[:-1] + (2,)); sa = mu[..., 0][None] + z[..., 0] * sd[..., 0][None]; so = mu[..., 1][None] + z[..., 1] * sd[..., 1][None]
        price = 1.0
        if self.pace: price = np.clip(1.0 + 2.0 * (ctx["spent_frac"] - k / ep.T), 0.3, 4.0)
        q = np.empty((S.N_ACT,) + ctx["shape"])
        for a in range(S.N_ACT):
            m = S.link_metrics(sa, so, np.full(sa.shape, a), prev_action=ctx["prev"][None], p=p, dist=ep.dist[k][None])
            r = p.w_thr * np.log2(1 + m["goodput"]) - p.lam_e * price * m["energy"] - p.lam_l * m["latency"] - p.kappa * (1 + self.risk) * m["viol"] - p.switch_cost * (a != ctx["prev"][None])
            q[a] = r.mean(0)
        return q.argmax(0)


def rollout(ep, policy, p=S.Params(), rng=None, collect_feats=None, with_dt=False, sample=False):
    """Run one batch of episodes. Returns metrics dict of (T,B,N) arrays (+ obs/act/logp when collect_feats)."""
    T, B, N = ep.T, ep.B, ep.N; prev = np.zeros((B, N), int); batt = np.full((B, N), ep.batt0); alive = np.ones((B, N), bool); spent = np.zeros((B, N))
    keys = ["goodput", "pdr", "ber", "energy", "latency", "viol", "reward"]; out = {k: np.zeros((T, B, N)) for k in keys}; acts = np.zeros((T, B, N), int)
    feats, logps = [], []
    for k in range(T):
        bf = batt / ep.batt0; ctx = dict(ep=ep, k=k, p=p, shape=(B, N), obs_ac=ep.obs_ac[k], obs_op=ep.obs_op[k], prev=prev, spent_frac=spent / ep.batt0)
        if collect_feats is not None:
            f = obs_features(ep, k, bf, prev, with_dt); feats.append(f); a, lp = collect_feats(f, sample); logps.append(lp)
        else: a = policy.act(ctx)
        a = np.asarray(a, int)
        m = S.link_metrics(ep.tr_ac[k], ep.tr_op[k], a, prev_action=prev, p=p, dist=ep.dist[k])
        dead_now = alive & (batt < m["energy"] - 1e-9); alive_k = alive & ~dead_now
        for key in keys:
            v = m[key]
            if key == "reward": v = np.where(alive_k, v, -p.kappa)
            elif key in ("goodput", "pdr", "energy"): v = np.where(alive_k, v, 0.0)
            elif key == "viol": v = np.where(alive_k, v, 1.0)
            out[key][k] = v
        batt = np.where(alive_k, batt - m["energy"], batt); spent = ep.batt0 - batt; alive = alive_k; prev = np.where(alive_k, a, prev); acts[k] = a
    out["act"] = acts; out["alive_node"] = ep.alive_node
    if collect_feats is not None: out["feats"] = np.stack(feats); out["logp"] = np.stack(logps)
    return out


def summarize(out, mask=None):
    w = out["alive_node"][None] if mask is None else mask
    f = lambda k: float((out[k] * w).sum() / (w.sum() * out[k].shape[0] if out[k].ndim == 3 else 1)) if True else 0
    n = w.sum() * out["reward"].shape[0]
    g = lambda k: float((out[k] * w).sum() / n)
    s = {k: g(k) for k in ("reward", "goodput", "pdr", "ber", "energy", "latency", "viol")}
    s["mode_ac"] = float((((out["act"] // 2) == 0) * w).sum() / n); s["mode_op"] = float((((out["act"] // 2) == 1) * w).sum() / n); s["mode_hyb"] = float((((out["act"] // 2) == 2) * w).sum() / n)
    sw = (out["act"][1:] != out["act"][:-1]) * w; s["switch_rate"] = float(sw.sum() / (w.sum() * (out["act"].shape[0] - 1)))
    return s


# ------------------------------------------------------------------ MAPPO (shared actor, mean-field centralised critic)
class AC(nn.Module):
    def __init__(self, fin, h=64):
        super().__init__(); self.pi = nn.Sequential(nn.Linear(fin, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, S.N_ACT))
        self.v = nn.Sequential(nn.Linear(2 * fin, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
    def value(self, f): return self.v(torch.cat([f, f.mean(-2, keepdim=True).expand_as(f)], -1)).squeeze(-1)


def make_actor_fn(net, greedy=False):
    def fn(f, sample):
        with torch.no_grad():
            lg = net.pi(torch.from_numpy(f)); dist = torch.distributions.Categorical(logits=lg)
            a = lg.argmax(-1) if (greedy or not sample) else dist.sample(); return a.numpy(), dist.log_prob(a).numpy()
    return fn


class MAPPOPolicy:
    def __init__(self, net, with_dt, greedy=True): self.net, self.with_dt, self.greedy = net, with_dt, greedy
    def act(self, ctx):
        ep = ctx["ep"]; bf = ep.batt0 - ctx["spent_frac"] * ep.batt0; f = obs_features(ep, ctx["k"], bf / ep.batt0, ctx["prev"], self.with_dt)
        return make_actor_fn(self.net, self.greedy)(f, not self.greedy)[0]


def train_mappo(make_episode, with_dt, iters=300, seed=0, p=S.Params(), lr=1e-3, gamma=0.95, lam=0.95, clip=0.2, epochs=4, log=None, init=None):
    torch.manual_seed(seed); np.random.seed(seed)
    ep0 = make_episode(0); fin = obs_features(ep0, 0, np.ones((ep0.B, ep0.N)), np.zeros((ep0.B, ep0.N), int), with_dt).shape[-1]
    net = init or AC(fin); opt = torch.optim.Adam(net.parameters(), lr=lr); hist = []; t0 = time.time()
    for it in range(iters):
        ep = ep0 if it == 0 else make_episode(it)
        out = rollout(ep, None, p, collect_feats=make_actor_fn(net), with_dt=with_dt, sample=True)
        F = torch.from_numpy(out["feats"]); R = torch.from_numpy(out["reward"].astype(np.float32)); LP = torch.from_numpy(out["logp"]); A = torch.from_numpy(out["act"])
        with torch.no_grad(): V = net.value(F)
        T = F.shape[0]; adv = torch.zeros_like(R); last = torch.zeros_like(R[0])
        for t in reversed(range(T)):
            nv = V[t + 1] if t + 1 < T else torch.zeros_like(V[0]); delta = R[t] + gamma * nv - V[t]; last = delta + gamma * lam * last; adv[t] = last
        ret = adv + V; advn = (adv - adv.mean()) / (adv.std() + 1e-6)
        Ff, Af, LPf, advf, retf = F.reshape(T * ep.B, ep.N, -1), A.reshape(T * ep.B, ep.N), LP.reshape(T * ep.B, ep.N), advn.reshape(T * ep.B, ep.N), ret.reshape(T * ep.B, ep.N)
        n = Ff.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 512):
                idx = perm[i:i + 512]; lg = net.pi(Ff[idx]); d = torch.distributions.Categorical(logits=lg); lp = d.log_prob(Af[idx]); ratio = torch.exp(lp - LPf[idx])
                l_pi = -torch.min(ratio * advf[idx], torch.clamp(ratio, 1 - clip, 1 + clip) * advf[idx]).mean(); l_v = ((net.value(Ff[idx]) - retf[idx]) ** 2).mean()
                loss = l_pi + 0.5 * l_v - 0.01 * d.entropy().mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        s = summarize(out); s["iter"] = it; s["episodes"] = (it + 1) * ep.B; s["time_s"] = time.time() - t0; hist.append(s)
        if log and it % 20 == 0: print(f"[{s['time_s']:5.0f}s] it {it} reward {s['reward']:.3f} pdr {s['pdr']:.3f} E {s['energy']:.3f} viol {s['viol']:.3f}", flush=True)
    return net, hist
