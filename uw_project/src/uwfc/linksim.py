"""Hybrid acoustic–optical link SIMULATOR (semi-synthetic).

WHAT IS REAL: the acoustic channel-gain fluctuations g_i(t) (dB, relative to each trace's mean) are taken from the MEASURED channel
recordings (Zenodo 10.5281/zenodo.21287414), one real receiver trace per node; real spatial correlation is kept by assigning
azimuth-neighbouring nodes to neighbouring receivers of the same recording.
WHAT IS ASSUMED (not measured anywhere in this project; see docs/SIMULATION_ASSUMPTIONS.md):
  link budgets (source level, absorption, noise), optical attenuation field and power budget, modem power, packet size,
  rates (optical rate = measured OFDM-UWVC 64-QAM @2 MS/s throughput), traffic/battery model, reward weights.
=> every PDR / energy / latency / throughput / controller result derived from this module is a SIMULATION result.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.special import erfc
from . import data as D

RATE, STRIDE = D.RATE, 4
STEP_S = STRIDE / RATE            # 0.25 s control interval
HIST, EP_LEN = 24, 96             # history window (6 s) and episode length (24 s) in control steps
TOT = HIST + EP_LEN
N_ACT = 6                         # action = mode*2 + power ; mode 0=AC 1=OP 2=HYB ; power 0=low 1=high
MODES = ["AC", "OP", "HYB"]


@dataclass
class Params:
    fc_khz: float = 12.0
    sl_db_per_w: float = 170.8 + 10 * np.log10(0.5)      # source level re 1 µPa @1 m for 1 W electrical (eff 0.5)
    noise_band_db: float = 118.0                         # in-band noise (shipping/harbour, 4 kHz) ASSUMED
    spreading_k: float = 1.5
    p_ac: tuple = (2.0, 10.0)                            # W electrical (low, high)
    p_op: tuple = (3.0, 8.0)
    snr_ref_op_db: float = 90.0                          # electrical SNR at 1 m, P_op low, no attenuation (ASSUMED)
    r_ac_kbps: float = 4.0
    r_op_kbps: float = 4490.0                            # MEASURED throughput (Dataset_6, 64-QAM, 2 MS/s)
    pkt_bits: int = 512
    w_thr: float = 0.6
    lam_e: float = 0.5                                   # reward per Joule
    lam_l: float = 2.0                                   # reward per second of latency
    kappa: float = 2.0                                   # penalty for reliability violation
    switch_cost: float = 0.3
    pdr_min: float = 0.9
    battery_frac: float = 0.45                           # battery = frac x energy of always (AC, high power)
    c_sound: float = 1500.0
    c_light: float = 2.25e8
    obs_noise_db: float = 0.7
    obs_delay: int = 4                                   # feedback delay in control steps (1 s): acoustic propagation + processing ASSUMED


def thorp_db_per_km(f):
    return 0.11 * f**2 / (1 + f**2) + 44 * f**2 / (4100 + f**2) + 2.75e-4 * f**2 + 0.003


def q_func(x): return 0.5 * erfc(x / np.sqrt(2))


def ber_bpsk(snr_db): return q_func(np.sqrt(2 * 10 ** (np.asarray(snr_db) / 10)))


def ber_qam64(snr_db):
    M = 64; s = 10 ** (np.asarray(snr_db) / 10)
    return np.clip((4 / np.log2(M)) * (1 - 1 / np.sqrt(M)) * q_func(np.sqrt(3 * s / (M - 1))), 1e-12, 0.5)


def pdr_from_ber(ber, bits): return np.exp(bits * np.log1p(-np.clip(ber, 0, 0.5 - 1e-9)))


# ---------------------------------------------------------------- scenarios
class Traces:
    """Real measured relative-gain traces grouped by site: dict site -> list of (T, R) arrays on the 16 Hz grid."""
    def __init__(self, sites):
        from . import protocol as P
        feats, site, _ = P.load_all(); self.by_site = {}
        for c, F in feats.items():
            if site[c] in sites:
                g = F[:, :, 0].copy(); g -= g.mean(0, keepdims=True)
                self.by_site.setdefault(site[c], []).append(g.astype(np.float32))
        self.sites = sorted(self.by_site)
    def stats(self):
        return {s: float(np.mean([g.std(0).mean() for g in v])) for s, v in self.by_site.items()}


def synth_trace(rng, T, R, rho=0.985, sigma=1.2):
    g = np.zeros((T, R), np.float32); e = rng.normal(size=(T, R)); g[0] = e[0] * sigma
    for t in range(1, T): g[t] = rho * g[t - 1] + sigma * np.sqrt(1 - rho**2) * e[t]
    return g


@dataclass
class Regime:
    """Environment regime (ranges used for in-distribution training vs OOD tests)."""
    log_c_mu: tuple = (-1.9, -0.9)          # optical attenuation c=exp(mu): 0.15 .. 0.41 /m
    near_range: tuple = (4.0, 22.0)         # optical-capable nodes (m)
    far_range: tuple = (150.0, 1800.0)      # acoustic-only nodes (m)
    near_frac: float = 0.5
    noise_shift_db: float = 0.0
    name: str = "train"


def make_scenarios(traces, n_env, n_nodes, rng, regime=Regime(), synth_frac=0.0, sites=None, real_ratio=None):
    """Returns dict of arrays (B,N,TOT): dist m, gain_ac dB (REAL traces), logc (optical turbidity field), noise dB (B,TOT), adjacency (B,N,N), pos."""
    sites = sites or traces.sites; B, N = n_env, n_nodes; T = TOT * STRIDE
    pos = rng.uniform(-1, 1, size=(B, N, 2)); ang = np.arctan2(pos[..., 1], pos[..., 0])
    near = rng.random((B, N)) < regime.near_frac
    d0 = np.where(near, rng.uniform(*regime.near_range, (B, N)), np.exp(rng.uniform(np.log(regime.far_range[0]), np.log(regime.far_range[1]), (B, N))))
    ph = rng.uniform(0, 2 * np.pi, (B, N)); amp = d0 * rng.uniform(0.0, 0.12, (B, N)); per = rng.uniform(20, 60, (B, N))
    t = np.arange(TOT) * STEP_S
    dist = np.maximum(d0[..., None] + amp[..., None] * np.sin(2 * np.pi * t / per[..., None] + ph[..., None]), 1.0)
    gain = np.zeros((B, N, TOT), np.float32); is_real = np.zeros((B, N), bool)
    for b in range(B):
        use_synth = rng.random() < synth_frac
        s = sites[rng.integers(len(sites))]; recs = traces.by_site[s]
        order = np.argsort(ang[b]); k = rng.integers(len(recs)); rr = recs[k]; off = rng.integers(0, max(rr.shape[0] - T, 1)); ridx = rng.integers(rr.shape[1])
        for j, i in enumerate(order):                       # azimuth neighbours -> neighbouring receivers of the same recording
            if use_synth: g = synth_trace(rng, T, 1)[:, 0]
            else:
                rec = recs[(k + (ridx + j) // rr.shape[1]) % len(recs)]; col = (ridx + j) % rec.shape[1]; o2 = min(off, max(rec.shape[0] - T, 0)); g = rec[o2:o2 + T, col]
                if len(g) < T: g = np.pad(g, (0, T - len(g)), mode="edge")
            gain[b, i] = g[::STRIDE][:TOT]; is_real[b, i] = not use_synth
    # optical attenuation field: log c = mu + spatially correlated OU process (time constant ~15 s)
    mu = rng.uniform(*regime.log_c_mu, (B, 1, 1))
    dd = np.linalg.norm(pos[:, :, None] - pos[:, None], axis=-1); Cov = np.exp(-dd / 0.6) + 1e-6 * np.eye(N); L_ = np.linalg.cholesky(Cov)
    z = np.zeros((B, N, TOT)); eps = np.einsum("bij,bjt->bit", L_, rng.normal(size=(B, N, TOT))); a = np.exp(-STEP_S / 15.0)
    z[..., 0] = eps[..., 0] * 0.35
    for k in range(1, TOT): z[..., k] = a * z[..., k - 1] + 0.35 * np.sqrt(1 - a**2) * eps[..., k]
    # advected turbidity plumes (ASSUMED physics): Gaussian blobs of extra attenuation moving across the area -> upstream neighbours foreshadow downstream nodes
    plume = np.zeros((B, N, TOT)); tt = np.arange(TOT)
    for b in range(B):
        for _ in range(rng.integers(1, 3)):
            A_ = rng.uniform(0.6, 1.4); w_ = rng.uniform(0.35, 0.6); th = rng.uniform(0, 2 * np.pi); v = rng.uniform(0.02, 0.045) * np.array([np.cos(th), np.sin(th)])
            x0 = -1.4 * np.array([np.cos(th), np.sin(th)]) + rng.uniform(-0.3, 0.3, 2) - v * rng.uniform(0, 30)
            cen = x0[None, :] + v[None, :] * tt[:, None]                                     # (TOT,2)
            r2 = ((pos[b][:, None, :] - cen[None]) ** 2).sum(-1); plume[b] += A_ * np.exp(-r2 / (2 * w_**2))
    z = z * (0.2 / 0.35)
    logc = mu + z + plume
    nz = np.zeros((B, TOT)); w = rng.normal(size=(B, TOT)); a2 = np.exp(-STEP_S / 30.0)
    for k in range(1, TOT): nz[:, k] = a2 * nz[:, k - 1] + 1.5 * np.sqrt(1 - a2**2) * w[:, k]
    nz += regime.noise_shift_db
    adj = np.zeros((B, N, N), np.float32); kk = min(3, N - 1)
    for b in range(B):
        for i in range(N): adj[b, i, np.argsort(dd[b, i])[1:kk + 1]] = 1
    adj = np.maximum(adj, adj.transpose(0, 2, 1)); adj /= adj.sum(-1, keepdims=True)
    return dict(dist=dist.astype(np.float32), gain=gain, logc=logc.astype(np.float32), noise=nz.astype(np.float32), adj=adj.astype(np.float32), pos=pos.astype(np.float32), is_real=is_real, near=near)


def snr_db(sc, p=Params(), mismatch=None):
    """True per-node SNRs (low-power setting) at every step. mismatch: dict of perturbations (sl_db, abs_scale, c_scale, noise_db)."""
    mm = mismatch or {}; d = sc["dist"]
    tl = 10 * p.spreading_k * np.log10(d) + thorp_db_per_km(p.fc_khz) * mm.get("abs_scale", 1.0) * d / 1000.0
    s_ac = p.sl_db_per_w + 10 * np.log10(p.p_ac[0]) + mm.get("sl_db", 0.0) - tl - (p.noise_band_db + sc["noise"][:, None, :] + mm.get("noise_db", 0.0)) + sc["gain"]
    c = np.exp(sc["logc"]) * mm.get("c_scale", 1.0)
    s_op = p.snr_ref_op_db - 20 * c * d * np.log10(np.e) - 40 * np.log10(d)
    return s_ac.astype(np.float32), s_op.astype(np.float32)


def link_metrics(s_ac_low, s_op_low, action, prev_action=None, p=Params(), dist=None):
    """Vectorised per-node outcome of `action` (int array). All quantities EXPECTED values (no packet sampling)."""
    mode, pw = action // 2, action % 2
    s_a = s_ac_low + 10 * np.log10(np.where(pw == 1, p.p_ac[1] / p.p_ac[0], 1.0)); s_o = s_op_low + 20 * np.log10(np.where(pw == 1, p.p_op[1] / p.p_op[0], 1.0))
    ber_a, ber_o = ber_bpsk(s_a), ber_qam64(s_o); pa, po = pdr_from_ber(ber_a, p.pkt_bits), pdr_from_ber(ber_o, p.pkt_bits)
    use_a, use_o = (mode != 1), (mode != 0)
    goodput = use_a * p.r_ac_kbps * pa + use_o * p.r_op_kbps * po
    pdr_eff = np.where(mode == 0, pa, np.where(mode == 1, po, 1 - (1 - pa) * (1 - po)))
    ber_eff = np.where(mode == 0, ber_a, np.where(mode == 1, ber_o, np.minimum(ber_a, ber_o)))
    energy = STEP_S * (use_a * np.where(pw == 1, p.p_ac[1], p.p_ac[0]) + use_o * np.where(pw == 1, p.p_op[1], p.p_op[0]))
    lat_a = (dist / p.c_sound if dist is not None else 0) + p.pkt_bits / (p.r_ac_kbps * 1e3); lat_o = (dist / p.c_light if dist is not None else 0) + p.pkt_bits / (p.r_op_kbps * 1e3)
    latency = np.where(mode == 0, lat_a, np.where(mode == 1, lat_o, po * lat_o + (1 - po) * lat_a))   # HYB: optical copy arrives first only if it succeeds, else the acoustic copy (BUGFIX: was min(lat_a, lat_o))
    viol = (pdr_eff < p.pdr_min).astype(np.float32)
    sw = 0.0 if prev_action is None else (action != prev_action).astype(np.float32)
    reward = p.w_thr * np.log2(1 + goodput) - p.lam_e * energy - p.lam_l * latency - p.kappa * viol - p.switch_cost * sw
    return dict(goodput=goodput, pdr=pdr_eff, ber=ber_eff, energy=energy, latency=latency, viol=viol, reward=reward, pdr_a=pa, pdr_o=po)


def battery_budget(p=Params()): return p.battery_frac * EP_LEN * STEP_S * p.p_ac[1]      # Joules per node per episode
