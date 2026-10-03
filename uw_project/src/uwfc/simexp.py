"""Shared helpers for the simulation experiments (scripts/12-16)."""
from pathlib import Path
import numpy as np, torch
from . import linksim as S, dt as DT, ctrl as C

R = Path(__file__).resolve().parents[2]; SIM = R / "results/sim"; MODELS = SIM / "models"; MODELS.mkdir(parents=True, exist_ok=True)
P = S.Params()
TRAIN_SITES, VAL_SITES, TEST_SITES = ["blue", "red"], ["yellow"], ["black", "purple"]
OOD = S.Regime(log_c_mu=(-0.6, 0.1), near_range=(22.0, 45.0), far_range=(2200.0, 3500.0), noise_shift_db=6.0, name="ood")
_traces = None
def traces():
    global _traces
    if _traces is None: _traces = S.Traces(TRAIN_SITES + VAL_SITES + TEST_SITES)
    return _traces


def gen(seed, sites, B, N=8, regime=None, synth_frac=0.0, mismatch=None, p=P, **bk):
    rng = np.random.default_rng(seed); sc = S.make_scenarios(traces(), B, N, rng, regime=regime or S.Regime(), synth_frac=synth_frac, sites=sites)
    return DT.build(sc, p, rng, mismatch=mismatch, **bk)


def datasets(B_train=64):
    return dict(train=gen(11, TRAIN_SITES, B_train), val=gen(12, VAL_SITES, 16), test_id=gen(1000, TEST_SITES, 48), test_ood=gen(2000, TEST_SITES, 48, regime=OOD))


def model_path(name, seed, tag=""): return MODELS / f"{tag}{name.replace(' ', '_').replace('-', 'm').replace('/', '_')}_s{seed}.pt"


def get_model(name, seed, tr, va, tag="", epochs=20, factory=None, force=False):
    mp = model_path(name, seed, tag); mk = factory or DT.SPECS[name][0]
    if mk is None: return None
    if mp.exists() and not force:
        m = mk(); m.load_state_dict(torch.load(mp)); m.eval(); return m
    m = DT.fit(name, tr, va, seed, epochs=epochs, factory=factory); torch.save(m.state_dict(), mp); return m


def load_ensemble(n=3, tag=""):
    return [get_model("PIGT-DT", s, None, None, tag=tag) for s in range(n)]


def dt_metrics(mu, sd, d):
    """Forecast quality in physical units on one dataset. Optical metrics only where the optical link is active (near node, SNR > -29 dB)."""
    tgt = d["tgt"]; e = mu - tgt; out = {}
    mk_op = [d["near"] & (tgt[..., 1] > -29), d["near"] & (tgt[..., 3] > -29)]
    for j, nm, m in ((0, "ac_h1", None), (2, "ac_h2", None), (1, "op_h1", mk_op[0]), (3, "op_h2", mk_op[1])):
        x = e[..., j] if m is None else e[..., j][m]; out[f"MAE_{nm}"] = float(np.abs(x).mean()); out[f"RMSE_{nm}"] = float(np.sqrt((x ** 2).mean()))
    dp, dtru = DT.derived(mu[..., 0], mu[..., 1]), DT.derived(tgt[..., 0], tgt[..., 1])
    m0 = mk_op[0]
    out["MAE_pdr_ac"] = float(np.abs(dp["pdr_ac"] - dtru["pdr_ac"]).mean()); out["MAE_pdr_op"] = float(np.abs(dp["pdr_op"][m0] - dtru["pdr_op"][m0]).mean())
    out["MAE_log10ber_ac"] = float(np.abs(np.log10(dp["ber_ac"] + 1e-9) - np.log10(dtru["ber_ac"] + 1e-9)).mean()); out["MAE_log10ber_op"] = float(np.abs(np.log10(dp["ber_op"][m0] + 1e-9) - np.log10(dtru["ber_op"][m0] + 1e-9)).mean())
    out["MAE_thr_op_kbps"] = float(np.abs(dp["thr_op"][m0] - dtru["thr_op"][m0]).mean()); out["MAE_thr_ac_kbps"] = float(np.abs(dp["thr_ac"] - dtru["thr_ac"]).mean())
    if sd is not None:
        from scipy.stats import norm
        z = e / np.maximum(sd, 1e-3); sel = np.ones_like(z, bool); sel[..., 1] = mk_op[0]; sel[..., 3] = mk_op[1]
        nll = 0.5 * np.log(2 * np.pi * np.maximum(sd, 1e-3) ** 2) + 0.5 * z ** 2; crps = np.maximum(sd, 1e-3) * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        out |= dict(NLL=float(nll[sel].mean()), CRPS=float(crps[sel].mean()), cov90=float((np.abs(z) <= 1.645)[sel].mean()), width90=float((2 * 1.645 * sd)[sel].mean()))
    return out
