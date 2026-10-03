"""Controller-level ablations / stress tests of the DT-based controller on the SIMULATED link: components, ensemble size, real/sim data ratio,
model mismatch, DT staleness, node failure, risk constraint. Reference = PIGT-DT (3-member ensemble mean forecast, greedy planner)."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
from uwfc import ctrl as C, simexp as X, dt as DT, linksim as S
torch.set_num_threads(3); p = X.P
while not all(X.model_path("PIGT-DT", s).exists() for s in range(5)): time.sleep(20)
def load(name, tag="", n=3):
    out = []
    for s in range(n):
        mp = X.model_path(name, s, tag)
        while not mp.exists(): time.sleep(20)
        out.append(X.get_model(name, s, None, None, tag=tag))
    return out
full = load("PIGT-DT"); full5 = [X.get_model("PIGT-DT", s, None, None) for s in range(5)]
rows = []
def run(exp, variant, level, members, ptype, pol, B=64, seed=3000, regime=None, mismatch=None, lag=None, fail_frac=0.0, drop=0.0, extra_noise=0.0, ctrl_p=None):
    rng = np.random.default_rng(seed); sc = S.make_scenarios(X.traces(), B, 8, rng, regime=regime or S.Regime(), sites=X.TEST_SITES)
    d = DT.build(sc, p, rng, mismatch=mismatch, lag=lag, drop=drop, extra_noise=extra_noise); fail = None
    if fail_frac > 0:
        fail = rng.random((B, 8)) < fail_frac; m = fail[d["env"]]; d["X"][m] = 0; d["persist"][m] = 0; d["prior"][m] = 0
    pred = DT.ens_predict(members, d, ptype) if members else None
    ep = C.Episode(d, B, p, pred, fail=fail); out = C.rollout(ep, pol, p); s = C.summarize(out)
    rows.append(dict(experiment=exp, variant=variant, level=level) | s); print(f"{exp:14s} {variant:26s} {str(level):8s} reward {s['reward']:.3f} viol {s['viol']:.3f}", flush=True)
mean_pol = C.Greedy("dt", use_unc=False)
# 1) component ablation (controller reward with each ablated DT)
run("component", "PIGT-DT (full)", 0, full, "physics", mean_pol); run("component", "Reactive (no DT)", 0, None, "physics", C.Greedy("obs")); run("component", "Oracle", 0, None, "physics", C.Greedy("oracle"))
for nm in ("- no physics prior", "- no graph", "- no temporal", "- no uncertainty"):
    run("component", nm, 0, load(nm), DT.SPECS[nm][1], mean_pol)
run("component", "+ uncertainty MC (full DT)", 0, full, "physics", C.Greedy("dt", use_unc=True))
# 2) risk constraint in the planner objective (kappa=0 removes the reliability-violation term)
class NoRisk(C.Greedy):
    def act(self, ctx):
        import dataclasses; ctx = dict(ctx); ctx["p"] = dataclasses.replace(ctx["p"], kappa=0.0); return super().act(ctx)
run("risk", "with risk constraint", 0, full, "physics", mean_pol); run("risk", "without risk constraint", 0, full, "physics", NoRisk("dt", use_unc=False))
run("risk", "risk-averse (+50% violation weight, uncertainty MC)", 0, full, "physics", C.Greedy("dt", use_unc=True, risk=0.5))
# 3) ensemble size
for k in (1, 2, 3, 5): run("ensemble", f"M={k}", k, full5[:k], "physics", mean_pol)
# 4) real vs synthetic acoustic traces used to train the DT
for frac in (0, 50, 100):
    ms = [X.get_model("PIGT-DT", s, None, None, tag=f"synth{frac}_") if X.model_path("PIGT-DT", s, f"synth{frac}_").exists() else None for s in range(2)]
    if all(m is not None for m in ms): run("real_sim_ratio", f"{100-frac}% real traces", 100 - frac, ms, "physics", mean_pol)
# 5) model mismatch (the 'real world' differs from the DT's nominal link budget)
MM = {"nominal": {}, "SL -3 dB": dict(sl_db=-3.0), "SL +3 dB": dict(sl_db=3.0), "absorption x1.5": dict(abs_scale=1.5), "noise +6 dB": dict(noise_db=6.0), "optical c x1.5": dict(c_scale=1.5), "optical c x0.7": dict(c_scale=0.7)}
for nm, mm in MM.items():
    for lab, mem, pol in (("PIGT-DT", full, mean_pol), ("Reactive", None, C.Greedy("obs")), ("Oracle", None, C.Greedy("oracle"))): run("mismatch", f"{lab}", nm, mem, "physics", pol, mismatch=mm)
# 6) DT staleness: feedback delay grows (DT trained at the nominal delay of 4 steps)
for delay in (1, 4, 8, 12, 16):
    for lab, mem, pol in (("PIGT-DT", full, mean_pol), ("Reactive", None, C.Greedy("obs")), ("Oracle", None, C.Greedy("oracle"))): run("staleness", lab, delay * S.STEP_S, mem, "physics", pol, lag=delay - 1)
# 7) node failure (failed nodes: all-zero observations; excluded from metrics) with/without graph
nog = load("- no graph")
for ff in (0.0, 0.125, 0.25, 0.5):
    run("node_failure", "PIGT-DT (graph)", ff, full, "physics", mean_pol, fail_frac=ff); run("node_failure", "no graph", ff, nog, "physics", mean_pol, fail_frac=ff); run("node_failure", "Reactive", ff, None, "physics", C.Greedy("obs"), fail_frac=ff)
pd.DataFrame(rows).to_csv(X.SIM / "ablation_controllers.csv", index=False); print("done")
