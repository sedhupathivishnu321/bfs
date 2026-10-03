"""Scalability (N nodes) and robustness of all controllers on the SIMULATED link. Needs: DT members, MAPPO and MAPPO+DT seed-0 policies."""
import sys, time, dataclasses, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
from uwfc import ctrl as C, simexp as X, dt as DT, linksim as S
from torch.profiler import profile, ProfilerActivity
ap = argparse.ArgumentParser(); ap.add_argument('--part', default='robust'); a = ap.parse_args()
torch.set_num_threads(3 if a.part == 'robust' else 4); p = X.P
for f in [X.MODELS / "mappo_s0.pt", X.MODELS / "mappo_dt_s0.pt"] + [X.model_path("PIGT-DT", s) for s in range(3)]:
    while not f.exists(): time.sleep(20)
members = X.load_ensemble(3); fin0 = 11; 
def load_mappo(name, with_dt):
    ep = X.gen(1, X.TRAIN_SITES, 2); pred = DT.ens_predict(members, ep, "physics") if with_dt else None; e = C.Episode(ep, 2, p, pred)
    fin = C.obs_features(e, 0, np.ones((2, 8)), np.zeros((2, 8), int), with_dt).shape[-1]; net = C.AC(fin); net.load_state_dict(torch.load(X.MODELS / name)); return net
mappo, mappo_dt = load_mappo("mappo_s0.pt", False), load_mappo("mappo_dt_s0.pt", True)
POL = {"Heuristic": (C.Heuristic(), False, False), "Reactive": (C.Greedy("obs"), False, False), "PIGT-DT": (C.Greedy("dt", use_unc=False), True, False), "MAPPO": (C.MAPPOPolicy(mappo, False), False, False),
       "MAPPO+DT": (C.MAPPOPolicy(mappo_dt, True), True, False), "Oracle": (C.Greedy("oracle"), False, False)}
rows = []; t0 = time.time()
def scen(kind, level, seed=7000, B=48, N=8, regime=None, mismatch=None, lag=None, drop=0.0, extra_noise=0.0, sites=None):
    rng = np.random.default_rng(seed); sc = S.make_scenarios(X.traces(), B, N, rng, regime=regime or S.Regime(), sites=sites or X.TEST_SITES)
    d = DT.build(sc, p, rng, mismatch=mismatch, lag=lag, drop=drop, extra_noise=extra_noise); pred = DT.ens_predict(members, d, "physics"); eps = {}
    for nm, (pol, needs_dt, _) in POL.items():
        ep = C.Episode(d, B, p, pred if (needs_dt or nm == "PIGT-DT") else None); ep.dt_mu = C.Episode(d, B, p, pred).dt_mu; ep.dt_sd = C.Episode(d, B, p, pred).dt_sd
        out = C.rollout(ep, pol, p); s = C.summarize(out); s["goodput_total_kbps"] = float(s["goodput"] * N); s["energy_total_J_step"] = float(s["energy"] * N)
        rows.append(dict(kind=kind, level=level, controller=nm, N=N) | s)
    print(f"[{time.time()-t0:5.0f}s] {kind} {level}", flush=True)
# ---------------- robustness
if a.part == 'robust':
  for dr in (0.0, 0.1, 0.3, 0.5): scen("sensor_dropout", dr, drop=dr)
  for ex in (0.0, 1.0, 2.0, 4.0): scen("noisy_observations", ex, extra_noise=ex)
  for dl in (1, 4, 8, 12, 16): scen("stale_dt", dl * S.STEP_S, lag=dl - 1)
  for nm, mm in {"nominal": {}, "SL -3 dB": dict(sl_db=-3.0), "SL +3 dB": dict(sl_db=3.0), "absorption x1.5": dict(abs_scale=1.5), "optical c x1.5": dict(c_scale=1.5), "optical c x0.7": dict(c_scale=0.7)}.items(): scen("channel_model_mismatch", nm, mismatch=mm)
  for nm, rg in {"nominal": S.Regime(), "noise +6 dB": S.Regime(noise_shift_db=6.0), "noise -6 dB": S.Regime(noise_shift_db=-6.0)}.items(): scen("environment_mismatch", nm, regime=rg)
  for nm, rg in {"seen (4-22 m / 150-1800 m)": S.Regime(), "unseen near 22-45 m": S.Regime(near_range=(22.0, 45.0)), "unseen far 2200-3500 m": S.Regime(far_range=(2200.0, 3500.0))}.items(): scen("unseen_distances", nm, regime=rg)
  for nm, rg in {"seen turbidity (c 0.15-0.41)": S.Regime(), "unseen clearer (c 0.08-0.15)": S.Regime(log_c_mu=(-2.5, -1.9)), "unseen turbid (c 0.55-1.1)": S.Regime(log_c_mu=(-0.6, 0.1))}.items(): scen("unseen_turbidity", nm, regime=rg)
  for st in ("black", "purple", "yellow", "blue", "red"): scen("unseen_acoustic_site", st, sites=[st], seed=7100)     # blue/red were in DT+MAPPO training (seen); black/purple/yellow unseen
  pd.DataFrame(rows).to_csv(X.SIM / "robustness_controllers.csv", index=False)
# ---------------- scalability (run alone on an idle CPU: it measures latency)
sc_rows = []; net = members[0]
for N in (() if a.part == 'robust' else (4, 8, 16, 32, 64, 128)):
    d = X.gen(8000, X.TEST_SITES, 16, N=N); x, adj = torch.from_numpy(d["X"][:1]), torch.from_numpy(d["adj"][:1])
    with torch.no_grad():
        for _ in range(5): net(x, adj)
        reps = 50; t = time.perf_counter()
        for _ in range(reps): net(x, adj)
        lat = (time.perf_counter() - t) / reps * 1e3
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof: net(x, adj)
        mem = sum(max(e.self_cpu_memory_usage, 0) for e in prof.key_averages()) / 2**20
    d2 = X.gen(8001, X.TRAIN_SITES, 16, N=N); v2 = X.gen(8002, X.VAL_SITES, 4, N=N); t = time.time(); DT.fit("PIGT-DT", d2, v2, 0, epochs=2); tr_s = (time.time() - t) / 2
    ctl = {}
    for nm, (pol, needs_dt, _) in POL.items():
        rng = np.random.default_rng(9000); sc = S.make_scenarios(X.traces(), 24, N, rng, sites=X.TEST_SITES); dd = DT.build(sc, p, rng); pred = DT.ens_predict(members, dd, "physics")
        ep = C.Episode(dd, 24, p, pred); s = C.summarize(C.rollout(ep, pol, p)); ctl[nm] = s
        sc_rows.append(dict(N=N, controller=nm, reward=s["reward"], pdr=s["pdr"], goodput_per_node=s["goodput"], goodput_total=s["goodput"] * N, energy_per_node=s["energy"], energy_total=s["energy"] * N, viol=s["viol"]))
    sc_rows.append(dict(N=N, controller="_DT", dt_infer_ms=lat, dt_infer_ms_per_node=lat / N, dt_ensemble3_ms=3 * lat, dt_mem_MB=mem, dt_train_s_per_epoch=tr_s, n_train_samples=len(d2["X"]))); print("scale", N, flush=True)
if sc_rows: pd.DataFrame(sc_rows).to_csv(X.SIM / "scalability_sim.csv", index=False)
print("done")
