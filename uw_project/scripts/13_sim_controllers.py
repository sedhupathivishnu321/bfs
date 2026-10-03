"""Controllers on the SIMULATED hybrid link. --mode baselines | mappo | mappo_dt  (--seed for mappo).
baselines: fixed AC/OP, tuned Heuristic, Reactive greedy, PIGT-DT (DT-forecast greedy; mean and uncertainty-MC variants), Oracle.
mappo / mappo_dt: shared-policy PPO with mean-field critic (optionally with DT forecast features); learning curves are logged per iteration."""
import argparse, sys, time, itertools
from pathlib import Path
import numpy as np, pandas as pd, torch
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
from uwfc import ctrl as C, simexp as X, dt as DT, linksim as S
ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="baselines"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--iters", type=int, default=250); ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args(); torch.set_num_threads(a.threads); p = X.P
while not all(X.model_path("PIGT-DT", s).exists() for s in range(5)): time.sleep(20)       # wait for DT benchmark to finish training members
members = X.load_ensemble(3)

def episode(seed, sites, B=16, N=8, regime=None, dt_members=members, **kw):
    d = X.gen(seed, sites, B, N=N, regime=regime, **kw); pred = DT.ens_predict(dt_members, d, "physics") if dt_members else None
    return C.Episode(d, B, p, pred)

def evaluate(pol, label, rows, series=None, seed=0, extra=None):
    for sname, sites, reg, sd in (("test_id", X.TEST_SITES, None, 3000), ("test_ood", X.TEST_SITES, X.OOD, 4000)):
        ep = episode(sd, sites, B=64, regime=reg); out = C.rollout(ep, pol, p); s = C.summarize(out)
        rows.append(dict(controller=label, testset=sname, seed=seed) | s | (extra or {}))
        if series is not None and sname == "test_id":
            w = out["alive_node"][None]; series[label] = {k: ((out[k] * w).sum((1, 2)) / w.sum()) for k in ("goodput", "energy", "reward", "viol", "pdr")} | {f"mode{m}": ((((out["act"] // 2) == m) * w).sum((1, 2)) / w.sum()) for m in range(3)} | {"act0": out["act"][:, 0, 0]}
rows, series = [], {}
if a.mode == "eval_all":
    # consolidated final evaluation with per-episode rewards (for paired bootstrap); includes every saved MAPPO / MAPPO+DT seed
    epv = episode(777, X.VAL_SITES, B=32); best, bs = None, -1e9
    for t in itertools.product((18, 22, 26), (10, 14, 18), (9, 11, 13)):
        sc_ = C.summarize(C.rollout(epv, C.Heuristic(*t), p))["reward"]
        if sc_ > bs: bs, best = sc_, t
    pols = {"AC-only": C.Fixed(1), "OP-only": C.Fixed(3), "HYB-low": C.Fixed(4), "Heuristic": C.Heuristic(*best), "Reactive": C.Greedy("obs"), "PIGT-DT": C.Greedy("dt", use_unc=False),
            "PIGT-DT (uncertainty MC)": C.Greedy("dt", use_unc=True), "PIGT-DT (risk-averse)": C.Greedy("dt", use_unc=True, risk=0.5), "Oracle": C.Greedy("oracle")}
    per_env = {}
    eps = {sn: episode(sd, sites, B=64, regime=reg) for sn, sites, reg, sd in (("test_id", X.TEST_SITES, None, 3000), ("test_ood", X.TEST_SITES, X.OOD, 4000))}
    _o = C.rollout(eps['test_id'], pols['Oracle'], p); _sw = ((_o['act'][1:] // 2) != (_o['act'][:-1] // 2)).sum(0); bi = np.unravel_index(_sw.argmax(), _sw.shape)   # node whose MODE (AC/OP/HYB) changes most under the oracle (for the switching plot)
    def go(label, pol, seed=0):
        for sn, ep in eps.items():
            out = C.rollout(ep, pol, p); s_ = C.summarize(out); rows.append(dict(controller=label, testset=sn, seed=seed) | s_)
            w = out["alive_node"][None]; per_env[f"{sn}__{label}__s{seed}"] = (out["reward"] * w).sum((0, 2)) / (w.sum((0, 2)) * out["reward"].shape[0] / out["reward"].shape[0] + 1e-9) / out["reward"].shape[0]
            if sn == "test_id" and seed == 0:
                series[label] = {k: ((out[k] * w).sum((1, 2)) / w.sum()) for k in ("goodput", "energy", "reward", "viol", "pdr")} | {f"mode{m}": ((((out["act"] // 2) == m) * w).sum((1, 2)) / w.sum()) for m in range(3)} | {"act0": out["act"][:, bi[0], bi[1]]}
    for nm, pol in pols.items(): go(nm, pol); print(nm, flush=True)
    for fam, nm, wd in (("mappo", "MAPPO", False), ("mappo_dt", "MAPPO+DT", True)):
        for sd_ in range(3):
            f = X.MODELS / f"{fam}_s{sd_}.pt"
            if not f.exists(): continue
            ep0 = eps["test_id"]; fin = C.obs_features(ep0, 0, np.ones((ep0.B, ep0.N)), np.zeros((ep0.B, ep0.N), int), wd).shape[-1]; net = C.AC(fin); net.load_state_dict(torch.load(f)); go(nm, C.MAPPOPolicy(net, wd), seed=sd_); print(nm, sd_, flush=True)
    pd.DataFrame(rows).to_csv(X.SIM / "controllers_eval_all.csv", index=False); np.savez_compressed(X.SIM / "per_env_rewards.npz", **per_env)
    np.savez_compressed(X.SIM / "series_eval_all.npz", **{f"{k}__{m}": v for k, d in series.items() for m, v in d.items()}); print("done"); sys.exit(0)
if a.mode == "baselines":
    # Heuristic thresholds tuned on VALIDATION-site episodes (fairness: baselines get tuning too)
    epv = episode(777, X.VAL_SITES, B=32); best, bs = None, -1e9
    for t in itertools.product((18, 22, 26), (10, 14, 18), (9, 11, 13)):
        s = C.summarize(C.rollout(epv, C.Heuristic(*t), p))["reward"]
        if s > bs: bs, best = s, t
    print("heuristic thresholds", best, round(bs, 3), flush=True)
    pols = {"AC-only": C.Fixed(1), "OP-only": C.Fixed(3), "HYB-low": C.Fixed(4), "Heuristic": C.Heuristic(*best), "Reactive": C.Greedy("obs"), "PIGT-DT": C.Greedy("dt", use_unc=False),
            "PIGT-DT (uncertainty MC)": C.Greedy("dt", use_unc=True), "PIGT-DT (risk-averse)": C.Greedy("dt", use_unc=True, risk=0.5), "Oracle": C.Greedy("oracle")}
    for nm, pol in pols.items(): evaluate(pol, nm, rows, series); print(nm, rows[-2]["reward"], flush=True)
    pd.DataFrame(rows).to_csv(X.SIM / "controllers_baselines.csv", index=False)
    np.savez_compressed(X.SIM / "series_baselines.npz", **{f"{k}__{m}": v for k, d in series.items() for m, v in d.items()})
else:
    with_dt = a.mode == "mappo_dt"; name = "MAPPO+DT" if with_dt else "MAPPO"
    mk = lambda it: episode(5000 + 1000 * a.seed + it, X.TRAIN_SITES, B=16)
    t0 = time.time(); net, hist = C.train_mappo(mk, with_dt, iters=a.iters, seed=a.seed, p=p, log=True)
    pd.DataFrame(hist).to_csv(X.SIM / f"mappo_hist_{a.mode}_s{a.seed}.csv", index=False); torch.save(net.state_dict(), X.SIM / f"models/{a.mode}_s{a.seed}.pt")
    evaluate(C.MAPPOPolicy(net, with_dt), name, rows, series, seed=a.seed, extra=dict(train_s=time.time() - t0))
    pd.DataFrame(rows).to_csv(X.SIM / f"controllers_{a.mode}_s{a.seed}.csv", index=False)
    np.savez_compressed(X.SIM / f"series_{a.mode}_s{a.seed}.npz", **{f"{k}__{m}": v for k, d in series.items() for m, v in d.items()})
print("done", flush=True)
