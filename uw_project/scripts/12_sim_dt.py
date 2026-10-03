"""Train + evaluate the digital-twin forecasters on the SIMULATED hybrid link. --part main | ablate | ratio  (resumable: models cached in results/sim/models)."""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
from uwfc import dt as DT, simexp as X, linksim as S
ap = argparse.ArgumentParser(); ap.add_argument("--part", default="main"); ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--envs", type=int, default=64)
a = ap.parse_args(); torch.set_num_threads(2)
D = X.datasets(a.envs); rows = []; preds = {}; t0 = time.time()
def evaluate(label, members, prior_type, kind, seed, tag, sets=("test_id", "test_ood")):
    for sname in sets:
        d = D[sname]; mu, sd = DT.ens_predict(members, d, prior_type) if len(members) > 1 else DT.predict(members[0], d, prior_type)
        r = dict(part=a.part, model=label, kind=kind, seed=seed, tag=tag, testset=sname) | X.dt_metrics(mu, sd, d); rows.append(r)
        if seed == 0 and kind in ("single", "ens"): preds[f"{sname}__{label}__mu"] = mu; preds[f"{sname}__{label}__sd"] = sd if sd is not None else np.zeros(0)
def log(msg): print(f"[{time.time()-t0:5.0f}s] {msg}", flush=True)
if a.part == "main":
    for name in ("PIGT-DT", "Physics-only", "Data-only", "LSTM", "GRU", "TCN", "Transformer", "GNN"):
        pt = DT.SPECS[name][1]
        if name == "Physics-only":
            evaluate(name, [None], pt, "single", 0, ""); log(name); continue
        ms = []
        for s in range(5 if name == "PIGT-DT" else 3):
            m = X.get_model(name, s, D["train"], D["val"], epochs=a.epochs); ms.append(m)
            if s < 3: evaluate(name, [m], pt, "single", s, "")
            log(f"{name} seed {s}")
        if name == "PIGT-DT":
            for k in (1, 2, 3, 5): evaluate(f"PIGT-DT ensemble x{k}", ms[:k], pt, "ens" if k == 3 else "ens_k", 0, "")
            evaluate("PIGT-DT", ms[:1], pt, "single", 0, "")
elif a.part == "ablate":
    for name in ("- no physics prior", "- no graph", "- no temporal", "- no uncertainty"):
        for s in range(3):
            m = X.get_model(name, s, D["train"], D["val"], epochs=a.epochs); evaluate(name, [m], DT.SPECS[name][1], "single", s, "ablation"); log(f"{name} seed {s}")
elif a.part == "ratio":
    for frac in (0.0, 0.5, 1.0):                      # fraction of SYNTHETIC (AR) acoustic traces replacing the REAL measured traces in DT training
        rng_seed = 11; dtr = X.gen(rng_seed + int(frac * 10), X.TRAIN_SITES, a.envs, synth_frac=frac)
        for s in range(2):
            m = X.get_model("PIGT-DT", s, dtr, D["val"], tag=f"synth{int(frac*100)}_", epochs=a.epochs); evaluate(f"PIGT-DT synth={frac:.2f}", [m], "physics", "single", s, f"ratio{frac}"); log(f"synth {frac} seed {s}")
out = X.SIM / f"dt_metrics_{a.part}.csv"; pd.DataFrame(rows).to_csv(out, index=False)
np.savez_compressed(X.SIM / f"dt_preds_{a.part}.npz", **preds); log("done")
