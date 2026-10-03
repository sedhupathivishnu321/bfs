"""Benchmark runner.
Protocol A (primary, cross-environment): leave-one-SITE-out. Test site never seen in training; validation = a different held-out site.
Protocol B (deployment-like): within every recording, chronological split 60/15/25 train/val/test with a purge gap >= max horizon+L.
Usage: python scripts/02_run_benchmark.py --protocol A --suite main --seeds 3
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
from uwfc import data as D, baselines as B, metrics as M, train as T

ap = argparse.ArgumentParser()
ap.add_argument("--protocol", default="A"); ap.add_argument("--suite", default="main", choices=["main", "ablation"])
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--fresh", action="store_true"); ap.add_argument("--suffix", default=""); ap.add_argument("--folds", default="")  # comma list of test sites (A) to restrict
a = ap.parse_args()

feats = {p.stem: D.load_recording(p)[0] for p in sorted((R / "data/processed/features").glob("*.npz"))}
site = {c: c.split("_")[0] for c in feats}
sites = sorted(set(site.values()))
print("recordings", len(feats), "sites", sites, flush=True)
names = [f"{n}_{h}s" for h in D.HZ for n in ("gain", "delay")]
GAIN_COL = 2  # gain @ 1.0 s


def stack(parts):
    parts = [p for p in parts if p is not None]
    return tuple(np.concatenate(z) for z in zip(*parts))


def windows(codes, lo=0.0, hi=1.0, stride=4):
    out = []
    for c in codes:
        F = feats[c]; T_ = len(F); out.append(D.make_windows(F, int(lo * T_), int(hi * T_), stride))
    return stack(out)


def splits():
    if a.protocol == "A":
        tests = a.folds.split(",") if a.folds else sites
        for ts in tests:
            others = [s for s in sites if s != ts]
            vs = others[sites.index(ts) % len(others)]  # deterministic validation site
            tr = [c for c in feats if site[c] not in (ts, vs)]
            yield ts, windows(tr), windows([c for c in feats if site[c] == vs]), windows([c for c in feats if site[c] == ts])
    else:
        yield "within", windows(list(feats), 0, .60), windows(list(feats), .60, .75), windows(list(feats), .75, 1.0)


def main_suite(seed):
    return {"Persistence": ("sk", lambda: B.Persistence()), "AR(ridge)": ("sk", lambda: B.AR()),
            "GBM": ("sk", lambda: B.GBM(seed)), "LSTM": ("nn", dict(name="LSTM")), "GRU": ("nn", dict(name="GRU")),
            "TCN": ("nn", dict(name="TCN")), "Transformer": ("nn", dict(name="Transformer")), "PCT (proposed)": ("nn", dict(name="PCT"))}


def ablation_suite(seed):
    P = lambda **k: ("nn", dict(name="PCT", **k))
    return {"PCT full": P(), "- AR prior": P(prior=False), "- gating": P(gate=False), "- depthwise-sep (dense conv)": P(sep=False),
            "- heteroscedastic head": P(hetero=False), "- attention pooling": P(pool=False),
            "- cross-receiver ctx": P(use_ctx=False), "- absolute level": P(use_abs=False),
            "gain-only inputs": P(only_gain=True), "narrow (h=16)": P(h=16), "wide (h=64)": P(h=64), "shallow (dil 1,4)": P(dil=(1, 4))}


out_csv = R / f"results/protocol{a.protocol}_{a.suite}{a.suffix}.csv"
errdir = R / f"results/errs/{a.protocol}_{a.suite}{a.suffix}"; errdir.mkdir(parents=True, exist_ok=True)
done = {}
if out_csv.exists() and not a.fresh:                       # resume after interruption (container restarts)
    for r_ in pd.read_csv(out_csv).to_dict("records"): done[(str(r_["fold"]), int(r_["seed"]), r_["model"])] = r_
def errfile(fold, seed, mname): return errdir / (f"{fold}__{seed}__" + "".join(ch if ch.isalnum() else "_" for ch in mname) + ".npy")
rows, errs = [], {}
t0 = time.time()
for fold, tr, va, te in splits():
    if tr is None or va is None or te is None: print("skip fold", fold); continue
    nrm = D.Norm().fit(*tr)
    thr = float(np.percentile(tr[2][:, GAIN_COL], 15))          # physical dB threshold from TRAIN only
    S = {k: nrm.tgt_scale(v[0]) for k, v in (('tr', tr), ('va', va), ('te', te))}
    for seed in range(a.seeds):
        suite = (main_suite if a.suite == "main" else ablation_suite)(seed)
        for mname, (kind, spec) in suite.items():
            key = (str(fold), seed, mname)
            if key in done:
                rows.append(done[key]); continue
            kw = dict(spec) if kind == "nn" else {}
            use_abs, use_ctx, only_gain = kw.pop("use_abs", True), kw.pop("use_ctx", True), kw.pop("only_gain", False)
            prep = lambda d: nrm.x(d[0], d[1], d[3], use_abs, use_ctx)
            Xtr, Xva, Xte = prep(tr), prep(va), prep(te)
            if only_gain:
                keep = lambda X: X[..., [0, D.NF, 2 * D.NF] if use_ctx and use_abs else [0]]
                Xtr, Xva, Xte = map(keep, (Xtr, Xva, Xte)); kw["n_own"] = 1
            Ytr, Yva, Yte = tr[2] / S['tr'], va[2] / S['va'], te[2] / S['te']
            t1 = time.time(); lv = None
            if kind == "sk":
                m = spec().fit(Xtr, Ytr); Yp = m.predict(Xte)
            else:
                nm = kw.pop("name"); m = T.fit_torch(nm, Xtr, Ytr, Xva, Yva, seed, epochs=a.epochs, **kw); Yp, lv = T.predict_torch(m, Xte)
            r = dict(protocol=a.protocol, suite=a.suite, fold=fold, seed=seed, model=mname, n_train=len(Xtr), n_test=len(Xte), fit_s=time.time() - t1)
            r |= M.reg_metrics(Yp, Yte, S['te'], names)
            r |= {f"ev_{k}": v for k, v in M.event_metrics(Yp, Yte, S['te'], GAIN_COL, thr).items()}
            r |= M.coverage(Yp, lv, Yte, S['te'])
            rows.append(r); pd.DataFrame(rows).to_csv(out_csv, index=False)
            np.save(errfile(fold, seed, mname), (np.abs(Yp - Yte)[:, GAIN_COL] * S['te'][:, GAIN_COL]).astype(np.float32))
            print(f"[{time.time()-t0:6.0f}s] {fold:7s} s{seed} {mname:28s} MAEg1s={r['MAE_gain_1.0s']:.4f} MAEd1s={r['MAE_delay_1.0s']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
pd.DataFrame(rows).to_csv(out_csv, index=False)
print("done", time.time() - t0)
