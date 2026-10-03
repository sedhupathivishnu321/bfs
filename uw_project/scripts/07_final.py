"""Final proposed model PCT-E and fairness controls, evaluated with the SAME protocols/splits/normalisation as 02_run_benchmark.py.
Decisions (loss=L1, dropout=0.2) come from validation-only selection (06_dev_select.py -> results/dev_selected_config.json).
PCT-E = mean of 3 PCT members (different seeds) + per-output shrinkage lambda_j in [0,1.5] fitted on the VALIDATION site/segment only.
Controls: single member (PCT-v2), single member + shrink, GBM + same shrink (so the shrink step cannot explain a win by itself).
Usage: python scripts/07_final.py --protocol A|B --seeds 3
"""
import argparse, json, sys, time, joblib, torch
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
from uwfc import data as D, baselines as B, metrics as M, train as T

ap = argparse.ArgumentParser(); ap.add_argument("--protocol", default="A"); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--epochs", type=int, default=25)
a = ap.parse_args()
cfg = json.load(open(R / "results/dev_selected_config.json"))["cfg"]
feats = {p.stem: D.load_recording(p)[0] for p in sorted((R / "data/processed/features").glob("*.npz"))}
site = {c: c.split("_")[0] for c in feats}; sites = sorted(set(site.values()))
names = [f"{n}_{h}s" for h in D.HZ for n in ("gain", "delay")]; GAIN_COL = 2
def stack(ps): ps = [p for p in ps if p is not None]; return tuple(np.concatenate(z) for z in zip(*ps))
def windows(codes, lo=0.0, hi=1.0, stride=4, meta=False):
    parts = []
    for ci, c in enumerate(codes):
        w = D.make_windows(feats[c], int(lo * len(feats[c])), int(hi * len(feats[c])), stride, return_meta=True)
        if w is not None: parts.append(w + (np.full(len(w[0]), ci),))
    z = tuple(np.concatenate(q) for q in zip(*parts))
    return z if meta else z[:4]
CODES = {}
def splits():
    if a.protocol == "A":
        for ts in sites:
            others = [s for s in sites if s != ts]; vs = others[sites.index(ts) % len(others)]
            tc = [c for c in feats if site[c] == ts]; CODES[ts] = tc
            yield ts, windows([c for c in feats if site[c] not in (ts, vs)]), windows([c for c in feats if site[c] == vs]), windows(tc, meta=True)
    else:
        CODES["within"] = list(feats)
        yield "within", windows(list(feats), 0, .60), windows(list(feats), .60, .75), windows(list(feats), .75, 1.0, meta=True)

def fit_lambda(Pva, Yva, Sva):
    grid = np.linspace(0, 1.5, 31); lam = np.zeros(Pva.shape[1])
    for j in range(Pva.shape[1]):
        lam[j] = grid[np.argmin([np.abs(g * Pva[:, j] - Yva[:, j]).__mul__(Sva[:, j]).mean() for g in grid])]
    return lam

out_csv = R / f"results/protocol{a.protocol}_final.csv"; errdir = R / f"results/errs/{a.protocol}_final"; errdir.mkdir(parents=True, exist_ok=True)
rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
seen = {(str(r["fold"]), int(r["seed"])) for r in rows}
san = lambda m: "".join(ch if ch.isalnum() else "_" for ch in m)
t0 = time.time()
for fold, tr, va, te in splits():
    nrm = D.Norm().fit(*tr); thr = float(np.percentile(tr[2][:, GAIN_COL], 15))
    S = {k: nrm.tgt_scale(v[0]) for k, v in (("tr", tr), ("va", va), ("te", te))}
    Xtr, Xva, Xte = (nrm.x(v[0], v[1], v[3]) for v in (tr, va, te)); Ytr, Yva, Yte = tr[2] / S["tr"], va[2] / S["va"], te[2] / S["te"]
    for seed in range(a.seeds):
        if (str(fold), seed) in seen: continue
        preds = {}                                                     # name -> (test pred, val pred, logvar)
        mem_te, mem_va, mem_lv, members = [], [], [], []
        for k in range(3):
            m = T.fit_torch("PCT", Xtr, Ytr, Xva, Yva, seed * 10 + k, epochs=a.epochs, loss=cfg.get("loss", "huber"), drop=cfg.get("drop", 0.0))
            members.append(m); pt, lv = T.predict_torch(m, Xte); pv, _ = T.predict_torch(m, Xva); mem_te.append(pt); mem_va.append(pv); mem_lv.append(lv)
            if k == 0: preds["PCT-v2 (single)"] = (pt, pv, lv)
        preds["PCT-E (ensemble x3)"] = (np.mean(mem_te, 0), np.mean(mem_va, 0), np.mean(mem_lv, 0))
        g = B.GBM(seed).fit(Xtr, Ytr); preds["GBM"] = (g.predict(Xte), g.predict(Xva), None)
        if seed == 0:                                                     # artifacts for plots / robustness (seed 0 only)
            adir = R / f"results/artifacts/{a.protocol}_{fold}"; adir.mkdir(parents=True, exist_ok=True)
            joblib.dump(g, adir / "GBM.joblib")
            for k, mm in enumerate(members): torch.save(mm.state_dict(), adir / f"PCT_member{k}.pt")
            for nm in ("LSTM", "GRU", "TCN", "Transformer"):
                mm = T.fit_torch(nm, Xtr, Ytr, Xva, Yva, 0, epochs=a.epochs); torch.save(mm.state_dict(), adir / f"{nm}.pt")
                pt, _ = T.predict_torch(mm, Xte); pv, _ = T.predict_torch(mm, Xva); preds[nm] = (pt, pv, None)
            preds["Persistence"] = (np.zeros_like(Yte), np.zeros_like(Yva), None)
            np.savez_compressed(adir / "predictions.npz", Yte=Yte, Ste=S["te"], Yva=Yva, Sva=S["va"], t=te[4], r=te[5], rec=te[6], codes=np.array(CODES[fold]),
                                last=te[3], member_te=np.stack(mem_te), member_va=np.stack(mem_va), member_lv=np.stack(mem_lv), thr=thr,
                                **{f"te__{k}": v[0] for k, v in preds.items()}, **{f"va__{k}": v[1] for k, v in preds.items()}, **{f"lv__{k}": v[2] for k, v in preds.items() if v[2] is not None})
        for base in ("PCT-v2 (single)", "PCT-E (ensemble x3)", "GBM"):
            pt, pv, lv = preds[base]; lam = fit_lambda(pv, Yva, S["va"])          # validation-only calibration
            preds[base + " + shrink"] = (pt * lam, pv * lam, lv)
        for mname, (pt, pv, lv) in preds.items():
            r = dict(protocol=a.protocol, suite="final", fold=fold, seed=seed, model=mname, n_train=len(Xtr), n_test=len(Xte))
            r |= M.reg_metrics(pt, Yte, S["te"], names); r |= {f"ev_{k}": v for k, v in M.event_metrics(pt, Yte, S["te"], GAIN_COL, thr).items()}
            r |= M.coverage(pt, lv, Yte, S["te"]); rows.append(r)
            np.save(errdir / f"{fold}__{seed}__{san(mname)}.npy", (np.abs(pt - Yte)[:, GAIN_COL] * S["te"][:, GAIN_COL]).astype(np.float32))
            print(f"[{time.time()-t0:5.0f}s] {fold:7s} s{seed} {mname:26s} MAEg1s={r['MAE_gain_1.0s']:.4f} g2s={r['MAE_gain_2.0s']:.4f} d1s={r['MAE_delay_1.0s']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
print("done")
