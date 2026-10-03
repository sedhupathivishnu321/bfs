"""Robustness of the trained seed-0 models (artifacts from 07_final.py) under test-time input corruption, Protocol A folds.
Corruptions are applied to RAW test windows before the (train-fitted) normaliser; targets are never altered.
 - sensor dropout : each input time step is missing with prob p (hold-last-value imputation)
 - observation noise : Gaussian noise with std = sigma x (window volatility) added to every channel
 - stale DT : the newest k seconds of observations are unavailable (last-known value held)
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch, joblib
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
from uwfc import data as D, protocol as P, train as T
from uwfc.models import make_model
cfg = json.load(open(R / "results/dev_selected_config.json"))["cfg"]
feats, site, sites = P.load_all(); rng = np.random.default_rng(0)

def hold(X, mask):               # mask True = missing; hold last available value (first step never missing)
    X = X.copy(); mask = mask.copy(); mask[:, 0] = False
    for t in range(1, X.shape[1]):
        m = mask[:, t]; X[m, t] = X[m, t - 1]
    return X

def corrupt(X, C, kind, v):
    if v == 0: return X, C
    if kind == "dropout":
        return hold(X, rng.random(X.shape[:2]) < v), hold(C, rng.random(C.shape[:2]) < v)
    if kind == "noise":
        s = np.diff(X, axis=1).std(1, keepdims=True); sc = np.diff(C, axis=1).std(1, keepdims=True)
        return X + rng.normal(size=X.shape) * v * s, C + rng.normal(size=C.shape) * v * sc
    if kind == "stale":
        k = int(round(v * D.RATE)); m = np.zeros(X.shape[:2], bool); m[:, X.shape[1] - k:] = True
        return hold(X, m), hold(C, m)
LEVELS = dict(dropout=[0, .1, .3, .5, .7], noise=[0, .1, .25, .5, 1.0], stale=[0, .5, 1, 2, 4])
rows = []
for fold in sites:
    ad = R / f"results/artifacts/A_{fold}"
    if not (ad / "predictions.npz").exists(): continue
    tr, va, te = P.fold_data(feats, site, sites, fold); nrm = D.Norm().fit(*tr); S = nrm.tgt_scale(te[0]); Y = te[2] / S
    fin = 3 * D.NF + 0
    Xn = nrm.x(te[0], te[1], te[3]); fin = Xn.shape[-1]
    models = {}
    for k in range(3):
        m = make_model("PCT", fin, 6, L=D.L, n_own=D.NF, drop=cfg.get("drop", 0.0)); m.load_state_dict(torch.load(ad / f"PCT_member{k}.pt")); models[f"PCT_member{k}"] = m.eval()
    for nm in ("LSTM", "GRU", "TCN", "Transformer"):
        m = make_model(nm, fin, 6, L=D.L, n_own=D.NF); m.load_state_dict(torch.load(ad / f"{nm}.pt")); models[nm] = m.eval()
    gbm = joblib.load(ad / "GBM.joblib")
    for kind, lv in LEVELS.items():
        for v in lv:
            Xc, Cc = corrupt(te[0], te[1], kind, v); X = nrm.x(Xc, Cc, te[3]); out = {}
            pm = [T.predict_torch(models[f"PCT_member{k}"], X)[0] for k in range(3)]
            out["PCT-E (ours)"] = np.mean(pm, 0); out["PCT-v2 (single)"] = pm[0]
            for nm in ("LSTM", "GRU", "TCN", "Transformer"): out[nm] = T.predict_torch(models[nm], X)[0]
            out["GBM"] = gbm.predict(X); out["Persistence"] = np.zeros_like(Y)
            for nm, p in out.items():
                e = np.abs(p - Y) * S
                rows.append(dict(fold=fold, kind=kind, level=v, model=nm, MAE_g1=e[:, 2].mean(), MAE_g2=e[:, 4].mean(), MAE_d1=e[:, 3].mean()))
    print("robustness done", fold, flush=True)
    pd.DataFrame(rows).to_csv(R / "results/robustness.csv", index=False)
