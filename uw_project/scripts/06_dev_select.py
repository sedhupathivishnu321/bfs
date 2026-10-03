"""VALIDATION-ONLY model selection for the proposed network. Test sites/segments are never touched here.
Each candidate config is trained on the train sites and scored on the held-out VALIDATION site of every fold (Protocol A);
the config with the lowest mean validation MAE (avg of gain@1s and gain@2s, plus delay@1s) is frozen for the final test run."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
from uwfc import data as D, train as T

feats = {p.stem: D.load_recording(p)[0] for p in sorted((R / "data/processed/features").glob("*.npz"))}
site = {c: c.split("_")[0] for c in feats}; sites = sorted(set(site.values()))
def stack(ps): ps = [p for p in ps if p is not None]; return tuple(np.concatenate(z) for z in zip(*ps))
def win(codes): return stack([D.make_windows(feats[c], 0, len(feats[c]), 4) for c in codes])
CANDS = {
 "base": dict(),
 "l1": dict(loss="l1"),
 "l1+drop.2": dict(loss="l1", drop=0.2),
 "l1+wd1e-2": dict(loss="l1", wd=1e-2),
}
out = R / 'results/dev_val_selection.csv'
rows = pd.read_csv(out).to_dict('records') if out.exists() else []   # resume
seen = {(r['fold'], r['cand']) for r in rows}; t0 = time.time()
for ts in sites:
    others = [s for s in sites if s != ts]; vs = others[sites.index(ts) % len(others)]
    tr = win([c for c in feats if site[c] not in (ts, vs)]); va = win([c for c in feats if site[c] == vs])
    nrm = D.Norm().fit(*tr); Str, Sva = nrm.tgt_scale(tr[0]), nrm.tgt_scale(va[0])
    Xtr, Xva = nrm.x(tr[0], tr[1], tr[3]), nrm.x(va[0], va[1], va[3]); Ytr, Yva = tr[2] / Str, va[2] / Sva
    for name, kw in CANDS.items():
        if (ts, name) in seen: continue
        kw = dict(kw); ep = kw.pop("epochs", 25); lossn = kw.pop("loss", "huber")
        m = T.fit_torch("PCT", Xtr, Ytr, Xva, Yva, 0, epochs=ep, loss=lossn, **kw)
        Yp, _ = T.predict_torch(m, Xva); e = np.abs(Yp - Yva) * Sva
        rows.append(dict(fold=ts, val_site=vs, cand=name, val_g1=e[:, 2].mean(), val_g2=e[:, 4].mean(), val_d1=e[:, 3].mean()))
        print(f"[{time.time()-t0:5.0f}s] test={ts:7s} val={vs:7s} {name:20s} g1={rows[-1]['val_g1']:.4f} g2={rows[-1]['val_g2']:.4f} d1={rows[-1]['val_d1']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
d = pd.DataFrame(rows); d["score"] = (d.val_g1 + d.val_g2) / 2
rank = d.groupby("cand")[["val_g1", "val_g2", "val_d1", "score"]].mean().sort_values("score"); print(rank.round(4))
best = rank.index[0]; json.dump(dict(best=best, cfg=CANDS[best], ranking=rank.round(5).to_dict()), open(R / "results/dev_selected_config.json", "w"), indent=1)
